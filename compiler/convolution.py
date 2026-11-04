"""Source-preserving convolution through checked counted window schedules.

Inputs are NHWC i8 and weights are [KH * KW * CI, CO] i8.  All local
storage belongs to this invocation. Completion alternatives share one effect
adapter: conservative drains remain available, while derived native ordering
facts admit checked compute and readout epochs. Border loads and zero writes
remain disjoint source effects within every schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionRhsPhaseLaw,
    ConvolutionRhsResidencyLaw,
    ConvolutionScheduleChoice,
    emit_convolution_schedule,
    plan_convolution_rhs_phases,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionInputRetentionLaw,
    ConvolutionComputeEpochLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionReadoutEpochLaw,
    CountedConvolutionPlan,
    Window2D,
)


def _select_schedule_mode(requested, settings, source, hardware_identity, *,
                          edge, operand_rows, accumulator_rows, gains):
    """Select an explicitly configured geometry region using library predicates.

    Regions carry shared dimension/resource guards, never source names. Empty
    or unmatched regions preserve sync. Installing a region is a separate
    empirical decision; this function establishes no performance qualification.
    """
    modes = ("sync", "issue", "epoch", "readout")
    if requested in modes:
        return requested, {"kind": "explicit", "mode": requested}
    if requested != "regional":
        raise ValueError("CONV_SCHEDULE must be sync, issue, epoch, readout or regional")
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate, ShapePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

    p = source.plan
    w = p.window
    # These values describe the actual original window and selected capacities.
    # Output precision remains a separate source property; gains are not folded.
    resources = dict(array_edge=edge, operand_rows=operand_rows,
        accumulator_rows=accumulator_rows, kernel_points=w.kernel[0]*w.kernel[1],
        reduction_elements=w.kernel[0]*w.kernel[1]*w.input_shape[-1],
        output_pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1],
        stride_height=w.stride[0], stride_width=w.stride[1],
        dilation_height=w.dilation[0], dilation_width=w.dilation[1],
        column_tiles=len(source.column_intervals),
        live_column_tiles=min(len(source.column_intervals), accumulator_rows//p.row_tile),
        uniform_scale=int(len(set(gains)) == 1))
    facts = PolicyFacts(AnalysisStamp(source.identity,
        PipelineContext(hardware=hardware_identity, resources=hardware_identity),
        "counted-convolution"), shapes=(tuple(w.input_shape),
        (w.kernel[0], w.kernel[1], w.input_shape[-1], p.output_channels)),
        resources=tuple(sorted(resources.items())))
    regions = settings.get("conv_schedule_regions", ())
    if not isinstance(regions, (list, tuple)) or len(regions) > 8:
        raise ValueError("convolution regional settings require at most eight guarded alternatives")
    for index, region in enumerate(regions):
        if not isinstance(region, dict) or set(region) - {"mode", "shapes", "resources"}:
            raise ValueError("convolution regions contain only mode and library predicate fields")
        mode = region.get("mode")
        if mode not in modes:
            raise ValueError("convolution region must select an existing schedule")
        guards = tuple(ShapePredicate(**g) for g in region.get("shapes", ()))
        resource_guards = tuple(ResourcePredicate(**g) for g in region.get("resources", ()))
        if any(g.resource not in resources for g in resource_guards):
            raise ValueError("unknown convolution resource predicate")
        guards += resource_guards
        if not guards or len(guards) > 16:
            raise ValueError("convolution regions require one to sixteen predicates")
        if all(g.match(facts) is True for g in guards):
            return mode, {"kind": "regional", "region": index,
                "mode": mode, "resources": resources}
    return "sync", {"kind": "regional", "region": None,
        "mode": "sync", "resources": resources}


@lru_cache(maxsize=4)
def _ordering_admission(facts_path, ordering_path):
    """Reopen selected native dependencies; unknown premises keep real drains.

    These checks mirror the public convolution-epoch recipe, restricted to
    the actual MVIN/MVIN2/MVIN3, PRELOAD and COMPUTE_PRELOADED encoding used
    below.  A dispatch overlap wait is sufficient even where its disjoint
    tracking probe is inconclusive; that residual is preserved in the receipt.
    """
    try:
        raw = Path(facts_path).read_bytes()
        facts = json.loads(raw)
        selected = json.loads(Path(ordering_path).read_bytes())
        if selected["base_facts"]["sha256"] != hashlib.sha256(raw).hexdigest():
            return {"admitted": False, "reason": "ordering record has different base facts"}
        order = selected["ordering"]
        actual_ordering_sha = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
        if actual_ordering_sha != selected["ordering_sha256"]:
            return {"admitted": False, "reason": "selected ordering record hash changed"}
        if order.get("status") != "derived":
            return {"admitted": False, "reason": "ordering facts are not derived"}
        cross = {(r["older"], r["younger"], r["space"]): r for r in order["cross_queue"]}
        accesses, extents = order["accesses"], order["extents"]
        loads = ("MVIN.spad_addr", "MVIN2.spad_addr", "MVIN3.spad_addr")
        readers = ("COMPUTE_PRELOADED.a_spad", "PRELOAD.weight_spad")
        descriptor, readout = "PRELOAD.c_addr", "MVOUT.acc_addr"
        used = (*loads, *readers, descriptor, readout)
        if any(extents[k].get("status") != "derived" for k in used):
            return {"admitted": False, "reason": "native access extents are not derived"}
        required = [(w, r, "scratchpad") for w in loads for r in readers]
        required += [(r, w, "scratchpad") for w in loads for r in readers]
        required += [(w, descriptor, "accumulator") for w in loads]
        required += [(descriptor, readout, "accumulator"), (readout, descriptor, "accumulator")]
        if any(cross.get(k, {}).get("overlap") != "completion" for k in required):
            return {"admitted": False, "reason": "a native local RAW/WAR dependency is unknown"}
        same = order["same_queue"]
        if (order["completion_order"]["execute"].get("order") != "in_order"
            or same["execute"].get("release") not in ("issue", "completion")
            or same["load"].get("release") not in ("issue", "completion")
            or same["store"].get("release") not in ("issue", "completion")):
            return {"admitted": False, "reason": "native issue/completion ordering is unknown"}
        queues = {}
        for row in order["funct_queue"]:
            queues.update({int(k): v for k, v in row.get("by_code", {}).items()})
        # ConfigEx, ConfigMvin and ConfigMvout must enter their own command lanes.
        if any(queues.get(k) != q for k, q in ((0, "execute"), (1, "load"), (2, "store"))):
            return {"admitted": False, "reason": "configuration does not share its native issue lane"}
        classes = (facts.get("facts", facts)["timing_structure"]["command_classes"]["classes"])
        capacities = {}
        for q, minimum in (("load", 1), ("execute", 2), ("store", 1)):
            entry = classes[q]["entries"]
            if entry.get("status") != "derived" or type(entry.get("value")) is not int or entry["value"] < minimum:
                return {"admitted": False, "reason": "finite native queue capacity is unknown"}
            capacities[q] = entry["value"]
        return {"admitted": True, "ordering_sha256": selected["ordering_sha256"],
            "required_completion_edges": [list(k) for k in required],
            "pending_compute_pairs": capacities["execute"] // 2,
            "pending_readouts": capacities["store"],
            "residual": [list(k) for k in required if cross[k].get("tracked") is None],
            "program_premises": "private disjoint rows; adjacent descriptor/data pair; load OR zero once; full ownership drains"}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"admitted": False, "reason": str(error)}


@lru_cache(maxsize=4)
def _resources(facts_path):
    from merlin.targetgen.address_space import (
        accumulator_store, derive_address_space, operand_store,
    )

    raw = Path(facts_path).read_bytes()
    facts = json.loads(raw)
    space = derive_address_space("gemmini", facts=facts)
    operand, accumulator = operand_store(space), accumulator_store(space)
    if operand.store is None or accumulator.store is None:
        raise ValueError("convolution requires derived operand and accumulator stores")
    edge = min(space.array_rows, space.array_cols)
    if operand.store.elems_per_row() < edge or accumulator.store.elems_per_row() < edge:
        raise ValueError("convolution primitive does not fit the selected store rows")
    return edge, operand.capacity_rows(), accumulator.capacity_rows(), hashlib.sha256(raw).hexdigest()


def _emit_pointwise_dense(enc, inp, weights, out, view, out_dtype, gains,
                          relu, bias, selected, selection):
    from dense import emit_dense
    enc.fence()
    enc.configure_execute()
    receipt = emit_dense(enc, inp, weights, out, *view.dimensions,
        out_dtype=out_dtype, scale=gains if len(set(gains)) != 1 else gains[0],
        relu=relu, bias=bias, alternative=selected)
    enc.fence()
    return {"pointwise_dense_view": view.record(), "pointwise_schedule": selected,
        "pointwise_selection": dict(selection, selected=selected), "dense_emission": receipt}


def emit_conv(enc, inp, weights, out, input_shape, weight_shape,
              strides=(1, 1), padding=(0, 0), dilation=(1, 1),
              out_dtype="i32", scale=1.0, relu=False, bias=None, *,
              kernel=None, output_channels=None, source_identity=None,
              facts_path=None, retain_weights=True, retain_input=True,
              schedule_mode=None, ordering_path=None, compute_pairs=None,
              pointwise_mode=None, readout_mode=None, readout_scratch=None,
              readout_scratch_bytes=0, input_transfer_mode=None, output_group_mode=None,
              rhs_transport_mode=None, array_output_rows=None):
    """Emit one original convolution and return its checked schedule receipt.

    ``weight_shape`` may be the original KH,KW,CI,CO declaration or the
    physical reduction,CO matrix when ``kernel=(KH,KW)`` is supplied.  Scale
    is a scalar or the complete original per-column vector; heterogeneous
    scales retain individual scalar readouts.  ``enc`` supplies the actual
    instructions and a FnBuilderControlAdapter as ``index_builder``. The
    default raw-i32 pointwise route uses a checked no-copy dense view only
    for unpadded unit-stride 1x1 windows with no bias, scale or activation.
    Shared dense capacity/order preflight and identical hardware facts remain
    mandatory; other contracts retain this window schedule.
    """
    shape = tuple(map(int, input_shape))
    if len(shape) != 4:
        raise ValueError("convolution input must be NHWC")
    ws = tuple(map(int, weight_shape))
    if len(ws) == 4:
        kh, kw, ci, co = ws
    elif len(ws) == 2 and kernel is not None:
        kh, kw = tuple(map(int, kernel))[:2]
        ci, co = shape[-1], ws[1]
        if ws[0] != kh * kw * ci:
            raise ValueError("convolution flattened weights disagree with original kernel")
    else:
        raise ValueError("convolution requires original kernel geometry")
    if ci != shape[-1] or output_channels is not None and int(output_channels) != co:
        raise ValueError("convolution source channel extents disagree")
    pad = tuple(map(int, padding))
    if len(pad) == 2:
        pad = pad + pad
    if len(pad) != 4:
        raise ValueError("convolution padding must be top,left,bottom,right")
    window = Window2D(shape, (kh, kw), tuple(strides), tuple(dilation), pad)
    if out_dtype not in ("i8", "i32"):
        raise ValueError("convolution supports exact i32 or scaled i8 readout")
    gains = tuple(float(x) for x in scale) if isinstance(scale, (list, tuple)) else (float(scale),) * co
    if len(gains) == 1:
        gains = gains * co
    if len(gains) != co:
        raise ValueError("convolution scales must be scalar or original output-column vector")
    path = str(Path(facts_path) if facts_path else Path(__file__).resolve().parent / "data/facts.json")
    edge, operand_rows, accumulator_rows, hardware_identity = _resources(path)
    settings_path = Path(__file__).parent / 'DENSE_SCHEDULE.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    from conv_array_policy import select_array_retention, row_block_limit
    array_request, array_region = select_array_retention(
        os.environ.get('CONV_ARRAY_RETENTION', settings.get('conv_array_retention', 'off')),
        window, co, edge=edge, accumulator_rows=accumulator_rows,
        hardware_identity=hardware_identity)
    array_row_blocks = row_block_limit(settings.get('conv_array_row_blocks', 4))
    vertical_rows = int(array_output_rows if array_output_rows is not None else
        os.environ.get('CONV_ARRAY_OUTPUT_ROWS', settings.get('conv_array_output_rows', 1)))
    if vertical_rows not in (1, 2, 4):
        raise ValueError('CONV_ARRAY_OUTPUT_ROWS must be 1, 2 or 4')
    # Explicit opt-in alternative for rows with no horizontal M reuse. This
    # is a geometry experiment; selected storage/order checks remain mandatory.
    if (vertical_rows > 1 and array_region['requested'] == 'regional'
            and window.output_spatial[1] <= edge and window.output_spatial[0] > 1):
        array_request = 'groups'
        array_region = dict(array_region, effective='groups',
            vertical_output_rows=vertical_rows,
            profitability=dict(status='experimental vertical grouping; timing UNKNOWN'),
            legality=dict(status='pending original ring/RHS/ACC/ordering checks'))
    if (array_request != 'groups' or (array_region['requested'] == 'regional'
            and window.output_spatial[1] > edge)):
        vertical_rows = 1
    vertical_rows = min(vertical_rows, window.output_spatial[0])
    # Keep the finite knob domain when a source has only three output rows.
    if vertical_rows == 3:
        vertical_rows = 2
    vertical_attempt = vertical_rows > 1
    rhs_transport_request = rhs_transport_mode or os.environ.get('CONV_RHS_TRANSPORT',
        settings.get('conv_rhs_transport','serial'))
    if rhs_transport_request not in ('serial','last_use'):
        raise ValueError('CONV_RHS_TRANSPORT must be serial or last_use')
    group_request = output_group_mode or os.environ.get('CONV_OUTPUT_GROUP',
        settings.get('conv_output_group','rows'))
    if group_request not in ('rows','compact_runs','strided_pointwise','regional_strided'):
        raise ValueError('CONV_OUTPUT_GROUP must be rows, compact_runs, strided_pointwise or regional_strided')
    compact = None
    group_selection = dict(requested=group_request, selected='rows')
    if group_request == 'compact_runs':
        if window.stride[1] != 1:
            group_selection['reason'] = 'compact halo runs require unit horizontal stride'
        else:
            from conv_groups import CompactConvolutionRuns
            compact = CompactConvolutionRuns(window,edge)
            window = compact.virtual
            group_selection.update(selected='compact_runs', mapping=compact.record())
    # Flatten only original independent output coordinates.  The shared gather
    # splits every source-row/image boundary, retaining the original DMA pitch.
    # The regional choice is a provisional geometry class, not a legality grant:
    # all ordinary gather, storage, numeric and completion checks still run.
    from merlin.transforms.pipeline_policy import ResourcePredicate
    from merlin.transforms.pipeline_policy import PolicyFacts
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    pointwise_resources = dict(valid_strided_pointwise=int(window.kernel == (1,1)
        and not any(window.padding) and window.stride[1] > 1),
        output_width=window.output_spatial[1], input_channels=ci,
        output_channels=co, row_padding=(-window.output_spatial[1]) % edge)
    pointwise_guards = tuple(ResourcePredicate(k, at_least=v) for k,v in
        [('valid_strided_pointwise',1),('output_width',edge),
         ('input_channels',edge),('output_channels',edge),('row_padding',1)])
    pointwise_facts = PolicyFacts(AnalysisStamp(
        hashlib.sha256(repr((window,co,edge)).encode()).hexdigest(),
        PipelineContext(hardware=hardware_identity,resources=hardware_identity),
        'convolution-pointwise-region'),resources=tuple(sorted(pointwise_resources.items())))
    regional_pointwise = all(g.match(pointwise_facts) is True for g in pointwise_guards)
    linear_pointwise = bool(pointwise_resources['valid_strided_pointwise']) and (
        group_request == 'strided_pointwise' or
        group_request == 'regional_strided' and regional_pointwise)
    if group_request in ('strided_pointwise','regional_strided'):
        group_selection.update(selected='strided_pointwise' if linear_pointwise else 'rows',
            resources=pointwise_resources,
            predicates=[dict(resource=g.resource,at_least=g.at_least) for g in pointwise_guards],
            source_row_stride_elements=window.stride[1]*ci,
            mechanism='shared ConvolutionGatherRuns; original row/image boundary fragments',
            profitability='provisional one large paired complete-model observation; wider geometry timing UNKNOWN')
    linear_iteration = compact is not None or linear_pointwise
    row_tile = edge if compact is not None else min(edge,window.output_spatial[1])
    requested_readout = readout_mode or os.environ.get('CONV_READOUT', settings.get('conv_readout','direct'))
    if requested_readout not in ('direct','staged_columns','staged_epochs','regional_staged','regional_epochs'):
        raise ValueError('CONV_READOUT must be direct, staged_columns, staged_epochs, regional_staged or regional_epochs')
    staged = None
    readout_selection = dict(requested=requested_readout, selected='direct')
    geometry_staged = requested_readout in ('staged_columns','staged_epochs')
    epoch_readout = requested_readout == 'staged_epochs'
    if requested_readout == 'regional_epochs':
        from conv_readout import regional_epochs_candidate
        regional = regional_epochs_candidate(ci,(kh,kw),co,edge,row_tile,
            hardware_identity,padding=pad,bias=bias is not None)
        readout_selection['regional'] = regional
        epoch_readout = regional['geometry_selected']
        geometry_staged = epoch_readout or regional['fallback']['geometry_selected']
    if requested_readout == 'regional_staged':
        from conv_readout import regional_staged_candidate
        regional = regional_staged_candidate(ci, (kh, kw), co, edge, row_tile,
                                              hardware_identity)
        readout_selection['regional'] = regional
        geometry_staged = regional['geometry_selected']
        if not geometry_staged:
            readout_selection['reason'] = 'outside provisional readout geometry region'
    if geometry_staged and out_dtype == 'i8' and len(set(gains)) > 1:
        if readout_scratch is None:
            readout_selection['reason'] = 'caller scratch not supplied'
        elif accumulator_rows <= row_tile:
            readout_selection['reason'] = 'no disjoint accumulator window for readout'
        else:
            from conv_readout import StagedColumnReadout, StagedEpochColumnReadout, staged_epoch_workspace
            if epoch_readout:
                workspace = staged_epoch_workspace(shape, ws, kernel=(kh,kw),
                    strides=strides,dilation=dilation,padding=pad,facts_path=path)
                staged = StagedEpochColumnReadout(readout_scratch,readout_scratch_bytes,
                    gains,co,row_tile,edge,accumulator_rows-row_tile,bool(relu),
                    slot_count=workspace['slot_count'])
            else:
                staged = StagedColumnReadout(readout_scratch, readout_scratch_bytes,
                    gains, co, row_tile, edge, accumulator_rows-row_tile, bool(relu))
            accumulator_rows -= row_tile
            readout_selection['selected'] = 'staged_epochs' if epoch_readout else 'staged_columns'
    # Heterogeneous original gains must never be averaged across a readout.
    column_tile = edge if staged is not None or len(set(gains)) == 1 else 1
    channel_tile = min(edge, ci)
    rhs_base = ((row_tile + edge - 1) // edge) * edge
    input_transfer_request = input_transfer_mode or os.environ.get('CONV_INPUT_TRANSFER',
        settings.get('conv_input_transfer','stream'))
    from conv_transfers import plan_input_transfer, bind_input_rows, regional_input_candidate
    # Resolve geometry before probing the physical ring. A short original CI
    # must probe primitive-pitched rows; only successful admission commits them.
    transfer_mode = input_transfer_request
    transfer_region = None
    if transfer_mode == 'regional_rolling':
        transfer_region = regional_input_candidate(window,co,edge,hardware_identity,
            out_dtype=out_dtype,bias=bias is not None,gains=gains,relu=bool(relu))
        transfer_mode = transfer_region['candidate']
    transfer_channel_tile = (edge if transfer_mode in
        ('rolling_narrow', 'rolling_wide') and ci < edge else channel_tile)
    # An expanded ring is useful only to the original direct output adapter.
    if compact is not None or staged is not None:
        vertical_rows = 1
    input_transfer, input_transfer_selection = plan_input_transfer(window,
        mode='stream' if linear_iteration else transfer_mode, edge=edge, operand_rows=operand_rows,
        channel_tile=transfer_channel_tile, retain_input=retain_input, facts_path=path,
        ordering_path=ordering_path, output_channels=co, output_rows=vertical_rows)
    if input_transfer is None and vertical_rows > 1:
        vertical_refusal = input_transfer_selection
        vertical_rows = 1
        input_transfer, input_transfer_selection = plan_input_transfer(window,
            mode='stream' if linear_iteration else transfer_mode, edge=edge, operand_rows=operand_rows,
            channel_tile=transfer_channel_tile, retain_input=retain_input, facts_path=path,
            ordering_path=ordering_path, output_channels=co, output_rows=1)
        input_transfer_selection['vertical_fallback'] = vertical_refusal
    input_transfer_selection['requested'] = input_transfer_request
    if transfer_region is not None:
        input_transfer_selection['regional'] = transfer_region
    if input_transfer is not None:
        channel_tile = transfer_channel_tile
        rhs_base = input_transfer['rhs_base']
        if (transfer_region is not None
                and transfer_region['region'] == 'primitive_pitch_tail'
                and array_region['requested'] == 'regional'):
            # Consume the existing bounded grouped kernel only after its input
            # owner admitted the primitive pitch. Explicit off/groups and the
            # row-block knob retain their meaning; group legality is still
            # established below by the unchanged checked group planner.
            array_request = 'groups'
            array_region = dict(schema='submission.conv_array_region.v1',
                requested='regional',effective='groups',
                input_tail_region=transfer_region,incumbent_region=array_region,
                legality=dict(status='pending checked group planner'),
                profitability=dict(status='provisional one large paired complete-model observation',
                    limitation='cutoff neighbors timing UNKNOWN; no measured speedup claim'))
    plan = CountedConvolutionPlan(window, co, edge, row_tile, column_tile,
        channel_tile, operand_rows, accumulator_rows, 0, rhs_base, 0,
        bias=bias is not None, linear_rows=linear_iteration)
    original = source_identity or hashlib.sha256(repr((shape, ws, window, gains,
        out_dtype, bool(relu), bias is not None)).encode()).hexdigest()
    intervals = tuple((j, min(column_tile, co-j)) for j in range(0, co, column_tile))
    source = ConvolutionExplorationSource(plan, intervals, original)
    input_rows = bind_input_rows(plan, source.identity, input_transfer)
    numeric = "signed-i8-products-i32-exact-source-readout"
    qualification = "private-convolution-rows-real-full-drain-per-effect"
    retention = ConvolutionInputRetentionLaw(hardware_identity, numeric,
        source.identity, qualification, len(intervals), input_preserved=True,
        independent_outputs=True, immutable_mapped_disjoint_buffers=True,
        independent_output_rows=rhs_transport_request == 'last_use' or array_request == 'groups',
        no_intermediate_observers=rhs_transport_request == 'last_use' or array_request == 'groups')
    pointwise = pointwise_mode or os.environ.get("CONV_POINTWISE", settings.get("conv_pointwise", "window"))
    if pointwise not in ("window", "retained", "epochs", "raw_i32_retained"):
        raise ValueError("CONV_POINTWISE must be window, retained, epochs or raw_i32_retained")
    pointwise_selection = {"requested": pointwise, "selected": "window"}
    if (staged is not None or input_rows is not None or compact is not None) and pointwise != "window":
        pointwise_selection['reason'] = 'selected readout/input transfer uses the checked window compute plan'
        pointwise = 'window'
    if pointwise == "raw_i32_retained" and (out_dtype != "i32" or
            bias is not None or bool(relu) or any(g != 1.0 for g in gains)):
        pointwise_selection["reason"] = "requires raw i32 without bias, scaling or activation"
        pointwise = "window"
    if pointwise != "window":
        from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
        view = pointwise_dense_view(window, co,
            output_shape=(shape[0]*window.output_spatial[0]*window.output_spatial[1], co))
        if view is not None:
            from dense import emit_dense, plan_dense
            selected_pointwise = "retained" if pointwise == "raw_i32_retained" else pointwise
            # Plan before issuing anything: capacity/order refusal preserves the
            # original window route rather than leaving partially emitted work.
            refusal = None
            if pointwise == "raw_i32_retained":
                dense_facts = Path(__file__).resolve().parent / "data/facts.json"
                if hashlib.sha256(dense_facts.read_bytes()).hexdigest() != hardware_identity:
                    refusal = "dense view and convolution use different hardware facts"
                else:
                    try:
                        plan_dense(*view.dimensions, bias=False, alternative="retained")
                    except ValueError as error:
                        refusal = str(error)
            if refusal is not None:
                pointwise_selection["reason"] = refusal
            else:
                return _emit_pointwise_dense(enc, inp, weights, out, view,
                    out_dtype, gains, relu, bias, selected_pointwise, pointwise_selection)
        else:
            pointwise_selection["reason"] = "requires unpadded unit-stride 1x1 view"
    requested = schedule_mode or os.environ.get("CONV_SCHEDULE", settings.get('conv_schedule','sync'))
    # An opt-in realization of the same shared readout schedule: combine only
    # adjacent original stores under its separate selected MVOUT block law.
    wide_readout_requested = requested == 'wide_readout'
    mode, selection = _select_schedule_mode('readout' if wide_readout_requested else requested, settings, source,
        hardware_identity, edge=edge, operand_rows=operand_rows,
        accumulator_rows=accumulator_rows, gains=gains)
    ordering = _ordering_admission(path, str(ordering_path or
        Path(__file__).resolve().parent / "data/command_ordering.json"))
    admitted_mode = mode if ordering["admitted"] else "sync"
    if requested_readout == 'regional_epochs' and epoch_readout and staged is not None:
        from conv_readout import staged_epoch_ordering_admission
        epoch_admission = staged_epoch_ordering_admission(ordering,str(ordering_path or
            Path(__file__).resolve().parent / 'data/command_ordering.json'))
        if admitted_mode != 'readout' or not epoch_admission['admitted']:
            # No instruction has issued. Rebuild the original incumbent plan;
            # a smaller direct-column plan must not inherit epoch ownership.
            fallback = emit_conv(enc,inp,weights,out,input_shape,weight_shape,
                strides=strides,padding=padding,dilation=dilation,out_dtype=out_dtype,
                scale=scale,relu=relu,bias=bias,kernel=kernel,
                output_channels=output_channels,source_identity=source_identity,
                facts_path=facts_path,retain_weights=retain_weights,retain_input=retain_input,
                schedule_mode=schedule_mode,ordering_path=ordering_path,compute_pairs=compute_pairs,
                pointwise_mode=pointwise_mode,readout_mode='regional_staged',
                readout_scratch=readout_scratch,readout_scratch_bytes=readout_scratch_bytes,
                input_transfer_mode=input_transfer_mode,output_group_mode=output_group_mode,
                rhs_transport_mode=rhs_transport_mode)
            fallback['readout_selection']['regional_epochs_refusal'] = (
                'actual shared schedule is not readout' if admitted_mode != 'readout'
                else epoch_admission['reason'])
            fallback['readout_selection']['requested'] = requested_readout
            return fallback
    max_live = min(len(intervals), accumulator_rows // row_tile) if retain_input else 1
    total_pairs = kh * kw * ((ci + channel_tile - 1) // channel_tile) * max_live
    pairs = int(compute_pairs) if compute_pairs is not None else max(2, total_pairs)
    if pairs <= 1:
        raise ValueError("convolution compute epoch requires at least two issued pairs")
    ids = (hardware_identity, numeric, source.identity,
        "native-ordering:" + str(ordering.get("ordering_sha256", "unavailable")))
    issue = epoch = readout = None
    if admitted_mode != "sync":
        issue = ConvolutionLoadIssueLaw(*ids,
            max(row_tile + 1 + max_live * (4 if array_request == "groups" else 1) * int(bias is not None),
                0 if compact is None else 3*(1+(row_tile+window.output_spatial[1]-2)//window.output_spatial[1])+1+max_live*int(bias is not None),
                0 if input_rows is None else input_rows.initial_load_commands),
            local_range_dependencies=True, ordered_load_configuration=True,
            paired_descriptor_compute=True, finite_queue_backpressure=True,
            complete_device_drain=True, immutable_mapped_disjoint_buffers=True)
    if admitted_mode in ("epoch", "readout"):
        epoch = ConvolutionComputeEpochLaw(*ids, pairs,
            local_read_after_write=True, local_write_after_read=True,
            local_write_after_write=True, accumulator_result_visibility=True,
            operand_consumption_retirement=True, ordered_configuration_capture=True,
            adjacent_descriptor_compute=True, finite_queue_pair_progress=True,
            complete_device_drain=True, exclusive_private_state=True,
            pending_compute_pairs=ordering["pending_compute_pairs"])
    if admitted_mode == "readout":
        readout = ConvolutionReadoutEpochLaw(*ids, max_live * (4 if array_request == "groups" else 1),
            ordering["pending_readouts"], local_read_after_write=True,
            accumulator_result_visibility=True, ordered_configuration_capture=True,
            finite_queue_progress=True, complete_device_drain=True,
            exclusive_private_state=True)
    if epoch_readout and staged is not None:
        staged.bind(plan, readout, ordering, str(ordering_path or
            Path(__file__).resolve().parent / 'data/command_ordering.json'))
    # Epoch emission uses the original column-block consumer even for a
    # one-column output, so its source retention premises remain explicit.
    schedule = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(live_tiles=None if retain_input else 1,
            load_issue=issue is not None, compute_pairs=pairs if epoch else 1),
        retention_law=retention if retain_input or epoch else None,
        issue_law=issue, epoch_law=epoch)
    rhs = None
    if retain_weights:
        law = ConvolutionRhsResidencyLaw(hardware_identity, numeric,
            source.identity, qualification, weight_transfer_only=True,
            immutable_mapped_disjoint_buffers=True, other_effects_preserve_rhs=True,
            exclusive_private_rows=True, entry_quiescence_and_complete_preparation=True)
        phase_law = ConvolutionRhsPhaseLaw(source.identity, qualification,
            independent_output_columns=True, immutable_inputs=True,
            no_intermediate_observers=True, full_completion_before_reprepare=True)
        rhs = plan_convolution_rhs_phases(schedule, operand_rows=operand_rows,
            law=law, phase_law=phase_law, input_rows=input_rows)

    array_group = array_law = None
    array_selection = None
    if array_request == "groups":
        from conv_array_retention import plan_array_retention
        if compact is not None or staged is not None:
            array_selection = dict(requested="groups", selected="incumbent", reason="requires original direct rolling output adapter")
        else:
            array_group, array_law, array_selection = plan_array_retention(
                schedule,rhs,input_rows,readout,ordering,facts_path=path,ordering_path=str(ordering_path or
                    Path(__file__).resolve().parent / "data/command_ordering.json"),
                max_row_blocks=array_row_blocks, max_output_rows=vertical_rows)

    # Planning has emitted no commands. Restore the complete original plan if
    # any later ownership/epoch/array check cannot use the expanded horizon.
    actual_groups = array_group if isinstance(array_group, tuple) else (array_group,)
    if vertical_attempt and not any(g is not None and g.output_rows > 1 for g in actual_groups):
        receipt = emit_conv(enc, inp, weights, out, input_shape, weight_shape,
            strides=strides, padding=padding, dilation=dilation, out_dtype=out_dtype,
            scale=scale, relu=relu, bias=bias, kernel=kernel,
            output_channels=output_channels, source_identity=source_identity,
            facts_path=facts_path, retain_weights=retain_weights, retain_input=retain_input,
            schedule_mode=schedule_mode, ordering_path=ordering_path, compute_pairs=compute_pairs,
            pointwise_mode=pointwise_mode, readout_mode=readout_mode,
            readout_scratch=readout_scratch, readout_scratch_bytes=readout_scratch_bytes,
            input_transfer_mode=input_transfer_mode, output_group_mode=output_group_mode,
            rhs_transport_mode=rhs_transport_mode, array_output_rows=1)
        receipt['vertical_group_fallback'] = array_selection
        return receipt

    lookahead = None
    lookahead_selection = dict(requested=rhs_transport_request,selected='serial')
    if rhs_transport_request == 'last_use' and array_group is None:
        if readout is None or compact is not None or staged is not None:
            lookahead_selection['reason'] = 'requires original rolling readout epoch without staged or gathered output adapter'
        else:
            from conv_lookahead import plan_rhs_lookahead
            lookahead, refusal = plan_rhs_lookahead(schedule,rhs,input_rows,ordering)
            if lookahead is None:
                lookahead_selection['reason'] = refusal
            else:
                lookahead_selection.update(selected='last_use',application=lookahead.record())

    b = enc.index_builder

    def add(x, y):
        return b.op("add", x, y)

    def mul(x, y):
        return b.op("mul", x, y)

    def lower_physical(kind, f):
        if kind in ('stage_input_rows','fill_input_rows'):
            ptr = enc.ptr_zero() if kind == 'fill_input_rows' else enc.offset(inp, f['first_element'])
            if kind == 'fill_input_rows' and f['value'] != 0:
                raise ValueError('resident convolution padding must preserve original integer zero')
            enc.load(ptr, f['local_row'], f['rows'], f['columns'],
                f.get('row_stride_elements', ci), stream=0, block_stride=f['block_stride_rows'])
        elif kind == "load_input":
            ptr = enc.ptr_zero() if f["zero"] else enc.offset(inp, f["first_element"])
            enc.load(ptr, f["local_row"], f["rows"], f["columns"],
                     f["row_stride_elements"], stream=0)
        elif kind == "load_weight":
            ptr = enc.offset(weights, add(mul(f["reduction_row"], co), f["output_column"]))
            enc.load(ptr, f["local_row"], f["rows"], f["columns"], co, stream=1)
        elif kind == "initialize_bias":
            ptr = enc.offset(bias, mul(f["output_column"], 4))
            # Zero DRAM stride broadcasts this original bias row to all M rows.
            enc.load(ptr, add(f["accumulator_row"], 1 << 31),
                     f["rows"], f["columns"], 0, stream=2, shrunk=False)
        elif kind == "compute":
            if f.get("fresh_weights") is False and array_law is None:
                raise ValueError("retained array operation needs the checked original grouped owner")
            def product(accumulate):
                if array_law is not None:
                    from conv_array_retention import emit_product
                    emit_product(enc,f,accumulate)
                    return
                target = add(f["accumulator_row"], (1 << 30) if accumulate else 0)
                enc.preload(f["rhs_row"], target, f["depth"], f["columns"], f["rows"])
                enc.compute(f["lhs_row"], f["depth"], f["rows"])
            if f["bias"]:
                product(True)
            else:
                b.branch(f["first_reduction"], lambda: product(False), lambda: product(True))
        elif kind == "store":
            if staged is not None:
                staged.emit(enc, out, f)
                return
            size = 4 if out_dtype == "i32" else 1
            ptr = enc.offset(out, mul(add(mul(f["output_row"], co), f["output_column"]), size))
            # Column intervals are static source partitions, including phases.
            gain = gains[f["output_column"]]
            enc.store(ptr, f["accumulator_row"], f["rows"], f["columns"],
                      co * size, dtype=out_dtype, scale=gain, relu=bool(relu))
        else:
            raise ValueError("unsupported checked convolution effect: " + kind)
        if admitted_mode == "sync":
            enc.fence()

    def lower(kind, fields):
        if compact is not None and kind == 'load_input':
            compact.emit_load(b,fields,lower_physical)
        else:
            lower_physical(kind,fields)

    enc.fence()
    # The convolution owns its execution configuration as well as its rows.
    # A preceding device group may leave strides, scaling or activation set.
    # Reestablish weight-stationary, unit-stride, exact-i32 execution after
    # draining that group, before preparing or consuming this invocation.
    enc.configure_execute()
    from conv_transfers import bind_readout_transfers
    readout_transfers, readout_transfer_selection = bind_readout_transfers(plan,
        requested=wide_readout_requested, gains=gains, out_dtype=out_dtype,
        relu=relu, staged=staged, readout_law=readout, facts_path=path,
        ordering_path=ordering_path)
    receipt = emit_convolution_schedule(schedule, b, lower,
        complete=(lambda kind, fields: staged.complete(enc,out,kind,fields))
            if epoch_readout and staged is not None
            else (lambda kind, fields: enc.fence()),
        synchronous_completion=admitted_mode == "sync", rhs=rhs, readout_law=readout,
        input_rows=input_rows, readout_transfer_law=readout_transfers,
        spatial_iteration='linear' if linear_iteration else 'rows',
        **(dict(output_group=array_group,array_law=array_law) if array_group is not None else {} if lookahead is None else dict(output_group=lookahead.groups,
            grouped_transport=True,rhs_phase_prefetch_law=lookahead.law)))
    enc.fence()
    if array_selection is not None:
        receipt["array_retention_selection"] = array_selection
    receipt["array_retention_region"] = array_region
    receipt["native_ordering_admission"] = ordering
    receipt["requested_mode"] = requested
    receipt["schedule_selection"] = selection
    receipt["emitted_mode"] = admitted_mode
    receipt["pointwise_selection"] = pointwise_selection
    receipt["input_transfer_selection"] = input_transfer_selection
    receipt['output_group_selection'] = group_selection
    receipt['readout_selection'] = readout_selection
    receipt['rhs_transport_selection'] = lookahead_selection
    receipt['readout_transfer_selection'] = readout_transfer_selection
    if staged is not None:
        receipt['staged_readout'] = staged.record()
    return receipt
