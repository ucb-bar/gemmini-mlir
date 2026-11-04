"""Bounded convolution halos consumed directly by signed pool readout.

The library checks both the eliminated intermediate's exact rounded boundary
and the accumulator halo layout. No full convolution image is allocated.
"""
from __future__ import annotations

import os
from pathlib import Path

from merlin.compile.logical_abi import pool_params
from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionRhsResidencyLaw,
    plan_convolution_rhs_residency, plan_convolution_schedule,
)
from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan, emit_spatial_producer
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.compile.scheduling.window_transfer import ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw
from merlin.compile.tuning_knobs import KnobError, knob

from convolution import _resources, _ordering_admission
from pool_schedule import emit_pool_load_issue
from merlin.compile.scheduling.spatial_residue_runs import SpatialResidueRuns
from merlin.compile.scheduling.spatial_packed_raw import PackedRawConvolution
from merlin.compile.scheduling.spatial_packed_raw_epochs import (
    propose_packed_raw_halo_epochs, emit_packed_raw_halo_epochs,
)
from merlin.llvmlower.window_preparation_receipt import emit_checked_window_preparation
from merlin.compile.scheduling.spatial_raw_retention import propose_spatial_raw_retention, emit_spatial_raw_retention


def _choice(name, default, values, attrs, attr, env):
    value = attrs.get(attr)
    if value is None:
        value = os.environ.get(env)
    if value is None:
        value = knob(name, default=default, values=values, root=Path(__file__).resolve().parent)
    return value


def _plan(input_shape, kernel, attrs):
    shape, kernel = tuple(map(int, input_shape)), tuple(map(int, kernel))
    if len(shape) != 4 or len(kernel) != 4 or shape[-1] != kernel[-2]:
        raise ValueError("fused convolution pool requires NHWC and KH,KW,CI,CO")
    kh, kw, ci, co = kernel
    stride = tuple(attrs.get("strides", attrs.get("stride", (1, 1))))
    padding = tuple(attrs.get("padding", (0, 0, 0, 0)))
    if len(padding) == 2:
        padding = padding + padding
    dilation = tuple(attrs.get("dilation", (1, 1)))
    window = Window2D(shape, (kh, kw), stride, dilation, padding)
    normal = dict(attrs)
    normal.setdefault("pool_in_dims", window.output_spatial)
    pool = pool_params(normal, op="fused convolution pool")
    if tuple(pool["pool_in_dims"]) != window.output_spatial:
        raise ValueError("pool input dimensions disagree with original convolution")
    stages = attrs.get("epilogue", ())
    dtype = attrs.get("out_dtype", attrs.get("output_dtype", "i8"))
    if dtype != "i8":
        raise ValueError("hardware spatial readout requires signed i8 output")
    relu = bool(attrs.get("relu", False) or "relu" in stages)
    gain = attrs.get("scale", attrs.get("acc_scale", 1.0))
    if "acc_scale_by_column" in attrs or isinstance(gain, (list, tuple)):
        raise ValueError("fused spatial halo currently requires one original scalar scale")
    gain = float(gain)
    size, step, pad = pool["pool_size"], pool["pool_stride"], pool["pool_padding"]
    if size[0] != size[1] or size[0] not in (1, 2, 3) or step[1] not in (1, 2, 3):
        raise ValueError("selected native pool fields do not admit this window/stride")
    pad_value = pool["pad_value"]
    effective_pad = pad_value
    if any(pad):
        if pad_value == 0 and relu:
            # Q1 is nonnegative. Every window intersects its source (checked
            # below), hence replacing zero padding by a duplicated valid Q1
            # sample cannot change its maximum.
            effective_pad = -128
        elif pad_value != -128:
            raise ValueError("padded fused pool requires minimum i8 or nonnegative Q1 with zero padding")
    path = str(attrs.get("facts_path") or Path(__file__).resolve().parent / "data/facts.json")
    edge, operand_rows, accumulator_rows, hardware = _resources(path)
    producer = CountedConvolutionPlan(window, co, edge, min(edge, window.output_spatial[1]),
        edge, min(edge, ci), operand_rows, accumulator_rows, 0, edge, 0,
        bias=attrs.get("bias") is not None)
    spatial = SpatialMaxReadout(window.output_spatial, size, step, pad, effective_pad)
    # The original Q1 narrowing is followed only by max, whose Q2 is identity.
    # prove_spatial_readout checks all signed-i32 transition boundaries.
    q1 = RoundedReadout((gain,), relu)
    chain = SpatialReadoutChain(q1, RoundedReadout((1.0,), False), q1)
    fit = (accumulator_rows // size[0] - size[1]) // step[1] + 1
    native_fit = (255 - size[1]) // step[1] + 1
    requested = _choice("conv_pool_stripe", 1, (1, 4, 7, 8, 15, 16, "derived"), attrs,
        "pool_stripe_width", "CONV_POOL_STRIPE")
    if requested == "derived":
        # One complete interior halo row per original primitive. This policy
        # has paired evidence for the 3x3/stride-2 pool class; other geometries
        # preserve the incumbent stripe rather than extrapolating that ranking.
        requested = (max(1, (producer.row_tile - size[1]) // step[1] + 1)
            if size == (3, 3) and step == (2, 2) else 1)
    requested = int(requested)
    if requested < 1:
        raise ValueError("fused pool stripe width must be positive")
    stripe = min(requested, fit, native_fit, spatial.output_hw[1])
    plan = SpatialProducerPlan(producer, spatial, (chain,) * ((co + edge - 1) // edge), stripe)
    return plan, gain, relu, hardware, pad_value


def supports_conv_pool(input_shape, kernel, **attrs):
    """Pure preflight, before ABI construction or target effects are emitted."""
    try:
        _plan(input_shape, kernel, attrs)
        return True
    except KnobError:
        # Invalid shared tuning configuration is a compiler configuration
        # error, not a geometry refusal granting a different scratch ABI.
        raise
    except (ValueError, TypeError, KeyError):
        return False


def packed_input_workspace_bytes(input_shape, kernel, **attrs):
    """Experimental CI-fragmented raw-panel preflight; zero means refusal.

    Budget is explicit private host storage, independent of device capacity.
    The ordinary native object preflight still checks the complete entry frame.
    """
    plan, _, _, _, _ = _plan(input_shape, kernel, attrs)
    p = plan.producer
    reduction = p.window.kernel[0] * p.window.kernel[1] * p.window.input_shape[-1]
    required = p.row_tile * reduction
    if (propose_spatial_raw_retention(plan) is None
            or not p.window.input_shape[-1] < p.primitive < reduction
            or not 0 < required <= 4096):
        return 0
    packed = PackedRawConvolution(p, required)
    packed.__post_init__()
    return required


def emit_conv_pool(enc, inp, weight, out, input_shape, kernel, **attrs):
    """Emit the original pure convolution/rounded-readout/pool composition.

    Caller establishes immutable mapped inputs, disjoint output and an
    unobserved intermediate. ``bias`` is the original i32 row pointer or None.
    Pool padding and all kernel attributes are the original source values.
    """
    plan, gain, relu, hardware, original_pad = _plan(input_shape, kernel, attrs)
    requested_schedule = _choice("conv_pool_schedule", "sync", ("sync", "issue", "issue_peel", "retain_raw", "retain_epochs", "packed_raw", "packed_shared", "regional_device_runs", "device_flat_runs"), attrs,
        "pool_schedule", "CONV_POOL_SCHEDULE")
    if requested_schedule not in ("device_flat_runs", "regional_device_runs", "device_runs", "sync", "issue", "issue_peel", "retain_raw", "retain_epochs", "packed_raw", "packed_shared", "packed_runs"):
        raise ValueError("unknown fused pool completion schedule")
    root = Path(__file__).resolve().parent
    ordering = _ordering_admission(str(attrs.get("facts_path") or root / "data/facts.json"),
        str(attrs.get("ordering_facts_path") or root / "data/command_ordering.json"))
    p, spatial = plan.producer, plan.spatial
    flat_requested = requested_schedule == "device_flat_runs"
    flat_runs = None
    if flat_requested:
        from pool_horizontal_flat import propose_pool_horizontal_flat
        flat_runs = (propose_pool_horizontal_flat(p)
            if ordering["admitted"] and propose_spatial_raw_retention(plan) is not None else None)
        requested_schedule = "device_runs" if flat_runs is not None else "regional_device_runs"
    regional_runs = requested_schedule == "regional_device_runs"
    if regional_runs:
        # Paired complete-call evidence covers this geometry class across small,
        # ragged and large images. Storage legality is checked independently by
        # RHS residency and SpatialResidueRuns, without extrapolating the ranking
        # to other kernel/stride classes.
        requested_schedule = ("device_runs" if p.window.input_shape[-1] == 3
            and p.window.kernel == (7, 7) and p.window.stride == (2, 2)
            and p.window.dilation[1] == 1
            and spatial.window == (3, 3) and spatial.stride == (2, 2)
            else "retain_epochs")
    if requested_schedule == "packed_runs" and p.window.dilation[1] != 1:
        requested_schedule = "packed_shared"  # Contiguous source taps are not available.
    retained = propose_spatial_raw_retention(plan) if requested_schedule in ("device_runs", "retain_raw", "retain_epochs", "packed_raw", "packed_shared", "packed_runs") else None
    packed = None
    if requested_schedule in ("packed_raw", "packed_shared", "packed_runs"):
        required = packed_input_workspace_bytes(input_shape, kernel, **attrs)
        if required and ordering["admitted"] and attrs.get("packed_input_workspace") is not None:
            packed = PackedRawConvolution(p, int(attrs.get("packed_input_workspace_bytes", 0)))
        else:
            requested_schedule = "retain_epochs"
    shared_packed = (propose_packed_raw_halo_epochs(plan, packed)
        if packed is not None and requested_schedule in ("packed_shared", "packed_runs") else None)
    if requested_schedule in ("packed_shared", "packed_runs") and shared_packed is None:
        packed = None
        requested_schedule = "retain_epochs"
    selected_schedule = ("issue_peel" if requested_schedule in ("device_runs", "retain_raw", "retain_epochs", "packed_raw", "packed_shared", "packed_runs") and retained is None else requested_schedule)
    # The promoted peeled schedule is confined to its measured pool geometry.
    # Explicit generic issue remains available for further independent studies.
    policy_admitted = (selected_schedule != "issue_peel"
        or spatial.window == (3, 3) and spatial.stride == (2, 2))
    issue_mode = selected_schedule != "sync" and ordering["admitted"] and policy_admitted
    ci, co = p.window.input_shape[-1], p.output_channels
    kh, kw = p.window.kernel
    b = enc.index_builder
    bias = attrs.get("bias")
    chunks = (ci + p.channel_tile - 1) // p.channel_tile
    if selected_schedule in ("device_runs", "retain_epochs", "packed_raw", "packed_shared", "packed_runs") and not 1 < kh * kw * chunks < 2**31:
        selected_schedule = "retain_raw"  # No multi-pair epoch or no bounded typed law.
    column_tiles = (co + p.column_tile - 1) // p.column_tile

    def op(name, x, y):
        return b.op(name, x, y)

    def offset(ptr, row, col=0):
        return enc.offset(ptr, op("add", op("mul", row, co), col))

    intervals = tuple((j, min(p.column_tile, co-j)) for j in range(0, co, p.column_tile))
    transport = p if packed is None else packed.packed
    transport_ci = transport.window.input_shape[-1]
    transport_chunks = (transport_ci + transport.channel_tile - 1) // transport.channel_tile
    source = ConvolutionExplorationSource(transport, intervals, plan.identity)
    lifetime = plan_convolution_schedule(source)
    law = ConvolutionRhsResidencyLaw(hardware, "i8-products-exact-Q1-max-i8",
        source.identity, "private-halo-no-observer-synchronous-full-completion",
        weight_transfer_only=True, immutable_mapped_disjoint_buffers=True,
        other_effects_preserve_rhs=True, exclusive_private_rows=True,
        entry_quiescence_and_complete_preparation=True)
    resident = (None if flat_runs is not None else
        plan_convolution_rhs_residency(lifetime, operand_rows=p.operand_rows, law=law))

    def resident_row(reduction, column):
        panel = op("add", op("mul", op("udiv", reduction, transport_ci), transport_chunks),
            op("udiv", op("urem", reduction, transport_ci), transport.channel_tile))
        slot = op("add", op("mul", panel, column_tiles), op("udiv", column, p.column_tile))
        return op("add", resident.layout.rhs_base, op("mul", slot, transport.channel_tile))

    device_runs = None
    if regional_runs and selected_schedule == "device_runs" and resident is None:
        selected_schedule = "retain_epochs"
    if selected_schedule == "device_runs":
        base = (resident.layout.rhs_base + resident.layout.rhs_rows if resident is not None
            else max(p.rhs_row+p.channel_tile,p.lhs_row+p.row_tile))
        try:
            device_runs = flat_runs if flat_runs is not None else SpatialResidueRuns(p, base)
        except ValueError:
            selected_schedule = "retain_epochs"

    enc.fence()
    if resident is not None:
        resident.verify_current()
        def tap(t):
            def chunk(c, depth):
                reduction = op("add", op("mul", t, transport_ci), op("mul", c, transport.channel_tile))
                for column, width in intervals:
                    enc.load(offset(weight, reduction, column), resident_row(reduction, column),
                        depth, width, co, stream=1)
            if transport_ci // transport.channel_tile:
                b.loop(0, transport_ci // transport.channel_tile, 1, lambda c: chunk(c, transport.channel_tile))
            if transport_ci % transport.channel_tile:
                chunk(transport_ci // transport.channel_tile, transport_ci % transport.channel_tile)
        b.loop(0, transport.window.kernel[0] * transport.window.kernel[1], 1, tap)
        # Preparation rows are pairwise disjoint; the full completion makes
        # them visible before the first original product and closes entry.
        enc.fence()

    pending = None
    host_preparations = []
    def lower(kind, f):
        nonlocal pending
        if kind == "pack_original_input":
            host_preparations.append(emit_checked_window_preparation(enc.fn,
                packed, inp, attrs["packed_input_workspace"], f,
                transport="contiguous_taps" if selected_schedule == "packed_runs" else "window_bytes"))
            # Publish the completed CPU stores before a device reads this
            # private panel. The prior raw tile already completed before reuse.
            enc.fence()
        elif kind == "load_input":
            source_input = attrs["packed_input_workspace"] if f.get("packed_input") else inp
            src = enc.ptr_zero() if f["zero"] else enc.offset(source_input, f["first_element"])
            enc.load(src, f["local_row"], f["rows"], f["columns"], f["row_stride_elements"])
        elif kind == "load_weight":
            if resident is not None:
                if pending is not None:
                    raise ValueError("fused halo lost original weight/product pairing")
                pending = f
                return
            enc.load(offset(weight, f["reduction_row"], f["output_column"]),
                f["local_row"], f["rows"], f["columns"], co, stream=1)
        elif kind == "initialize_bias":
            enc.load(enc.offset(bias, op("mul", f["output_column"], 4)),
                op("add", f["accumulator_row"], 1 << 31), f["rows"], f["columns"],
                0, stream=2, shrunk=False)
        elif kind == "compute":
            rhs = f["rhs_row"]
            if resident is not None:
                if pending is None:
                    raise ValueError("fused halo has no paired original RHS")
                rhs = resident_row(pending["reduction_row"], pending["output_column"])
                pending = None
            def product(accumulate):
                enc.preload(rhs, f["accumulator_row"], f["depth"],
                    f["columns"], f["rows"], accumulate=accumulate)
                enc.compute(f["lhs_row"], f["depth"], f["rows"])
            if p.bias:
                product(True)
            else:
                b.branch(f["first_reduction"], lambda: product(False), lambda: product(True))
        elif kind == "finish_raw_tile":
            pass  # The checked raw ACC halo remains live for its original pool.
        elif kind == "store_spatial_max":
            enc.store(offset(out, f["output_row"], f["output_column"]),
                f["accumulator_row"], 1, f["columns"], co, dtype="i8",
                scale=gain, relu=relu, pool_out_dim=f["output_pixels"],
                pool_size=f["window_height"], pool_stride=f["stride_width"],
                porows=1, pocols=f["output_pixels"], orows=f["local_height"],
                ocols=f["local_width"], upad=0, lpad=0)
        else:
            raise ValueError("unimplemented direct spatial producer effect: " + kind)
        if not issue_mode:
            enc.fence()

    if issue_mode:
        issue_law = ConvolutionLoadIssueLaw(hardware, "i8-products-exact-Q1-max-i8",
            p.identity, "native-ordering:" + ordering["ordering_sha256"],
            max(p.row_tile, device_runs.required_rows if device_runs is not None else 0) + 1 + int(p.bias), local_range_dependencies=True,
            ordered_load_configuration=True, paired_descriptor_compute=True,
            finite_queue_backpressure=True, complete_device_drain=True,
            immutable_mapped_disjoint_buffers=True)
        if retained is not None:
            epoch_law = None
            pairs = kh * kw * chunks
            if selected_schedule in ("device_runs", "retain_epochs", "packed_raw", "packed_shared", "packed_runs") and pairs > 1:
                epoch_law = ConvolutionComputeEpochLaw(hardware, "i8-products-exact-Q1-max-i8",
                    p.identity, "native-ordering:" + ordering["ordering_sha256"], pairs,
                    local_read_after_write=True, local_write_after_read=True,
                    local_write_after_write=True, accumulator_result_visibility=True,
                    operand_consumption_retirement=True, ordered_configuration_capture=True,
                    adjacent_descriptor_compute=True, finite_queue_pair_progress=True,
                    complete_device_drain=True, exclusive_private_state=True,
                    pending_compute_pairs=ordering["pending_compute_pairs"])
            if shared_packed is not None:
                receipt = emit_packed_raw_halo_epochs(shared_packed, b, lower,
                    lambda kind, fields: enc.fence(), law=issue_law,
                    epoch_law=epoch_law, compute_pairs=pairs)
            else:
                receipt = emit_spatial_raw_retention(retained, b, lower, lambda kind, fields: enc.fence(),
                    law=issue_law, epoch_law=epoch_law, compute_pairs=pairs if epoch_law else None, packed_consumer=packed, tile_consumer=device_runs)
        else:
            receipt = emit_pool_load_issue(plan, b, lower,
                lambda kind, fields: enc.fence(), law=issue_law,
                peel=selected_schedule == "issue_peel")
    else:
        receipt = emit_spatial_producer(plan, b, lower, synchronous_completion=True)
    if pending is not None:
        raise ValueError("fused halo leaves an unmatched weight transfer")
    if resident is not None:
        resident.verify_current()
    enc.fence()
    receipt["device_runs"] = None if device_runs is None else dict(identity=device_runs.identity, private_rows=device_runs.required_rows, host_payload_access=False)
    receipt["packed_input"] = None if packed is None else dict(identity=packed.identity, workspace_bytes=packed.workspace_bytes, reduction=packed.reduction, original_reduction_order="ky,kx,ci")
    if host_preparations:
        receipt["host_preparations"] = host_preparations
    receipt["rhs_residency"] = None if resident is None else resident.identity
    receipt["horizontal_flat"] = None if flat_runs is None else dict(identity=flat_runs.identity, rhs_layout="original-nonresident-HWCF", host_payload_access=False)
    receipt["requested_schedule"] = "device_flat_runs" if flat_requested else requested_schedule
    receipt["realized_schedule"] = ("device_flat_runs" if flat_runs is not None else selected_schedule) if issue_mode else "sync"
    receipt["ordering_admission"] = ordering
    receipt["schedule_geometry_admitted"] = policy_admitted
    receipt["original_pool_pad_value"] = original_pad
    receipt["padding_replacement"] = ("nonnegative-Q1-all-windows-intersect" if original_pad == 0 and any(spatial.padding) else None)
    return receipt
