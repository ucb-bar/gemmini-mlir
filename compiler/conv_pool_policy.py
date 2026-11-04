"""Optional private-pool-patch input retention through existing shared emitters.

Every original patch still owns complete accumulator rows through its real
StoreController max observation. Retention shares one input row run across
output-channel tiles. It does not cache convolution values across overlapping
pool patches or claim that fewer transfers imply a latency improvement.
"""
from pathlib import Path
from merlin.compile.tuning_knobs import knob

VALUES=('off','retained_input','retained_input_pairs16')
DECLARATION={
    'values':list(VALUES),
    'mechanism':'pooled convolution private-patch input retention and fact-backed epochs',
    'note':'Off preserves incumbent. Alternatives share input runs across channel tiles, retain original pooling/readout fences, and require paired complete-cycle validation.',
}


def selected():
    return knob('conv_pool_schedule',default='off',values=VALUES,root=Path(__file__).parent)


def options(plan,*,pool_size,per_column=False,request=None):
    """Resolve a finite actual route; decline before issuing any target command."""
    from conv_policy import selected as storage,epoch_selection,completion_laws
    request=selected() if request is None else request
    if request not in VALUES:
        raise ValueError('unknown pooled convolution schedule')
    current=(storage(),epoch_selection())
    receipt=dict(requested=request,applied=False,storage=current[0],completion_epochs=current[1],
        latency_evidence='unmeasured pooled alternative; no default promotion')
    def fallback(reason):return current[0],current[1],dict(receipt,reason=reason)
    if request=='off':return fallback('disabled')
    if current!=('synchronous','off'):return fallback('explicit_convolution_options')
    edge=plan.primitive
    tiles=(plan.output_channels+edge-1)//edge
    patch_rows=pool_size*pool_size
    if (per_column or plan.column_tile!=edge):return fallback('nonuniform_readout_partition')
    if not 1<=pool_size<=3:return fallback('unsupported_pool_patch')
    # A single N tile cannot share an input across columns, but the checked
    # epoch emitter still coalesces its real reduction PRELOAD/COMPUTE pairs.
    # Keep plain retention's original fallback; admit one tile only when that
    # distinct completion schedule was explicitly requested and proven below.
    if tiles<2 and request!='retained_input_pairs16':
        return fallback('no_cross_column_input_reuse')
    # Pooling multiplies static column partitions by up to three patch rows
    # and nine clipped border families. Keep expansion below the general
    # nonpool emitter's 32-column budget, independently of image dimensions.
    if tiles>8:return fallback('bounded_pool_column_expansion')
    if plan.accumulator_row+tiles*patch_rows>plan.accumulator_rows:
        return fallback('private_patch_accumulator_capacity')
    mode='pairs16' if request=='retained_input_pairs16' else 'off'
    if mode!='off' and completion_laws(plan,plan.identity,tiles,mode=mode) is None:
        return fallback('selected_ordering_facts_unavailable')
    return 'retained_input',mode,dict(receipt,applied=True,reason=('single_column_tile_compute_epochs' if tiles==1 else 'private_patch_retained_windows'),
        storage='retained_input',completion_epochs=mode,live_column_tiles=tiles,
        private_accumulator_rows=tiles*patch_rows,
        retained_lifetime=('one original input row run across reduction compute epochs' if tiles==1 else 'one original input row run across independent output-column tiles'),
        observation_lifetime='each private full patch through its source maxpool readout fence')


# Independent opt-in: existing private-patch choices above remain unchanged.
MATERIALIZATION_VALUES = ('off', 'quantized_once')
MATERIALIZATION_DECLARATION = {
    'values': list(MATERIALIZATION_VALUES),
    'mechanism': 'complete quantized convolution once followed by library spatial-max strips',
    'note': 'Requires an externally mapped full prepool i8 workspace. No default promotion or timing claim.',
}


def materialization_selected():
    return knob('conv_pool_materialization', default='off',
                values=MATERIALIZATION_VALUES, root=Path(__file__).parent)


def plan_quantized_once(*, batches, input_hw, channels, pool_size, pool_stride,
                        padding=(0, 0, 0, 0), pad_value=-128, relu=False):
    """Plan the second stage; caller preserves the complete original Q1 producer.

    input_hw describes the *prepool convolution output*. The producer must write
    contiguous NHWC signed i8 with its original bias, exact f32 scale(s), rounding,
    saturation and ReLU before this stage. No scale is moved across maximum.
    Padding zero is equivalent to minimum padding only after source ReLU.
    A returned plan owns accumulator rows [0, temporary_rows) until each fence.
    """
    import json
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max
    from isa import DIM
    if any(type(x) is not int or x <= 0 for x in (batches, channels, *input_hw)):
        raise ValueError('quantized-once requires positive static NHWC extents')
    if type(relu) is not bool or len(padding) != 4:
        raise ValueError('explicit source ReLU and top/left/bottom/right padding required')
    if not 1 <= pool_size <= 3 or not 1 <= pool_stride <= 3:
        raise ValueError('selected StoreController supports square pool sizes/strides one through three')
    if any(padding) and not (pad_value == -128 or (pad_value == 0 and relu)):
        raise ValueError('clamped valid-cell staging requires minimum padding, or zero after source ReLU')
    facts = json.loads(Path(__file__).with_name('facts.json').read_text())['facts']
    acc = next(x for x in facts['memory_address_spaces'] if x['name'] == 'accumulator')
    if acc['lanes'] != DIM or acc['lane_bits'] != 32:
        raise ValueError('selected accumulator is not signed i32 sixteen-lane storage')
    source = SpatialMaxReadout(tuple(input_hw), (pool_size, pool_size),
                              (pool_stride, pool_stride), tuple(padding), -128)
    p = plan_striped_spatial_max(source, batches=batches, channels=channels,
        channel_block=DIM, transfer_rows=DIM, capacity_rows=acc['bytes']//acc['row_bytes'])
    p.revalidate()
    # Actual CONFIG_ST fields are eight-bit dimensions. Larger strips decline
    # instead of truncating fields or silently changing the source partition.
    if max(*p.stripe_hw, *source.output_hw) >= 256 or channels >= 1 << 32:
        raise ValueError('spatial strip exceeds selected StoreController dimension/stride fields')
    if batches * input_hw[0] * input_hw[1] * channels >= 1 << 63:
        raise ValueError('prepool workspace exceeds signed i64 byte addressing')
    return p


def emit_quantized_once(builder, plan, *, intermediate, output, workspace_bytes,
                        emit_producer):
    """Emit real shared producer then hardware maximum, without host tensor math.

    ``emit_producer(intermediate_pointer)`` must invoke the ordinary convolution
    emitter exactly once, with pooling removed and the original quantized i8
    prepool output shape/readout attributes preserved. The parent supplies a
    mapped, nonaliasing workspace (not a large stack allocation) and owns its
    lifetime. This API deliberately does not infer pointers or rewrite source.
    It can handle scalar or per-column Q1 because the completed i8 image is the
    sole input of the identity-readout max stage.
    """
    from isa import ACC
    from merlin.compile.scheduling.llvm_program import emit_index_program
    checked = plan.revalidate()
    canonical = plan_quantized_once(batches=plan.batches, input_hw=plan.source.input_hw,
        channels=plan.channels, pool_size=plan.source.window[0],
        pool_stride=plan.source.stride[0], padding=plan.source.padding,
        pad_value=plan.source.pad_value)
    if plan != canonical:
        raise ValueError('pool plan differs from the selected hardware-bound canonical strip')
    # Exhaust the complete second-stage input domain, not sample accumulators.
    # Q1 already narrowed to i8; sign-extension and identity readout therefore
    # retain each exact value before the signed maximum.
    from merlin.llvmlower.requantization import quantized
    if any(quantized(x, (1.0,), False) != x for x in range(-128, 128)):
        raise ValueError('selected identity readout does not preserve signed i8')
    h, w = plan.source.input_hw
    n = plan.channels
    required = plan.batches * h * w * n
    if type(workspace_bytes) is not int or workspace_bytes < required:
        raise ValueError('mapped prepool workspace is smaller than the original NHWC i8 image')
    if intermediate == output:
        raise ValueError('prepool workspace must not alias the pooled output')
    if not callable(emit_producer):
        raise ValueError('a shared complete original quantized convolution producer is required')
    # Producer completion is a real host-memory visibility and ACC ownership
    # boundary. Following loads overwrite every observed strip cell explicitly.
    producer_receipt = emit_producer(intermediate)
    builder.fence()
    builder.config_ld(n, shrink=True, id=2)
    kh, stripe_width = plan.stripe_hw
    _, ow = plan.source.output_hw
    builder.config_st(n, activation=0, scale=1.0,
        pool_stride=plan.source.stride[0], pool_size=plan.source.window[0],
        pool_out_dim=ow, porows=1, pocols=ow, orows=kh, ocols=stripe_width,
        upad=0, lpad=0)

    def address(base, row, column):
        return builder.ptr_offset(base, builder.op('add', builder.op('mul', row, n), column))

    def lower(name, a):
        if name == 'stage_spatial_rows':
            # shrink=True loads signed i8 into signed i32 accumulator lanes.
            # Every staged byte comes from Q1, including duplicated border cells.
            builder.mvin(address(intermediate, a['source_row'], a['first_column']),
                builder.op('or', a['local_row'], ACC),
                cols=a['columns'], rows=a['rows'], id=2)
            builder.fence()
        elif name == 'read_spatial_strip':
            builder.mvout(address(output, a['output_row'], a['first_column']),
                          ACC, cols=a['columns'], rows=0)
            builder.fence()
        else:
            raise ValueError('unknown checked spatial strip effect: ' + name)

    emitted = emit_index_program(plan.kernel, builder, lower)
    return dict(route='quantized_once_then_library_spatial_max_strips',
        workspace_bytes=required, producer=producer_receipt, library_plan=checked,
        emission=emitted, target_execution=False, timing=None,
        numerical_contract='original Q1 unchanged; identity f32 readout is exact on signed i8; signed maximum observes those exact Q1 values',
        padding_contract='duplicate valid border cells; minimum signed i8 padding, or source zero padding after ReLU',
        lifetime='complete full original producer, then complete every strip load/readout before reuse')
