"""Opt-in convolution family comparison with exact traces and no cycle score.

All address traversal belongs to Merlin. This target adapter interprets only
verified primitive fields. Requested DMA bytes are payloads, not physical bus
transactions or cache misses. A Pareto ranking is a structural search screen;
command service, banking and overlap remain unpriced and performance UNKNOWN.
"""
from collections import Counter
from dataclasses import asdict

from merlin.llvmlower.static_llvm_cfg import StaticInt, StaticPointer, trace_static_function
from .golden_device_lower import lower
from .golden_flat_conv import GoldenFlatConv
from .golden_resident_conv import GoldenResidentConv, choose_compact_resident
from .ir import gemmini_dialect as G
from .tables import isa, rtl_facts as F


class ConvFootprintError(ValueError):
    pass


def _union_bytes(intervals):
    end = -1
    total = 0
    for lo, hi in sorted(intervals):
        if hi > end:
            total += hi - max(lo, end)
            end = hi
    return total


def footprint(generator, *, max_steps=10000000):
    """Consume a complete current primitive CFG; partial traces never count.

    Dense NHWC/HWIO and one direct output are required. The constructors prove
    scratchpad/accumulator legality. Every requested DRAM row is additionally
    checked against its typed operand extent, including tails and zero fills.
    """
    if type(generator) not in (GoldenFlatConv, GoldenResidentConv):
        raise ConvFootprintError('No exact footprint adapter for selected family')
    if generator.store_plan is not None:
        raise ConvFootprintError('Paired output footprint requires a separate exact adapter')
    shape = generator.conv
    if shape.explicit_halo:
        raise ConvFootprintError('Source-proven unpadded NHWC input required')
    # Build a fresh emitter. Tracing must not append to the caller builder.
    module = generator.with_emission_options().build()
    module.verify()
    lower(module.clone()).verify()
    fn = module.body.block.first_op
    if len(fn.body.blocks.first.args) != 3:
        raise ConvFootprintError('Three typed direct operands required')
    output_width = 4 if shape.output_dtype == 'i32' else 1
    sizes = (shape.h * shape.w * shape.cin, 9 * shape.cin * shape.cout,
             shape.oh * shape.ow * shape.cout * output_width)
    arguments = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    counts = Counter()
    requests = Counter()
    intervals = {i: [] for i in range(3)}
    load_configs = {}
    store_config = None
    last_preload = None
    geometry = Counter()
    zero_bytes = 0
    total_commands = 0
    for step in trace_static_function(fn, arguments, observe=lambda op: isinstance(op, G._GemminiOp),
                                      pointer_index_bits=64, max_steps=max_steps):
        op = step.operation
        if type(op) not in (G.ConfigExOp, G.ConfigLdOp, G.ConfigStOp, G.MvinOp, G.MvoutOp, G.PreloadOp, G.ComputeOp, G.FlushOp, G.FenceOp):
            raise ConvFootprintError('Primitive command has no exact cost adapter')
        total_commands += 1
        counts[op.name.removeprefix('gemmini.')] += 1
        if isinstance(op, G.ConfigExOp):
            if op.a('dataflow') != isa.WEIGHT_STATIONARY or op.a('a_transpose', 0) or op.a('b_transpose', 0):
                raise ConvFootprintError('Unsupported dataflow/transpose geometry')
        elif isinstance(op, G.ConfigLdOp):
            if op.a('scale', 1.0) != 1.0:
                raise ConvFootprintError('Scaled operand DMA not priced')
            if op.a('pixel_repeats', 1) != 1:
                raise ConvFootprintError('Repeated-pixel DMA not priced')
            load_configs[op.a('load_id')] = (op.a('stride'), op.a('shrunk', 0))
        elif isinstance(op, G.ConfigStOp):
            if any(op.a(key, 0) for key in ('pool_stride', 'pool_size', 'pool_out_dim')):
                raise ConvFootprintError('Pooling store not priced')
            store_config = op.a('stride')
        elif isinstance(op, (G.MvinOp, G.MvoutOp)):
            if len(step.inputs) != 1 or not isinstance(step.inputs[0], StaticPointer):
                raise ConvFootprintError('DMA pointer unresolved')
            ptr = step.inputs[0]
            rows, cols = op.a('rows'), op.a('cols')
            width = 1
            if isinstance(op, G.MvinOp):
                stream = op.a('load_id', 0)
                if stream not in load_configs:
                    raise ConvFootprintError('DMA load configuration missing')
                stride, shrunk = load_configs[stream]
                if shrunk:
                    raise ConvFootprintError('Shrunk operand DMA not priced')
                if ptr.base is None:
                    if ptr.offset.value != 0:
                        raise ConvFootprintError('Absolute nonzero DMA pointer not priced')
                    zero_bytes += rows * cols
                    counts['mvin_zero'] += 1
                    continue
                if ptr.base not in (0, 1):
                    raise ConvFootprintError('Load does not borrow the declared typed inputs')
                counts['mvin_a' if ptr.base == 0 else 'mvin_b'] += 1
            else:
                if ptr.base != 2 or store_config is None:
                    raise ConvFootprintError('Store does not target the declared typed output')
                width = 4 if op.a('local') & isa.ACC_FULL_ROW_BIT else 1
                if width != output_width:
                    raise ConvFootprintError('Store width differs from typed output')
                stride = store_config
            requests[ptr.base] += rows * cols * width
            for row in range(rows):
                lo = ptr.offset.signed + row * stride
                hi = lo + cols * width
                if lo < 0 or hi > sizes[ptr.base]:
                    raise ConvFootprintError('DMA request exceeds typed operand storage')
                intervals[ptr.base].append((lo, hi))
        elif isinstance(op, G.PreloadOp):
            last_preload = (op.a('c_rows'), op.a('c_cols'), op.a('bd_rows'), op.a('bd_cols'))
        elif isinstance(op, G.ComputeOp):
            if last_preload is None:
                raise ConvFootprintError('Compute has no resolved output geometry')
            geometry[(op.a('a_rows'), op.a('a_cols'), last_preload[1])] += 1
    unique = {str(i): _union_bytes(intervals[i]) for i in range(3)}
    if unique['2'] != sizes[2] or requests[2] != sizes[2]:
        raise ConvFootprintError('Output trace is incomplete or overlaps writes')
    logical_macs = shape.oh * shape.ow * shape.cin * shape.cout * 9
    issued_macs = sum(m * k * n * count for (m, k, n), count in geometry.items())
    # Full DIM is explicitly a padded issue proxy, not resolved controller time.
    result = dict(complete=True, family=type(generator).__name__, source_shape=asdict(shape),
                  typed_storage_bytes={str(i): n for i, n in enumerate(sizes)},
                  requested_payload_bytes={str(i): requests[i] for i in range(3)},
                  unique_requested_bytes=unique, zero_fill_payload_bytes=zero_bytes,
                  command_counts=dict(sorted(counts.items())), primitive_commands=total_commands,
                  compute_geometry=[dict(m=m, k=k, n=n, commands=count) for (m, k, n), count in sorted(geometry.items())],
                  logical_padded_convolution_MAC=logical_macs, issued_MAC=issued_macs,
                  padded_DIM_issue_proxy=counts['compute'] * F.DIM,
                  actual_A_feed_rows=sum(m * count for (m, _, _), count in geometry.items()),
                  performance='UNKNOWN', timing_claim=False,
                  unpriced=['physical transactions/cache misses', 'CPU instruction service',
                            'bank arbitration', 'preload/compute dispatch pairing', 'DMA/execute overlap'])
    result['rank_components'] = dict(
        requested_a=requests[0], requested_b=requests[1], requested_output=requests[2],
        zero_fill=zero_bytes, issued_MAC=issued_macs, padded_DIM_issue=result['padded_DIM_issue_proxy'],
        A_feed_rows=result['actual_A_feed_rows'],
        **{'commands_' + name: counts[name] for name in ('config_ex', 'config_ld', 'config_st', 'flush', 'fence',
                                                        'mvin', 'preload', 'compute', 'mvout')})
    return result


def choose_compact_frontier(control, *, source_virtual_padding=False, prefetch_b=False,
                            compact_commands=False, weight_issue_tiles=None, max_steps=10000000,
                            allow_command_repartition=False):
    """Select only a componentwise nonworse candidate; tradeoffs stay UNKNOWN.

    This opt-in policy is a resource/structural ranking. It does not claim that
    equal command counts or fewer requested bytes predict hardware cycles.
    The default compiler path does not invoke it. Explicit command repartition
    permission can generate a candidate with lower A traffic and nonworse payload,
    arithmetic, configuration and synchronization bounds. Its increased transfer,
    preload, compute and store command counts remain unpriced, so this is a search
    candidate requiring complete measurement, never a profitability proof.
    """
    if any(type(v) is not bool for v in (source_virtual_padding, prefetch_b, compact_commands, allow_command_repartition)):
        raise ValueError('Family selection requires explicit boolean source and prefetch facts')
    decision = dict(applied=False, automatic_policy=False, performance='UNKNOWN', timing_claim=False,
                    policy='compact_channel_planes_frontier', rank='componentwise nonworse; no scalar cycle score')
    if not source_virtual_padding:
        return control, dict(decision, refusal='Source-proven virtual padding required')
    if type(control) not in (GoldenFlatConv, GoldenResidentConv):
        return control, dict(decision, refusal='No exact footprint adapter for selected family')
    try:
        candidate, options = choose_compact_resident(control.conv, prefetch_b=prefetch_b)
        if compact_commands:
            candidate = candidate.with_emission_options(compact_commands=True)
        if weight_issue_tiles is not None:
            from .golden_resident_conv import issue_resident_weight_packets
            candidate, packets = issue_resident_weight_packets(candidate, tiles=weight_issue_tiles)
            if not packets['applied']:
                raise ConvFootprintError(packets['refusal'])
        if getattr(control, 'tail_before_last_full', False):
            raise ConvFootprintError('Compact candidate has no proved equivalent spatial tail ordering')
        before = footprint(control, max_steps=max_steps)
        after = footprint(candidate, max_steps=max_steps)
    except ValueError as failure:
        return control, dict(decision, refusal=str(failure))
    if before['typed_storage_bytes'] != after['typed_storage_bytes'] or before['logical_padded_convolution_MAC'] != after['logical_padded_convolution_MAC']:
        return control, dict(decision, refusal='Candidate semantic footprint differs')
    old, new = before['rank_components'], after['rank_components']
    better = [key for key in old if new[key] < old[key]]
    worse = [key for key in old if new[key] > old[key]]
    decision.update(control=before, candidate=after, candidate_options=asdict(candidate.emission_options),
                    lower_components=better, higher_components=worse,
                    resource_legality='Both actual constructors and complete typed DMA bounds pass',
                    unpriced=before['unpriced'])
    decision['applied'] = bool(better) and not worse
    if allow_command_repartition:
        repartitioned = {'commands_mvin', 'commands_preload', 'commands_compute',
                         'commands_mvout', 'padded_DIM_issue'}
        protected_worse = [key for key in worse if key not in repartitioned]
        permitted_candidate = (new['requested_a'] < old['requested_a']
                               and not protected_worse)
        decision['command_repartition_permission'] = dict(
            explicit=True, candidate=permitted_candidate,
            protected_higher_components=protected_worse,
            unpriced_higher_components=[key for key in worse if key in repartitioned],
            primitive_commands_delta=after['primitive_commands']-before['primitive_commands'],
            command_count_deltas={key: after['command_counts'].get(key,0)-before['command_counts'].get(key,0)
                                  for key in sorted(before['command_counts'].keys()|after['command_counts'].keys())},
            unpriced_higher_command_counts={key: after['command_counts'].get(key,0)-before['command_counts'].get(key,0)
                                           for key in sorted(before['command_counts'].keys()|after['command_counts'].keys())
                                           if after['command_counts'].get(key,0)>before['command_counts'].get(key,0)},
            profitability='UNKNOWN; complete paired measurement required')
        if permitted_candidate and not decision['applied']:
            decision.update(applied=True, comparison='UNPRICED_COMMAND_REPARTITION_CANDIDATE',
                            refusal=None)
            return candidate, decision
    decision['comparison'] = 'STRUCTURAL_DOMINANCE' if decision['applied'] else ('UNKNOWN_TRADEOFF' if better and worse else 'NO_STRUCTURAL_GAIN')
    decision['refusal'] = None if decision['applied'] else ('Unpriced service/overlap and opposing components require complete paired measurement' if better and worse else 'No componentwise strict reduction')
    return (candidate if decision['applied'] else control), decision
