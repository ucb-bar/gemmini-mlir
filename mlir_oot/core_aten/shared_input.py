"""Shared-input sibling contractions: one resident input, independent readouts.

Merlin selects the sibling group (``find_shared_input_groups``) and its checked
logical schedule (``schedule_shared_input``). This module realizes that schedule
as one primitive stream: every tile of the shared input is moved in once per M
block, each sibling's weights stream through, and each sibling drains under its
own readout configuration (its own multiplier bits and activation, or a full
i32 exit). Constant weights are stored as ``[ceil(N/w)][K][w]`` column panels
with the panel width ``w`` derived from the synced contract and RTL facts.

Admission never trusts the emitter's records. ``admit`` decodes the emitted
loads, descriptors, computes, stores and configuration registers back into
logical effects, re-plans every sibling readout from the source, and passes
the decoded effects to Merlin's independent shared-input checker. Anything
unproved declines to the ordinary per-contraction kernels.
"""

from __future__ import annotations

import numpy as np

from .source import module_of
from .match import Decline, Match, ResultBuffer, dense_values, shape
from .stream import ACC, ACC_ROWS, DIM, GARBAGE, NO_ACT, SP, SPAD_ROWS, Arg

OPERATION = 'merlin.shared_input'


def panel_width():
    """Widest whole-tile panel one movement row carries; ``ValueError`` when underived.

    ``memory_model.dma.max_transfer_bytes`` is the contract's declared payload
    bound; the tile edge and the operand width come from the RTL facts.
    """
    from merlin.targetgen.rtl.facts import load_facts
    from .readout_encoding import contract
    from .storage import itemsize

    declared = ((contract().get('memory_model') or {}).get('dma') or {}).get('max_transfer_bytes')
    facts = (load_facts(contract()['name']) or {}).get('facts') or {}
    arrays = facts.get('arrays') or []
    operand = next((d.get('dtype') for d in facts.get('datapaths') or [] if d.get('name') == 'input'), None)
    if type(declared) is not int or declared <= 0 or not arrays or operand is None:
        raise ValueError('UNKNOWN panel width: transfer payload, array edge or operand width is undeclared')
    edge = arrays[0].get('cols')
    if edge != DIM:
        raise ValueError('RTL array edge differs from the provider tile primitive')
    tiles = declared // (edge * itemsize(operand))
    if tiles < 1:
        raise ValueError('declared transfer payload holds no whole operand tile row')
    return tiles * edge


def _views(group):
    """Private scratchpad/accumulator assignments for Merlin's schedule search."""
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseResources

    tiles_m, tiles_k = -(-group.m // DIM), -(-group.k // DIM)
    lhs_rows = min(tiles_m * tiles_k * DIM, (SPAD_ROWS // 2) // DIM * DIM)
    storage = DenseLoopStorage(
        DenseMemoryView('shared_input', 'scratchpad', DIM, 0, lhs_rows, DIM),
        DenseMemoryView('sibling_weights', 'scratchpad', DIM, lhs_rows, SPAD_ROWS - lhs_rows, DIM),
        DenseMemoryView('sibling_outputs', 'accumulator', DIM * 4, 0, ACC_ROWS, DIM))
    tiles_n = sum(-(-n // DIM) for n in group.widths)
    resources = DenseResources(SPAD_ROWS * DIM, ACC_ROWS * DIM * 4,
                               max_m_blocks=min(tiles_m, 8), max_n_blocks=min(tiles_n, 16),
                               max_k_blocks=min(tiles_k, 32),
                               partial_output_tiles=True, partial_output_blocks=True)
    return (DIM, DIM, DIM), storage, resources


class _Member:
    def __init__(self, ordinal, op, plan, chain):
        self.ordinal, self.op, self.plan, self.chain = ordinal, op, plan, chain
        self.k, self.n = shape(op.operands[1])
        proof = plan.decision.get('proof') or {}
        self.narrow = plan.route == 'epilogue'
        self.scale_operand = proof.get('scale_operand') if self.narrow else None
        self.scale_ref = None
        self.bits = (tuple(int(v) for v in proof['multiplier_bits'])
                     if self.narrow and self.scale_operand is None else None)
        self.relu = self.narrow and 'relu' in plan.decision['stages']
        self.result = chain.ops[-1].results[0] if self.narrow else op.results[0]
        self.consumed = list(chain.ops) if self.narrow else []
        if self.narrow and np.any(np.asarray(proof.get('offsets', 0))):
            raise Decline('shared-input siblings do not seed accumulator offsets')


def _bind_scales(module, members, operands, args):
    for member in members:
        if member.scale_operand is not None:
            from .runtime_scale import bind_scale_operand
            member.scale_ref = bind_scale_operand(member.scale_operand, module, operands, args)


def _readout_matches(member, scale, column, columns):
    if member.scale_ref is not None:
        return scale == member.scale_ref
    return set(member.bits[column:column + columns]) == {scale}


def _reference(member, acc, stream, buffers):
    from merlin.runtime.exact_readout import f32_readout

    if not member.narrow:
        return acc.astype(np.int32)
    if member.scale_ref is not None:
        from .runtime_scale_replay import runtime_scale_value
        scale = runtime_scale_value(member.scale_ref, stream.args, buffers)
    else:
        scale = np.array(member.bits, np.uint32).view(np.float32)
    return f32_readout(acc, scale, 'i8', relu=member.relu)


def _members(module, group, readout_options):
    from xdsl.dialects import func
    from .integer_regions import _readout_plan

    function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    body = tuple(function.body.block.ops)
    members = []
    for index, ordinal in enumerate(group.contraction_ordinals):
        op = body[ordinal]
        planned = _readout_plan(module, op, readout_options)
        if planned is None:
            raise Decline(f'sibling {index} has no proved readout composition')
        if planned[0].route not in ('epilogue', 'int32_readout'):
            raise Decline(f'sibling {index} has an unsupported shared-input readout route: {planned[0].route}')
        members.append(_Member(ordinal, op, *planned))
    return body, members


def _layouts(members):
    from merlin.compile.scheduling.panel_layout import PanelLayout
    try:
        width = panel_width()
    except ValueError:
        width = None
    layouts = []
    for member in members:
        values = dense_values(member.op.operands[1])
        layouts.append(PanelLayout(member.k, member.n, width) if values is not None and width else None)
    return width, layouts


def _stride(layout, member):
    return layout.row_pitch if layout is not None else member.n


def _offset(layout, member, row, col):
    return layout.offset(row, col) if layout is not None else row * member.n + col


def match_shared_input(module, *, readout_options):
    """Fused matches for every admissible sibling group, plus decline records."""
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups, grow_shared_input_group

    if readout_options is None:
        return [], []
    try:
        groups = find_shared_input_groups(module)
    except ValueError as exc:
        return [], [dict(op=OPERATION, reason=str(exc))]
    matches, declines = [], []
    for group in groups:
        for prefix in reversed(tuple(grow_shared_input_group(group))):
            try:
                match = _fuse(module, prefix, readout_options)
            except (Decline, ValueError) as exc:
                declines.append(dict(op=OPERATION, ordinals=list(prefix.contraction_ordinals), reason=str(exc)))
                continue
            if match is not None:
                matches.append(match)
                break
    return matches, declines


def _fuse(module, group, readout_options):
    from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input
    from .ops import Kernel, _begin
    from .readout_encoding import activation

    body, members = _members(module, group, readout_options)
    tile, storage, resources = _views(group)
    schedule = schedule_shared_input(module, group, primitive_tile=tile, storage=storage, resources=resources)
    if schedule is None:
        raise Decline('no shared-input schedule fits the private storage')
    width, layouts = _layouts(members)
    mixed = len({m.narrow for m in members}) > 1
    dtype = 'i8' if members[0].narrow else 'i32'
    bases, total = [], 0
    for member in members:
        bases.append(total)
        total += group.m * member.n
    args = [Arg('a', 'i8', (group.m, group.k), 'input')]
    args += [Arg(f'w{i}', 'i8', (m.k, m.n), 'input') for i, (m, layout) in enumerate(zip(members, layouts))
             if layout is None]
    operands = [('a', members[0].op.operands[0])] + [
        (f'w{i}', m.op.operands[1]) for i, (m, l) in enumerate(zip(members, layouts)) if l is None]
    _bind_scales(module, members, operands, args)
    if mixed:
        args += [Arg(f'out{i}', 'i8' if m.narrow else 'i32', (group.m, m.n), 'output')
                 for i, m in enumerate(members)]
    else:
        args.append(Arg('out', dtype, (total,), 'output'))
    s = _begin(args)
    for i, (member, layout) in enumerate(zip(members, layouts)):
        if layout is not None:
            weight = dense_values(member.op.operands[1]).astype(np.int8)
            s.const(f'w{i}', layout.pack(weight))
    s.config_ex()
    s.config_ld(0, group.k)
    load_stride = store_config = None
    for effect in schedule.operations:
        member, op = effect.member, effect.operation
        kind = type(op).__name__
        if kind == 'Load' and member is None:
            s.mvin(s.at('a', op.dram_row * group.k + op.dram_col), SP(op.row), op.cols, op.rows, 0)
        elif kind == 'Load':
            sibling, layout = members[member], layouts[member]
            stride = _stride(layout, sibling)
            if stride != load_stride:
                s.config_ld(1, stride)
                load_stride = stride
            s.mvin(s.at(f'w{member}', _offset(layout, sibling, op.dram_row, op.dram_col)),
                   SP(op.row), op.cols, op.rows, 1)
        elif kind == 'Preload':
            s.preload(GARBAGE if op.weight_row is None else SP(op.weight_row),
                      ACC(op.accumulator_row, accumulate=op.accumulate), op.cols, op.weight_rows, op.cols, op.rows)
        elif kind == 'Compute':
            s.compute(SP(op.input_row), op.cols, op.rows, accumulate=not op.fresh_weights)
        elif kind == 'Store':
            sibling = members[member]
            if sibling.narrow:
                if sibling.scale_ref is not None:
                    config = (sibling.n, activation(sibling.relu), None, sibling.scale_ref)
                else:
                    selected = set(sibling.bits[op.dram_col:op.dram_col + op.cols])
                    if len(selected) != 1:
                        raise Decline('sibling store tile spans two readout multipliers')
                    config = (sibling.n, activation(sibling.relu), selected.pop(), None)
            else:
                config = (sibling.n * 4, NO_ACT, None, None)
            if config != store_config:
                if config[3] is None:
                    s.config_st(config[0], config[1], multiplier_bits=config[2])
                else:
                    s.config_st(config[0], config[1], scale_operand=config[3])
                store_config = config
            element = 1 if sibling.narrow else 4
            offset = ((0 if mixed else bases[member]) + op.dram_row * sibling.n + op.dram_col) * element
            s.mvout(s.at(f'out{member}' if mixed else 'out', offset), ACC(op.accumulator_row, full=not sibling.narrow), op.cols, op.rows)
        else:
            raise Decline('unknown shared-input schedule effect')
    s.fence()
    meta = dict(
        shared_input=dict(group=group.record(), tile=list(tile), block=[schedule.block.m, schedule.block.k,
                          schedule.block.n], realization=schedule.realization, panel_width=width,
                          members=[dict(source_ordinal=m.ordinal, n=m.n, base=b, readout=m.plan.route,
                                        relu=m.relu, weight_layout=None if l is None else l.record(),
                                        composition_plan=m.plan.record())
                                   for m, b, l in zip(members, bases, layouts)]),
        integer_producer=True, integer_readout=any(m.narrow for m in members), raw_integer=all(not m.narrow for m in members),
        dimensions=dict(batch=0, m=group.m, n=sum(group.widths), k=group.k), seed='zero', bias=False,
        output_dtype='mixed' if mixed else dtype, relu=False,
        readout='mixed' if mixed else ('f32_product_rne' if members[0].narrow else 'full_i32'),
        device_readout='mixed' if mixed else ('f32_product_rne' if members[0].narrow else 'full_i32'),
        accumulator_min=-group.k * 16384, accumulator_max=group.k * 16384)
    kernel = Kernel(OPERATION, s, meta)
    consumed = [body[i] for i in group.source_op_indices] + [op for m in members for op in m.consumed]
    outputs = [(m.result, b, (group.m, m.n), m.op) for m, b in zip(members, bases)]
    return Match(OPERATION, members[0].op, consumed, operands,
                 (('out0', members[0].result) if mixed else ('out', members[-1].result)), kernel,
                 numeric=dict(integer_producer=True, shared_input=True, seed=None, runtime_seed=False,
                              raw_integer=all(not m.narrow for m in members), integer_readout=any(m.narrow for m in members)),
                 outputs=[] if mixed else outputs,
                 result_buffers=[ResultBuffer(m.result, f'out{i}', (group.m, m.n), m.op)
                                 for i, m in enumerate(members)] if mixed else [])


# ---------------------------------------------------------------------------------------------
# Admission: decode the emitted stream, never the emitter's plan
# ---------------------------------------------------------------------------------------------

def _decoded_load(command, load_state, names, layouts, members, group):
    from merlin.compile.scheduling.block_schedule import Load
    from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect
    from .stream_proof import load_model, local, _matches

    fields = command.fields
    place = local(fields['local_addr'])
    rows, cols = fields['num_rows'], fields['num_cols']
    if place.garbage or place.accumulator or command.arg is None or not 1 <= cols <= DIM or not 1 <= rows <= DIM:
        raise ValueError('shared-input stream seeds the accumulator or loads a multi-tile row')
    state = load_state.get(fields['state'])

    def require_load(stride):
        if state is None:
            raise ValueError('effective load configuration differs from the declared layout')
        _matches(state, {**load_model(), 'dram_stride': stride}, ('state_id',),
                 'effective load configuration differs from the declared layout')
    if command.arg == names['a']:
        require_load(group.k)
        row, col = divmod(command.off, group.k)
        effect = SharedInputEffect(None, Load('lhs', (row // DIM, col // DIM), row, col, rows, cols,
                                                    place.row, col // DIM))
    elif command.arg in layouts:
        member, layout = layouts[command.arg]
        sibling = members[member]
        require_load(layout.row_pitch if layout is not None else sibling.n)
        row, col = layout.source(command.off, cols) if layout is not None else divmod(command.off, sibling.n)
        effect = SharedInputEffect(member, Load('weight', (row // DIM, col // DIM), row, col, rows,
                                                      cols, place.row, row // DIM))
    else:
        raise ValueError('shared-input stream loads an unbound argument')
    return effect


def _decoded_store(command, store_state, names, members, bases, element, bits):
    from merlin.compile.scheduling.block_schedule import Store
    from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect
    from .readout_encoding import activation, code
    from .stream_proof import local

    fields = command.fields
    place = local(fields['local_addr'])
    outputs = [names['out']] if 'out' in names else [names[f'out{i}'] for i in range(len(members))]
    if (place.garbage or not place.accumulator or place.accumulate or command.arg not in outputs
            or store_state is None):
        raise ValueError('shared-input store is not an accumulator readout of a bound output')
    owner = None
    if 'out' not in names:
        member = owner = next(i for i in range(len(members)) if command.arg == names[f'out{i}'])
        element = 1 if members[member].narrow else 4
        position, remainder = divmod(command.off, element)
        position += bases[member]
    else:
        position, remainder = divmod(command.off, element)
    member = next((i for i in range(len(members) - 1, -1, -1) if bases[i] <= position), None)
    if remainder or member is None:
        raise ValueError('shared-input store is misaligned with every sibling result')
    if owner is not None and member != owner:
        raise ValueError('shared-input store position belongs to a different sibling output')
    sibling = members[member]
    row, col = divmod(position - bases[member], sibling.n)
    cols = fields['num_cols']
    stride, act, scale_bits = store_state['stride'], store_state['activation'], store_state['acc_scale']
    if place.full == sibling.narrow or stride != sibling.n * element:
        raise ValueError('sibling store width or pitch differs from its exit')
    if sibling.narrow:
        if not _readout_matches(sibling, scale_bits, col, cols) or act != activation(sibling.relu):
            raise ValueError('decoded sibling readout differs from its proved multiplier bits')
    elif act != NO_ACT or scale_bits != bits['f1']:
        raise ValueError('full-width sibling store applies a readout stage')
    if store_state['cmd_type'] != code('config_subtype', 'CONFIG_ST') or any(
            v for k, v in store_state.items() if k not in ('cmd_type', 'stride', 'activation', 'acc_scale')):
        raise ValueError('effective store configuration sets an unproved readout field')
    return SharedInputEffect(member, Store(row, col, fields['num_rows'], cols, place.row))


def _decoded_effects(stream, names, layouts, members, group, bases, element, bits):
    """Logical sibling effects of the emitted words, through ``stream_proof.decode``.

    Effective register state is validated at every command that depends on it:
    a configuration changed between uses and restored is still caught. Every
    decoded field must be one this realization models (an allowlist).
    """
    from merlin.compile.scheduling.block_schedule import Compute, Preload
    from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect
    from .stream_proof import decode, execute_model, local, _matches

    execute_state, load_state, store_state = None, {}, None
    effects, owners, stationary = [], {}, None
    pending_member = None
    unfenced = False
    for ins in stream.ins:
        command = decode(ins)
        kind, fields = command.kind, command.fields
        if kind == 'FENCE':
            unfenced = False
        if kind in ('FENCE', 'FLUSH'):
            continue
        if kind in ('PRELOAD', 'COMPUTE_PRELOADED', 'COMPUTE_ACCUMULATE', 'MVOUT'):
            if execute_state is None:
                raise ValueError('effective execute configuration is unset')
            _matches(execute_state, execute_model(), (),
                     'effective execute configuration differs from untransposed unit-stride WS')
        if kind == 'CONFIG_EX':
            execute_state = fields
        elif kind == 'CONFIG_LD':
            load_state[fields['state_id']] = fields
        elif kind == 'CONFIG_ST':
            store_state = fields
        elif kind in ('MVIN', 'MVIN2', 'MVIN3'):
            effect = _decoded_load(command, load_state, names, layouts, members, group)
            effects.append(effect)
            place = local(fields['local_addr'])
            rows = fields['num_rows']
            for r in range(rows):
                owners[place.row + r] = effects[-1].member, effects[-1].operation.dram_row + r, effects[-1].operation.dram_col
        elif kind == 'PRELOAD':
            bd, c = local(fields['bd_local_addr']), local(fields['c_local_addr'])
            if c.garbage or not c.accumulator or c.full or (not bd.garbage and bd.accumulator):
                raise ValueError('shared-input descriptor addresses the wrong local memory')
            if bd.garbage:
                if stationary is None:
                    raise ValueError('shared-input descriptor retains no resident weights')
                member, block, weight_row = stationary, None, None
            else:
                owner = owners.get(bd.row)
                if owner is None or owner[0] is None:
                    raise ValueError('shared-input descriptor reads unowned weight rows')
                member, block, weight_row = owner[0], (owner[1] // DIM, owner[2] // DIM), bd.row
                stationary = member
            pending_member = member
            effects.append(SharedInputEffect(member, Preload(weight_row, block, c.row, c.accumulate,
                                                             fields['c_num_rows'], fields['c_num_cols'],
                                                             fields['bd_num_rows'])))
            if fields['bd_num_cols'] != fields['c_num_cols']:
                raise ValueError('shared-input descriptor weight and output widths differ')
        elif kind in ('COMPUTE_PRELOADED', 'COMPUTE_ACCUMULATE'):
            a, bd = local(fields['a_local_addr']), local(fields['bd_local_addr'])
            if not bd.garbage or a.garbage or a.accumulator or pending_member is None:
                raise ValueError('shared-input compute differs from the independent dot product')
            owner = owners.get(a.row)
            block = None if owner is None else (owner[1] // DIM, owner[2] // DIM)
            effects.append(SharedInputEffect(pending_member, Compute(a.row, block, fields['a_num_rows'],
                                                                     fields['a_num_cols'],
                                                                     kind == 'COMPUTE_PRELOADED')))
            pending_member = None
        elif kind == 'MVOUT':
            effects.append(_decoded_store(command, store_state, names, members, bases, element, bits))
            unfenced = True
        else:
            raise ValueError('shared-input stream contains an unknown command')
    if unfenced:
        raise ValueError('shared-input stream ends with unfenced stores')
    return effects


def _replayed_outputs(replay, group, members, weights, bases, total):
    """Every output element is one store of one accumulator row holding exactly its source products."""
    from .readout_encoding import activation
    from .stream_proof import _f1, is_zero, require

    if 'out' in replay.content:
        content = replay.content['out']
    else:
        content = {}
        for i, sibling in enumerate(members):
            piece = replay.content.get(f'out{i}', {})
            size = group.m * sibling.n
            require(len(piece) == size and set(piece) == set(range(size)),
                    'stores do not write every sibling output element')
            content.update({bases[i] + flat: state for flat, state in piece.items()})
    require(len(content) == total and all(0 <= flat < total for flat in content),
            'stores do not write every sibling output element')
    require(sum(len(store.rows) * store.columns for store in replay.stores) == total,
            'stores write some sibling output element more than once')
    groups = {}
    for flat, (index, r, c) in content.items():
        groups.setdefault((index, r), []).append((c, flat))
    for (index, r), columns in groups.items():
        columns.sort()
        c0, flat0 = columns[0]
        require(all(c == c0 + i and f == flat0 + i for i, (c, f) in enumerate(columns)),
                'stored row is split across sibling output positions')
        cols = len(columns)
        member = max(i for i, base in enumerate(bases) if base <= flat0)
        sibling = members[member]
        row, n0 = divmod(flat0 - bases[member], sibling.n)
        require(n0 + cols <= sibling.n, 'stored row leaves its sibling output row')
        store = replay.stores[index]
        require(store.accumulator and store.full != sibling.narrow, 'sibling store width differs from its exit')
        if sibling.narrow:
            require(store.config['activation'] == activation(sibling.relu) and
                    _readout_matches(sibling, store.config['acc_scale'], n0, cols),
                    'decoded sibling readout differs from its proved multiplier bits')
        else:
            require(store.config['activation'] == NO_ACT and store.config['acc_scale'] == _f1(),
                    'full-width sibling store applies a readout stage')
        state = store.rows[r]
        require(state.seed is None or (is_zero(state.seed) and state.seed.width >= c0 + cols),
                'sibling accumulator holds a seed the source does not have')
        spans = []
        for a_tag, w_tags, depth, width in state.terms:
            require(a_tag.name == 'a' and a_tag.content is None and a_tag.width >= depth,
                    'mesh input is not the shared source input')
            a_row, k0 = divmod(a_tag.flat, group.k)
            require(a_row == row and k0 + depth <= group.k, 'mesh row reads a different shared input row')
            require(width >= c0 + cols and len(w_tags) == depth, 'mesh tile does not cover the stored columns')
            for j, tag in enumerate(w_tags):
                owner = weights.get(tag.name)
                require(owner is not None and owner[0] == member and tag.content is None and tag.width >= c0 + cols,
                        'stationary rows are not this sibling\'s source weight')
                layout = owner[1]
                k, col = layout.source(tag.flat, tag.width) if layout is not None else divmod(tag.flat, sibling.n)
                require(k == k0 + j and col + c0 == n0 and col + tag.width <= sibling.n,
                        'stationary operand differs from the source contraction')
            spans.append((k0, k0 + depth))
        position = 0
        for lo, hi in sorted(spans):
            require(lo == position, 'reduction tiles do not cover K exactly once')
            position = hi
        require(position == group.k, 'reduction tiles do not cover K exactly once')


def source_reference(match, options):
    """Plan each bound source result independently of the emitted sibling metadata."""
    from .integer_regions import _readout_plan

    plans = []
    outputs = ([(b.value, 0, b.shape, b.root) for b in match.result_buffers]
               if match.result_buffers else match.outputs)
    for value, offset, extent, op in outputs:
        planned = _readout_plan(module_of(op), op, options)
        if planned is None:
            raise ValueError('sibling result has no source readout plan')
        member = _Member(0, op, *planned)
        if member.result is not value:
            raise ValueError('sibling result differs from its source readout')
        plans.append((member, offset, extent))

    _bind_scales(module_of(match.root), [m for m, _, _ in plans], list(match.operands),
                 list(match.kernel.stream.args))

    def evaluate(inputs):
        buffers = {i: inputs[a.name] for i, a in enumerate(match.kernel.stream.args) if a.role == 'input'}
        bound = {value: inputs[name] for name, value in match.operands}
        pieces = []
        for member, offset, extent in plans:
            lhs = bound[member.op.operands[0]].astype(np.int64)
            weight = bound.get(member.op.operands[1])
            if weight is None:
                weight = dense_values(member.op.operands[1])
            if weight is None:
                raise ValueError('sibling source weight is unbound')
            acc = lhs @ np.asarray(weight, np.int64)
            result = _reference(member, acc, match.kernel.stream, buffers)
            if (not match.result_buffers and offset != sum(p.size for p in pieces)) or tuple(result.shape) != tuple(extent):
                raise ValueError('sibling source output slices are not contiguous')
            pieces.append(result.reshape(-1))
        if match.result_buffers:
            return {b.argument: p.reshape(b.shape) for b, p in zip(match.result_buffers, pieces)}
        return {'out': np.concatenate(pieces)}
    return evaluate


def _backstop(stream, members, weights, bases, trials=3):
    """Execute the decoded commands and compare bit for bit with Merlin's exact readout reference."""
    from .stream_proof import execute, require
    from .runtime_scale_replay import runtime_scale_trial

    scale_args = sorted({m.scale_ref.arg_index for m in members if m.scale_ref is not None})
    scale_offsets = {index: k for k, index in enumerate(scale_args)}
    rng = np.random.default_rng(0x51b1)
    names = {a.name: i for i, a in enumerate(stream.args)}
    for trial in range(trials):
        def operand(shape):
            if trial < 2:
                return np.full(shape, (-128, 127)[trial], np.int8)
            return rng.integers(-128, 128, shape).astype(np.int8)
        buffers, sources = {}, {}
        for index, arg in enumerate(stream.args):
            if arg.role == 'const':
                data = stream.consts[arg.name]
            elif arg.role == 'input':
                data = sources[arg.name] = (runtime_scale_trial(trial, offset=scale_offsets[index])
                                            if index in scale_offsets else operand(arg.shape))
            else:
                data = np.full(arg.shape, 0x5A if arg.dtype == 'i8' else 0x5A5A5A5A,
                               np.int8 if arg.dtype == 'i8' else np.int32)
            buffers[index] = np.ascontiguousarray(data).copy()
        execute(stream, buffers)
        actual = buffers[names['out']].reshape(-1) if 'out' in names else None
        lhs = sources['a'].astype(np.int64)
        for member, (sibling, base) in enumerate(zip(members, bases)):
            name = f'w{member}'
            weight = (sources[name] if weights[name][1] is None
                      else dense_values(sibling.op.operands[1])).astype(np.int64)
            acc = lhs @ weight
            require(np.all((acc >= -(1 << 31)) & (acc < (1 << 31))), 'reference accumulator leaves i32')
            reference = _reference(sibling, acc, stream, buffers)
            got = (actual[base:base + reference.size] if actual is not None
                   else buffers[names[f'out{member}']].reshape(-1))
            require(np.array_equal(got, reference.reshape(-1)),
                    f'instruction model differs from the exact reference on differential trial {trial}')


def admit(match, *, readout_options):
    """Return a proof record for ``match``'s emitted stream or raise ``ValueError``."""
    from merlin.compile.scheduling.panel_layout import PanelLayout
    from merlin.compile.scheduling.shared_input_groups import SharedInputGroup
    from merlin.compile.scheduling.shared_input_verification import verify_shared_input
    from .readout_encoding import flags

    record = match.kernel.meta['shared_input']
    fields = record['group']
    group = SharedInputGroup(fields['source_ir_sha256'], tuple(fields['source_op_indices']),
                             tuple(fields['contraction_ordinals']), fields['m'], fields['k'], tuple(fields['widths']))
    module = module_of(match.root)
    body, members = _members(module, group, readout_options)  # re-planned from source, not from meta
    stream = match.kernel.stream
    names = {a.name: i for i, a in enumerate(stream.args)}
    rebound_operands, rebound_args = list(match.operands), list(stream.args)
    _bind_scales(module, members, rebound_operands, rebound_args)
    if rebound_operands != match.operands or rebound_args != stream.args:
        raise ValueError('runtime scale argument differs from its source operand')
    bound = dict(match.operands)
    if bound.get('a') is not members[0].op.operands[0] or stream.args[names['a']].shape != (group.m, group.k):
        raise ValueError('shared input argument differs from the source group input')
    element = 1 if members[0].narrow else 4
    bases, total = [], 0
    for member in members:
        bases.append(total)
        total += group.m * member.n
    if match.result_buffers:
        expected = [ResultBuffer(m.result, f'out{i}', (group.m, m.n), m.op) for i, m in enumerate(members)]
        if match.outputs or match.result_buffers != expected:
            raise ValueError('sibling result buffers differ from the source results')
        from .catalog import _validate_result_buffers
        _validate_result_buffers(match)
    else:
        out = stream.args[names['out']]
        if out.shape != (total,) or out.dtype != ('i8' if members[0].narrow else 'i32'):
            raise ValueError('flat sibling output differs from the source results')
        if [(id(v), b, s, id(op)) for v, b, s, op in match.outputs] != [
                (id(m.result), b, (group.m, m.n), id(m.op)) for m, b in zip(members, bases)]:
            raise ValueError('sibling result slices differ from the source results')
    layouts = {}
    for i, member in enumerate(members):
        name = f'w{i}'
        arg = stream.args[names[name]]
        if arg.role == 'const':
            source = dense_values(member.op.operands[1])
            width = record['panel_width']
            if source is None or type(width) is not int or arg.shape != (-(-member.n // width), member.k, width):
                raise ValueError('packed sibling weight lacks a source constant or declared panel')
            layout = PanelLayout(member.k, member.n, width)
            if not np.array_equal(layout.unpack(stream.consts[name]).astype(np.int64), source.astype(np.int64)):
                raise ValueError('packed sibling weight is not the source weight')
            layouts[names[name]] = (i, layout)
        else:
            if bound.get(name) is not member.op.operands[1] or arg.shape != (member.k, member.n):
                raise ValueError('sibling weight argument differs from its source contraction')
            layouts[names[name]] = (i, None)
    from .stream_proof import Replay, require
    replay = Replay(stream).run()
    require(not replay.unfenced, 'shared-input stream ends with unfenced stores')
    weights = {stream.args[index].name: (member, layout) for index, (member, layout) in layouts.items()}
    _replayed_outputs(replay, group, members, weights, bases, total)
    # Merlin's independent sibling checker, over the same words decoded as logical effects.
    effects = _decoded_effects(stream, names, layouts, members, group, bases, element, flags())
    tile, storage, resources = _views(group)
    realization = verify_shared_input(tuple(effects), module=module, group=group, primitive_tile=tile,
                                      resources=resources, storage=storage)
    _backstop(stream, members, weights, bases)
    return dict(group=group.record(), decoded_effects=len(effects), realization=realization,
                members=[dict(n=m.n, exit='f32_product_rne' if m.narrow else 'full_i32', relu=m.relu,
                              multiplier_bits=None if m.bits is None else sorted(set(m.bits)),
                              **(dict(scale_operand=m.scale_operand, scale_ref=list(m.scale_ref))
                                 if m.scale_ref is not None else {}),
                              weight_layout='panel' if layouts[names[f'w{i}']][1] is not None else 'row_major')
                         for i, m in enumerate(members)],
                panel_width=record['panel_width'])
