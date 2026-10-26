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

from .match import Decline, Match, dense_values, shape
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
        self.bits = tuple(int(v) for v in proof['multiplier_bits']) if self.narrow else None
        self.relu = self.narrow and 'relu' in plan.decision['stages']
        self.result = chain.ops[-1].results[0] if self.narrow else op.results[0]
        self.consumed = list(chain.ops) if self.narrow else []
        if self.narrow and np.any(np.asarray(proof.get('offsets', 0))):
            raise Decline('shared-input siblings do not seed accumulator offsets')


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
        members.append(_Member(ordinal, op, *planned))
    if len({m.narrow for m in members}) != 1:
        raise Decline('shared-input siblings must share one exit width')
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
    element = 1 if members[0].narrow else 4
    dtype = 'i8' if members[0].narrow else 'i32'
    bases, total = [], 0
    for member in members:
        bases.append(total)
        total += group.m * member.n
    args = [Arg('a', 'i8', (group.m, group.k), 'input')]
    args += [Arg(f'w{i}', 'i8', (m.k, m.n), 'input') for i, (m, layout) in enumerate(zip(members, layouts))
             if layout is None]
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
                selected = set(sibling.bits[op.dram_col:op.dram_col + op.cols])
                if len(selected) != 1:
                    raise Decline('sibling store tile spans two readout multipliers')
                config = (sibling.n, activation(sibling.relu), selected.pop())
            else:
                config = (sibling.n * 4, NO_ACT, None)
            if config != store_config:
                s.config_st(config[0], config[1], multiplier_bits=config[2])
                store_config = config
            offset = (bases[member] + op.dram_row * sibling.n + op.dram_col) * element
            s.mvout(s.at('out', offset), ACC(op.accumulator_row, full=not sibling.narrow), op.cols, op.rows)
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
        integer_producer=True, integer_readout=members[0].narrow, raw_integer=not members[0].narrow,
        dimensions=dict(batch=0, m=group.m, n=sum(group.widths), k=group.k), seed='zero', bias=False,
        output_dtype=dtype, relu=False, readout='f32_product_rne' if members[0].narrow else 'full_i32',
        device_readout='f32_product_rne' if members[0].narrow else 'full_i32',
        accumulator_min=-group.k * 16384, accumulator_max=group.k * 16384)
    kernel = Kernel(OPERATION, s, meta)
    consumed = [body[i] for i in group.source_op_indices] + [op for m in members for op in m.consumed]
    operands = [('a', members[0].op.operands[0])] + [
        (f'w{i}', m.op.operands[1]) for i, (m, l) in enumerate(zip(members, layouts)) if l is None]
    outputs = [(m.result, b, (group.m, m.n), m.op) for m, b in zip(members, bases)]
    return Match(OPERATION, members[0].op, consumed, operands, ('out', members[-1].result), kernel,
                 numeric=dict(integer_producer=True, shared_input=True, seed=None, runtime_seed=False,
                              raw_integer=not members[0].narrow, integer_readout=members[0].narrow),
                 outputs=outputs)


# ---------------------------------------------------------------------------------------------
# Admission: decode the emitted stream, never the emitter's plan
# ---------------------------------------------------------------------------------------------

def _module_of(op):
    while op.parent is not None:
        op = op.parent
    return op


def _decoded_config_st(rec):
    from .readout_encoding import unpack
    return (unpack(rec.values[1], 'ConfigMvoutRs2', 'stride'), unpack(rec.values[0], 'ConfigMvoutRs1', 'activation'),
            unpack(rec.values[1], 'ConfigMvoutRs2', 'acc_scale'))


def _canonical(kind, *args, **kwargs):
    """The encoded register words of one configuration this pattern depends on."""
    from .emit import encode
    from .stream import Stream
    probe = Stream([])
    getattr(probe, kind)(*args, **kwargs)
    return encode(probe.ins[0])


def admit(match, *, readout_options):
    """Return a proof record for ``match``'s emitted stream or raise ``ValueError``."""
    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
    from merlin.compile.scheduling.panel_layout import PanelLayout
    from merlin.compile.scheduling.shared_input_groups import SharedInputGroup
    from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect
    from merlin.compile.scheduling.shared_input_verification import verify_shared_input
    from .readout_encoding import activation, flags
    from ..tables import isa

    record = match.kernel.meta['shared_input']
    fields = record['group']
    group = SharedInputGroup(fields['source_ir_sha256'], tuple(fields['source_op_indices']),
                             tuple(fields['contraction_ordinals']), fields['m'], fields['k'], tuple(fields['widths']))
    module = _module_of(match.root)
    body, members = _members(module, group, readout_options)  # re-planned from source, not from meta
    stream = match.kernel.stream
    names = {a.name: i for i, a in enumerate(stream.args)}
    bound = dict(match.operands)
    if bound.get('a') is not members[0].op.operands[0] or stream.args[names['a']].shape != (group.m, group.k):
        raise ValueError('shared input argument differs from the source group input')
    element = 1 if members[0].narrow else 4
    out = stream.args[names['out']]
    bases, total = [], 0
    for member in members:
        bases.append(total)
        total += group.m * member.n
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
    from .emit import encode

    bits = flags()
    address_flags = bits['acc_i8'] | bits['acc_accum'] | bits['full_c_bit']
    # Effective register state, validated at every command that depends on it:
    # a configuration changed between uses and restored is still caught.
    execute_state, load_state, store_state = None, {}, None
    execute_expected = _canonical('config_ex')
    effects, owners, stationary = [], {}, None
    pending_member = None
    for ins in stream.ins:
        kind = ins.kind
        if kind in ('fence', 'flush'):
            continue
        if kind in ('preload', 'compute') and execute_state != execute_expected:
            raise ValueError('effective execute configuration differs from untransposed unit-stride WS')
        if kind == 'config_ex':
            execute_state = encode(ins)
        elif kind == 'config_ld':
            load_state[ins.a('load_id')] = encode(ins)
        elif kind == 'config_st':
            store_state = encode(ins)
        elif kind == 'mvin':
            local, rows, cols = ins.a('local'), ins.a('rows'), ins.a('cols')
            if local & bits['acc_i8'] or ins.a('zero') or cols > DIM:
                raise ValueError('shared-input stream seeds the accumulator or loads a multi-tile row')
            state = load_state.get(ins.a('load_id'))

            def require_load(stride):
                if state != _canonical('config_ld', ins.a('load_id'), stride):
                    raise ValueError('effective load configuration differs from the declared layout')
            if ins.arg == names['a']:
                require_load(group.k)
                row, col = divmod(ins.off, group.k)
                effects.append(SharedInputEffect(None, Load('lhs', (row // DIM, col // DIM), row, col, rows, cols,
                                                            local, col // DIM)))
            elif ins.arg in layouts:
                member, layout = layouts[ins.arg]
                sibling = members[member]
                require_load(layout.row_pitch if layout is not None else sibling.n)
                row, col = layout.source(ins.off, cols) if layout is not None else divmod(ins.off, sibling.n)
                effects.append(SharedInputEffect(member, Load('weight', (row // DIM, col // DIM), row, col, rows,
                                                              cols, local, row // DIM)))
            else:
                raise ValueError('shared-input stream loads an unbound argument')
            for r in range(rows):
                owners[local + r] = effects[-1].member, effects[-1].operation.dram_row + r, effects[-1].operation.dram_col
        elif kind == 'preload':
            bd, c = ins.a('bd'), ins.a('c')
            if not c & bits['acc_i8'] or (bd != isa.GARBAGE_ADDR and bd & bits['acc_i8']):
                raise ValueError('shared-input descriptor addresses the wrong local memory')
            if bd == isa.GARBAGE_ADDR:
                if stationary is None:
                    raise ValueError('shared-input descriptor retains no resident weights')
                member, block, weight_row = stationary, None, None
            else:
                owner = owners.get(bd)
                if owner is None or owner[0] is None:
                    raise ValueError('shared-input descriptor reads unowned weight rows')
                member, block, weight_row = owner[0], (owner[1] // DIM, owner[2] // DIM), bd
                stationary = member
            pending_member = member
            effects.append(SharedInputEffect(member, Preload(weight_row, block, c & ~address_flags,
                                                             bool(c & bits['acc_accum']), ins.a('c_rows'),
                                                             ins.a('c_cols'), ins.a('bd_rows'))))
            if ins.a('bd_cols') != ins.a('c_cols'):
                raise ValueError('shared-input descriptor weight and output widths differ')
        elif kind == 'compute':
            a = ins.a('a')
            if ins.a('bd') != isa.GARBAGE_ADDR or a & bits['acc_i8'] or pending_member is None:
                raise ValueError('shared-input compute differs from the independent dot product')
            owner = owners.get(a)
            block = None if owner is None else (owner[1] // DIM, owner[2] // DIM)
            effects.append(SharedInputEffect(pending_member, Compute(a, block, ins.a('a_rows'), ins.a('a_cols'),
                                                                     not ins.a('accumulate'))))
            pending_member = None
        elif kind == 'mvout':
            local = ins.a('local')
            if not local & bits['acc_i8'] or ins.arg != names['out'] or store_state is None:
                raise ValueError('shared-input store is not an accumulator readout of the flat output')
            position, remainder = divmod(ins.off, element)
            member = next((i for i in range(len(members) - 1, -1, -1) if bases[i] <= position), None)
            if remainder or member is None:
                raise ValueError('shared-input store is misaligned with every sibling result')
            sibling = members[member]
            row, col = divmod(position - bases[member], sibling.n)
            cols = ins.a('cols')
            full = bool(local & bits['full_c_bit'])
            stride, act, scale_bits = _decoded_config_st(store_state)
            if full == sibling.narrow or stride != sibling.n * element:
                raise ValueError('sibling store width or pitch differs from its exit')
            if sibling.narrow:
                expected = set(sibling.bits[col:col + cols])
                if expected != {scale_bits} or act != activation(sibling.relu):
                    raise ValueError('decoded sibling readout differs from its proved multiplier bits')
                canonical = _canonical('config_st', stride, act, multiplier_bits=scale_bits)
            elif act != isa.NO_ACTIVATION or scale_bits != bits['f1']:
                raise ValueError('full-width sibling store applies a readout stage')
            else:
                canonical = _canonical('config_st', stride)
            if store_state != canonical:
                raise ValueError('effective store configuration sets an unproved readout field')
            effects.append(SharedInputEffect(member, Store(row, col, ins.a('rows'), cols, local & ~address_flags)))
        else:
            raise ValueError('shared-input stream contains an unknown command')
    tile, storage, resources = _views(group)
    realization = verify_shared_input(tuple(effects), module=module, group=group, primitive_tile=tile,
                                      resources=resources, storage=storage)
    return dict(group=group.record(), decoded_effects=len(effects), realization=realization,
                members=[dict(n=m.n, exit='f32_product_rne' if m.narrow else 'full_i32', relu=m.relu,
                              multiplier_bits=None if m.bits is None else sorted(set(m.bits)),
                              weight_layout='panel' if layouts[names[f'w{i}']][1] is not None else 'row_major')
                         for i, m in enumerate(members)],
                panel_width=record['panel_width'])
