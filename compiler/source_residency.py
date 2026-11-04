"""Source-proved sharing of administrative immutable RHS handle lifetimes.

This exposes reuse to dense_residency; it grants no private-cell retention law.
Foreign dense work can invalidate the backend's physical resident state, which
must retain its ordinary cold-load fallback.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

KNOB_DECLARATION = {'source_rhs_lifetime': {
    'values': ['off', 'shared_rhs_lifetime'],
    'mechanism': 'same immutable source SSA RHS administrative handle lifetime sharing',
    'note': 'Default off. No physical retention claim; backend scratchpad invalidation remains mandatory.'}}


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('source_rhs_lifetime', default='off', values=('off', 'shared_rhs_lifetime'),
                root=Path(__file__).resolve().parent)


def _attrs(cb, index):
    return {**cb['commands'][index].get('attributes', {}),
            **cb.get('params', {}).get('command_source_attribution', {}).get(str(index), {})}


def _pure_tensor_source(op):
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.traits import get_effects
    effects = get_effects(op)
    if effects is not None:
        return not effects and all(child.name == 'linalg.yield' or _pure_tensor_source(child)
            for region in op.regions for block in region.blocks for child in block.ops)
    # Linalg tensor semantics return new immutable SSA values; memref Linalg
    # and unknown scalar/call effects cannot use this exception.
    if (isinstance(op, LinalgStructuredOperation)
            and all(isinstance(v.type, TensorType) for v in (*op.operands, *op.results))):
        return all(child.name == 'linalg.yield' or _pure_tensor_source(child)
                   for region in op.regions for block in region.blocks for child in block.ops)
    return False


def _signed_product(op, rhs, resolve):
    """Bind the actual original signed scalar program, not command labels."""
    inputs = tuple(getattr(op, 'inputs', ()))
    if (len(inputs) != 2 or resolve(inputs[1]) is not resolve(rhs) or len(op.results) != 1
            or tuple(str(v.type.element_type) for v in (*inputs, *op.results)) != ('i8', 'i8', 'i32')):
        return False
    if op.name == 'linalg.matmul':
        return True
    if op.name != 'linalg.generic':
        return False
    from linalg_lowering import _body, _commutes, UnsupportedLinalg
    a, b, c = ('arg', 0), ('arg', 1), ('arg', 2)
    product = ('arith.muli', ('arith.extsi', a), ('arith.extsi', b))
    try:
        expression = _body(op)
    except (UnsupportedLinalg, ValueError, AttributeError):
        return False
    return (_commutes(expression, 'arith.addi', c, product)
            or _commutes(expression, 'arith.addi', c, ('arith.muli', product[2], product[1])))


def share_rhs_lifetimes(source, command_buffer, *, mode=None):
    """Return a changed clone and source-bound admission/refusal report.

    Administrative replacements use identity ROW_MAJOR_VIEW commands, preserving
    command positions, task/primary-op provenance and all output data effects.
    Physical RHS identity requires the same materialized tensor name AND exact
    original source value binding. Neither matching shapes nor provenance labels
    establish equality. Static unencoded i8 RHS and i8*i8->i32 consumers only.
    """
    mode = setting() if mode is None else mode
    if mode not in ('off', 'shared_rhs_lifetime'):
        raise ValueError('unknown source RHS lifetime mode: ' + str(mode))
    cb = deepcopy(command_buffer)
    report = {'schema': 'source_immutable_rhs_lifetime_v1', 'mode': mode,
              'groups': [], 'refusals': [], 'scope':
              'administrative logical handles only; physical backend invalidation/cold fallback required'}
    if mode == 'off':
        return cb, report
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    from merlin.llvmlower.tensor_preparation_identity import canonical_tensor_read_view, validate_tensor_read_view
    from source_attribution import task_definitions, materialized_task_accesses
    source = source.decode() if isinstance(source, bytes) else source
    inventory = source_operation_inventory(source)
    report['source_sha256'] = inventory['source_sha256']
    params = cb.get('params', {})
    if params.get('source_sha256') != inventory['source_sha256']:
        report['refusals'].append({'reason': 'missing or mismatched exact source digest'})
        return cb, report
    bindings = params.get('tensor_source_map', {})
    module = parse_mlir_text(source)
    owner = next(op for op in module.body.block.ops if op.name == 'func.func'
                 and op.sym_name.data == inventory['entry'])
    ops = tuple(op for op in owner.body.block.ops if op.name != 'func.return')
    # Re-derive immutable source equalities instead of trusting serialized alias
    # claims. This is needed when actual source preparation already coalesced
    # two original views into the very same physical RHS tensor name.
    from source_preparation import prepare_source_layouts
    proven = prepare_source_layouts(module, entry=inventory['entry'])
    aliases = {(a['source_ordinal'], a['result_index']):
               (a['retained_ordinal'], a['retained_result_index']) for a in proven.aliases}
    op_indices = {op: i for i, op in enumerate(ops)}
    def resolve(value):
        index = op_indices.get(value.owner)
        if index is None:
            return value
        key = (index, tuple(value.owner.results).index(value))
        if key in aliases:
            retained, result = aliases[key]
            return ops[retained].results[result]
        return value
    tasks = task_definitions(cb)
    task_map = {task['task_index']: task for task in tasks}
    access = {}
    for task in tasks:
        reads, writes = materialized_task_accesses(inventory, cb, task['source_op_indices'])
        access[task['task_index']] = (set(task.get('reads', reads)), set(task.get('writes', writes)))
    commands = cb['commands']
    packs = defaultdict(list)
    for index, command in enumerate(commands):
        if command['opcode'] == 'RES_PACK':
            packs[command['operands']['src']].append(index)
    report['original_handle_count'] = sum(map(len, packs.values()))
    # This is a supported command protocol/effect domain, not an eligibility map.
    supported = {'RES_PACK', 'EVICT', 'MATMUL_RESIDENT', 'MATMUL', 'COMMIT',
                 'VECTOR_MAP', 'ROW_MAJOR_VIEW'}
    for rhs, indices in packs.items():
        if len(indices) < 2:
            continue
        reason = None
        spec = cb.get('tensors', {}).get(rhs, {})
        origin = bindings.get(rhs)
        if spec.get('dtype') != 'i8' or len(spec.get('shape', ())) != 2 or any(type(d) is not int or d <= 0 for d in spec.get('shape', ())):
            reason = 'unsupported RHS numeric type or nonstatic rank-two shape'
        elif not isinstance(origin, dict):
            reason = 'RHS lacks exact original SSA binding'
        else:
            try:
                if set(origin) == {'arg_index'}:
                    value = owner.body.block.args[origin['arg_index']]
                elif set(origin) == {'op_index', 'result_index'}:
                    value = ops[origin['op_index']].results[origin['result_index']]
                else:
                    raise ValueError('unsupported source value identity')
                view = canonical_tensor_read_view(value)
                validate_tensor_read_view(view)
                if getattr(value.owner, 'name', '') == 'linalg.transpose':
                    raise ValueError('transposed source RHS not supported')
                if (str(value.type.element_type) != spec['dtype']
                        or list(value.type.get_shape()) != spec['shape']):
                    raise ValueError('physical tensor differs from bound source type')
            except (ValueError, IndexError, KeyError) as exc:
                reason = 'RHS source identity unavailable: ' + str(exc)
        lifetimes = []
        if reason is None:
            for start in indices:
                pack = commands[start]
                handle = pack['operands']['dst']
                if _attrs(cb, start).get('layout') != 'packed_rhs':
                    reason = 'unsupported pack layout'
                    break
                uses, end = [], None
                for index in range(start + 1, len(commands)):
                    command = commands[index]
                    operands = command.get('operands', {})
                    if command['opcode'] == 'EVICT' and operands.get('handle') == handle:
                        end = index
                        break
                    if handle not in operands.values():
                        continue
                    if (command['opcode'] not in ('MATMUL', 'MATMUL_RESIDENT')
                            or operands.get('rhs') != handle or _attrs(cb, index).get('rhs_transposed', False)):
                        reason = 'unknown or transposed handle consumer'
                        break
                    if _attrs(cb, index).get('source_task_index') not in task_map:
                        reason = 'consumer source task ownership unavailable'
                        break
                    lhs = cb.get('tensors', {}).get(operands.get('lhs'), {})
                    if lhs.get('dtype') != 'i8' or len(lhs.get('shape', ())) != 2 or lhs['shape'][-1] != spec['shape'][0]:
                        reason = 'consumer is not exact rank-two signed i8 contraction'
                        break
                    commits = [j for j in range(index + 1, len(commands))
                        if commands[j]['opcode'] == 'COMMIT' and commands[j].get('operands', {}).get('src') == operands.get('dst')]
                    if len(commits) != 1 or _attrs(cb, commits[0]).get('output_dtype') not in (None, 'i32'):
                        reason = 'consumer readout numeric contract unsupported'
                        break
                    output = commands[commits[0]]['operands'].get('dst')
                    if cb.get('tensors', {}).get(output, {}).get('dtype') != 'i32':
                        reason = 'readout physical tensor is not signed i32'
                        break
                    result_origin = bindings.get(commands[commits[0]]['operands'].get('dst'), {})
                    source_index = result_origin.get('op_index')
                    if (type(source_index) is not int or not 0 <= source_index < len(ops)
                            or not _signed_product(ops[source_index], value, resolve)):
                        reason = 'original consumer signed sum-product or exact RHS SSA identity not established'
                        break
                    uses.append(index)
                if reason is not None:
                    break
                if end is None or not uses:
                    reason = 'unclosed or unused resident handle'
                    break
                if any(handle in c.get('operands', {}).values() for c in commands[:start] + commands[end + 1:]):
                    reason = 'handle use outside its original lifetime'
                    break
                # Handle names are lexical identities and may not be rebound.
                if sum(c['opcode'] == 'RES_PACK' and c['operands'].get('dst') == handle for c in commands) != 1:
                    reason = 'resident handle name rebound'
                    break
                lifetimes.append((start, end, handle, uses))
        if reason is None:
            first, last = lifetimes[0][0], lifetimes[-1][1]
            if any(not (a[1] < b[0]) for a, b in zip(lifetimes, lifetimes[1:])):
                reason = 'overlapping preexisting lifetimes unsupported'
            elif any(first <= t['before_command_index'] <= last for t in params.get('source_transport_tasks', [])):
                reason = 'source transport crosses extended lifetime'
            else:
                for index in range(first, last + 1):
                    command = commands[index]
                    operands = command.get('operands', {})
                    if command['opcode'] not in supported:
                        reason = 'unknown command effects across extended lifetime: ' + command['opcode']
                        break
                    if command['opcode'] == 'ROW_MAJOR_VIEW' and operands.get('src') != operands.get('dst'):
                        reason = 'nonidentity physical view crosses lifetime'
                        break
                    if command['opcode'] not in ('RES_PACK', 'EVICT', 'MATMUL', 'MATMUL_RESIDENT', 'ROW_MAJOR_VIEW') and operands.get('dst') == rhs:
                        reason = 'RHS write crosses extended lifetime'
                        break
                if reason is None:
                    member_tasks = {_attrs(cb, i).get('source_task_index') for i in (first, last)}
                    if None in member_tasks or any(t not in task_map for t in member_tasks):
                        reason = 'source task ordering unavailable'
                    else:
                        begin, end = min(member_tasks), max(member_tasks)
                        for task in tasks:
                            if not begin < task['task_index'] < end:
                                continue
                            if rhs in access[task['task_index']][1]:
                                reason = 'source task writes RHS across extended lifetime'
                                break
                            if task['kind'] == 'host_source' and any(not _pure_tensor_source(ops[i]) for i in task['source_op_indices']):
                                reason = 'unknown host source effects cross extended lifetime'
                                break
        if reason is not None:
            report['refusals'].append({'rhs': rhs, 'pack_command_indices': indices, 'reason': reason})
            continue
        canonical = lifetimes[0][2]
        consumer_indices = [i for x in lifetimes for i in x[3]]
        consumer_tasks = sorted({_attrs(cb, i)['source_task_index'] for i in consumer_indices})
        for ordinal, (start, end, handle, uses) in enumerate(lifetimes):
            if ordinal:
                commands[start]['opcode'] = 'ROW_MAJOR_VIEW'
                commands[start]['operands'] = {'src': rhs, 'dst': rhs}
            for index in uses:
                commands[index]['operands']['rhs'] = canonical
            if ordinal + 1 < len(lifetimes):
                commands[end]['opcode'] = 'ROW_MAJOR_VIEW'
                commands[end]['operands'] = {'src': rhs, 'dst': rhs}
            else:
                commands[end]['operands']['handle'] = canonical
        report['groups'].append({'rhs': rhs, 'source_value': origin, 'shape': spec['shape'],
            'dtype': spec['dtype'], 'retained_handle': canonical,
            'original_handles': [x[2] for x in lifetimes],
            'pack_command_indices': indices,
            'consumer_command_indices': consumer_indices,
            'consumer_task_indices': consumer_tasks,
            'requires_cross_task_emission_owner': len(consumer_tasks) > 1,
            'first_command_index': lifetimes[0][0], 'last_command_index': lifetimes[-1][1],
            'physical_retention': 'backend dependent; foreign dense/scratchpad work may force cold fallback'})
    report['transformed_handle_count'] = sum(c['opcode'] == 'RES_PACK' for c in commands)
    if not report['groups']:
        report['no_opportunity_reason'] = ('no repeated RES_PACK of the same physical source-bound RHS tensor'
            if all(len(indices) < 2 for indices in packs.values())
            else 'all repeated RHS candidates refused; see source-bound refusal records')
    cb.setdefault('params', {})['source_rhs_lifetime'] = report
    return cb, report
