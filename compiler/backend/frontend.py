"""Structural linalg tensor frontend with source-bound device/host task seams.

Only exact typed scalar semantics select a device form. Provenance labels are
retained for diagnostics and never participate in device dispatch.
"""
from __future__ import annotations

from typing import Any
from merlin.targetgen.contract import linalg_iface as li
from merlin.targetgen.oot_starterkit.plan import _parsed_source, source_operation_inventory
from .frontend_layout import layout_device_semantics, layout_device_program


def _name(op):
    return li._op_name(op)


def _operands(op):
    # xDSL's reviewed context represents some standard named contractions as
    # unregistered generic operations. Their fixed operand arity still defines
    # the input/destination split structurally.
    if _name(op) in ('linalg.batch_matmul', 'linalg.matmul_transpose_b') and len(op.operands) == 3:
        return list(op.operands[:2]), list(op.operands[2:])
    return li._ins_outs(op)


def _initialization_only(op):
    from xdsl.dialects.builtin import TensorType
    if _name(op) == 'tensor.empty':
        return True
    if _name(op) not in ('arith.constant', 'tensor.splat', 'linalg.fill'):
        return False
    if _name(op) == 'arith.constant' and not any(isinstance(v.type, TensorType) for v in op.results):
        return True
    for result in op.results:
        if not isinstance(result.type, TensorType):
            return False
        for use in result.uses:
            ins, outs = _operands(use.operation)
            if result in ins or result not in outs:
                return False
            # A nonzero destination is data, not an eliminable initializer.
            if _constant(result) != 0 and _matmul_semantics(use.operation):
                return False
    return bool(op.results)


def _constant(value):
    direct = li._scalar_const(value)
    if direct is not None:
        return direct
    owner = value.owner
    if getattr(owner, 'name', None) == 'linalg.fill' and owner.operands:
        return _constant(owner.operands[0])
    return None


def _integer_fill_program(op):
    """Materialize an exact integer linalg.fill seed with the existing DMA ABI."""
    if _name(op) != 'linalg.fill' or len(op.results) != 1:
        return None
    ins, outs = _operands(op)
    if len(ins) != 1 or len(outs) != 1:
        return None
    result = op.results[0]
    shape, dtype = li._shape(result.type), li._dtype(result.type)
    value = _constant(ins[0])
    if (dtype not in ('i8', 'i32') or li._dtype(ins[0].type) != dtype
        or outs[0].type != result.type or type(value) is not int
        or not shape or any(dim <= 0 for dim in shape)):
        return None
    bits = 8 if dtype == 'i8' else 32
    value = (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
    return [({}, {'output_dtype': dtype,
                  'layout_fill': {'schema': 'constant_layout_fill_v1', 'value': value,
                                  'result_shape': shape, 'dtype': dtype},
                  'layout_proof': 'typed source integer fill initializes every destination element exactly'})]


def _defined_destination(value):
    """Conservatively prove a materialized destination has defined elements."""
    from xdsl.ir import BlockArgument
    if isinstance(value, BlockArgument):
        return True
    if _constant(value) is not None:
        return True
    owner = value.owner
    if _name(owner) == 'tensor.empty':
        return False
    ins, outs = _operands(owner)
    # Exact products preserve defined seeds; generic pointwise producers below
    # may ignore an empty output destination because every element is replaced.
    if _matmul_semantics(owner):
        return True
    if _name(owner) == 'linalg.generic' and len(outs) == 1:
        from xdsl.ir.affine import AffineMap, AffineDimExpr
        rank = len(li._shape(value.type))
        maps = {**owner.attributes, **owner.properties}.get('indexing_maps')
        if (maps is not None and maps[-1].data == AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank)))
            and li._iterator_types(owner, ins) == ['parallel'] * rank):
            body = owner.regions[0].block
            if not body.args[-1].uses:
                return all(_defined_destination(v) for v in ins)
    # Layout descriptors identify all actual data sources, excluding unused
    # tensor.empty destination placeholders of full writes.
    layout = layout_device_semantics(owner)
    if layout:
        return all(_defined_destination(v) for v in layout[1].values())
    return False


def _matmul_semantics(op):
    """Return M,K,N for exact signed i8 products with defined i32 seeds."""
    ins, outs = _operands(op)
    if len(ins) != 2 or len(outs) != 1 or len(op.results) != 1:
        return None
    a, b, c = ins[0].type, ins[1].type, op.results[0].type
    ash, bsh, csh = li._shape(a), li._shape(b), li._shape(c)
    if (li._dtype(a), li._dtype(b), li._dtype(c)) != ('i8', 'i8', 'i32'):
        return None
    if len(ash) != 2 or len(bsh) != 2 or len(csh) != 2:
        return None
    m, k = ash
    kb, n = bsh
    if (min(m, k, n) <= 0 or k != kb or csh != [m, n]
        or outs[0].type != c or not _defined_destination(outs[0])):
        return None
    kind = _name(op)
    if kind == 'linalg.matmul':
        return m, k, n
    if kind != 'linalg.generic':
        return None
    # Structural map equality, independent of argument names or whitespace.
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    attrs = {**op.attributes, **op.properties}
    maps = attrs.get('indexing_maps')
    expected = [(0, 2), (2, 1), (0, 1)]
    if maps is None or len(maps) != 3:
        return None
    if any(a.data != AffineMap(3, 0, tuple(AffineDimExpr(i) for i in dims))
           for a, dims in zip(maps, expected)):
        return None
    if li._iterator_types(op, ins) != ['parallel', 'parallel', 'reduction']:
        return None
    return (m, k, n) if _signed_accumulate_body(op) else None


def _signed_accumulate_body(op):
    block = op.regions[0].block
    nodes = list(block.ops)
    if len(block.args) != 3 or len(nodes) != 5:
        return False
    exta, extb, mul, add, yield_ = nodes
    if [_name(x) for x in nodes] != ['arith.extsi', 'arith.extsi', 'arith.muli', 'arith.addi', 'linalg.yield']:
        return False
    if (list(exta.operands) != [block.args[0]] or list(extb.operands) != [block.args[1]]
        or list(mul.operands) != [exta.results[0], extb.results[0]]
        or set(add.operands) != {mul.results[0], block.args[2]}
        or list(yield_.operands) != [add.results[0]]
        or any(str(x.results[0].type) != 'i32' for x in (exta, extb, mul, add))):
        return False
    return True


def _other_contraction_semantics(op):
    """Static zero-initialized batch products and transposed-RHS products."""
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    ins, outs = _operands(op)
    if len(ins) != 2 or len(outs) != 1 or len(op.results) != 1:
        return None
    if [li._dtype(v.type) for v in (*ins, op.results[0])] != ['i8', 'i8', 'i32'] or _constant(outs[0]) != 0:
        return None
    ash, bsh, csh = [li._shape(v.type) for v in (*ins, op.results[0])]
    if not ash or not bsh or min(*ash, *bsh, *csh) <= 0:
        return None
    kind = _name(op)
    generic = kind == 'linalg.generic'
    maps = ({**op.attributes, **op.properties}).get('indexing_maps')
    def has_maps(expected, ndim):
        return (maps is not None and len(maps) == len(expected)
                and all(a.data == AffineMap(ndim, 0, tuple(AffineDimExpr(i) for i in dims))
                        for a, dims in zip(maps, expected)))
    if len(ash) == len(bsh) == len(csh) == 3:
        batch, m, k = ash
        bb, kb, n = bsh
        if bb != batch or kb != k or csh != [batch, m, n]:
            return None
        if kind == 'linalg.batch_matmul' or (generic
            and has_maps([(0, 1, 3), (0, 3, 2), (0, 1, 2)], 4)
            and li._iterator_types(op, ins) == ['parallel', 'parallel', 'parallel', 'reduction']
            and _signed_accumulate_body(op)):
            return 'BATCHED_MATMUL', {'a': ins[0], 'w': ins[1]}, {'output_dtype': 'i32'}
    if len(ash) == len(bsh) == len(csh) == 2:
        m, k = ash
        n, kb = bsh
        if kb != k or csh != [m, n]:
            return None
        if kind == 'linalg.matmul_transpose_b' or (generic
            and has_maps([(0, 2), (1, 2), (0, 1)], 3)
            and li._iterator_types(op, ins) == ['parallel', 'parallel', 'reduction']
            and _signed_accumulate_body(op)):
            return 'ATTENTION_QK', {'q': ins[0], 'k': ins[1]}, {'output_dtype': 'i32'}
    return None


def _map_device_semantics(op):
    """Recognize only exact identity movement and per-column raw i32 bias."""
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    from xdsl.dialects.builtin import AffineMapAttr
    if _name(op) not in ('linalg.generic', 'linalg.map') or len(op.results) != 1:
        return None
    ins, outs = _operands(op)
    if len(outs) != 1 or not ins:
        return None
    shape = li._shape(op.results[0].type)
    rank = len(shape)
    if not rank or min(shape) <= 0 or li._reduction_dims(op):
        return None
    maps = li._indexing_maps(op, ins, outs)
    identity = str(AffineMapAttr(
        AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank)))))
    # Named map defines identities; generic must supply the same exact maps.
    nodes = list(op.regions[0].block.ops)
    args = list(op.regions[0].block.args)
    if len(ins) == 1 and li._shape(ins[0].type) == shape and maps == [identity, identity]:
        source_dtype, output_dtype = li._dtype(ins[0].type), li._dtype(op.results[0].type)
        if source_dtype == 'i8' and output_dtype == 'i32' and len(nodes) == 2:
            if (_name(nodes[0]) == 'arith.extsi' and list(nodes[0].operands) == [args[0]]
                and _name(nodes[1]) == 'linalg.yield' and list(nodes[1].operands) == [nodes[0].results[0]]):
                return 'MOVEMENT', {'src': ins[0]}, {'output_dtype': 'i32', 'semantic': 'mvin_mvout'}
        if source_dtype == output_dtype == 'i8' and len(nodes) == 1:
            if _name(nodes[0]) == 'linalg.yield' and list(nodes[0].operands) == [args[0]]:
                return 'MOVEMENT', {'src': ins[0]}, {'output_dtype': 'i8', 'semantic': 'mvin_mvout'}
    if len(ins) != 2 or li._dtype(op.results[0].type) != 'i32' or any(li._dtype(v.type) != 'i32' for v in ins):
        return None
    if len(nodes) != 2 or [_name(x) for x in nodes] != ['arith.addi', 'linalg.yield']:
        return None
    if set(nodes[0].operands) != set(args[:2]) or list(nodes[1].operands) != [nodes[0].results[0]]:
        return None
    column = str(AffineMapAttr(
        AffineMap(rank, 0, (AffineDimExpr(rank - 1),))))
    for full, bias in ((0, 1), (1, 0)):
        if (li._shape(ins[full].type) == shape and li._shape(ins[bias].type) == [shape[-1]]
            and maps[full] == identity and maps[bias] == column and maps[-1] == identity):
            return 'BIAS_ADD', {'src': ins[full], 'bias': ins[bias]}, {'output_dtype': 'i32', 'epilogue': []}
    return None


def _host_module(op, folded, names):
    """Clone a source task into a closed tensor function, preserving scalar DAGs."""
    from xdsl.ir import Block, Region
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.builtin import ModuleOp, TensorType
    from xdsl.printer import Printer
    from io import StringIO
    selected = [*folded, op]
    produced = {v for node in selected for v in node.results}
    external = []
    for node in selected:
        # Include captures used inside linalg scalar regions, not just operands.
        for inner in node.walk():
            for v in inner.operands:
                if v in produced or v in external:
                    continue
                owner = v.owner
                if owner in selected or any(owner is b for parent in selected for region in parent.regions for b in region.blocks):
                    continue
                if v in names:
                    external.append(v)
                elif getattr(owner, 'parent_op', lambda: None)() in selected:
                    continue
                elif v not in {r for p in selected for i in p.walk() for r in i.results}:
                    raise ValueError(f'host task has unresolved scalar capture {v}')
    if any(not isinstance(v.type, TensorType) for v in external):
        raise ValueError('host ABI requires materialized tensor captures')
    block = Block(arg_types=[v.type for v in external])
    mapping = dict(zip(external, block.args))
    for node in selected:
        block.add_op(node.clone(value_mapper=mapping))
    block.add_op(ReturnOp(*(mapping[v] for v in op.results)))
    fn = FuncOp.from_region('host_region', [v.type for v in external], [v.type for v in op.results], Region(block))
    stream = StringIO()
    Printer(stream=stream).print_op(ModuleOp([fn]))
    return stream.getvalue(), [names[v] for v in external]


def _host_placement_reason(op):
    """Explicit numerical hardware seams; unresolved integer forms stay unknown."""
    operands, _ = _operands(op)
    dtypes = {li._dtype(v.type) for v in (*operands, *op.results)}
    body = li._body_op_names(op)
    if _name(op).startswith('quant_ext.'):
        return ('source quantization/dequantization preserves float scale, zero point, '
                'rounding and saturation; no standalone Gemmini quantization instruction'), False
    if any(name.startswith('math.') for name in body) or _name(op).startswith('math.'):
        return 'floating nonlinear source semantics require host arithmetic', False
    if any(dtype.startswith(('f', 'bf')) for dtype in dtypes):
        return 'floating tensor/scalar semantics outside signed-i8 Gemmini datapath', False
    if dtypes and dtypes.issubset({'i64', 'index', 'i1'}):
        return 'index, predicate or wide-integer source semantics outside Gemmini element datapath', False
    if _name(op) in ('tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'):
        return 'exact tensor storage/layout transformation retained at the source boundary', None
    return 'typed host operation; target eligibility requires per-operation proof', None


def _boundary_bytes(names, tensors):
    from math import prod
    widths = {'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
              'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8}
    counts = [prod(tensors[name]['shape']) * widths[tensors[name]['dtype']]
              if tensors[name]['dtype'] in widths else None for name in dict.fromkeys(names)]
    return None if any(value is None for value in counts) else sum(counts)


def lower_linalg(text: str) -> dict[str, Any]:
    """Build a complete CB; unimplemented legal device contractions fail closed."""
    from xdsl.dialects.builtin import TensorType
    inventory = source_operation_inventory(text)
    parsed = li.parse_linalg_mlir(text)
    entry, block, direct, returned = _parsed_source(text.encode('utf-8'))
    if not direct:
        raise ValueError('empty linalg program')
    indices = {op: i for i, op in enumerate(direct)}
    payload = [op for op in direct if not _initialization_only(op)]
    if not payload:
        raise ValueError('linalg program has no observable tensor computation')
    folded_set = set(direct) - set(payload)
    result_owners = {v: op for op in direct for v in op.results}
    fold_dependencies = {}
    for op in payload:
        found = set()
        def visit(node):
            for inner in node.walk():
                for v in inner.operands:
                    producer = result_owners.get(v)
                    if producer in folded_set and producer not in found:
                        found.add(producer)
                        visit(producer)
        visit(op)
        fold_dependencies[op] = sorted(found, key=indices.__getitem__)
    # Initialization ops are folded once into their earliest consuming task.
    # Shared constants may appear in multiple closed host functions, while their
    # direct source operation has exactly one plan owner.
    owners = {op: [] for op in payload}
    for init in sorted(folded_set, key=indices.__getitem__):
        consumers = [op for op in payload if init in fold_dependencies[op]]
        owners[(consumers or payload)[0]].append(init)
    tensors, names, source_values = {}, {}, []
    output_values = list(returned.operands)
    # Public logical ABI metadata is authoritative when present. Some legacy
    # tensor sources carry only the standard ordered Y0,Y1,... result ABI.
    # SSA hints are labels for input bindings, never operation dispatch keys.
    fn_attrs = block.parent_op().attributes
    def logical_names(key, count):
        value = fn_attrs.get(key)
        if value is None:
            return None
        selected = [getattr(item, 'data', None) for item in value]
        if (len(selected) != count or any(type(name) is not str or not name for name in selected)
            or len(set(selected)) != len(selected)):
            raise ValueError(f'{key} differs from complete ordered source ABI')
        return selected
    input_names = logical_names('merlin.logical_inputs', len(block.args))
    output_names = logical_names('merlin.logical_outputs', len(output_values))
    input_roles_attr = fn_attrs.get('merlin.logical_input_roles')
    input_roles = [getattr(item, 'data', None) for item in input_roles_attr] if input_roles_attr is not None else None
    if input_roles is not None and (len(input_roles) != len(block.args) or any(role not in ('input', 'weight', 'bias', 'scale') for role in input_roles)):
        raise ValueError('merlin.logical_input_roles differs from ordered source argument ABI')
    if input_names is None:
        input_names = [value.name_hint or f'arg{i}' for i, value in enumerate(block.args)]
    if output_names is None:
        output_names = [f'Y{i}' for i in range(len(output_values))]
    if len(set(input_names + output_names)) != len(input_names) + len(output_names):
        raise ValueError('source logical ABI aliases tensor names')
    reserved_names = set(input_names + output_names)
    output_binding_names = dict(zip(output_values, output_names))
    for i, value in enumerate(block.args):
        if not isinstance(value.type, TensorType):
            raise ValueError('linalg entry scalar arguments are not supported by tensor ABI')
        name = input_names[i]
        names[value] = name
        tensors[name] = {'shape': li._shape(value.type), 'dtype': li._dtype(value.type),
                         'role': input_roles[i] if input_roles is not None else 'input'}
    for op in payload:
        for j, value in enumerate(op.results):
            if not isinstance(value.type, TensorType):
                raise ValueError(f'observable non-tensor result in {_name(op)}')
            name = output_binding_names.get(value, f'v{indices[op]}_{j}')
            while value not in output_binding_names and name in reserved_names:
                name = '_' + name
            reserved_names.add(name)
            names[value] = name
            tensors[name] = {'shape': li._shape(value.type), 'dtype': li._dtype(value.type),
                             'role': 'output' if value in output_values else 'intermediate'}
            source_values.append({'op_index': indices[op], 'result_index': j, 'tensor': name})
    if any(v not in names or v in block.args for v in output_values):
        raise ValueError('returned aliases/initializer values need an explicit source-owned materialization')
    commands, tasks, placements = [], [], []
    for tid, op in enumerate(payload):
        source_i = indices[op]
        start = len(commands)
        attrs = {'task_index': tid, 'source_op_index': source_i}
        source_provenance = li._prov(op)
        for key, value in {**op.attributes, **op.properties}.items():
            if key.startswith('prov.') and key[5:] not in source_provenance:
                source_provenance[key[5:]] = str(value)
        if source_provenance.get('region_id'):
            attrs['source_region_id'] = source_provenance['region_id']
        attrs['source_provenance'] = source_provenance
        source_inputs, source_destinations = _operands(op)
        attrs['source_effects'] = li._effects(op)
        attrs['source_indexing_maps'] = li._indexing_maps(op, source_inputs, source_destinations)
        attrs['source_iterator_types'] = li._iterator_types(op, source_inputs)
        attrs['source_semantic_attributes'] = li._semantic_attributes(op)
        attrs['source_numeric_boundary'] = {
            'inputs': [{'shape': li._shape(v.type), 'dtype': li._dtype(v.type)} for v in source_inputs],
            'outputs': [{'shape': li._shape(v.type), 'dtype': li._dtype(v.type)} for v in op.results],
            'destination_constants': [_constant(v) for v in source_destinations],
        }
        outputs = [names[v] for v in op.results]
        device = _matmul_semantics(op)
        device_map = (_map_device_semantics(op) or _other_contraction_semantics(op)
                      or layout_device_semantics(op))
        device_program = layout_device_program(op) or _integer_fill_program(op)
        if device:
            ins, destinations = _operands(op)
            inputs = [names[v] for v in ins]
            seeded = _constant(destinations[0]) != 0
            if seeded:
                seed = names[destinations[0]]
                inputs.append(seed)
                attrs = {**attrs, 'bias': seed, 'bias_broadcast': False}
            handle, accum = f'packed{source_i}', f'acc{source_i}'
            commands.extend([
                {'opcode': 'RES_PACK', 'operands': {'src': inputs[1], 'dst': handle}, 'attributes': {**attrs, 'layout': 'packed_rhs'}},
                {'opcode': 'MATMUL_RESIDENT', 'operands': {'lhs': inputs[0], 'rhs': handle, 'dst': accum}, 'attributes': attrs.copy()},
                {'opcode': 'COMMIT', 'operands': {'src': accum, 'dst': outputs[0]}, 'attributes': {**attrs, 'epilogue': [], 'output_dtype': 'i32'}},
                {'opcode': 'EVICT', 'operands': {'handle': handle}, 'attributes': attrs.copy()},
            ])
            kind, reason = 'device', ('exact signed i8 product with defined full-matrix i32 initialization'
                                      if seeded else 'exact signed i8 product with zero i32 initialization')
        elif device_program:
            inputs = []
            for values, properties in device_program:
                inputs.extend(names[v] for v in values.values())
                operands = {key: names[value] for key, value in values.items()}
                commands.append({'opcode': 'MOVEMENT', 'operands': {**operands, 'dst': outputs[0]},
                                 'attributes': {**attrs, **properties}})
            inputs = list(dict.fromkeys(inputs))
            kind, reason = 'device', device_program[0][1]['layout_proof']
        elif device_map:
            opcode, values, properties = device_map
            inputs = [names[v] for v in values.values()]
            operands = {key: names[value] for key, value in values.items()}
            commands.append({'opcode': opcode, 'operands': {**operands, 'dst': outputs[0]},
                             'attributes': {**attrs, **properties}})
            kind, reason = 'device', properties.get('layout_proof', 'exact typed movement, bias, batched or transposed contraction')
        else:
            # Reject known hardware-eligible integer contractions instead of
            # misrepresenting missing frontend/lowering work as a host seam.
            ins, _ = _operands(op)
            integer_reduction = (
                _name(op) == 'linalg.generic' and bool(li._reduction_dims(op))
                and 'arith.muli' in li._body_op_names(op)
            )
            if ((_name(op) in ('linalg.matmul', 'linalg.matmul_transpose_b', 'linalg.batch_matmul', 'linalg.conv_2d_nchw_fchw', 'linalg.conv_2d_nhwc_hwcf')
                 or integer_reduction)
                and len(ins) >= 2 and all(li._dtype(v.type) == 'i8' for v in ins[:2])):
                raise ValueError(f'eligible integer contraction frontend gap at source op {source_i}: {_name(op)}')
            module_ir, inputs = _host_module(op, fold_dependencies[op], names)
            reason, eligible_device = _host_placement_reason(op)
            commands.append({'opcode': 'VECTOR_MAP',
                             'operands': {**{f'input{i}': n for i, n in enumerate(inputs)},
                                          **{f'output{i}': n for i, n in enumerate(outputs)}},
                             'attributes': {**attrs, 'op': 'host_region', 'host_module_ir': module_ir,
                                            'input_tensors': inputs, 'output_tensors': outputs,
                                            'reason': reason, 'host_reason': reason, 'eligible_device': eligible_device}})
            kind = 'host'
        owned_indices = sorted([source_i, *(indices[x] for x in owners[op])])
        region_ids = sorted({inventory['operations'][i]['region_id'] for i in owned_indices if inventory['operations'][i]['region_id']})
        tasks.append({'task_index': tid, 'kind': kind, 'source_op_indices': owned_indices,
                      'source_region_ids': region_ids, 'instruction_start': start, 'instruction_end': len(commands),
                      'reads': list(dict.fromkeys(inputs)), 'writes': outputs})
        placements.append({'source_op_index': source_i, 'operation': _name(op), 'placement': kind,
                           'reason': reason, 'region_ids': region_ids,
                           'source_provenance': source_provenance,
                           'input_dtypes': [tensors[name]['dtype'] for name in inputs],
                           'output_dtypes': [tensors[name]['dtype'] for name in outputs],
                           'static_boundary_bytes': _boundary_bytes(inputs + outputs, tensors),
                           'measured_host_cycles': None,
                           'device_eligible': True if kind == 'device' else eligible_device,
                           'selected_device_form': commands[start]['opcode'] if kind == 'device' else None,
                           'remaining_gap': 'eligibility proof' if kind == 'host' and eligible_device is None else None})
    output_names = [names[v] for v in output_values]
    plan = {'schema': 'mixed_program_plan_v1', 'source_sha256': inventory['source_sha256'],
            'source_op_count': inventory['source_op_count'], 'tasks': tasks,
            'schedule_instruction_count': len(commands), 'prologue_instruction_range': [0, 0],
            'epilogue_instruction_range': [len(commands), len(commands)],
            'entry_bindings': [names[v] for v in block.args], 'source_values': source_values,
            'output_bindings': output_names}
    abi_args = [{'tensor': name, 'access': 'write' if spec['role'] == 'output' else 'read' if spec['role'] in ('input', 'weight', 'bias', 'scale') else 'readwrite'}
                for name, spec in tensors.items()]
    # The public lane-cost/census vocabulary is separate from the executable
    # source plan. Declare the proposed lane for each actual source-owning task;
    # these records grant neither hardware eligibility nor execution evidence.
    # Keep the public lane/family/reason fields, with exact source identity in
    # the human-readable reason rather than inventing a second binding schema.
    lane_placement = [
        {'lane': 'unit' if task['kind'] == 'device' else 'host',
         'family': placement['operation'],
         'reason': (f"{placement['reason']}; source_sha256={inventory['source_sha256']}; "
                    f"task_index={task['task_index']}; source_op_indices={task['source_op_indices']}; "
                    f"source_region_ids={task['source_region_ids']}")}
        for task, placement in zip(tasks, placements)
    ]
    return {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': commands,
            'kernel_abi': {'kind': 'whole_program', 'args': abi_args, 'outputs': output_names},
            'params': {'global_program_plan': plan, 'source_inventory': inventory,
                       'linalg_inventory': parsed, 'placement_diagnostics': placements,
                       'lane_placement': lane_placement,
                       'placement_summary': {
                           'source_payload_operations': len(placements),
                           'known_device_eligible': sum(row['device_eligible'] is True for row in placements),
                           'covered_device_operations': sum(row['placement'] == 'device' for row in placements),
                           'eligible_but_host': sum(row['device_eligible'] is True and row['placement'] == 'host' for row in placements),
                           'unknown_eligibility': sum(row['device_eligible'] is None for row in placements),
                           'hardware_unsupported_host_operations': sum(row['device_eligible'] is False for row in placements),
                           'measured_host_cycles': None, 'measured_transfer_bytes': None}}}
