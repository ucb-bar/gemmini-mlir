"""Typed source readers and conservative logical command lowering.

The residency grammar is parsed by the shared canonical parser. Linalg source
is parsed structurally; only proved operations become logical device commands.
"""
from __future__ import annotations

from typing import Any, Callable


class FrontendError(ValueError):
    pass


def parse_source(text: str, *, linalg_lowerer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Read source into a logical command buffer.

    ``linalg_lowerer(text, inventory)`` optionally handles complete typed graphs;
    it receives original source bytes and the shared structural inventory.
    Unsupported source never becomes an empty command stream or CPU fallback.
    """
    from merlin.targetgen.oot_starterkit.iface import parse_interface, parse_linalg

    if 'merlin_iface.' in text:
        return parse_interface(text)
    if linalg_lowerer is not None:
        return linalg_lowerer(text, parse_linalg(text))
    return lower_linalg(text)


def _type(value) -> dict[str, Any]:
    from xdsl.dialects.builtin import TensorType, NoneAttr
    if not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr):
        raise FrontendError(f"expected an ordinary ranked tensor, got {value.type}")
    shape = list(value.type.get_shape())
    if any(d <= 0 for d in shape):
        raise FrontendError("dynamic or empty tensor shape needs target lowering")
    return {'shape': shape, 'dtype': str(value.type.element_type)}


def _provenance(op) -> dict[str, Any]:
    from xdsl.dialects.builtin import ArrayAttr, StringAttr
    def data(attr):
        if isinstance(attr, StringAttr):
            return attr.data
        if isinstance(attr, ArrayAttr):
            return [data(a) for a in attr]
        return str(attr)
    return {key: data(val) for key, val in op.attributes.items() if key.startswith('prov.')}


def _zero(value) -> bool:
    owner = value.owner
    if getattr(owner, 'name', '') == 'linalg.fill':
        scalar = owner.inputs[0]
    elif getattr(owner, 'name', '') == 'tensor.splat':
        scalar = owner.operands[0]
    else:
        return False
    op = scalar.owner
    attr = op.properties.get('value', op.attributes.get('value'))
    return op.name == 'arith.constant' and getattr(getattr(attr, 'value', None), 'data', None) == 0 and str(scalar.type) == 'i32'


def _saturating_cast(op) -> Any | None:
    """Prove pointwise max(x,-128), min(x,127), trunc-to-i8 exactly."""
    if op.name != 'linalg.generic' or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    source = op.inputs[0]
    if _type(source)['dtype'] != 'i32' or _type(op.results[0]) != {**_type(source), 'dtype': 'i8'}:
        return None
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    rank = len(_type(source)['shape'])
    ident = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank)))
    if tuple(a.data for a in op.indexing_maps) != (ident, ident) or any(str(a.data.value) != 'parallel' for a in op.iterator_types):
        return None
    body = op.regions[0].block
    ops = list(body.ops)
    if [o.name for o in ops] != ['arith.maxsi', 'arith.minsi', 'arith.trunci', 'linalg.yield']:
        return None
    def constant(v, expected):
        owner = v.owner
        attr = getattr(owner, 'properties', {}).get('value', getattr(owner, 'attributes', {}).get('value'))
        return getattr(owner, 'name', '') == 'arith.constant' and getattr(getattr(attr, 'value', None), 'data', None) == expected
    lo, hi, cast, yld = ops
    if body.args[0] not in lo.operands or not any(constant(v, -128) for v in lo.operands):
        return None
    if lo.results[0] not in hi.operands or not any(constant(v, 127) for v in hi.operands):
        return None
    if list(cast.operands) != [hi.results[0]] or list(yld.operands) != [cast.results[0]]:
        return None
    return source



def _bias_add(op):
    if op.name != 'linalg.generic' or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    a, b = op.inputs
    at, bt, rt = _type(a), _type(b), _type(op.results[0])
    if len(at['shape']) != 2 or at['dtype'] != 'i32' or bt != {'shape': [at['shape'][1]], 'dtype': 'i32'} or rt != at:
        return None
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity, columns = AffineMap(2, 0, (d0, d1)), AffineMap(2, 0, (d1,))
    if tuple(a.data for a in op.indexing_maps) != (identity, columns, identity) or any(a.data.value != 'parallel' for a in op.iterator_types):
        return None
    body = op.regions[0].block
    ops = list(body.ops)
    if [x.name for x in ops] != ['arith.addi', 'linalg.yield']:
        return None
    add, yld = ops
    if set(add.operands) != {body.args[0], body.args[1]} or list(yld.operands) != [add.results[0]] or add.overflow_flags.data:
        return None
    return a, b


def _batched_matmul(op):
    """Prove identical batch axes plus the shared signed scalar recurrence."""
    if op.name != 'linalg.generic' or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    a, b = op.inputs
    at, bt, rt = _type(a), _type(b), _type(op.results[0])
    ash, bsh = at['shape'], bt['shape']
    if len(ash) < 3 or len(ash) != len(bsh):
        return None
    if ash[:-2] != bsh[:-2] or ash[-1] != bsh[-2] or rt != {'shape': [*ash[:-2], ash[-2], bsh[-1]], 'dtype': 'i32'}:
        return None
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr, IteratorType
    from merlin.frontends.linalg_patterns import recognize_signed_i8_i32_matmul
    batch_rank = len(ash) - 2
    dims = tuple(AffineDimExpr(i) for i in range(batch_rank + 3))
    batch, m, n, k = dims[:batch_rank], dims[-3], dims[-2], dims[-1]
    expected = (AffineMap(len(dims), 0, (*batch, m, k)), AffineMap(len(dims), 0, (*batch, k, n)), AffineMap(len(dims), 0, (*batch, m, n)))
    if tuple(a.data for a in op.indexing_maps) != expected or tuple(a.data.value for a in op.iterator_types) != ('parallel',) * (batch_rank + 2) + ('reduction',):
        return None
    # Axes above are proved against the original operation. The established
    # rank-two recognizer additionally proves scalar widths, signed extensions,
    # product/add/yield SSA connections, purity and absent overflow promises.
    proof = op.clone()
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    proof.properties['indexing_maps'] = ArrayAttr(AffineMapAttr(x) for x in (AffineMap(3, 0, (d0,d2)), AffineMap(3, 0, (d2,d1)), AffineMap(3, 0, (d0,d1))))
    proof.properties['iterator_types'] = ArrayAttr(IteratorTypeAttr(x) for x in (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION))
    try:
        recognize_signed_i8_i32_matmul(proof)
    except ValueError:
        return None
    finally:
        proof.drop_all_references()
    return a, b, op.outputs[0]

def lower_linalg(text: str) -> dict[str, Any]:
    """Lower proved signed contractions, contiguous input views and readouts.

    Scalar arithmetic and indexing are checked by shared dense recognition.
    Rank-two runtime i32 initializers use the existing accumulator preload.
    Full mixed programs can use parse_source's linalg_lowerer callback.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.frontends.linalg_patterns import recognize_signed_i8_i32_matmul
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    module = parse_mlir_text(text)
    definitions = [op for op in module.walk() if op.name == 'func.func' and op.body.blocks]
    forward = [op for op in definitions if op.sym_name.data == 'forward']
    if len(forward or definitions) != 1:
        raise FrontendError('source requires one unambiguous defined entry function')
    fn = (forward or definitions)[0]
    if len(fn.body.blocks) != 1:
        raise FrontendError('multi-block source needs complete control-flow lowering')
    block = fn.body.block
    term = block.ops.last
    if term is None or term.name != 'func.return':
        raise FrontendError('entry must end with func.return')
    def names(key):
        attr = fn.attributes.get(key)
        return [a.data for a in attr] if isinstance(attr, ArrayAttr) and all(isinstance(a, StringAttr) for a in attr) else None
    input_names, output_names = names('merlin.logical_inputs'), names('merlin.logical_outputs')
    if input_names is not None and len(input_names) != len(block.args):
        raise FrontendError('logical input names differ from entry arity')
    if output_names is not None and len(output_names) != len(term.operands):
        raise FrontendError('logical output names differ from return arity')
    roles = names('merlin.logical_input_roles')
    env, tensors, commands, accumulators, initializers = {}, {}, [], {}, set()
    views, aliases, bindings, view_proofs, view_bindings = {}, {}, {}, {}, []
    source_owners = {v: (i, j) for i, op in enumerate(block.ops) if op.name != 'func.return' for j, v in enumerate(op.results)}
    for i, value in enumerate(block.args):
        name = input_names[i] if input_names else (value.name_hint or f'arg{i}')
        if name in tensors:
            raise FrontendError(f'duplicate logical input name {name}')
        env[value] = name
        tensors[name] = {**_type(value), 'role': roles[i] if roles and len(roles) == len(block.args) else 'input'}
    returned = {v: output_names[i] if output_names else f'Y{i}' for i, v in enumerate(term.operands)}
    def command(code, operands, attrs=None, source_index=None):
        item = {'opcode': code, 'operands': operands}
        if source_index is None:
            source_index = ordinal
        item['attributes'] = {**(attrs or {}), 'merlin.global_task': 0, 'merlin.source_op_index': source_index}
        commands.append(item)
        return item
    def materialize(value):
        if value in views:
            root = views[value]
            while root in views:
                root = views[root]
            if root in block.args:
                if value in returned:
                    raise FrontendError('returned input view requires an actual output writer')
                if value not in env:
                    name = f'view{source_owners[value][0]}'
                    while name in tensors or name in aliases or name in returned.values():
                        name += '_'
                    source = materialize(root)
                    aliases[name] = {'source': source, **_type(value)}
                    env[value] = name
                    chain, step = [], value
                    while step in views:
                        chain.append({**view_proofs[step], 'op_index': source_owners[step][0],
                                      'result_index': source_owners[step][1]})
                        step = views[step]
                    view_bindings.append({'tensor': name, 'source': source, 'byte_offset': 0,
                                          'steps': list(reversed(chain))})
                return env[value]
            if value not in returned:
                raise FrontendError('intermediate computed view requires typed storage alias lowering')
            env[value] = materialize(root)
            return env[value]
        if value in env:
            return env[value]
        if value in accumulators:
            name = returned.get(value, f'tmp{len(env)}')
            command('COMMIT', {'src': accumulators[value], 'dst': name}, {'epilogue': [], 'output_dtype': 'i32'}, source_index=source_owners[value][0])
            env[value] = name
            bindings[name] = source_owners[value]
            return name
        raise FrontendError('consumer reads a value without target storage lowering')
    for ordinal, op in enumerate(block.ops):
        if op.name == 'func.return':
            break
        if op.name in ('tensor.collapse_shape', 'tensor.expand_shape'):
            from .source_views import prove_contiguous_reassociation, SourceViewError
            try:
                proof = prove_contiguous_reassociation(op)
            except SourceViewError as exc:
                raise FrontendError(str(exc)) from exc
            view_proofs[op.results[0]] = {**proof, 'attributes': _provenance(op)}
            views[op.results[0]] = op.operands[0]
            continue
        if op.name == 'arith.constant':
            if any(hasattr(v.type, 'get_shape') for v in op.results):
                raise FrontendError('observable tensor constant needs constant storage lowering')
            continue
        if op.name in ('tensor.empty', 'linalg.fill', 'tensor.splat'):
            initializers.update(op.results)
            continue
        batched = _batched_matmul(op)
        if batched is not None:
            lhs, rhs, init = batched
            if not _zero(init):
                raise FrontendError('batched contraction requires explicit zero initialization')
            result = op.results[0]
            name = returned.get(result, f'tmp{ordinal}')
            command('BATCHED_MATMUL', {'a': materialize(lhs), 'w': materialize(rhs), 'dst': name}, {'output_dtype': 'i32', **_provenance(op)})
            env[result] = name
            bindings[name] = source_owners[result]
            continue
        narrow = _saturating_cast(op) if op.name == 'linalg.generic' else None
        if narrow is not None and narrow in accumulators:
            result = op.results[0]
            name = returned.get(result, f'tmp{ordinal}')
            attrs = {'epilogue': [], 'output_dtype': 'i8', **_provenance(op)}
            command('COMMIT', {'src': accumulators[narrow], 'dst': name}, attrs)
            env[result] = name
            bindings[name] = source_owners[result]
            continue
        bias = _bias_add(op) if op.name == 'linalg.generic' else None
        if bias is not None:
            result = op.results[0]
            name = returned.get(result, f'tmp{ordinal}')
            command('BIAS_ADD', {'src': materialize(bias[0]), 'bias': materialize(bias[1]), 'dst': name}, {'output_dtype': 'i32', **_provenance(op)})
            env[result] = name
            bindings[name] = source_owners[result]
            continue
        if op.name == 'linalg.generic':
            try:
                lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
            except ValueError as exc:
                raise FrontendError(f'{op.name} at entry operation {ordinal}: {exc}') from exc
            a, b, out = _type(lhs), _type(rhs), _type(op.results[0])
            if len(a['shape']) != 2 or len(b['shape']) != 2 or a['shape'][1] != b['shape'][0] or out['shape'] != [a['shape'][0], b['shape'][1]]:
                raise FrontendError('contraction tensor geometry differs from reduction maps')
            resident, acc = f'res{ordinal}', f'acc{ordinal}'
            attrs = _provenance(op)
            if not _zero(init):
                if _type(init) != out:
                    raise FrontendError('contraction initializer differs from its i32 result type')
                attrs['accumulator_init'] = materialize(init)
            command('RES_PACK', {'src': materialize(rhs), 'dst': resident}, {'layout': 'packed_rhs'})
            command('MATMUL_RESIDENT', {'lhs': materialize(lhs), 'rhs': resident, 'dst': acc}, attrs)
            accumulators[op.results[0]] = acc
            command('EVICT', {'handle': resident})
            continue
        raise FrontendError(f'{op.name} at entry operation {ordinal} needs target lowering')
    return_roots = []
    for value in term.operands:
        root = value
        while root in views:
            root = views[root]
        return_roots.append(root)
    if len(set(return_roots)) != len(return_roots):
        raise FrontendError('returned values sharing view storage require independent output writers')
    for i, value in enumerate(term.operands):
        name = materialize(value)
        output = returned[value]
        if name != output:
            if value not in views or name in [env[a] for a in block.args]:
                raise FrontendError('returned alias requires an actual output writer')
            root_value = views[value]
            while root_value in views:
                root_value = views[root_value]
            aliases[name] = {'source': output, **_type(root_value)}
            tensors.pop(name, None)
            tensors[output] = {**_type(value), 'role': 'output'}
            bindings.pop(name, None)
            bindings[output] = source_owners[value]
    params = {
        'frontend_source_values': [{'op_index': i, 'result_index': j, 'tensor': name} for name, (i, j) in bindings.items()],
        'frontend_entry_bindings': [env[a] for a in block.args],
        'frontend_output_bindings': [returned[v] for v in term.operands],
    }
    if aliases:
        params['tensor_aliases'] = aliases
    if view_bindings:
        params['frontend_view_bindings'] = view_bindings
    return {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': commands,
            'outputs': [returned[v] for v in term.operands], 'params': params}


def infer_tensors(cb: dict[str, Any]) -> dict[str, Any]:
    """Return a copied command buffer with exact tensor boundary/intermediate types.

    Resident and accumulator handles are tracked internally, never exposed as
    kernel arguments. Explicit declarations are checked and preserved verbatim.
    Outputs subsequently consumed become intermediate buffers when undeclared.
    """
    from copy import deepcopy
    result = deepcopy(cb)
    tensors = result.setdefault('tensors', {})
    aliases = result.get('params', {}).get('tensor_aliases', {})
    residents: dict[str, str] = {}
    accumulators: dict[str, dict[str, Any]] = {}
    commands = result.get('commands', [])
    consumed = {name for c in commands for key, name in c.get('operands', {}).items() if key != 'dst'}

    def tensor(name):
        name = residents.get(name, name)
        if name in aliases:
            return aliases[name]
        if name not in tensors:
            raise FrontendError(f'command reads unknown tensor or resident {name!r}')
        return tensors[name]
    def positive(shape):
        if any(type(x) is not int or x <= 0 for x in shape):
            raise FrontendError(f'inferred nonpositive/dynamic tensor geometry {shape}')
        return list(shape)
    def define(name, shape, dtype):
        inferred = {'shape': positive(shape), 'dtype': dtype}
        if name in aliases:
            if any(aliases[name].get(key) != val for key, val in inferred.items()):
                raise FrontendError(f'view storage {name!r} differs from produced tensor type')
            return
        if name in tensors:
            if any(tensors[name].get(key) != val for key, val in inferred.items()):
                raise FrontendError(f'explicit tensor {name!r} differs from source-derived type: {tensors[name]} versus {inferred}')
            return
        tensors[name] = {**inferred, 'role': 'intermediate' if name in consumed else 'output'}
    def pooled(shape, attrs, spatial=None):
        if 'maxpool' not in attrs.get('epilogue', []) and attrs.get('_pool_op') is not True:
            return shape
        h, w = attrs['pool_in_dims']
        if spatial is not None and [h, w] != list(spatial):
            raise FrontendError('pool_in_dims differs from convolution output geometry')
        if shape[-2] % (h * w):
            raise FrontendError('pooled flat output rows disagree with pool_in_dims')
        batches = shape[-2] // (h * w)
        ph, pw = attrs['pool_size']
        sh, sw = attrs['pool_stride']
        pt, pl, pb, pr = attrs.get('pool_padding', [0, 0, 0, 0])
        ho, wo = (h + pt + pb - ph) // sh + 1, (w + pl + pr - pw) // sw + 1
        return [*shape[:-2], batches * ho * wo, shape[-1]]
    for index, cmd in enumerate(commands):
        code, operands, attrs = cmd['opcode'], cmd.get('operands', {}), cmd.get('attributes', {})
        dtype = attrs.get('output_dtype', 'i32')
        if code == 'RES_PACK':
            tensor(operands['src'])
            residents[operands['dst']] = residents.get(operands['src'], operands['src'])
        elif code == 'EVICT':
            residents.pop(operands['handle'], None)
        elif code == 'MATMUL_RESIDENT':
            a, b = tensor(operands['lhs']), tensor(operands['rhs'])
            if len(a['shape']) != 2 or len(b['shape']) != 2 or a['shape'][1] != b['shape'][0]:
                raise FrontendError('resident matmul requires matching rank-two reduction axes')
            accumulators[operands['dst']] = {'shape': [a['shape'][0], b['shape'][1]], 'dtype': 'i32'}
        elif code == 'COMMIT':
            if operands['src'] not in accumulators:
                raise FrontendError(f'commit reads unknown accumulator {operands["src"]!r}')
            define(operands['dst'], pooled(accumulators[operands['src']]['shape'], attrs), dtype)
        elif code == 'CONV2D':
            a, b = tensor(operands['ifm']), tensor(operands['weight'])
            if attrs.get('layout', 'nhwc') != 'nhwc' or len(a['shape']) != 4:
                raise FrontendError('convolution requires explicit NHWC geometry')
            batch, h, w, ci = a['shape']
            kh, kw, channels, co = attrs['kernel']
            if ci != channels or b['shape'] != [kh * kw * ci, co]:
                raise FrontendError('convolution input/filter geometry differs from kernel attributes')
            sh, sw = attrs.get('stride', [1, 1])
            dh, dw = attrs.get('dilation', [1, 1])
            pt, pl, pb, pr = attrs.get('padding', [0, 0, 0, 0])
            ho, wo = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1, (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
            define(operands['dst'], pooled([batch * ho * wo, co], attrs, [ho, wo]), dtype)
        elif code == 'BATCHED_MATMUL':
            a, b = tensor(operands['a'])['shape'], tensor(operands['w'])['shape']
            if len(a) < 3 or len(a) != len(b) or a[:-2] != b[:-2] or a[-1] != b[-2]:
                raise FrontendError('batched matmul requires identical batch axes and matching reduction axes')
            define(operands['dst'], [*a[:-2], a[-2], b[-1]], dtype)
        elif code in ('ATTENTION_QK', 'ATTENTION_PV'):
            # Named attention has a narrower ABI than the general dense
            # readout: no bias or spatial pooling, and no reordered/repeated
            # stages. Check before inferring any destination geometry.
            stages = attrs.get('epilogue', [])
            attention_stages = ('acc_scale', 'requant', 'relu')
            if (not isinstance(stages, (list, tuple))
                    or any(stage not in attention_stages for stage in stages)
                    or len(set(stages)) != len(stages)
                    or list(stages) != sorted(stages, key=attention_stages.index)):
                raise FrontendError('attention epilogue must be an ordered subset of [acc_scale, requant, relu]')
            keys = ('q', 'k') if code == 'ATTENTION_QK' else ('p', 'v')
            a, b = (tensor(operands[k])['shape'] for k in keys)
            reduction, n = (b[-1], b[-2]) if code == 'ATTENTION_QK' else (b[-2], b[-1])
            if len(a) != 2 or len(b) != 2 or a[-1] != reduction:
                raise FrontendError('attention contraction requires matching rank-two reduction axes')
            define(operands['dst'], pooled([a[-2], n], attrs), dtype)
        elif code == 'STRIDED_MOVEMENT':
            if code not in result.get('params', {}).get('source_internal_opcodes', []):
                raise FrontendError('strided movement is an internal extracted-source callback')
            src=tensor(operands['src']);perm=attrs.get('permutation',[])
            raw_float=(src['dtype']=='f32' and attrs.get('storage_dtype')=='i32'
                       and attrs.get('semantic')=='bit_preserving_permutation')
            if (not 2<=len(src['shape'])<=4 or any(type(i) is not int for i in perm)
                    or sorted(perm)!=list(range(len(src['shape']))) or not (src['dtype'] in ('i8','i32') or raw_float)
                    or attrs.get('output_dtype',src['dtype'])!=src['dtype']):
                raise FrontendError('strided movement requires an exact rank2-4 integer or raw-bit f32 permutation')
            define(operands['dst'],[src['shape'][i] for i in perm],src['dtype'])
        elif code == 'RAW_BIT_REPEAT':
            if code not in result.get('params', {}).get('source_internal_opcodes', []):
                raise FrontendError('raw-bit repeat is an internal extracted-source callback')
            if set(operands)!={'src','dst'} or set(attrs)!={'axis','repeats','output_dtype','storage_dtype','semantic'}:
                raise FrontendError('raw-bit repeat requires exact copy contract')
            src=tensor(operands['src'])
            if (src['dtype']!='f32' or attrs.get('output_dtype')!='f32'
                or attrs.get('storage_dtype')!='i32' or attrs.get('semantic')!='bit_preserving_repeat'):
                raise FrontendError('raw-bit repeat requires unchanged f32 metadata')
            from .bit_repeat import plan_repeat, BitRepeatError
            try:plan=plan_repeat(src['shape'],attrs['axis'],attrs['repeats'])
            except BitRepeatError as exc:raise FrontendError(str(exc)) from exc
            define(operands['dst'],list(plan.output_shape),'f32')
        elif code == 'RAW_BIT_SLICE':
            if code not in result.get('params', {}).get('source_internal_opcodes', []):
                raise FrontendError('raw-bit slice is an internal extracted-source callback')
            if set(operands)!={'src','dst'}:
                raise FrontendError('raw-bit slice requires one source and one destination')
            src=tensor(operands['src']);rank=len(src['shape'])
            offsets=attrs.get('offsets',[]);sizes=attrs.get('sizes',[]);strides=attrs.get('strides',[])
            if (not 2<=rank<=4 or src['dtype']!='f32'
                    or attrs.get('output_dtype')!='f32' or attrs.get('storage_dtype')!='i32'
                    or attrs.get('semantic')!='bit_preserving_slice'
                    or len(offsets)!=rank or len(sizes)!=rank or strides!=[1]*rank
                    or any(type(v) is not int or v<0 for v in offsets)
                    or any(type(v) is not int or v<=0 for v in sizes)
                    or any(o+n>extent for o,n,extent in zip(offsets,sizes,src['shape']))):
                raise FrontendError('raw-bit slice requires exact static unit-stride f32 source geometry')
            define(operands['dst'],sizes,'f32')
        elif code == 'RAW_BIT_CONCAT':
            if code not in result.get('params', {}).get('source_internal_opcodes', []):
                raise FrontendError('raw-bit concat is an internal extracted-source callback')
            count=len(operands)-1
            if count<2 or set(operands)!={'dst',*(f'src{i}' for i in range(count))}:
                raise FrontendError('raw-bit concat requires ordered source pointers and one destination')
            sources=[tensor(operands[f'src{i}']) for i in range(count)]
            shape=list(sources[0]['shape']);rank=len(shape);axis=attrs.get('axis')
            if (not 2<=rank<=4 or type(axis) is not int or not 0<=axis<rank
                    or attrs.get('output_dtype')!='f32' or attrs.get('storage_dtype')!='i32'
                    or attrs.get('semantic')!='bit_preserving_concat'
                    or any(src['dtype']!='f32' or len(src['shape'])!=rank
                           or any(d<=0 for d in src['shape']) for src in sources)
                    or any(any(src['shape'][d]!=shape[d] for d in range(rank) if d!=axis)
                           for src in sources)):
                raise FrontendError('raw-bit concat requires exact static f32 shapes and concat axis')
            shape[axis]=sum(src['shape'][axis] for src in sources)
            define(operands['dst'],shape,'f32')
        elif code == 'ACCUMULATOR_READOUT':
            if code not in result.get('params', {}).get('source_internal_opcodes', []):
                raise FrontendError('accumulator readout is an internal extracted-source callback')
            src = tensor(operands['src'])
            if src['dtype'] != 'i32' or attrs.get('output_dtype') != 'i8':
                raise FrontendError('internal accumulator readout requires i32-to-i8 tensors')
            define(operands['dst'], src['shape'], 'i8')
        elif code in ('MOVEMENT', 'BIAS_ADD', 'RESIDUAL_ADD', 'MAXPOOL2D'):
            src = tensor(operands['lhs'] if code == 'RESIDUAL_ADD' else operands['src'])
            if code == 'RESIDUAL_ADD' and tensor(operands['rhs'])['shape'] != src['shape']:
                raise FrontendError('residual addition operands have different shapes')
            if code == 'BIAS_ADD':
                bias = tensor(operands['bias'])
                if bias['shape'] != [src['shape'][-1]] or bias['dtype'] != src['dtype']:
                    raise FrontendError('bias shape/dtype differs from source accumulator columns')
            pool_attrs = {**attrs, '_pool_op': True} if code == 'MAXPOOL2D' else attrs
            if code == 'MAXPOOL2D' and len(src['shape']) == 4:
                b, h, w, c = src['shape']
                if [h, w] != attrs['pool_in_dims']:
                    raise FrontendError('NHWC pool geometry differs from its input axes')
                pt, pl, pb, pr = attrs.get('pool_padding', [0, 0, 0, 0])
                ph, pw = attrs['pool_size']; sh, sw = attrs['pool_stride']
                out_shape = [b, (h + pt + pb - ph) // sh + 1, (w + pl + pr - pw) // sw + 1, c]
            else:
                out_shape = pooled(src['shape'], pool_attrs)
            define(operands['dst'], out_shape, attrs.get('output_dtype', src['dtype']))
        else:
            raise FrontendError(f'no tensor inference for logical opcode {code!r} at command {index}')
    return result


def prepare_direct_program_abi(cb: dict[str, Any]) -> dict[str, Any]:
    """Declare source tensors as whole-program pointers before target emission."""
    result = infer_tensors(cb)
    params = result.get('params', {})
    entries, outputs = params.get('frontend_entry_bindings'), params.get('frontend_output_bindings')
    if entries is None or outputs is None:
        raise FrontendError('direct program ABI requires typed frontend source bindings')
    order = [*entries, *[n for n in result['tensors'] if n not in entries and n not in outputs], *outputs]
    if len(order) != len(set(order)):
        raise FrontendError('whole-program source input/output alias requires an explicit copy')
    result['kernel_abi'] = {'kind': 'whole_program',
        'args': [{'tensor': n, 'access': 'read' if n in entries else 'write' if n in outputs else 'readwrite'} for n in order],
        'outputs': list(outputs)}
    return result


def build_direct_program_plan(source: str, cb: dict[str, Any], *, schedule_instruction_count: int,
                              instruction_start: int = 0, instruction_end: int | None = None) -> dict[str, Any]:
    """Bind one emitted device task to exact source operations and actual ranges.

    Call prepare_direct_program_abi before compilation; invoke this function once
    the emitted schedule's real instruction count and task bounds are available.
    It validates declaration/source consistency through the public preflight.
    """
    from copy import deepcopy
    from pathlib import Path
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan
    result = deepcopy(cb)
    inventory = source_operation_inventory(source)
    params = result.setdefault('params', {})
    end = schedule_instruction_count if instruction_end is None else instruction_end
    if not 0 <= instruction_start < end <= schedule_instruction_count:
        raise FrontendError('direct task range must be a real nonempty subset of emitted instructions')
    entries, outputs = params['frontend_entry_bindings'], params['frontend_output_bindings']
    writes = [name for name in result['tensors'] if name not in entries]
    plan = {'schema': 'mixed_program_plan_v1', 'source_sha256': inventory['source_sha256'],
        'source_op_count': inventory['source_op_count'], 'schedule_instruction_count': schedule_instruction_count,
        'prologue_instruction_range': [0, instruction_start], 'epilogue_instruction_range': [end, schedule_instruction_count],
        'entry_bindings': list(entries), 'output_bindings': list(outputs),
        'source_values': deepcopy(params['frontend_source_values']),
        'tasks': [{'task_index': 0, 'kind': 'device', 'source_op_indices': list(range(inventory['source_op_count'])),
                   'source_region_ids': sorted({row['region_id'] for row in inventory['operations'] if row['region_id'] is not None}),
                   'instruction_start': instruction_start, 'instruction_end': end,
                   'reads': list(entries), 'writes': writes}]}
    params['global_program_plan'] = plan
    check = validate_mixed_program_plan(source, result, schema_path=Path(__file__).parent / 'data/mixed_program_plan.schema.json')
    if not check['ok']:
        raise FrontendError('direct program plan failed public source preflight: ' + '; '.join(check['findings']))
    return result


def declare_logical_pointer_abi(cb: dict[str, Any]) -> dict[str, Any]:
    """Make the selected logical boundary explicit for a canonical command graph.

    Uses the shared boundary reader's declared input and published result order.
    Internal consumption does not erase a published result or change its access.
    Existing explicit whole-program ABIs are returned unchanged in the copy.
    """
    from copy import deepcopy
    from pathlib import Path
    from merlin.compile.logical_abi import logical_abi, logical_interface
    result = deepcopy(cb)
    if result.get('kernel_abi', {}).get('kind') == 'whole_program':
        return result
    boundary = logical_interface(result, logical_abi(Path(__file__).parent / 'data/abi.yaml'))
    if any(row.kind not in ('input', 'output') for row in boundary):
        raise FrontendError('canonical logical boundary contains an undeclared pointer access class')
    tensors = result.setdefault('tensors', {})
    for row in boundary:
        spec = tensors.get(row.name)
        typ = {'shape': list(row.shape), 'dtype': row.dtype}
        if spec is None:
            tensors[row.name] = {**typ, 'role': row.kind}
        elif any(spec.get(k) != v for k, v in typ.items()):
            raise FrontendError(f'logical boundary type differs from tensor declaration {row.name!r}')
        elif row.kind == 'output':
            # A canonical COMMIT publishes its destination even when a later
            # command consumes it. Keep that observable result in the ABI.
            spec['role'] = 'output'
    result['kernel_abi'] = {
        'kind': 'whole_program',
        'args': [{'tensor': row.name, 'access': 'read' if row.kind == 'input' else 'write'} for row in boundary],
        'outputs': [row.name for row in boundary if tensors[row.name].get('role') == 'output'],
    }
    # Re-read the explicit form to ensure publication has preserved pointer
    # order, shape, dtype and access under the same selected ABI contract.
    explicit = logical_interface(result, logical_abi(Path(__file__).parent / 'data/abi.yaml'))
    if [(b.name, b.shape, b.dtype) for b in explicit] != [(b.name, b.shape, b.dtype) for b in boundary]:
        raise FrontendError('explicit canonical pointer ABI changed its selected logical boundary')
    return result
