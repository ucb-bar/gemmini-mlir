"""Source-proven constant integer padding composed from checked DMA forms."""
from math import prod


def constant_pad(op, spec, scalar_constant):
    """Recognize only same-rank, unit-stride insertion into a typed splat.

    No input range or floating-point conversion is inferred. The destination
    must actually declare its fill scalar in the source.
    """
    from xdsl.dialects.builtin import NoneAttr
    if op.name != 'tensor.insert_slice' or len(op.operands) != 2 or len(op.results) != 1:
        return None
    src, base = op.operands
    a, b, out = map(spec, (src, base, op.results[0]))
    rank = len(a['shape'])
    if (not 1 <= rank <= 4 or len(b['shape']) != rank or b != out
            or a['dtype'] not in ('i8', 'i32') or a['dtype'] != b['dtype']
            or any(not isinstance(v.type.encoding, NoneAttr) for v in (src, base, op.results[0]))):
        return None
    owner = base.owner
    if getattr(owner, 'name', '') != 'tensor.splat' or len(owner.operands) != 1:
        return None
    fill = scalar_constant(owner.operands[0])
    bits = 8 if a['dtype'] == 'i8' else 32
    if (str(owner.operands[0].type) != a['dtype'] or type(fill) is not int
            or not -(1 << (bits-1)) <= fill < (1 << (bits-1))):
        return None
    geometry = []
    for key in ('static_offsets', 'static_sizes', 'static_strides'):
        attr = op.properties.get(key)
        if attr is None or not hasattr(attr, 'get_values'):
            return None
        geometry.append(list(attr.get_values()))
    offsets, sizes, strides = geometry
    if (sizes != a['shape'] or len(offsets) != rank or strides != [1]*rank
            or any(o < 0 or o+n > d for o,n,d in zip(offsets, sizes, b['shape']))):
        return None
    stages = pad_stages(sizes, b['shape'], offsets, a['dtype'])
    if stages is None:
        return None
    attrs = dict(stages=stages, fill=fill, output_dtype=a['dtype'])
    proof = dict(form='source_constant_integer_pad', offsets=offsets, sizes=sizes,
                 strides=strides, fill=fill, stages=stages,
                 semantics='disjoint constant border slabs around unchanged source coordinates')
    return src, attrs, proof


def pad_stages(sizes, output, offsets, dtype):
    """Construct checked constant-border concat partitions of admitted geometry."""
    from .source_concat import plan_concat, ConcatError
    shape = list(sizes); stages = []
    for axis in range(len(sizes)):
        lo, hi = offsets[axis], output[axis]-offsets[axis]-sizes[axis]
        if not lo and not hi:
            continue
        shapes = []
        if lo:
            side = list(shape); side[axis] = lo; shapes.append(side)
        shapes.append(list(shape))
        if hi:
            side = list(shape); side[axis] = hi; shapes.append(side)
        try:
            checked = plan_concat(shapes, axis, dtype)
        except ConcatError:
            return None
        stages.append(dict(axis=axis, low=lo, high=hi, shapes=shapes,
                           output=list(checked.output), address_projection=checked.proof))
        shape = list(checked.output)
    if not stages or shape != output:
        return None
    return stages


def emit_pad_buffer(tensors, constants, src, dst, attrs):
    """Expand a proven pad into existing INTEGER_CONCAT commands only."""
    commands = []; current = src; dtype = attrs['output_dtype']
    for index, stage in enumerate(attrs['stages']):
        parts = []
        for side, width in (('low', stage['low']), ('source', 0), ('high', stage['high'])):
            if side == 'source':
                parts.append(current)
            elif width:
                shape = list(tensors[current]['shape']); shape[stage['axis']] = width
                name = f'_pad_{index}_{side}'
                tensors[name] = dict(shape=shape, dtype=dtype, role='intermediate')
                constants[name] = dict(shape=shape, dtype=dtype, values=[attrs['fill']]*prod(shape))
                parts.append(name)
        target = dst if index == len(attrs['stages'])-1 else f'_pad_stage_{index}'
        if target != dst:
            tensors[target] = dict(shape=stage['output'], dtype=dtype, role='intermediate')
        commands.append(dict(opcode='INTEGER_CONCAT',
            operands={**{f'src{i}':name for i,name in enumerate(parts)}, 'dst':target},
            attributes=dict(axis=stage['axis'], inputs=len(parts), output_dtype=dtype)))
        current = target
    return commands
