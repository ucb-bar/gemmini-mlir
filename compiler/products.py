"""Normalize projected integer contractions to the checked dense accelerator path.

CPU loops only rearrange elements. Every multiply and reduction executes on the
Gemmini array; dimension projections retain arbitrary row-major logical layouts.
"""
from __future__ import annotations
from math import prod
import re
try:
    from .dense import lower_matmul
    from . import target
except ImportError:
    from dense import lower_matmul
    import target


def _projection(text):
    match = re.fullmatch(r'affine_map<\((.*?)\)\s*->\s*\((.*?)\)>', text.strip())
    if not match:
        raise ValueError('contraction requires symbol-free projected dimension maps')
    dimensions = [s.strip() for s in match[1].split(',') if s.strip()]
    outputs = [s.strip() for s in match[2].split(',') if s.strip()]
    if dimensions != [f'd{i}' for i in range(len(dimensions))]:
        raise ValueError('noncanonical contraction map domain')
    if any(not re.fullmatch(r'd[0-9]+', s) for s in outputs):
        raise ValueError('contraction map must project dimensions without arithmetic')
    result = tuple(int(s[1:]) for s in outputs)
    if len(set(result)) != len(result) or any(i >= len(dimensions) for i in result):
        raise ValueError('contraction map has repeated or missing domain dimensions')
    return len(dimensions), result


def _integer_product(record):
    """Match the exact signed-i8, modular-i32 multiply-add scalar DAG."""
    body = record.get('scalar_body') or {}
    if body.get('arguments') != ['i8', 'i8', 'i32'] or body.get('captures'):
        return False
    env = {'arg:0': ('a',), 'arg:1': ('b',), 'arg:2': ('old',)}
    for i, op in enumerate(body.get('operations', ())):
        args = [env.get(a) for a in op.get('operands', ())]
        if op.get('results') != ['i32'] or op.get('regions', 0):
            return False
        name = op['op']
        if name == 'arith.extsi' and args in ([('a',)], [('b',)]):
            value = ('wide', args[0])
        elif name == 'arith.muli' and len(args) == 2 and set(args) == {('wide', ('a',)), ('wide', ('b',))}:
            value = ('product',)
        elif name == 'arith.addi' and len(args) == 2 and set(args) == {('old',), ('product',)}:
            value = ('sum',)
        else:
            return False
        env[f'op:{i}:0'] = value
    return len(body.get('yields', ())) == 1 and env.get(body['yields'][0]) == ('sum',)


def _reorder(b, src, dst, source_dims, destination_dims, extents, dtype):
    """Copy one projected layout into another using a counted index loop."""
    if set(source_dims) != set(destination_dims):
        raise ValueError('layout copy must preserve all logical dimensions')
    count = prod(extents[d] for d in destination_dims)
    def copy(index):
        coordinates = {}
        stride = count
        for dim in destination_dims:
            stride //= extents[dim]
            coordinates[dim] = b.urem_i(b.udiv_i(index, b.const(stride)), b.const(extents[dim]))
        offset = b.const(0)
        for dim in source_dims:
            offset = b.add_i(b.mul_i(offset, b.const(extents[dim])), coordinates[dim])
        value = b.load_i64(b.pointer(src), offset, dtype)
        b.store_i64(value, b.pointer(dst), index, dtype)
    b.loop(0, count, 1, copy)


def _contract(b, cb, operands, pointers, maps, rank, attrs, record=None):
    lhs, rhs, dst = operands
    specs = [cb['tensors'][name] for name in operands]
    if [s['dtype'] for s in specs[:2]] != ['i8', 'i8']:
        raise ValueError('contraction accelerator requires signed i8 operands')
    if specs[2]['dtype'] not in ('i32', 'i8'):
        raise ValueError('contraction readout requires i32 or declared i8 scaling')
    stages = tuple(attrs.get('epilogue', ()))
    if any(stage not in ('acc_scale', 'relu') for stage in stages):
        raise ValueError('product readout has an unsupported epilogue stage')
    if specs[2]['dtype'] == 'i32' and (stages or any(float(attrs.get(key, 1.)) != 1. for key in ('scale', 'acc_scale'))):
        raise ValueError('full-width product readout cannot discard scaling or activation')
    extents = {}
    for spec, dims in zip(specs, maps, strict=True):
        if len(spec['shape']) != len(dims):
            raise ValueError('contraction indexing map rank differs from tensor rank')
        for dim, size in zip(dims, spec['shape'], strict=True):
            if type(size) is not int or size <= 0 or extents.get(dim, size) != size:
                raise ValueError('contraction dimension has inconsistent static extents')
            extents[dim] = size
    if set(extents) != set(range(rank)):
        raise ValueError('contraction has unconstrained dimensions')
    a, w, y = map(set, maps)
    batch = tuple(d for d in maps[2] if d in a & w)
    mdims = tuple(d for d in maps[2] if d in a - w)
    ndims = tuple(d for d in maps[2] if d in w - a)
    kdims = tuple(d for d in maps[0] if d in a & w - y)
    if a != set(batch + mdims + kdims) or w != set(batch + kdims + ndims) or y != set(batch + mdims + ndims) or not kdims:
        raise ValueError('indexing maps do not describe a projected dense contraction')
    if record:
        reductions = set(record.get('reduction_dims', ()))
        if reductions != set(kdims):
            raise ValueError('contraction iterator kinds disagree with indexed reduction')
        outs = record.get('outs', ())
        if len(outs) != 1 or outs[0].get('const_value') != 0:
            raise ValueError('contraction requires an explicit zero accumulator initializer')
    batches, m, k, n = [prod(extents[d] for d in dims) for dims in (batch, mdims, kdims, ndims)]
    canonical = (batch + mdims + kdims, batch + kdims + ndims, batch + mdims + ndims)
    packed = []
    for name, dims, order, spec in zip(operands[:2], maps[:2], canonical[:2], specs[:2], strict=True):
        ptr = pointers[name]
        if dims != order:
            tmp = b.alloca(prod(spec['shape']), 'i8')
            _reorder(b, ptr, tmp, dims, order, extents, 'i8')
            ptr = tmp
        packed.append(ptr)
    raw = pointers[dst] if maps[2] == canonical[2] and specs[2]['dtype'] == 'i32' else b.alloca(batches*m*n, 'i32')
    def lower_batch(index):
        lower_matmul(b, b.ptr(packed[0], b.mul_i(index, b.const(m*k))),
                     b.ptr(packed[1], b.mul_i(index, b.const(k*n))),
                     b.ptr(raw, b.mul_i(index, b.const(m*n*4))), m, k, n)
    b.loop(0, batches, 1, lower_batch)
    output = raw
    if specs[2]['dtype'] == 'i8':
        scale = attrs.get('acc_scale', attrs.get('scale'))
        if scale is None:
            raise ValueError('narrow contraction output needs an explicit accumulator scale')
        output = b.alloca(batches*m*n, 'i8')
        target.cfg_ld(b, n*4, shrunk=False)
        target.cfg_st(b, n, scale=float(scale), activation=int('relu' in attrs.get('epilogue', ())))
        def row(r):
            for col in range(0,n,target.DIM):
                nc = min(target.DIM,n-col)
                off = b.add_i(b.mul_i(r,b.const(n)),b.const(col))
                target.mvin(b,b.ptr(raw,b.mul_i(off,b.const(4))),target.ACC,1,nc)
                b.fence()
                target.mvout(b,b.ptr(output,off),target.ACC,1,nc)
                b.fence()
        b.loop(0,batches*m,1,row)
    if output is not pointers[dst]:
        _reorder(b,output,pointers[dst],canonical[2],maps[2],extents,specs[2]['dtype'])
    return True


def lower_product(b, cb, command, pointers):
    """Return False for nonproducts; emit admitted products or raise on invalid contracts."""
    opcode = command['opcode']; o = command.get('operands', {}); attrs = command.get('attributes', {})
    record = attrs.get('semantic_record')
    if opcode in ('VREDUCE', 'MATMUL') and record and record.get('scalar_body'):
        if not _integer_product(record):
            return False
        parsed = [_projection(x) for x in record.get('indexing_maps', ())]
        if len(parsed) != 3 or len({x[0] for x in parsed}) != 1:
            raise ValueError('product requires two operand maps and one result map')
        return _contract(b,cb,(o['in0'],o['in1'],o['dst0']),pointers,tuple(x[1] for x in parsed),parsed[0][0],attrs,record)
    if opcode not in ('BATCHED_MATMUL', 'ATTENTION_QK', 'ATTENTION_PV'):
        return False
    left, right = ('a','w') if opcode == 'BATCHED_MATMUL' else ('q','k') if opcode == 'ATTENTION_QK' else ('p','v')
    names = (o[left],o[right],o['dst'])
    shapes = [cb['tensors'][x]['shape'] for x in names]
    rank = len(shapes[0])
    if rank < 2 or any(len(s)!=rank for s in shapes) or any(s[:-2] != shapes[0][:-2] for s in shapes):
        raise ValueError('product operands require equal static batch prefixes')
    prefix = tuple(range(rank-2)); md,kd,nd = rank-2,rank-1,rank
    maps = (prefix+(md,kd),prefix+((nd,kd) if opcode=='ATTENTION_QK' else (kd,nd)),prefix+(md,nd))
    return _contract(b,cb,names,pointers,maps,rank+1,attrs,record)
