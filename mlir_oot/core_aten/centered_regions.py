"""Proven byte-origin i32 operands, decomposed into exact signed-byte products.

The source's i32 extension/subtraction/layout graph remains authoritative. A
closed integer interpreter proves each operand is a selector plus a constant;
no centered value is truncated. Constant planes are literal signed bytes.
"""
import hashlib

import numpy as np

from . import linmap, ops
from .exactness import require
from .limbs import IntegerRegion
from .match import etype, shape, producer, generic_maps, identity_map, iterators
from .recipes import movement as mv
from .recipes.gemm import blocked_matmul
from .stream import Arg


def extension(value):
    op = producer(value)
    if (etype(value) != 'i32' or op is None or op.name != 'linalg.generic' or
            len(op.operands) != 2 or len(op.results) != 1 or
            etype(op.operands[0]) != 'i8' or shape(value) != shape(op.operands[0]) or
            not all(identity_map(m) for m in generic_maps(op)) or
            any(t != 'parallel' for t in iterators(op))):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.ops)
    if (len(body) != 2 or body[0].name not in ('arith.extsi', 'arith.extui') or
            body[1].name != 'linalg.yield' or list(body[0].operands) != [block.args[0]] or
            str(body[0].results[0].type) != 'i32' or
            list(body[1].operands) != [body[0].results[0]]):
        return None
    return op.operands[0], body[0].name == 'arith.extui'


def operand_selectors(op):
    """Each i32 operand as one byte selector plus a literal constant, from the source.

    Returns ``(sources, selectors, constants, ops)``: ``sources[i]`` is ``(raw byte
    value, unsigned)``; selector ``(which, indices)`` picks ``sources[which]`` at
    flat ``indices`` (``which < 0``: none); unsigned selections read the packed
    signed byte, so their constants carry the +128. Emission and admission share it.
    """
    require(all(int(np.prod(shape(v))) <= 1 << 23 for v in (*op.operands, *op.results)),
            'centered contraction exceeds static operand-map capacity')
    sources, keys, carriers = [], {}, []

    def leaf(value):
        found = extension(value)
        if found is None:
            return None
        raw, unsigned = found
        require(int(np.prod(shape(raw))) <= 1 << 23,
                'centered byte source exceeds static operand-map capacity')
        key = (raw, unsigned)
        if key not in keys:
            keys[key] = len(sources)
            sources.append(key)
        carriers.append(producer(value))
        # XOR-0x80 packing is the existing exact unsigned-byte ABI. Thus
        # extui(raw) = signed(packed) + 128, for every storage bit pattern.
        return keys[key], 1, 128 if unsigned else 0

    region = IntegerRegion(leaf)
    try:
        values = [region.tensor(v) for v in op.operands[:2]]
    except linmap.NotLinear as exc:
        require(False, 'centered byte operand proof: ' + str(exc))
    require(sources, 'i32 contraction operands have no proven byte extension')
    selectors, constants = [], []
    for root, value in zip(op.operands, values):
        require(isinstance(value, linmap.Sym), 'centered operand is not byte-derived')
        found = linmap.selector(value, shape(root), compact=True)
        require(found is not None, 'centered operand must select at most one byte with unit coefficient')
        which, indices, constant = found
        constant = constant.reshape(shape(root))
        require(all(int(c) == c and abs(c) <= 255 for c in constant.flat),
                'centered byte correction is not a bounded integer constant')
        selectors.append((which, indices))
        constants.append(np.asarray(constant, np.int64))

    return sources, selectors, constants, list(dict.fromkeys([*region.ops, *carriers]))


def kernel(op, dims, batch, bias, seed, bias_rows, seed_range):
    sources, selectors, constants, producers = operand_selectors(op)

    # Every source intermediate was independently range checked by IntegerRegion.
    # The expanded absolute sum proves *all* schedule prefixes, including seed,
    # literal planes, each K block, and all byte-plane pairs without cancellation.
    magnitudes = [(128*(sel[0] >= 0)).reshape(c.shape) + np.abs(c)
                  for sel, c in zip(selectors, constants)]
    bound = int(np.max(np.matmul(magnitudes[0].astype(object), magnitudes[1].astype(object))))
    lo, hi = -bound, bound
    if bias:
        lower, upper = (int(np.min(seed)), int(np.max(seed))) if seed is not None else seed_range
        lo += min(0, lower); hi += max(0, upper)
    require(-(1 << 31) <= lo <= hi < (1 << 31),
            'centered byte plane accumulation has no nonoverflow i32 bound')

    operands = [(f'x{i}', v) for i, (v, _) in enumerate(sources)]
    args = [Arg(name, 'i8', shape(v), 'input') for name, v in operands]
    if bias and seed is None:
        args.append(Arg('bias', 'i32', shape(op.operands[2]) if bias_rows else (dims.n,), 'input'))
    args.append(Arg('out', 'i32', shape(op.results[0]), 'output'))
    s = ops._begin(args)
    names = []
    for i, (v, unsigned) in enumerate(sources):
        names.append(s.signed_scratch(f'x{i}', shape(v)) if unsigned else f'x{i}')
    planes = []
    for axis, (root, (which, indices), constant) in enumerate(zip(op.operands, selectors, constants)):
        prefix = f'operand{axis}'
        parts = []
        if np.any(which >= 0):
            # Materialize the source's exact index map using scratchpad byte DMA.
            # Pad positions contribute zero here; their literal correction is
            # also zero, because padding was applied after source centering.
            s._index[prefix] = len(s.args);s.args.append(Arg(prefix, 'i8', shape(root), 'scratch'))
            s.const(prefix+'__zero', np.zeros(which.size, np.int8))
            s.fence()
            mv.gather_elements(s, which, indices, 1, prefix, names, prefix+'__zero')
            s.fence()
            parts.append(prefix)
        remaining = constant.copy()
        while np.any(remaining):
            literal = np.clip(remaining, -127, 127).astype(np.int8)
            name = prefix+f'__constant{len(parts)}'
            s.const(name, literal)
            parts.append(name)
            remaining -= literal.astype(np.int64)
        require(parts, 'centered contraction has an identically zero operand')
        planes.append(parts)
    bname = None
    if bias:
        bname = 'bias'
        if seed is not None:
            bname = 'bias_c';s.const(bname, np.asarray(seed, np.int32))
    initialized = False
    for a in planes[0]:
        for b in planes[1]:
            for bi in range(max(batch, 1)):
                s.fence()
                use_seed = initialized or bias
                seed_name = 'out' if initialized else bname
                rows = initialized or bias_rows
                blocked_matmul(s, (a, bi*dims.m*dims.k), dims.k,
                               (b, bi*dims.k*dims.n), dims.n,
                               ('out', bi*dims.m*dims.n*4), dims.n, dims.m, dims.k, dims.n,
                               bias=(seed_name, bi*dims.m*dims.n*4 if rows else 0) if use_seed else None,
                               bias_rows=rows, out_i32=True)
            initialized = True
    proof = dict(version=1, rule='closed_byte_selectors_plus_literal_integer_planes',
                 unsigned_packing='xor_0x80_then_add_128',
                 operand_planes=[len(p) for p in planes], product_count=len(planes[0])*len(planes[1]),
                 operand_bounds=[int(np.max(m)) for m in magnitudes],
                 accumulator_min=lo, accumulator_max=hi,
                 prefix_bound='abs(seed)+sum_k(sum_plane_abs(A)*sum_plane_abs(B))',
                 selector_sha256=[hashlib.sha256(w.tobytes()+i.tobytes()+c.tobytes()).hexdigest()
                                  for (w,i),c in zip(selectors,constants)],
                 unsigned_sources=[name for (name,_),(_,unsigned) in zip(operands,sources) if unsigned],
                 source_operand_types=[str(v.type) for v in op.operands[:2]],
                 rounding='none', device_scales='unit', readout='full_i32')
    result = ops._end('aten.bmm.default' if batch else 'aten.mm.default', s,
                      accumulator_min=lo, accumulator_max=hi, centered_byte_contraction=proof)
    return result, operands, producers
