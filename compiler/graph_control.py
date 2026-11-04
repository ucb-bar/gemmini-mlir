"""Bounded tensor storage and exact host CFG for source SCF control.

Capacity is an allocation bound, never a replacement for a runtime tensor
extent. Bounds are proved from source integer ranges; unknown bounds refuse.
"""
from __future__ import annotations
import math
from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType, IndexType, IntegerType, IntegerAttr, i64
from xdsl.ir import Block, Operation
from merlin.llvmlower.host_builder import PTR


def _name(op):
    return getattr(getattr(op, 'op_name', None), 'data', getattr(op, 'name', 'block_argument'))


def _attr(op, key):
    return op.properties.get(key, op.attributes.get(key))


def _range(value, env=None):
    env = env or {}
    if value in env:
        return env[value]
    ty = value.type.element_type if isinstance(value.type, TensorType) else value.type
    if isinstance(ty, IntegerType) and ty.width.data == 1:
        return (0, 1)
    op = value.owner
    if not isinstance(op, Operation):
        raise ValueError('no source range proof for dynamic extent block argument')
    n = _name(op)
    if n == 'arith.constant':
        a = _attr(op, 'value')
        if isinstance(a, IntegerAttr):
            return (a.value.data, a.value.data)
        if hasattr(a, 'get_values'):
            values = tuple(a.get_values())
            return min(values), max(values)
    if n == 'arith.extsi':
        src = op.operands[0]
        if isinstance(src.type, IntegerType) and src.type.width.data == 1:
            return (-1, 0)
        return _range(src, env)
    if n == 'arith.extui':
        result = _range(op.operands[0], env)
        if result[0] < 0:
            raise ValueError('unsigned dynamic extent extension lacks a nonnegative range proof')
        return result
    if n in ('arith.index_cast', 'tensor.extract',
             'tensor.splat', 'tensor.cast', 'tensor.collapse_shape', 'tensor.expand_shape'):
        return _range(op.operands[0], env)
    if n in ('arith.addi', 'arith.muli'):
        a, b = (_range(v, env) for v in op.operands)
        candidates = [x+y if n == 'arith.addi' else x*y for x in a for y in b]
        result = min(candidates), max(candidates)
        width = 64 if isinstance(ty, IndexType) else ty.width.data
        if result[0] < -(1 << (width-1)) or result[1] >= 1 << (width-1):
            raise ValueError('dynamic extent interval may overflow source integer arithmetic')
        return result
    if n == 'linalg.generic':
        block = op.regions[0].block
        local = dict(env)
        for arg, operand in zip(block.args, op.operands):
            if arg.uses:
                local[arg] = _range(operand, env)
        for inner in block.ops:
            if _name(inner) == 'linalg.yield':
                return _range(inner.operands[value.index], local)
            for result in inner.results:
                local[result] = _range(result, local)
    if n == 'linalg.reduce':
        block = op.regions[0].block
        body = tuple(block.ops)
        if (len(body) != 2 or _name(body[0]) != 'arith.addi' or
                set(body[0].operands) != set(block.args) or
                tuple(body[1].operands) != tuple(body[0].results)):
            raise ValueError('dynamic extent reduction needs exact source integer sum')
        source, seed = op.operands
        lo, hi = _range(source, env)
        slo, shi = _range(seed, env)
        dims = list(_attr(op, 'dimensions').get_values())
        count = math.prod(source.type.get_shape()[i] for i in dims)
        result = (slo + lo*count, shi + hi*count)
        width = ty.width.data
        if result[0] < -(1 << (width-1)) or result[1] >= 1 << (width-1):
            raise ValueError('source dynamic-size reduction may overflow')
        return result
    raise ValueError(f'no source integer range proof for {_name(op)}')


def bounded_shape(value):
    """Return allocation capacities justified by the original SSA definitions."""
    shape = tuple(value.type.get_shape())
    if all(x >= 0 for x in shape):
        return shape
    owner = value.owner
    if not isinstance(owner, Operation):
        parent = owner.parent_op()
        if _name(parent) == 'scf.for' and value.index > 0:
            return bounded_shape(parent.operands[value.index + 2])
        raise ValueError('dynamic tensor argument has no bounded storage declaration')
    n = _name(owner)
    if n == 'tensor.empty':
        dims = iter(owner.operands)
        result = []
        for dim in shape:
            if dim >= 0:
                result.append(dim)
            else:
                low, high = _range(next(dims))
                if low < 0 or high < low:
                    raise ValueError('dynamic tensor extent has no nonnegative finite source bound')
                result.append(high)
        return tuple(result)
    if n == 'scf.for':
        return bounded_shape(owner.operands[3 + value.index])
    if n == 'scf.if':
        choices = [bounded_shape(r.block.last_op.operands[value.index]) for r in owner.regions]
        return tuple(max(dims) for dims in zip(*choices))
    if n in ('tensor.insert', 'tensor.cast'):
        return bounded_shape(owner.operands[1 if n == 'tensor.insert' else 0])
    raise ValueError(f'no bounded tensor allocation rule for {n}')


def _init(g):
    if not hasattr(g, '_dynamic_dims'):
        g._dynamic_dims = {}


def dimensions(g, value):
    """Actual LLVM i64 extents, including carried dynamic lengths."""
    _init(g)
    shape = value.type.get_shape()
    if value in g._dynamic_dims:
        return g._dynamic_dims[value]
    if any(x < 0 for x in shape):
        raise ValueError('runtime tensor extent was not bound by its defining operation')
    return tuple(g.b.const(x) for x in shape)


def _ensure(g, value):
    if value not in g.ptrs:
        ty = value.type.element_type
        count = max(1, math.prod(bounded_shape(value)))
        # Generic LLVM global type supports i1 and floating host tensors too.
        symbol = f'gemmini_control_scratch_{len(g.b.globals)}'
        g.b.globals.append(llvm.GlobalOp(llvm.LLVMArrayType(count, ty), symbol,
                                        'internal', alignment=64))
        g.ptrs[value] = g.b.add(llvm.AddressOfOp(symbol, PTR)).results[0]
    return g.ptrs[value]


def _flat(g, value, indices):
    result = g.b.const(0)
    for dim, index in zip(dimensions(g, value), indices):
        result = g.b.add_i(g.b.mul_i(result, dim), index)
    return result


def _loop(g, start, stop, step, body):
    """A runtime bounded host loop, with no source iteration substitution."""
    b = g.b
    header, active, after = Block(arg_types=[i64]), Block(), Block()
    for block in (header, active, after): b.region.add_block(block)
    b.add(llvm.BrOp(header, start))
    b.blk = header
    iv = header.args[0]
    pred = b.add(llvm.ICmpOp(iv, stop, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))).results[0]
    b.add(llvm.CondBrOp(pred, active, [], after, []))
    b.blk = active
    body(iv)
    latch = b.add(llvm.BrOp(header, b.add_i(iv, step)))
    from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
    disable_loop_unroll(latch)
    b.blk = after


def _copy(g, src, dst):
    if src == dst: return
    _ensure(g, src); _ensure(g, dst)
    ty = src.type.element_type
    if ty != dst.type.element_type:
        raise ValueError('control tensor copy changes element dtype')
    dims = dimensions(g, src)
    g._dynamic_dims[dst] = dims
    count = g.b.const(1)
    for dim in dims: count = g.b.mul_i(count, dim)
    def body(i):
        x = g.b.add(llvm.LoadOp(g.b.gep(g.ptrs[src], i, ty), ty)).results[0]
        g.b.add(llvm.StoreOp(x, g.b.gep(g.ptrs[dst], i, ty)))
    if g.ptrs[src] is not g.ptrs[dst]:
        different = g.b.add(llvm.ICmpOp(g.b.ptr(g.ptrs[src]), g.b.ptr(g.ptrs[dst]),
            IntegerAttr(llvm.ICmpPredicateFlag.NE.to_int(), i64))).results[0]
        # A carried scratch pointer commonly already owns this insert's result
        # allocation. Skip the identity copy, preserving its source SSA length.
        g.b.branch(different,
            lambda: _loop(g, g.b.const(0), count, g.b.const(1), body),
            lambda: None)


def _exclusive(first, second):
    """True only when operations are in opposite source scf.if arms."""
    ancestry = {}
    child = first
    while child.parent_op() is not None:
        parent = child.parent_op()
        ancestry[parent] = child.parent_region()
        child = parent
    child = second
    while child.parent_op() is not None:
        parent = child.parent_op()
        if (_name(parent) == 'scf.if' and parent in ancestry and
                ancestry[parent] is not child.parent_region()):
            return True
        child = parent
    return False


def _check_insert_lifetime(src, insert):
    # One scratch instance per source insert is reusable across loop iterations
    # only when the carried old version has no later live read on that arm.
    if isinstance(src.owner, Block) and _name(src.owner.parent_op()) == 'scf.for':
        for use in src.uses:
            if use.operation is insert:
                continue
            if _name(use.operation) != 'scf.yield' or not _exclusive(insert, use.operation):
                raise ValueError('loop-carried tensor insert has a simultaneously live old tensor version')


def _values(g, values):
    result = []
    for value in values:
        if isinstance(value.type, TensorType):
            result.append(_ensure(g, value))
            result.extend(d for s,d in zip(value.type.get_shape(), dimensions(g,value)) if s < 0)
        else:
            result.append(g.scalars[value])
    return result


def _types(values):
    result = []
    for value in values:
        ty = value.type
        if isinstance(ty, TensorType):
            result.append(PTR)
            result.extend(i64 for d in ty.get_shape() if d < 0)
        else:
            result.append(i64 if isinstance(ty, IndexType) else ty)
    return result


def _bind(g, source_values, lowered):
    lowered = iter(lowered)
    for value in source_values:
        if isinstance(value.type, TensorType):
            g.ptrs[value] = next(lowered)
            g._dynamic_dims[value] = tuple(next(lowered) if d < 0 else g.b.const(d)
                                           for d in value.type.get_shape())
        else:
            g.scalars[value] = next(lowered)


def _region(g, region):
    block = region.block
    for op in block.ops:
        if _name(op) == 'scf.yield':
            return _values(g, op.operands)
        g.operation(op)
    raise ValueError('source control region has no scf.yield')


def _result(g, op, lowered):
    # Named top-level buffers belong to the logical plan. Materialize into those
    # exact buffers, while retaining the runtime descriptor's actual extent.
    original = {v:g.ptrs[v] for v in op.results if v in g.ptrs and isinstance(v.type, TensorType)}
    _bind(g, op.results, lowered)
    for value, pointer in original.items():
        current = g.ptrs[value]
        if current is pointer: continue
        ty = value.type.element_type
        dims = dimensions(g,value)
        count = g.b.const(1)
        for d in dims: count = g.b.mul_i(count,d)
        def body(i):
            x = g.b.add(llvm.LoadOp(g.b.gep(current,i,ty),ty)).results[0]
            g.b.add(llvm.StoreOp(x,g.b.gep(pointer,i,ty)))
        _loop(g,g.b.const(0),count,g.b.const(1),body)
        g.ptrs[value] = pointer


def lower_control(g, op):
    """Lower explicit host control/tensor ops, or return False to the graph lane."""
    _init(g)
    n = _name(op)
    b = g.b
    if n == 'tensor.empty' and any(d < 0 for d in op.results[0].type.get_shape()):
        value = op.results[0]
        bounded_shape(value)  # allocation proof, independent of runtime length
        dynamic = iter(op.operands)
        g._dynamic_dims[value] = tuple(g.scalars[next(dynamic)] if d < 0 else b.const(d)
                                       for d in value.type.get_shape())
        _ensure(g,value)
        return True
    if n == 'tensor.dim':
        src, axis = op.operands
        # Dimension indices are structural source constants in this contract.
        low,high = _range(axis)
        if low != high or not 0 <= low < len(src.type.get_shape()):
            raise ValueError('tensor.dim requires a source constant in-range axis')
        g.scalars[op.results[0]] = dimensions(g,src)[low]
        return True
    if n == 'tensor.extract':
        src,*indices = op.operands
        ty = src.type.element_type
        index = _flat(g,src,[g.scalars[v] for v in indices])
        g.scalars[op.results[0]] = b.add(llvm.LoadOp(b.gep(_ensure(g,src),index,ty),ty)).results[0]
        return True
    if n == 'tensor.insert':
        scalar,src,*indices = op.operands
        dst = op.results[0]
        _check_insert_lifetime(src, op)
        _copy(g,src,dst)
        index = _flat(g,dst,[g.scalars[v] for v in indices])
        b.add(llvm.StoreOp(g.scalars[scalar],b.gep(g.ptrs[dst],index,dst.type.element_type)))
        return True
    if n == 'cf.assert':
        good,bad,after = Block(),Block(),Block()
        for block in (good,bad,after): b.region.add_block(block)
        b.add(llvm.CondBrOp(g.scalars[op.operands[0]],good,[],bad,[]))
        b.blk=bad
        b.add(llvm.CallIntrinsicOp('llvm.trap',[],[]))
        # llvm.trap never returns. Retain an ordinary CFG edge afterward so
        # the selected public task-plan auditor can traverse every block.
        b.add(llvm.BrOp(after))
        b.blk=good
        b.add(llvm.BrOp(after))
        b.blk=after
        return True
    if n == 'scf.if':
        if len(op.regions) != 2:
            raise ValueError('result-carrying source if requires both branches')
        yes,no = Block(),Block()
        after = Block(arg_types=_types(op.results))
        for block in (yes,no,after): b.region.add_block(block)
        b.add(llvm.CondBrOp(g.scalars[op.operands[0]],yes,[],no,[]))
        for block,region in zip((yes,no),op.regions):
            b.blk=block
            yielded = _region(g,region)
            b.add(llvm.BrOp(after,*yielded))
        b.blk=after
        _result(g,op,after.args)
        return True
    if n == 'scf.for':
        start,stop,step,*initial = op.operands
        lo,hi = _range(step)
        if lo <= 0 or lo != hi:
            raise ValueError('source scf.for requires a proved positive constant step')
        header = Block(arg_types=[i64,*_types(initial)])
        active = Block()
        after = Block(arg_types=_types(initial))
        for block in (header,active,after): b.region.add_block(block)
        b.add(llvm.BrOp(header,g.scalars[start],*_values(g,initial)))
        b.blk=header
        pred = b.add(llvm.ICmpOp(header.args[0],g.scalars[stop],
                       IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
        b.add(llvm.CondBrOp(pred,active,[],after,list(header.args[1:])))
        b.blk=active
        block=op.regions[0].block
        g.scalars[block.args[0]]=header.args[0]
        _bind(g,block.args[1:],header.args[1:])
        yielded = _region(g,op.regions[0])
        latch=b.add(llvm.BrOp(header,b.add_i(header.args[0],g.scalars[step]),*yielded))
        from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
        disable_loop_unroll(latch)
        b.blk=after
        _result(g,op,after.args)
        return True
    return False
