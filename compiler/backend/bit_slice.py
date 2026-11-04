"""Checked static f32 rectangular slice materialization with raw accumulator DMA.

The source retains its dense row pitch. The separate destination is densely
written once; no source view is reinterpreted as a dense allocation or copied by
CPU arithmetic. Every payload word passes through full-width raw transport.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from merlin.sched.ir import Call, Const, Kernel, Loop, Sum, Scaled, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class BitSliceError(ValueError):
    pass


def _call(name, **args):
    return Call(name, tuple((key, as_expr(value)) for key, value in args.items()))


def _strides(shape):
    return tuple(prod(shape[i + 1:]) for i in range(len(shape)))


def _geometry(shape, offsets, sizes):
    shape, offsets, sizes = tuple(shape), tuple(offsets), tuple(sizes)
    if (not 2 <= len(shape) <= 4 or len(offsets) != len(shape) or len(sizes) != len(shape)
            or any(type(v) is not int or v <= 0 for v in (*shape, *sizes))
            or any(type(v) is not int or v < 0 for v in offsets)
            or any(o + n > extent for o, n, extent in zip(offsets, sizes, shape))
            or max(shape[-1], sizes[-1]) * 4 >= 1 << 32
            or prod(shape) * 4 >= 1 << 63):
        raise BitSliceError('slice requires bounded static rank2-4 dense f32 geometry and unit source strides')
    return shape, offsets, sizes


@dataclass(frozen=True)
class BitSlicePlan:
    shape: tuple
    offsets: tuple
    sizes: tuple
    kernel: Kernel
    proof: dict

    def revalidate(self):
        return check_slice(self.kernel, self.shape, self.offsets, self.sizes)


def plan_slice(shape, offsets, sizes, dtype='f32'):
    if dtype != 'f32':
        raise BitSliceError('raw slice transport requires f32 source/output metadata')
    shape, offsets, sizes = _geometry(shape, offsets, sizes)
    src_strides, dst_strides = _strides(shape), _strides(sizes)
    src_base = sum(offset * stride for offset, stride in zip(offsets, src_strides))
    src_outer = sum((Var(f'd{i}') * src_strides[i] for i in range(len(shape) - 2)), Const(src_base))
    dst_outer = sum((Var(f'd{i}') * dst_strides[i] for i in range(len(shape) - 2)), Const(0))

    def tile(row, col, rows, cols):
        return (_call('slice_mvin', offset=src_outer + row * shape[-1] + col,
                      addr=isa.acc_addr(0), rows=rows, cols=cols),
                _call('slice_drain'),
                _call('slice_mvout', offset=dst_outer + row * sizes[-1] + col,
                      addr=isa.acc_addr(0, full=True), rows=rows, cols=cols),
                _call('slice_drain'))

    def columns(row, rows):
        full, tail = divmod(sizes[-1], isa.DIM)
        body = []
        if full:
            body.append(Loop('col', full, tile(row, Var('col') * isa.DIM, rows, isa.DIM)))
        if tail:
            body.extend(tile(row, Const(full * isa.DIM), rows, tail))
        return tuple(body)

    full, tail = divmod(sizes[-2], isa.DIM)
    body = []
    if full:
        body.append(Loop('row', full, columns(Var('row') * isa.DIM, isa.DIM)))
    if tail:
        body.extend(columns(Const(full * isa.DIM), tail))
    body = tuple(body)
    for i in reversed(range(len(shape) - 2)):
        body = (Loop(f'd{i}', sizes[i], body),)
    kernel = Kernel('raw_bit_slice', (TensorArg('src', shape, 'f32', 'read'),
                    TensorArg('dst', sizes, 'f32', 'write')),
                    (_call('slice_drain'), _call('slice_config'), *body))
    return BitSlicePlan(shape, offsets, sizes, kernel, check_slice(kernel, shape, offsets, sizes))


def check_slice(kernel, shape, offsets, sizes):
    """Prove the actual DMA map, bounds, uniqueness and complete dense writes."""
    import z3
    shape, offsets, sizes = _geometry(shape, offsets, sizes)
    expected = (TensorArg('src', shape, 'f32', 'read'), TensorArg('dst', sizes, 'f32', 'write'))
    if (kernel.args != expected or kernel.body[:2] != (_call('slice_drain'), _call('slice_config'))):
        raise BitSliceError('slice tensor boundaries or raw transport configuration changed')
    sites = []

    def integer(call, key):
        expr = call.arg(key)
        if not isinstance(expr, Const):
            raise BitSliceError('slice DMA descriptor must be constant')
        return expr.value

    def walk(body, scope):
        at = 0
        while at < len(body):
            op = body[at]
            if isinstance(op, Loop):
                if (op.var in dict(scope) or op.extent <= 0 or op.iter_args or op.yields or op.results):
                    raise BitSliceError('slice requires ordinary bounded counted loops')
                walk(op.body, scope + ((op.var, op.extent),)); at += 1
                continue
            group = body[at:at + 4]
            if (len(group) != 4 or any(not isinstance(o, Call) for o in group)
                    or [o.instr for o in group] != ['slice_mvin', 'slice_drain', 'slice_mvout', 'slice_drain']):
                raise BitSliceError('slice staging completion or DMA order changed')
            load, first_drain, store, last_drain = group
            rows, cols = integer(load, 'rows'), integer(load, 'cols')
            if (not 0 < rows <= min(isa.DIM, isa.ACC_ROWS) or not 0 < cols <= isa.DIM
                    or integer(store, 'rows') != rows or integer(store, 'cols') != cols
                    or integer(load, 'addr') != isa.acc_addr(0)
                    or integer(store, 'addr') != isa.acc_addr(0, full=True)
                    or set(dict(load.args)) != {'offset', 'addr', 'rows', 'cols'}
                    or set(dict(store.args)) != {'offset', 'addr', 'rows', 'cols'}
                    or first_drain.args or last_drain.args):
                raise BitSliceError('slice raw DMA descriptor changed')
            sites.append((load, store, scope, rows, cols)); at += 4

    walk(kernel.body[2:], ())
    if sum(prod(n for _, n in scope) * rows * cols for _, _, scope, rows, cols in sites) != prod(sizes):
        raise BitSliceError('slice write cardinality differs from complete destination')

    def expr(e, env):
        if isinstance(e, Const): return z3.IntVal(e.value)
        if isinstance(e, Var): return env[e.name]
        if isinstance(e, Sum): return sum(expr(v, env) for v in e.terms)
        if isinstance(e, Scaled): return e.factor * expr(e.expr, env)
        raise BitSliceError('unproved slice address expression')

    def event(site, label):
        load, store, scope, rows, cols = site
        env = {var: z3.Int(label + '_' + var) for var, _ in scope}
        rr, cc = z3.Int(label + '_r'), z3.Int(label + '_c')
        bounds = [b for var, n in scope for b in (env[var] >= 0, env[var] < n)]
        bounds += [rr >= 0, rr < rows, cc >= 0, cc < cols]
        src = expr(load.arg('offset'), env) + rr * shape[-1] + cc
        dst = expr(store.arg('offset'), env) + rr * sizes[-1] + cc
        return bounds, src, dst, tuple(env[v] for v, _ in scope) + (rr, cc)

    queries = 0
    source_strides, output_strides = _strides(shape), _strides(sizes)
    for i, site in enumerate(sites):
        bounds, src, dst, identity = event(site, f'a{i}')
        expected_src = sum(((dst / stride) % size + offset) * source_stride
                           for stride, size, offset, source_stride in zip(output_strides, sizes, offsets, source_strides))
        solver = z3.Solver(); solver.set(timeout=5000)
        solver.add(*bounds, z3.Or(dst < 0, dst >= prod(sizes), src < 0, src >= prod(shape), src != expected_src))
        queries += 1
        if solver.check() != z3.unsat:
            raise BitSliceError('slice source bounds or exact source coordinate map not proved')
        for j, other in enumerate(sites[:i + 1]):
            other_bounds, _, other_dst, other_identity = event(other, f'b{j}')
            distinct = z3.Or(*(a != b for a, b in zip(identity, other_identity))) if i == j else z3.BoolVal(True)
            solver = z3.Solver(); solver.set(timeout=5000)
            solver.add(*(bounds + other_bounds), dst == other_dst, distinct)
            queries += 1
            if solver.check() != z3.unsat:
                raise BitSliceError('slice destination uniqueness not proved')
    return {'schema': 'submission.raw_bit_slice.v1', 'kernel_sha256': kernel.digest(),
            'source_shape': list(shape), 'offsets': list(offsets), 'sizes': list(sizes),
            'source_strides': list(source_strides), 'destination_strides': list(output_strides),
            'actual_dma_sites': len(sites), 'solver_queries': queries,
            'coverage': 'actual DMA writes are injective, in dense destination bounds, with exact output cardinality',
            'source_map': 'sum((floor(dst/output_stride[i])%size[i]+offset[i])*source_stride[i])',
            'transport': 'raw i32 preload and full-width readout preserve every f32 IEEE payload bit',
            'ownership': 'separate source and destination; full drain between DMA stages and staging reuse',
            'requested_load_bytes': prod(sizes) * 4, 'requested_store_bytes': prod(sizes) * 4}


def lower_slice(ctx, operands, attributes):
    if (set(operands) != {'src', 'dst'} or set(attributes) != {'offsets', 'sizes', 'strides', 'output_dtype', 'storage_dtype', 'semantic'}
            or attributes.get('output_dtype') != 'f32' or attributes.get('storage_dtype') != 'i32'
            or attributes.get('semantic') != 'bit_preserving_slice'):
        raise BitSliceError('slice requires exact source raw-bit contract without numeric attributes')
    src, dst = operands['src'], operands['dst']
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    if src == dst or src in aliases or dst in aliases:
        raise BitSliceError('slice requires separate materialized destination, without alias views')
    specs = [ctx.spec(src), ctx.spec(dst)]
    if any(s['dtype'] != 'f32' or s.get('strides') or s.get('offset', 0) for s in specs):
        raise BitSliceError('slice requires ordinary dense f32 storage boundaries')
    if attributes['strides'] != [1] * len(specs[0]['shape']):
        raise BitSliceError('slice source strides must be exactly one')
    plan = plan_slice(specs[0]['shape'], attributes['offsets'], attributes['sizes'])
    if list(plan.sizes) != specs[1]['shape']:
        raise BitSliceError('slice destination shape differs from exact source sizes')
    from xdsl.dialects.builtin import IntegerType
    bridge = FnBuilderIndexAdapter(ctx.fb)

    def emit(name, arguments):
        if name == 'slice_drain': ctx.fence(); return
        if name == 'slice_config':
            ctx.emit(*isa.config_ld(plan.shape[-1] * 4, shrink=False))
            ctx.emit(*isa.config_st(plan.sizes[-1] * 4)); return
        a = {key: bridge.value(value) for key, value in arguments.items()}
        tensor = src if name == 'slice_mvin' else dst
        ptr = ctx.fb.gep(ctx.buffer(tensor), a['offset'], IntegerType(32))
        packed = isa.pack_matrix(a['addr'], a['rows'], a['cols'], fb=ctx.fb)
        ctx.emit(isa.FUNCT['mvin' if name == 'slice_mvin' else 'mvout'], ptr, packed)

    emitted = emit_index_program(plan.kernel, bridge, emit)
    return {'proof': plan.proof, 'emission': emitted, 'kernel_sha256': plan.kernel.digest(),
            'storage_view': {'logical_dtype': 'f32', 'transfer_dtype': 'i32', 'element_bytes': 4,
                             'mapping': 'original ABI pointers and exact checked four-byte source/destination offsets'}}
