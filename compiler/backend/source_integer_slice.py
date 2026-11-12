"""Source-bound static integer slices through checked accumulator DMA.

Only rank-preserving dense slices are admitted. The original source/result ABI,
axis order, offsets, sizes and strides remain explicit. Counted host code computes
addresses and issues target commands; it never copies or computes tensor words.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from functools import lru_cache

from merlin.compiler.kernel_ir.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class IntegerSliceError(ValueError):
    pass


def _strides(shape):
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def _call(name, **args):
    return Call(name, tuple((key, as_expr(value)) for key, value in args.items()))


@dataclass(frozen=True)
class SlicePlan:
    shape: tuple
    offsets: tuple
    sizes: tuple
    strides: tuple
    dtype: str
    kernel: Kernel
    proof: dict

    def revalidate(self):
        return check_slice_projection(self.kernel, self.shape, self.offsets,
                                      self.sizes, self.strides, self.dtype)


def _geometry(shape, offsets, sizes, strides, dtype):
    shape, offsets, sizes, strides = map(tuple, (shape, offsets, sizes, strides))
    rank = len(shape)
    if (not 1 <= rank <= 4 or any(len(a) != rank for a in (offsets, sizes, strides))
            or dtype not in ('i8', 'i32')
            or any(type(n) is not int or n <= 0 for n in (*shape, *sizes, *strides))
            or any(type(n) is not int or n < 0 for n in offsets)
            or any(o + (n-1)*s >= d for d, o, n, s in zip(shape, offsets, sizes, strides, strict=True))):
        raise IntegerSliceError('slice requires in-bounds rank1-4 static positive-stride integer geometry')
    width = 1 if dtype == 'i8' else 4
    if max(prod(shape)*width, prod(sizes)*width) >= 1 << 63:
        raise IntegerSliceError('slice byte footprint exceeds signed i64')
    return shape, offsets, sizes, strides


def plan_slice(shape, offsets, sizes, strides, dtype):
    return _plan_slice(tuple(shape), tuple(offsets), tuple(sizes), tuple(strides), dtype)


@lru_cache(maxsize=128)
def _plan_slice(shape, offsets, sizes, strides, dtype):
    shape, offsets, sizes, strides = _geometry(shape, offsets, sizes, strides, dtype)
    width = 1 if dtype == 'i8' else 4
    ss, ds = _strides(shape), _strides(sizes)
    # Grow only the output suffix with unit source displacement. Any remaining
    # axis is transferred as strided rows, preserving gaps in the original input.
    run, start = 1, len(shape)
    for axis in range(len(shape)-1, -1, -1):
        if sizes[axis] != 1 and strides[axis]*ss[axis] != run:
            break
        run *= sizes[axis]
        start = axis
    row_axis = start-1
    row_extent = sizes[row_axis] if row_axis >= 0 else 1
    source_pitch = strides[row_axis]*ss[row_axis] if row_axis >= 0 else run
    if max(source_pitch, run)*width >= 1 << 32:
        raise IntegerSliceError('slice DMA pitch exceeds selected configuration field')
    prefix = range(max(0, row_axis))
    src_base = sum((Var(f'p{i}')*strides[i]*ss[i] for i in prefix),
                   Const(sum(o*s for o, s in zip(offsets, ss, strict=True))))
    dst_base = sum((Var(f'p{i}')*ds[i] for i in prefix), Const(0))
    def tile(row, rows, column, columns):
        return (_call('slice_mvin', offset=src_base+row*source_pitch+column,
                      addr=isa.acc_addr(0), rows=rows, cols=columns),
                _call('slice_drain'),
                _call('slice_mvout', offset=dst_base+row*run+column,
                      addr=isa.acc_addr(0, full=dtype == 'i32'), rows=rows, cols=columns),
                _call('slice_drain'))
    def columns(row, rows):
        full, tail = divmod(run, isa.DIM)
        body = []
        if full:
            body.append(Loop('column', full, tile(row, rows, Var('column')*isa.DIM, isa.DIM)))
        if tail:
            body.extend(tile(row, rows, Const(full*isa.DIM), tail))
        return tuple(body)
    full, tail = divmod(row_extent, min(isa.DIM, isa.ACC_ROWS))
    body = []
    if full:
        body.append(Loop('row', full, columns(Var('row')*min(isa.DIM, isa.ACC_ROWS), min(isa.DIM, isa.ACC_ROWS))))
    if tail:
        body.extend(columns(Const(full*min(isa.DIM, isa.ACC_ROWS)), tail))
    body = tuple(body)
    for i in reversed(tuple(prefix)):
        body = (Loop(f'p{i}', sizes[i], body),)
    configs = (_call('slice_drain'), _call('slice_config_load', stride=source_pitch*width),
               _call('slice_config_store', stride=run*width))
    kernel = Kernel('integer_slice', (TensorArg('src', shape, dtype, 'read'),
                    TensorArg('dst', sizes, dtype, 'write')), configs+body)
    proof = check_slice_projection(kernel, shape, offsets, sizes, strides, dtype)
    return SlicePlan(shape, offsets, sizes, strides, dtype, kernel, proof)


def check_slice_projection(kernel, shape, offsets, sizes, strides, dtype):
    """Prove the actual DMA descriptors implement every original slice word."""
    import z3
    shape, offsets, sizes, strides = _geometry(shape, offsets, sizes, strides, dtype)
    width = 1 if dtype == 'i8' else 4
    ss, ds, count = _strides(shape), _strides(sizes), prod(sizes)
    if kernel.args != (TensorArg('src', shape, dtype, 'read'), TensorArg('dst', sizes, dtype, 'write')):
        raise IntegerSliceError('slice boundary tensor identity changed')
    first = kernel.body[:3]
    if [o.instr for o in first] != ['slice_drain', 'slice_config_load', 'slice_config_store']:
        raise IntegerSliceError('slice configuration/completion prologue changed')
    def integer(call, key):
        value = call.arg(key)
        if not isinstance(value, Const):
            raise IntegerSliceError('slice static descriptor became dynamic')
        return value.value
    sp, dp = integer(first[1], 'stride'), integer(first[2], 'stride')
    if min(sp, dp) <= 0 or max(sp, dp) >= 1 << 32 or sp % width or dp % width:
        raise IntegerSliceError('slice DMA pitch is invalid')
    sites = []
    def walk(body, scope):
        at = 0
        while at < len(body):
            op = body[at]
            if isinstance(op, Loop):
                if op.var in dict(scope) or op.extent <= 0 or op.iter_args or op.yields or op.results:
                    raise IntegerSliceError('slice counted loop changed')
                walk(op.body, scope+((op.var, op.extent),)); at += 1; continue
            group = body[at:at+4]
            if len(group) != 4 or [o.instr for o in group] != ['slice_mvin', 'slice_drain', 'slice_mvout', 'slice_drain']:
                raise IntegerSliceError('slice local ownership/completion changed')
            load, _, store, _ = group
            rows, cols = integer(load, 'rows'), integer(load, 'cols')
            if (not 0 < rows <= min(isa.DIM, isa.ACC_ROWS) or not 0 < cols <= isa.DIM
                    or integer(store, 'rows') != rows or integer(store, 'cols') != cols
                    or integer(load, 'addr') != isa.acc_addr(0)
                    or integer(store, 'addr') != isa.acc_addr(0, full=dtype == 'i32')
                    or set(dict(load.args)) != {'offset', 'rows', 'cols', 'addr'}
                    or set(dict(store.args)) != {'offset', 'rows', 'cols', 'addr'}):
                raise IntegerSliceError('slice descriptor or local footprint changed')
            sites.append((load, store, scope, rows, cols)); at += 4
    walk(kernel.body[3:], ())
    if not sites or sum(prod(n for _, n in scope)*r*c for _, _, scope, r, c in sites) != count:
        raise IntegerSliceError('slice DMA cardinality differs from original result')
    queries = 0
    def check(bounds, bad):
        nonlocal queries
        solver = z3.Solver(); solver.set(timeout=5000); solver.add(*bounds, bad); queries += 1
        status = solver.check()
        if status != z3.unsat:
            raise IntegerSliceError('slice address projection not proved: '+str(status))
    def expr(e, env):
        if isinstance(e, Const): return z3.IntVal(e.value)
        if isinstance(e, Var): return env[e.name]
        if isinstance(e, Sum): return sum(expr(x, env) for x in e.terms)
        if isinstance(e, Scaled): return e.factor*expr(e.expr, env)
        raise IntegerSliceError('slice address expression is unsupported')
    def event(site, label):
        load, store, scope, rows, cols = site
        env = {v: z3.Int(label+'_'+v) for v, _ in scope}
        row, col = z3.Int(label+'_row'), z3.Int(label+'_col')
        bounds = [b for v, n in scope for b in (env[v] >= 0, env[v] < n)]+[row >= 0, row < rows, col >= 0, col < cols]
        src = expr(load.arg('offset'), env)+row*(sp//width)+col
        dst = expr(store.arg('offset'), env)+row*(dp//width)+col
        return bounds, src, dst, tuple(env[v] for v, _ in scope)+(row, col)
    for i, site in enumerate(sites):
        bounds, src, dst, _ = event(site, 'a'+str(i))
        expected = sum((o+((dst/d)%n)*step)*s for o, d, n, step, s in zip(offsets, ds, sizes, strides, ss, strict=True))
        check(bounds, z3.Or(src < 0, src >= prod(shape), dst < 0, dst >= count, src != expected))
        for j, other in enumerate(sites[:i+1]):
            rhs, _, out, ids = event(other, 'b'+str(j))
            lhs, _, _, myids = event(site, 'a'+str(i))
            distinct = z3.Or(*(a != b for a, b in zip(myids, ids, strict=True))) if i == j else z3.BoolVal(True)
            check(lhs+rhs, z3.And(dst == out, distinct))
    return {'schema':'submission.integer_slice_projection.v1', 'kernel_sha256':kernel.digest(),
            'source_shape':list(shape), 'source_offsets':list(offsets), 'source_sizes':list(sizes),
            'source_strides':list(strides), 'output_elements':count, 'solver_queries':queries,
            'actual_dma_sites':len(sites), 'requested_load_bytes':count*width, 'requested_store_bytes':count*width,
            'source_row_stride_bytes':sp, 'destination_row_stride_bytes':dp,
            'coverage':'in-bounds original slice mapping, injective result addresses and exact result cardinality',
            'numeric':'signed i8 identity readout over all 256 input words' if dtype == 'i8' else 'full-width raw i32 load/readout',
            'completion':'MVIN, full drain, MVOUT, full drain before private row reuse',
            'alias_contract':'distinct materialized source/result ABI buffers'}


def integer_slice_candidate(op):
    """Recognize actual static tensor.extract_slice and return a typed plan."""
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import _type
    from .source_groups import _standalone_member, _standalone_capability
    if (op_name(op) != 'tensor.extract_slice' or len(op.operands) != 1
            or len(op.results) != 1 or op.regions
            or getattr(getattr(op, 'parent', None), 'parent_op', lambda: None)() is None
            or op.parent.parent_op().name != 'func.func'):
        return None
    a, b = _type(op.operands[0]), _type(op.results[0])
    if not a['dense'] or not b['dense'] or a['dtype'] != b['dtype'] or a['dtype'] not in ('i8', 'i32'):
        return None
    try:
        def values(key):
            attr = op.properties.get(key, op.attributes.get(key))
            return tuple(attr.get_values())
        offsets, sizes, strides = values('static_offsets'), values('static_sizes'), values('static_strides')
        plan = plan_slice(tuple(a['shape']), offsets, sizes, strides, a['dtype'])
        if list(sizes) != b['shape']:
            return None
        attrs = {'source_offsets':list(offsets), 'source_sizes':list(sizes),
                 'source_strides':list(strides), 'output_dtype':a['dtype']}
        proof = {'form':'static_integer_extract_slice', 'address_projection':plan.proof,
                 'source_ids':{k:str(v) for k,v in op.attributes.items() if k.startswith(('prov.', 'merlin.source'))}}
        candidate = _standalone_member(op, [op], 'STRIDED_SLICE', {'src':op.operands[0]}, attrs, proof)
        if candidate is not None:
            _standalone_capability(candidate)
        return candidate
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def lower_slice(ctx, operands, attributes):
    """Emit the same checked original-slice Kernel through target DMA calls."""
    src, dst = operands['src'], operands['dst']
    a, b = ctx.spec(src), ctx.spec(dst)
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    alias_sources = {v.get('source') for v in aliases.values() if isinstance(v, dict)}
    if src == dst or {src, dst} & (set(aliases) | alias_sources):
        raise IntegerSliceError('slice needs distinct materialized source/result buffers')
    allowed = {'source_offsets', 'source_sizes', 'source_strides', 'output_dtype'}
    if set(attributes) - allowed or a['dtype'] != b['dtype'] or attributes.get('output_dtype') != a['dtype']:
        raise IntegerSliceError('slice cannot change dtype or apply arithmetic/activation')
    plan = plan_slice(tuple(a['shape']), tuple(attributes['source_offsets']), tuple(attributes['source_sizes']),
                      tuple(attributes['source_strides']), a['dtype'])
    if list(plan.sizes) != b['shape']:
        raise IntegerSliceError('slice result shape differs from original source sizes')
    plan.revalidate()
    bridge = FnBuilderIndexAdapter(ctx.fb)
    def emit(name, arguments):
        if name == 'slice_drain': ctx.fence(); return
        if name == 'slice_config_load':
            ctx.emit(*isa.config_ld(arguments['stride'], shrink=plan.dtype == 'i8')); return
        if name == 'slice_config_store':
            ctx.emit(*isa.config_st(arguments['stride'], scale=1.0, activation=0)); return
        fields = {k:bridge.value(v) for k, v in arguments.items()}
        packed = isa.pack_matrix(fields['addr'], fields['rows'], fields['cols'], fb=ctx.fb)
        if name == 'slice_mvin': ctx.emit(isa.FUNCT['mvin'], ctx.ptr(src, fields['offset']), packed)
        elif name == 'slice_mvout': ctx.emit(isa.FUNCT['mvout'], ctx.ptr(dst, fields['offset']), packed)
        else: raise IntegerSliceError('unknown checked slice command')
    emitted = emit_index_program(plan.kernel, bridge, emit)
    receipt = {'proof':plan.proof, 'emission':emitted, 'kernel_sha256':plan.kernel.digest()}
    ctx.metadata.setdefault('integer_slice', []).append(receipt)
    return receipt
