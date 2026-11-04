"""Exact static integer concatenation through private accumulator DMA.

Each input keeps its own ABI pointer.  Checked counted descriptors partition
the original output along the source axis; no physical adjacency is assumed.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from math import prod

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var, as_expr
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa


class IntegerConcatError(ValueError):
    pass


def _call(name, **args):
    return Call(name, tuple((k, as_expr(v)) for k, v in args.items()))


def _geometry(shapes, axis, dtype):
    shapes = tuple(tuple(s) for s in shapes)
    if not 1 <= len(shapes) <= 64 or not 1 <= len(shapes[0]) <= 8:
        raise IntegerConcatError('concat requires 1-64 ranked inputs of rank1-8')
    rank = len(shapes[0])
    if (type(axis) is not int or not -rank <= axis < rank or dtype not in ('i8', 'i32')
            or any(len(s) != rank or any(type(n) is not int or n <= 0 for n in s) for s in shapes)):
        raise IntegerConcatError('concat requires static positive integer geometry')
    axis %= rank
    if any(any(s[i] != shapes[0][i] for i in range(rank) if i != axis) for s in shapes):
        raise IntegerConcatError('concat non-axis dimensions differ')
    out = list(shapes[0]); out[axis] = sum(s[axis] for s in shapes)
    width = 1 if dtype == 'i8' else 4
    if prod(out)*width >= 1 << 63 or prod(out[axis:])*width >= 1 << 32:
        raise IntegerConcatError('concat footprint or DMA pitch exceeds selected fields')
    return shapes, axis, tuple(out)


@dataclass(frozen=True)
class ConcatPlan:
    shapes: tuple
    axis: int
    dtype: str
    output_shape: tuple
    kernel: Kernel
    proof: dict

    def revalidate(self):
        return check_concat_projection(self.kernel, self.shapes, self.axis, self.dtype)


def plan_concat(shapes, axis, dtype):
    return _plan_concat(tuple(tuple(s) for s in shapes), axis, dtype)


@lru_cache(maxsize=128)
def _plan_concat(shapes, axis, dtype):
    shapes, axis, out = _geometry(shapes, axis, dtype)
    width = 1 if dtype == 'i8' else 4
    outer, inner, output_pitch = prod(out[:axis]), prod(out[axis+1:]), prod(out[axis:])
    body = []; start = 0
    for member, shape in enumerate(shapes):
        run = shape[axis]*inner
        def tile(row, rows, column, cols):
            return (_call(f'concat_mvin_{member}', offset=row*run+column,
                          addr=isa.acc_addr(0), rows=rows, cols=cols),
                    _call('concat_drain'),
                    _call('concat_mvout', offset=row*output_pitch+start+column,
                          addr=isa.acc_addr(0, full=dtype == 'i32'), rows=rows, cols=cols),
                    _call('concat_drain'))
        def columns(row, rows):
            full, tail = divmod(run, isa.DIM); result = []
            if full: result.append(Loop('column', full, tile(row, rows, Var('column')*isa.DIM, isa.DIM)))
            if tail: result.extend(tile(row, rows, Const(full*isa.DIM), tail))
            return tuple(result)
        body.extend((_call('concat_drain'), _call('concat_config_load', stride=run*width),
                     _call('concat_config_store', stride=output_pitch*width)))
        full, tail = divmod(outer, min(isa.DIM, isa.ACC_ROWS))
        if full: body.append(Loop('row', full, columns(Var('row')*min(isa.DIM, isa.ACC_ROWS), min(isa.DIM, isa.ACC_ROWS))))
        if tail: body.extend(columns(Const(full*min(isa.DIM, isa.ACC_ROWS)), tail))
        start += run
    args = tuple(TensorArg(f'src{i}', s, dtype, 'read') for i, s in enumerate(shapes))+(TensorArg('dst', out, dtype, 'write'),)
    kernel = Kernel('integer_concat', args, tuple(body))
    proof = check_concat_projection(kernel, shapes, axis, dtype)
    return ConcatPlan(shapes, axis, dtype, out, kernel, proof)


def check_concat_projection(kernel, shapes, axis, dtype):
    """Check emitted descriptors, exact source mapping and result partition."""
    import z3
    shapes, axis, out = _geometry(shapes, axis, dtype)
    width = 1 if dtype == 'i8' else 4
    outer, inner, pitch = prod(out[:axis]), prod(out[axis+1:]), prod(out[axis:])
    args = tuple(TensorArg(f'src{i}', s, dtype, 'read') for i, s in enumerate(shapes))+(TensorArg('dst', out, dtype, 'write'),)
    if kernel.args != args: raise IntegerConcatError('concat boundary identities changed')
    def integer(call, key):
        value = call.arg(key)
        if not isinstance(value, Const): raise IntegerConcatError('concat static descriptor became dynamic')
        return value.value
    def expr(e, env):
        if isinstance(e, Const): return z3.IntVal(e.value)
        if isinstance(e, Var): return env[e.name]
        if isinstance(e, Sum): return sum(expr(t, env) for t in e.terms)
        if isinstance(e, Scaled): return e.factor*expr(e.expr, env)
        raise IntegerConcatError('unsupported concat index expression')
    sites = []; member = -1; at = 0
    def walk(body, scope, owner):
        i = 0
        while i < len(body):
            op = body[i]
            if isinstance(op, Loop):
                if op.var in dict(scope) or op.extent <= 0 or op.iter_args or op.yields or op.results:
                    raise IntegerConcatError('concat counted loop changed')
                walk(op.body, scope+((op.var, op.extent),), owner); i += 1; continue
            group = body[i:i+4]
            if len(group) != 4 or any(isinstance(x, Loop) for x in group) or [x.instr for x in group] != [f'concat_mvin_{owner}', 'concat_drain', 'concat_mvout', 'concat_drain']:
                raise IntegerConcatError('concat local ownership/completion changed')
            load, _, store, _ = group
            rows, cols = integer(load, 'rows'), integer(load, 'cols')
            if (not 0 < rows <= min(isa.DIM, isa.ACC_ROWS) or not 0 < cols <= isa.DIM
                    or integer(store, 'rows') != rows or integer(store, 'cols') != cols
                    or integer(load, 'addr') != isa.acc_addr(0)
                    or integer(store, 'addr') != isa.acc_addr(0, full=dtype == 'i32')
                    or set(dict(load.args)) != {'offset','addr','rows','cols'}
                    or set(dict(store.args)) != {'offset','addr','rows','cols'}):
                raise IntegerConcatError('concat DMA footprint changed')
            sites.append((owner,load,store,scope,rows,cols)); i += 4
    while at < len(kernel.body):
        member += 1
        prologue = kernel.body[at:at+3]
        if member >= len(shapes) or any(isinstance(x, Loop) for x in prologue) or [x.instr for x in prologue] != ['concat_drain','concat_config_load','concat_config_store']:
            raise IntegerConcatError('concat configuration transition changed')
        if integer(prologue[1],'stride') != prod(shapes[member][axis:])*width or integer(prologue[2],'stride') != pitch*width:
            raise IntegerConcatError('concat DMA pitches changed')
        end = at+3
        while end < len(kernel.body) and (isinstance(kernel.body[end],Loop) or kernel.body[end].instr != 'concat_config_load'):
            # A prologue starts with drain immediately before the load config.
            if (not isinstance(kernel.body[end],Loop) and kernel.body[end].instr == 'concat_drain'
                    and end+1 < len(kernel.body) and not isinstance(kernel.body[end+1],Loop)
                    and kernel.body[end+1].instr == 'concat_config_load'): break
            end += 1
        walk(kernel.body[at+3:end], (), member); at = end
    if member+1 != len(shapes): raise IntegerConcatError('concat member count changed')
    queries = 0
    def check(bounds, bad):
        nonlocal queries
        solver = z3.Solver(); solver.set(timeout=5000); solver.add(*bounds,bad); queries += 1
        status = solver.check()
        if status != z3.unsat: raise IntegerConcatError('concat projection not proved: '+str(status))
    def event(site, label):
        owner, load, store, scope, rows, cols = site
        env = {v:z3.Int(label+'_'+v) for v,_ in scope}; r,c = z3.Int(label+'_r'),z3.Int(label+'_c')
        bounds = [b for v,n in scope for b in (env[v]>=0,env[v]<n)]+[r>=0,r<rows,c>=0,c<cols]
        src = expr(load.arg('offset'),env)+r*prod(shapes[owner][axis:])+c
        dst = expr(store.arg('offset'),env)+r*pitch+c
        return bounds,src,dst,tuple(env[v] for v,_ in scope)+(r,c)
    for owner,shape in enumerate(shapes):
        if sum(prod(n for _,n in s[3])*s[4]*s[5] for s in sites if s[0]==owner) != prod(shape):
            raise IntegerConcatError('concat member cardinality differs')
    for i,site in enumerate(sites):
        owner=site[0]; run=prod(shapes[owner][axis:]); start=sum(prod(s[axis:]) for s in shapes[:owner])
        bounds,src,dst,ids=event(site,'a'+str(i))
        check(bounds,z3.Or(src<0,src>=prod(shapes[owner]),dst<0,dst>=prod(out),dst != (src/run)*pitch+start+src%run))
        for j,other in enumerate(sites[:i+1]):
            rhs,_,outaddr,otherids=event(other,'b'+str(j))
            distinct=z3.Or(*(a!=b for a,b in zip(ids,otherids,strict=True))) if i==j else z3.BoolVal(True)
            check(bounds+rhs,z3.And(dst==outaddr,distinct))
    return {'schema':'submission.integer_concat_projection.v1','kernel_sha256':kernel.digest(),
            'source_shapes':[list(s) for s in shapes],'axis':axis,'output_shape':list(out),
            'output_elements':prod(out),'solver_queries':queries,'actual_dma_sites':len(sites),
            'coverage':'original member coordinates, in-bounds source and destination, disjoint complete output partition',
            'numeric':'signed i8 identity at scale1' if dtype=='i8' else 'full-width raw i32 transfer',
            'completion':'MVIN, full drain, MVOUT, full drain before private accumulator reuse',
            'alias_contract':'each source retains its actual pointer; output distinct from all sources'}


def integer_concat_candidate(op):
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import _type
    from .source_groups import _standalone_member, _standalone_capability
    if (op_name(op) != 'tensor.concat' or not op.operands or len(op.results)!=1 or op.regions
            or getattr(getattr(op,'parent',None),'parent_op',lambda:None)() is None
            or op.parent.parent_op().name != 'func.func'): return None
    specs=[_type(v) for v in (*op.operands,*op.results)]
    if any(not s['dense'] or s['dtype']!=specs[0]['dtype'] for s in specs): return None
    try:
        dim=op.properties.get('dim',op.attributes.get('dim'))
        axis=int(dim.value.data)
        plan=plan_concat([s['shape'] for s in specs[:-1]],axis,specs[0]['dtype'])
        if list(plan.output_shape)!=specs[-1]['shape']: return None
        attrs={'axis':plan.axis,'output_dtype':plan.dtype}
        proof={'form':'static_integer_concat','address_projection':plan.proof,
               'source_ids':{k:str(v) for k,v in op.attributes.items() if k.startswith(('prov.','merlin.source'))}}
        candidate=_standalone_member(op,[op],'INTEGER_CONCAT',{f'src{i}':v for i,v in enumerate(op.operands)},attrs,proof)
        if candidate is not None: _standalone_capability(candidate)
        return candidate
    except (ValueError,TypeError,AttributeError,OverflowError): return None


def lower_concat(ctx, operands, attributes):
    keys=tuple(f'src{i}' for i in range(len(operands)-1))
    if set(operands)!=set(keys)|{'dst'} or set(attributes)!={'axis','output_dtype'}:
        raise IntegerConcatError('concat operands or arithmetic attributes changed')
    sources=[operands[k] for k in keys]; dst=operands['dst']
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    alias_sources={v.get('source') for v in aliases.values() if isinstance(v,dict)}
    if dst in sources or set(sources+[dst])&(set(aliases)|alias_sources):
        raise IntegerConcatError('concat output must be distinct materialized storage')
    specs=[ctx.spec(s) for s in sources]; output=ctx.spec(dst)
    dtype=attributes['output_dtype']
    if any(s['dtype']!=dtype for s in specs+[output]): raise IntegerConcatError('concat dtype changed')
    plan=plan_concat([s['shape'] for s in specs],attributes['axis'],dtype)
    if list(plan.output_shape)!=output['shape']: raise IntegerConcatError('concat output shape changed')
    plan.revalidate()
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='concat_drain': ctx.fence(); return
        if name=='concat_config_load': ctx.emit(*isa.config_ld(arguments['stride'],shrink=dtype=='i8')); return
        if name=='concat_config_store': ctx.emit(*isa.config_st(arguments['stride'],scale=1.0,activation=0)); return
        fields={k:bridge.value(v) for k,v in arguments.items()}
        packed=isa.pack_matrix(fields['addr'],fields['rows'],fields['cols'],fb=ctx.fb)
        if name.startswith('concat_mvin_'):
            source=sources[int(name[len('concat_mvin_'):])]
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(source,fields['offset']),packed)
        elif name=='concat_mvout': ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,fields['offset']),packed)
        else: raise IntegerConcatError('unknown checked concat command')
    emitted=emit_index_program(plan.kernel,bridge,emit)
    receipt={'proof':plan.proof,'emission':emitted,'kernel_sha256':plan.kernel.digest()}
    ctx.metadata.setdefault('integer_concat',[]).append(receipt)
    return receipt
