"""Independent original dense rows share one physical primitive M tile.

Original pointers supply each LHS slice directly to disjoint scratchpad rows;
original outputs receive their exact accumulator row slices directly. There
is no temporary DRAM packing and no host tensor arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import isa
from .dense_readout import store_tile


def partition_groups(ctx, groups):
    """Bound consecutive source-independent members by primitive row capacity."""
    result = {}
    for members in groups.values():
        pending=[];rows=0
        def finish():
            if len(pending)>=2:result[pending[0]['command_index']]=tuple(pending)
        for member in members:
            shape=ctx.spec(member['lhs']).get('shape',())
            m=shape[-2] if len(shape)==2 else isa.DIM+1
            if type(m)is not int or m<=0 or m>isa.DIM:
                finish();pending=[];rows=0
                continue
            if rows+m>isa.DIM:
                finish();pending=[];rows=0
            pending.append(member);rows+=m
        finish()
    return result


def _prove_projection(schedule,total):
    """Prove whole-M LHS/store shape for all actual counted-loop values."""
    import z3
    from merlin.sched.ir import Const,Var,Sum,Scaled,Select,Loop
    counts={'lhs':0,'store':0,'queries':0}
    def expr(e,env):
        if type(e)is Const:return z3.IntVal(e.value)
        if type(e)is Var:return env[e.name]
        if type(e)is Sum:return z3.Sum([expr(t,env)for t in e.terms])
        if type(e)is Scaled:return expr(e.expr,env)*e.factor
        if type(e)is Select:return z3.If(env[e.var]==e.equals,expr(e.then,env),expr(e.other,env))
        raise ValueError('unsupported projected row expression')
    def walk(items,scopes=()):
        for node in items:
            if type(node)is Loop:
                if node.iter_args or node.yields or node.results:raise ValueError('row projection rejects carried loops')
                walk(node.body,(*scopes,(node.var,node.extent)))
            elif node.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
                a=dict(node.args);env={v:z3.Int(v)for v,_ in scopes}
                solver=z3.Solver();solver.set(timeout=2000)
                solver.add(*(z3.And(env[v]>=0,env[v]<extent)for v,extent in scopes))
                solver.add(z3.Or(expr(a['dram_row'],env)!=0,expr(a['rows'],env)!=total))
                counts['queries']+=1
                if solver.check()!=z3.unsat:raise ValueError('schedule splits or offsets projected M rows')
                counts['lhs'if node.instr=='gemmini_mvin_lhs'else'store']+=1
    schedule.revalidate();walk(schedule.kernel.body)
    if not counts['lhs'] or not counts['store']:raise ValueError('row projection did not find original loads and stores')
    return counts


@dataclass
class RowProjection:
    members: tuple
    total: int
    proof: dict | None = None

    def validate(self,schedule,m,k,n):
        try:
            if m!=self.total or not 0<m<=isa.DIM:return False
            self.proof=_prove_projection(schedule,m)
            return True
        except (ValueError,AttributeError,KeyError):return False

    def load(self,ctx,a,k):
        fb=ctx.fb
        for lhs,dst,prefix,rows in self.members:
            local=fb.add_i(a['addr'],fb.const(prefix))
            ctx.emit(*isa.mvin(ctx.ptr(lhs,a['dram_col']),local,rows,a['cols'],fb=fb))

    def store(self,ctx,a,n,attrs):
        fb=ctx.fb
        for lhs,dst,prefix,rows in self.members:
            local=fb.add_i(a['acc'],fb.const(prefix))
            store_tile(ctx,dst,local,rows,a['cols'],n,0,a['dram_col'],attrs)

    def record(self):
        return dict(mechanism='independent_rhs_products_in_one_primitive_m_tile',
            rows=self.total,members=[dict(lhs=a,dst=b,prefix=p,rows=r)for a,b,p,r in self.members],
            proof=self.proof,movement='original pointers directly to/from private local row slices',
            completion='original checked dense producer and per-member readout drains retained')


def lower_row_batch(ctx,members):
    """Return False before mutation for any unsupported source or schedule."""
    if len(members)<2:return False
    try:
        rhs=members[0]['rhs'];b=ctx.spec(rhs)
        if b['dtype']!='i8'or len(b['shape'])!=2:return False
        k,n=b['shape'];total=0;mapping=[];reads={rhs};writes=set()
        attrs_out=None
        if ctx.cb.get('params',{}).get('tensor_aliases'):return False
        for member in members:
            lhs,dst=member['lhs'],member['dst'];a,c=ctx.spec(lhs),ctx.spec(dst)
            attrs=dict(member.get('attributes',{}))
            if (member['rhs']!=rhs or a['dtype']!='i8'or c['dtype']!='i32'
                    or len(a['shape'])!=2 or a['shape'][1]!=k
                    or tuple(c['shape'])!=(a['shape'][0],n)
                    or attrs.get('output_dtype','i32')!='i32'
                    or attrs.get('bias') or attrs.get('accumulator_init')
                    or attrs.get('acc_scale_by_column')is not None
                    or float(attrs.get('acc_scale',1))!=1 or attrs.get('relu')
                    or attrs.get('requant_shift')is not None
                    or any(stage not in ('acc_scale',)for stage in attrs.get('epilogue',()))):return False
            if any(key in spec for spec in (a,b,c)for key in ('strides','physical_strides','physical_views','offset')):return False
            m=a['shape'][0]
            if type(m)is not int or m<=0 or total+m>isa.DIM or dst in writes:return False
            mapping.append((lhs,dst,total,m));total+=m;reads.add(lhs);writes.add(dst)
            attrs_out=attrs if attrs_out is None else attrs_out
        if reads&writes:return False
        # SSA buffer identity catches undeclared source aliases as well.
        buffers=getattr(ctx,'buffers',{})
        # Admission must not allocate a missing buffer through ctx.buffer.
        # Internal outputs may be allocated later by the ordinary owner.
        outputs=[buffers[name]for name in writes if name in buffers]
        inputs=[buffers[name]for name in reads if name in buffers]
        if any(p is q for p in outputs for q in inputs):return False
        if any(p is q for i,p in enumerate(outputs)for q in outputs[i+1:]):return False
        projection=RowProjection(tuple(mapping),total)
        from .dense import lower_dense
        receipt=lower_dense(ctx,mapping[0][0],rhs,mapping[0][1],m=total,k=k,n=n,
            attributes=dict(attrs_out,output_dtype='i32'),row_projection=projection)
        if receipt is False:return False
        return dict(projection.record(),dense_schedule=receipt,original_product_count=len(members),
            source_commit_indices=[member['command_index']for member in members])
    except (ValueError,KeyError,TypeError):
        # All source/capacity refusals precede lower_dense's first mutation.
        # Emission errors propagate: never fall back after a partial program.
        if 'projection' in locals():raise
        return False
