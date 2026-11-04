"""Copy-free row concatenation of independent products with an immutable RHS.

A checked ordinary dense schedule works over a logical row axis. Only pointers
are projected back to the original disjoint matrices; no tensor is packed or
scattered and each source result keeps its original pointer ABI.
"""
from __future__ import annotations

from . import isa


def _check_rows(schedule, member_m, total_m, k, n, *, fragmented=False):
    """Prove actual DMA rectangles stay within one original matrix."""
    import z3
    from merlin.sched.ir import Const, Var, Sum, Scaled, Select, Loop
    queries = 0
    def expr(e, env):
        if isinstance(e, Const): return z3.IntVal(e.value)
        if isinstance(e, Var): return env[e.name]
        if isinstance(e, Sum): return z3.Sum([expr(t, env) for t in e.terms])
        if isinstance(e, Scaled): return e.factor*expr(e.expr, env)
        if isinstance(e, Select): return z3.If(env[e.var]==e.equals,expr(e.then,env),expr(e.other,env))
        raise ValueError('unsupported row-stack schedule index')
    def walk(body, scope=()):
        nonlocal queries
        for op in body:
            if isinstance(op, Loop):
                walk(op.body, (*scope,(op.var,op.extent)))
                continue
            if op.instr not in ('gemmini_mvin_lhs','gemmini_mvout','gemmini_dense_store_epoch_mvout'): continue
            env={v:z3.Int(v) for v,_ in scope}
            a={key:expr(value,env) for key,value in op.args}
            row, col, rows, cols=(a[x] for x in ('dram_row','dram_col','rows','cols'))
            width=k if op.instr=='gemmini_mvin_lhs' else n
            solver=z3.Solver();solver.set(timeout=2000)
            solver.add(*(z3.And(env[v]>=0,env[v]<extent) for v,extent in scope))
            solver.add(z3.Or(row<0,row+rows>total_m,rows<=0,rows>isa.DIM,
                z3.BoolVal(False) if fragmented else row%member_m+rows>member_m,
                col<0,cols<=0,col+cols>width))
            queries+=1
            if solver.check()!=z3.unsat: raise ValueError('row-stack DMA crosses source matrix boundary')
    schedule.revalidate()
    walk(schedule.kernel.body)
    return {'kernel_sha256':schedule.kernel.digest(),'symbolic_rectangle_queries':queries,
        'projection':'member=floor(row/M); local_row=row mod M; column unchanged',
        'coverage':'ordinary checked dense source projection; each original row appears exactly once',
        'movement':'no packing, replication, gather payload loads or scatter copies'}


class _RowView:
    def __init__(self,ctx,parts,m,k,n,dtype):
        self._ctx=ctx
        self._parts=parts
        self._m=m
        self._shapes={'_row_stack_lhs':(len(parts)*m,k), '_row_stack_dst':(len(parts)*m,n)}
        self._dtype=dtype
    def __getattr__(self,key): return getattr(self._ctx,key)
    @property
    def current_source_index(self): return self._ctx.current_source_index
    @current_source_index.setter
    def current_source_index(self,value): self._ctx.current_source_index=value
    def spec(self,name):
        if name not in self._shapes:return self._ctx.spec(name)
        return {'shape':list(self._shapes[name]),'dtype':'i8' if name=='_row_stack_lhs' else self._dtype}
    def ptr(self,name,elem_offset=0):
        if name not in self._shapes:return self._ctx.ptr(name,elem_offset)
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr,i64
        from xdsl.ir import Block
        from merlin.llvmlower.host_builder import PTR
        fb=self.fb
        offset=fb.const(elem_offset) if isinstance(elem_offset,int) else elem_offset
        span=self._m*self._shapes[name][1]
        member=fb.udiv_i(offset,fb.const(span))
        local=fb.urem_i(offset,fb.const(span))
        key='lhs' if name=='_row_stack_lhs' else 'dst'
        # Balanced ordinary control flow selects only a pointer. Payload bytes
        # remain wholly on the accelerator; no CPU tensor arithmetic occurs.
        merge=Block(arg_types=[PTR])
        def choose(lo,hi):
            if hi-lo==1:
                fb.add(llvm.BrOp(merge,self._ctx.ptr(self._parts[lo][key],local)))
                return
            mid=(lo+hi)//2
            left,right=Block(),Block();fb.region.add_block(left);fb.region.add_block(right)
            pred=fb.add(llvm.ICmpOp(member,fb.const(mid),IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
            fb.add(llvm.CondBrOp(pred,left,[],right,[]))
            fb.blk=left;choose(lo,mid)
            fb.blk=right;choose(mid,hi)
        choose(0,len(self._parts))
        fb.region.add_block(merge);fb.blk=merge
        return merge.args[0]


def lower_stacked_rhs_products(ctx,members):
    """Return a complete checked receipt, or None before any IR mutation.

    The caller supplies the same independence/lifetime proof used by the shared
    RHS epoch grouping. Uniform full primitive row boundaries permit direct
    projection of every input/output DMA into a caller-owned matrix.
    """
    from .dense import lower_dense
    from .dense_pipeline import retained_schedule
    from .dense_store_epochs import select_store_epochs
    members=tuple(members)
    if not 2<=len(members)<=32:return None
    try:
        rhs=members[0]['rhs']
        if any(row['rhs']!=rhs for row in members):return None
        attrs=dict(members[0].get('attributes',{}))
        # Source attribution is kept separately; all numeric/readout settings
        # must match exactly, including scale identities and epilogue order.
        def numeric(a):return {k:v for k,v in a.items() if k not in ('merlin.source_op_index','prov.region_id')}
        if any(numeric(row.get('attributes',{}))!=numeric(attrs) for row in members):return None
        if attrs.get('bias') or attrs.get('accumulator_init') or attrs.get('block_scale') is not None:return None
        if any(e not in ('acc_scale','relu') for e in attrs.get('epilogue',())):return None
        if attrs.get('requant_shift') is not None:return None
        outputs=[row['dst'] for row in members]
        inputs={row[key] for row in members for key in ('lhs','rhs')}
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        alias_sources={a.get('source') for a in aliases.values() if isinstance(a,dict)}
        if len(set(outputs))!=len(outputs) or set(outputs)&inputs or (inputs|set(outputs))&(set(aliases)|alias_sources):return None
        a,b,c=(ctx.spec(members[0][key]) for key in ('lhs','rhs','dst'))
        if a['dtype']!='i8' or b['dtype']!='i8' or c['dtype'] not in ('i8','i32'):return None
        if len(a['shape'])!=2 or len(b['shape'])!=2:return None
        m,k=a['shape'];bk,n=b['shape']
        if any(type(v) is not int or v<=0 for v in (m,k,bk,n)):return None
        if k!=bk or c['shape']!=[m,n]:return None
        fragmented=bool(m%isa.DIM)
        if fragmented:
            if m>=isa.DIM:return None
            from pathlib import Path
            from merlin.compile.tuning_knobs import knob
            mode=knob('RHS_RAGGED_STACK',default='off',values=('off','on','decode_rows'),
                    root=Path(__file__).resolve().parent.parent)
            if mode=='off' or mode=='decode_rows' and m!=1:return None
        if attrs.get('output_dtype',c['dtype'])!=c['dtype']:return None
        for row in members:
            # Role/provenance metadata is irrelevant to physical geometry.
            aa,cc=ctx.spec(row['lhs']),ctx.spec(row['dst'])
            if (aa['shape'],aa['dtype'],cc['shape'],cc['dtype'])!=(a['shape'],a['dtype'],c['shape'],c['dtype']):return None
            if any(key in row and row[key]!=value for key,value in (('m',m),('k',k),('n',n))):return None
        scales=attrs.get('acc_scale_by_column')
        if scales is not None and (len(scales)!=n or len(set(map(float,scales)))>1):return None
        total_m=len(members)*m
        if max(total_m*k,total_m*n*4)>=1<<63:return None
        schedule,refusal=retained_schedule(total_m,k,n,False,False)
        if schedule is None:return None
        # Check the consumed memory descriptors after the exact same optional
        # store-epoch transformation as lower_dense. Its revalidate() also
        # proves that only completion/renaming changed, never store operands.
        consumed,store_epochs=select_store_epochs(schedule,attrs,n,c['dtype'])
        proof=_check_rows(consumed,m,total_m,k,n,fragmented=fragmented)
        if fragmented:
            from .attention_row_fragments import check_fragment_projection
            proof['fragment_projection']=check_fragment_projection(len(members),m,k,n)
            proof['movement']='per-member fragmented device DMA; no host payload preparation or scatter'
        proof['retained_source_kernel_sha256']=schedule.kernel.digest()
        proof['store_epochs_applied']=bool(store_epochs and store_epochs.get('applied'))
    except (ValueError,KeyError,TypeError,OverflowError):return None
    if fragmented:
        from .attention_row_fragments import FragmentRowView
        view=FragmentRowView(ctx,members,m,k,n,c['dtype'])
    else:
        view=_RowView(ctx,members,m,k,n,c['dtype'])
    # Force the same planned family checked above. Completion and storage
    # ownership remain with the original retained schedule and terminal encoder.
    receipt=lower_dense(view,'_row_stack_lhs',rhs,'_row_stack_dst',m=total_m,k=k,n=n,
        attributes={**attrs,'dense_pipeline':'retained'})
    if receipt.get('kernel_sha256')!=proof['kernel_sha256']:
        raise ValueError('row-stack consumed dense schedule differs from checked DMA projection')
    ctx.fence()
    report={'mechanism':'shared_rhs_row_stack','applied':True,'member_count':len(members),
        'member_shape':[m,k,n],'logical_shape':[total_m,k,n],'proof':proof,
        'members':[{'lhs':row['lhs'],'dst':row['dst'],
                    'source_op_index':row.get('attributes',{}).get('merlin.source_op_index'),
                    'source_region_id':row.get('attributes',{}).get('prov.region_id')} for row in members],
        'rhs':rhs,'dense_schedule':receipt,'preparation_bytes':0,'scatter_bytes':0,'fragmented_rows':fragmented,
        'output_contract':'all original disjoint matrices stored directly in original source ABI order'}
    ctx.metadata.setdefault('rhs_row_stacks',[]).append(report)
    return report
