"""Coalesce independent primitive-row products through exact DMA pointer views.

Every member keeps its own ABI pointer. Only the logical accelerator row axis
is combined, so shared weights serve several rows in one ordinary dense MAC.
"""
from __future__ import annotations

from . import isa
from .dense import _schedule
from .dense_readout import _pack


def _check_rows(kernel, count, rows_per_member=1):
    """Prove the exact emitted row span before replacing its DMA descriptors."""
    import z3
    from merlin.sched.ir import Loop, Const, Var, Sum, Scaled, Select
    queries=0
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(t,env) for t in e.terms)
        if isinstance(e,Scaled):return e.factor*expr(e.expr,env)
        if isinstance(e,Select):return z3.If(env[e.var]==e.equals,expr(e.then,env),expr(e.other,env))
        raise ValueError('unsupported grouped row address')
    def prove(scope,bad):
        nonlocal queries
        env={v:z3.Int(v) for v,_ in scope}
        s=z3.Solver();s.set(timeout=2000)
        s.add(*(z3.And(env[v]>=0,env[v]<n) for v,n in scope));s.add(bad(env));queries+=1
        if s.check()!=z3.unsat:raise ValueError('grouped row span is not the complete single M tile')
    def dma_span(a,env):
        row,height=expr(a['dram_row'],env),expr(a['rows'],env)
        if rows_per_member==1:
            return z3.Or(row!=0,height!=count)
        return z3.Or(row<0,height<=0,height>isa.DIM,
            row+height>count*rows_per_member,
            row%rows_per_member+height>rows_per_member)
    sites={'loads':0,'stores':0,'computes':0}
    def walk(body,scope=()):
        for op in body:
            if isinstance(op,Loop):
                walk(op.body,(*scope,(op.var,op.extent)))
                continue
            a=dict(op.args)
            if op.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
                prove(scope,lambda env:dma_span(a,env))
                sites['loads' if op.instr=='gemmini_mvin_lhs' else 'stores']+=1
            elif op.instr in ('gemmini_preload','gemmini_compute_preloaded'):
                def compute_span(env):
                    rows=expr(a['rows'],env)
                    return rows!=count if rows_per_member==1 else z3.Or(rows<=0,rows>isa.DIM)
                prove(scope,compute_span)
                sites['computes']+=int(op.instr=='gemmini_compute_preloaded')
            elif op.instr not in ('gemmini_mvin_weight','merlin_dense_panel_drain'):
                raise ValueError('unsupported grouped dense effect: '+op.instr)
    walk(kernel.body)
    if not all(sites.values()):raise ValueError('incomplete grouped dense program')
    return {'kernel_sha256':kernel.digest(),'symbolic_row_queries':queries,
        'row_span':[0,count*rows_per_member],'rows_per_member':rows_per_member,'source_row_mapping':'member = dense_row // M; source row = dense_row % M',
        'dma_projection':('one count-row load/store replaced by count disjoint one-row descriptors' if rows_per_member==1 else 'every descriptor is proved contained in one original member; original rows/cols retained'),
        'local_projection':('row i has original tile base+i; columns, K order and completion unchanged' if rows_per_member==1 else 'original local addresses, columns, K order and completion unchanged'),
        'complete_mac_projection_preserved':True,'sites':sites}


def lower_row_products(ctx,members):
    """Return None before mutation unless a checked raw primitive-row group applies."""
    from .source_pointer_groups import plan_disjoint_row_groups
    try:
        plan=plan_disjoint_row_groups(ctx.cb,members)
        if plan is None:return None
        m,k,n=plan.rows_per_member,plan.k,plan.n
        count=len(members)
        # Combine primitive heads; already multi-tile members retain their
        # incumbent schedule. The paired panel regresses when M exceeds DIM.
        if m not in (1,isa.DIM) or not 2<=count<=isa.DIM:return None
        attrs=dict(members[0].get('attributes',{}))
        if (ctx.spec(members[0]['dst'])['dtype']!='i32'
                or attrs.get('bias') or attrs.get('accumulator_init')
                or attrs.get('epilogue') or attrs.get('acc_scale_by_column') is not None
                or float(attrs.get('acc_scale',1.0))!=1.0 or attrs.get('relu')
                or attrs.get('requant_shift') is not None):return None
        from .dense_pipeline import retained_schedule
        schedule,_=retained_schedule(count*m,k,n,False,False,False)
        advanced=schedule is not None
        if schedule is None:schedule=_schedule(count*m,k,n,False,False)
        proof=_check_rows(schedule.kernel,count,m)
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    fb=ctx.fb
    saved_index=ctx.current_source_index
    # Group-wide commands have many source owners. Exact member provenance is
    # recorded below; only a statically selected member DMA gets one source ID.
    ctx.current_source_index=None
    rhs=members[0]['rhs']
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(k,channel=0))
    ctx.emit(*isa.config_ld(n,channel=1))
    ctx.emit(*isa.config_st(n*4))
    descriptor_columns=[None]
    def member_pointer(kind,dense_row,column,stride):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr,i64
        member=fb.udiv_i(dense_row,fb.const(m))
        local=fb.urem_i(dense_row,fb.const(m))
        offset=fb.add_i(fb.mul_i(local,fb.const(stride)),column)
        pointer=ctx.ptr(members[0][kind],offset)
        for i,row in enumerate(members[1:],1):
            condition=fb.add(llvm.ICmpOp(member,fb.const(i),IntegerAttr(llvm.ICmpPredicateFlag.EQ.to_int(),i64))).results[0]
            pointer=fb.add(llvm.SelectOp(condition,ctx.ptr(row[kind],offset),pointer)).results[0]
        return pointer
    def source(row):
        ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',saved_index)
    def emit(instruction,a,metadata=None):
        ctx.current_source_index=None
        a={key:fb.const(value) if isinstance(value,int) else value for key,value in a.items()}
        if instruction=='merlin_dense_panel_drain':ctx.fence();return
        if instruction=='gemmini_mvin_lhs':
            if m==1:
                for i,row in enumerate(members):
                    source(row)
                    local=fb.add_i(a['addr'],fb.const(i))
                    ctx.emit(*isa.mvin(ctx.ptr(row['lhs'],a['dram_col']),local,1,a['cols'],channel=0,fb=fb))
            else:
                pointer=member_pointer('lhs',a['dram_row'],a['dram_col'],k)
                ctx.emit(*isa.mvin(pointer,a['addr'],a['rows'],a['cols'],channel=0,fb=fb))
        elif instruction=='gemmini_mvin_weight':
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
            ctx.emit(*isa.mvin(ctx.ptr(rhs,offset),a['addr'],a['rows'],a['cols'],channel=1,fb=fb))
        elif instruction=='gemmini_preload':
            descriptor_columns[0]=a['cols']
            caddr=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,caddr,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            if descriptor_columns[0] is None:raise ValueError('grouped compute lacks preload')
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,fb.const(isa.GARBAGE),a['depth'],descriptor_columns[0]))
            if not advanced:ctx.fence()
        elif instruction=='gemmini_mvout':
            ctx.fence()
            if m==1:
                for i,row in enumerate(members):
                    source(row)
                    local=fb.add_i(a['acc'],fb.const(i))
                    ctx.emit(*isa.mvout(ctx.ptr(row['dst'],a['dram_col']),isa.acc_addr(local,full=True,fb=fb),1,a['cols'],fb=fb))
            else:
                pointer=member_pointer('dst',a['dram_row'],a['dram_col'],n)
                ctx.emit(*isa.mvout(pointer,isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
            ctx.fence()
        else:raise ValueError('unknown grouped dense effect '+instruction)
    schedule.emit_llvm(fb,emit)
    ctx.current_source_index=None
    ctx.fence()
    ctx.current_source_index=saved_index
    report={'mechanism':'disjoint_row_rhs_coalescing','members':count,
        'shape':{'m':count*m,'k':k,'n':n},'projection':proof,
        'dense_realization':dict(schedule.realization),'source_pointer_projection':plan.record(),
        'source_commands':[r['command_index'] for r in members],
        'payload_preparation':'none; each DMA descriptor stays within its original ABI pointer',
        'completion':'ordinary dense epochs; full drain before and after grouped disjoint readouts',
        'requested_lhs_rows':count*m,'rows_per_member':m,'shared_rhs_members':count}
    ctx.metadata.setdefault('attention_row_products',[]).append(report)
    return report
