"""Shared source RHS products partitioned into disjoint output-N rectangles."""
from dataclasses import dataclass,asdict
from math import isfinite,prod
from . import isa

@dataclass(frozen=True)
class SharedRhsPanelPlan:
    m: tuple[int,...]
    k: int
    n: int
    panel_n: int
    lhs_rows: int


def plan_shared_rhs_panels(shapes):
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    shapes=tuple(shapes)
    if not 2<=len(shapes)<=32:return None
    if any(len(s)!=3 or any(type(x) is not int or x<=0 for x in s) for s in shapes):return None
    k,n=shapes[0][1:]
    if any(s[1:]!=(k,n) for s in shapes):return None
    d=isa.DIM;kt=(k+d-1)//d
    lhs_rows=sum(((m+d-1)//d)*kt*d for m,_,_ in shapes)
    nt=min((isa.SPAD_ROWS-d-lhs_rows)//(kt*d),1024//kt)
    if nt<1:return None
    width=min(n,nt*d)
    if any(plan_dense_rhs_residency(m,k,width,block=d,operand_rows=isa.SPAD_ROWS,
        accumulator_rows=isa.ACC_ROWS,rhs_base=d) is None for m,_,_ in shapes):return None
    return SharedRhsPanelPlan(tuple(s[0] for s in shapes),k,n,width,lhs_rows)


def lower(ctx,members):
    from .resident_rhs import lower_resident_products,ResidentPhysicalPitches,_rewrite
    from .dense import _schedule
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    members=tuple(members)
    if not 2<=len(members)<=32:return None
    try:
        rhs=members[0]['rhs']
        if any(r['rhs']!=rhs for r in members):return None
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        sources={a.get('source') for a in aliases.values() if isinstance(a,dict)}
        names={r[key] for r in members for key in ('lhs','rhs','dst')}
        outputs=[r['dst'] for r in members]
        if names&(set(aliases)|sources) or len(set(outputs))!=len(outputs):return None
        if set(outputs)&{r[key] for r in members for key in ('lhs','rhs')}:return None
        shapes=[];attributes=[]
        for row in members:
            a,b,c=(ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
            if a['dtype']!='i8' or b['dtype']!='i8' or c['dtype'] not in ('i8','i32'):return None
            if len(a['shape'])!=2 or len(b['shape'])!=2:return None
            m,k=a['shape'];bk,n=b['shape']
            if k!=bk or c['shape']!=[m,n] or any(key in row and row[key]!=v for key,v in [('m',m),('k',k),('n',n)]):return None
            if max(m*k,k*n,4*m*n)>=1<<63:return None
            attrs=dict(row.get('attributes',{}))
            if attrs.get('output_dtype',c['dtype'])!=c['dtype']:return None
            if attrs.get('bias') or attrs.get('accumulator_init') or attrs.get('requant_shift') is not None:return None
            if any(e not in ('acc_scale','relu') for e in attrs.get('epilogue',())):return None
            scales=attrs.get('acc_scale_by_column')
            if scales is not None and (len(scales)!=n or len(set(map(float,scales)))!=1):return None
            scale=float(scales[0] if scales else attrs.get('acc_scale',1.0))
            if not isfinite(scale) or scale<0:return None
            if c['dtype']=='i32' and (scale!=1 or scales is not None or attrs.get('relu') or 'relu' in attrs.get('epilogue',())):return None
            if scales is not None:attrs.pop('acc_scale_by_column');attrs['acc_scale']=scale
            shapes.append((m,k,n));attributes.append(attrs)
        plan=plan_shared_rhs_panels(shapes)
        if plan is None:return None
        # Full-fit keeps the exact original source/terminal route, including its
        # original auxiliary and metadata handling. No extra loop is introduced.
        if plan.panel_n==plan.n:
            receipt=lower_resident_products(ctx,members)
            if receipt is not None:ctx.metadata.setdefault('resident_rhs_panels',[]).append({'plan':asdict(plan),'panels':1,'route':'original complete RHS','source_members':outputs})
            return receipt
        widths={plan.panel_n,plan.n%plan.panel_n}-{0}
        # Pure source/adapter checks precede any pointer or device emission.
        for m,_,_ in shapes:
            for width in widths:
                storage=plan_dense_rhs_residency(m,plan.k,width,block=isa.DIM,operand_rows=isa.SPAD_ROWS,accumulator_rows=isa.ACC_ROWS,rhs_base=isa.DIM)
                schedule=_schedule(m,plan.k,width,False,False)
                _rewrite(schedule,storage)
    except (ValueError,KeyError,TypeError,OverflowError):return None
    fb=ctx.fb;saved=ctx.current_source_index;receipts=[]
    def alias(name,offset,shape):
        label=f'_rhs_panel_{name}_{len(ctx.tensors)}'
        ctx.register(label,ctx.ptr(name,offset),shape,ctx.spec(name)['dtype']);return label
    def panel(start,width):
        common=alias(rhs,start,(plan.k,width));rows=[]
        for original,attrs,m in zip(members,attributes,plan.m):
            rows.append({**original,'rhs':common,'dst':alias(original['dst'],start,(m,width)),
                'm':m,'k':plan.k,'n':width,'attributes':attrs})
        result=lower_resident_products(ctx,rows,physical_pitches=ResidentPhysicalPitches(plan.n,plan.n))
        ctx.current_source_index=saved
        if result is None:raise ValueError('prechecked shared RHS panel lost resident admission')
        receipts.append(result)
    count=plan.n//plan.panel_n
    fb.for_loop(0,count,1,lambda index:panel(fb.mul_i(index,fb.const(plan.panel_n)),plan.panel_n),retain_loop=True)
    if plan.n%plan.panel_n:panel(fb.const(count*plan.panel_n),plan.n%plan.panel_n)
    receipt={'mechanism':'source_shared_rhs_n_panels','plan':asdict(plan),'panels':(plan.n+plan.panel_n-1)//plan.panel_n,
        'rhs':rhs,'source_members':outputs,'physical_rhs_pitch':plan.n,'physical_output_pitch':plan.n,
        'partition':'disjoint N rectangles covering every original member output exactly once',
        'completion':'existing resident full completion closes every panel before window reuse',
        'members':receipts}
    ctx.metadata.setdefault('resident_rhs_panels',[]).append(receipt)
    return receipt
