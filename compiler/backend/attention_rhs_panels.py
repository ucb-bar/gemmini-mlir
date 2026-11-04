"""Capacity-bounded shared RHS rectangles across grouped attention heads.

Each panel partitions N and retains all K. External addresses keep their full
matrix pitch; checked resident schedules operate on the logical rectangle.
"""
from dataclasses import dataclass, asdict
from math import isfinite
from . import isa


@dataclass(frozen=True)
class AttentionRhsPanelPlan:
    batches: int
    query_heads: int
    kv_heads: int
    m: int
    k: int
    n: int
    panel_n: int
    lhs_rows: int

    @property
    def group(self): return self.query_heads // self.kv_heads

    def projection(self):
        """Independent integer source coordinates for every member rectangle."""
        for kv in range(self.batches*self.kv_heads):
            for start in range(0,self.n,self.panel_n):
                width=min(self.panel_n,self.n-start)
                for local in range(self.group):
                    head=kv*self.group+local
                    yield head,kv,start,width,head*self.m*self.k,kv*self.k*self.n+start,head*self.m*self.n+start


def plan_attention_rhs_panels(geometry):
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    b,hq,hkv,m,k,n=geometry
    if any(type(x) is not int or x<=0 for x in geometry) or hq%hkv:return None
    group=hq//hkv
    if not 2<=group<=32:return None
    d=isa.DIM;kt=(k+d-1)//d
    # Reserve the complete immutable LHS of every head, allowing the existing
    # checked resident compute-epoch transformation to use disjoint windows.
    lhs_rows=group*((m+d-1)//d)*kt*d
    nt=min((isa.SPAD_ROWS-d-lhs_rows)//(kt*d),1024//kt)
    if nt<1:return None
    width=min(n,nt*d)
    if plan_dense_rhs_residency(m,k,width,block=d,operand_rows=isa.SPAD_ROWS,
            accumulator_rows=isa.ACC_ROWS,rhs_base=d) is None:return None
    return AttentionRhsPanelPlan(b,hq,hkv,m,k,n,width,lhs_rows)


def lower(ctx,left,right,dst,stationary,geometry,attrs):
    from .attention import _alias, AttentionError
    from .resident_rhs import lower_resident_products, ResidentPhysicalPitches
    plan=plan_attention_rhs_panels(geometry)
    if plan is None:return False
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    sources={a.get('source') for a in aliases.values() if isinstance(a,dict)}
    if len({left,right,dst})!=3 or {left,right,dst}&(set(aliases)|sources):return False
    if attrs.get('bias') or attrs.get('accumulator_init') or attrs.get('requant_shift') is not None:return False
    if any(x not in ('acc_scale','relu') for x in attrs.get('epilogue',())):return False
    scales=attrs.get('acc_scale_by_column')
    if scales is not None and (len(scales)!=plan.n or len(set(map(float,scales)))!=1):return False
    scale=float(scales[0] if scales else attrs.get('acc_scale',1.0))
    if not isfinite(scale) or scale<0:return False
    if ctx.spec(dst)['dtype']=='i32' and (scale!=1 or scales is not None or attrs.get('relu') or 'relu' in attrs.get('epilogue',())):return False
    member_attrs=dict(attrs)
    # A uniform scale has identical semantics on each logical rectangle.
    if scales is not None:
        member_attrs.pop('acc_scale_by_column');member_attrs['acc_scale']=scale
    saved=ctx.current_source_index
    if saved is not None:member_attrs.setdefault('merlin.source_op_index',saved)
    fb=ctx.fb; receipts=[]
    def kv_head(kv):
        def panel(start,width):
            common=_alias(ctx,stationary,fb.add_i(fb.mul_i(kv,fb.const(plan.k*plan.n)),start),(plan.k,width))
            members=[]
            for local in range(plan.group):
                head=fb.add_i(fb.mul_i(kv,fb.const(plan.group)),fb.const(local))
                members.append({'lhs':_alias(ctx,left,fb.mul_i(head,fb.const(plan.m*plan.k)),(plan.m,plan.k)),
                    'rhs':common,'dst':_alias(ctx,dst,fb.add_i(fb.mul_i(head,fb.const(plan.m*plan.n)),start),(plan.m,width)),
                    'm':plan.m,'k':plan.k,'n':width,'attributes':member_attrs})
            receipt=lower_resident_products(ctx,members,physical_pitches=ResidentPhysicalPitches(plan.n,plan.n))
            ctx.current_source_index=saved
            if receipt is None:raise AttentionError('checked attention panel rejected by resident source adapter')
            receipts.append(receipt)
        count=plan.n//plan.panel_n
        if count:fb.for_loop(0,count,1,lambda index:panel(fb.mul_i(index,fb.const(plan.panel_n)),plan.panel_n),retain_loop=True)
        if plan.n%plan.panel_n:panel(fb.const(count*plan.panel_n),plan.n%plan.panel_n)
    fb.for_loop(0,plan.batches*plan.kv_heads,1,kv_head,retain_loop=True)
    ctx.current_source_index=saved
    ctx.metadata.setdefault('attention_rhs_panels',[]).append({'mechanism':'grouped_rhs_n_panels','applied':True,
        'plan':asdict(plan),'group_size':plan.group,'panels_per_kv':(plan.n+plan.panel_n-1)//plan.panel_n,
        'rhs_physical_pitch':plan.n,'output_physical_pitch':plan.n,
        'head_partition':'global_query_head = global_kv_head * group + local_head',
        'N_partition':'disjoint [panel_start,min(panel_start+panel_n,N)) rectangles covering N',
        'completion':'existing resident full completion boundaries; all readouts complete before RHS panel reuse',
        'resident_kernel_sha256':[[x['resident_kernel_sha256'] for x in r['members']] for r in receipts]})
    return True
