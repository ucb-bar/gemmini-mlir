"""Source-owned complete-RHS groups with checked full-K member epochs.

The same selected RTL law as dense_epochs orders local accesses, ACC commits,
readout and finite queue progress. The library loads W once, retains full-K A
panels across N, and closes every member before reusing private A/ACC rows.
This is complete-region grouping, not a per-call cache or a promised speedup.
Parent source grouping proves an immutable explicit handle and no intervening
observer/writer. All original inputs/outputs and exact byte pointers are checked.
Raw i32 exits need no extra readout declaration. Narrow exits require an exact
externally qualified scalar readout; this module never invents qualification.
"""
from dataclasses import dataclass,field,replace,asdict
import json
from merlin.sched.ir import Const,Ptr,as_expr
from merlin.transforms.dense_planning import DenseProblem
from merlin.compile.scheduling.block_schedule import Load,Store
from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
from merlin.compile.scheduling.dense_shared_rhs_check import DenseRHSReuseLaw
from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission

KNOB_DECLARATION={'dense_shared_rhs_schedule':{'values':['off','drained','epochs'],
    'mechanism':'source-bound dense_shared_rhs full-K LHS and complete RHS with checked member completion epochs',
    'note':'Opt-in independent immutable source products, complete original outputs; capacity/readout/alias refusal retains original route. No cycle qualification.'}}

@dataclass(frozen=True)
class RHSProduct:
    lhs:str
    rhs:str
    output:str
    m:int
    k:int
    n:int
    attrs:dict=field(default_factory=dict)
    dependencies:tuple[str,...]=()
    zero_seed:bool=True
    rhs_transposed:bool=False
    bias:str|None=None


def _record(products,source_identity):
    return json.dumps({'source_identity':source_identity,'products':[vars(p) for p in products]},sort_keys=True)


class _MemberAdapter:
    def __init__(self,base,problems,readouts):
        self.base=base;self.problems=problems;self.readouts=readouts
    def encode(self,member,op):
        p=self.problems[0 if member is None else member]
        r=None if member is None else self.readouts[member]
        call=self.base.encode(op)
        if type(op) is Store and r is not None:
            if op.readout!=r.identity:raise ValueError('member store lost exact qualified readout')
            call=replace(call,instr=r.instruction_name)
        args=[('_member',Const(-1 if member is None else member))]
        if type(op) is Load:
            weight=op.role=='weight'
            args.append(('_external',Ptr('rhs' if weight else 'lhs'+str(member),
                (as_expr(op.dram_row)*(p.n if weight else p.k)+op.dram_col))))
        elif type(op) is Store:
            args.append(('_external',Ptr('output'+str(member),
                (as_expr(op.dram_row)*p.n+op.dram_col)*(4 if r is None else 1))))
        return replace(call,args=call.args+tuple(args))
    def project(self,call):
        m=dict(call.args).get('_member')
        if type(m) is not Const or not -1<=m.value<len(self.problems):
            raise ValueError('shared RHS actual member is not a literal source owner')
        member=None if m.value==-1 else m.value
        r=None if member is None else self.readouts[member]
        narrow=r is not None and call.instr==r.instruction_name
        op=self.base.project(replace(call,instr='store' if narrow else call.instr,
            args=tuple((k,v) for k,v in call.args if not k.startswith('_'))))
        if narrow:op=replace(op,readout=r.identity)
        return member,op


@dataclass
class RHSGroup:
    result:object
    products:tuple[RHSProduct,...]
    source_identity:str
    source_record:str
    member_adapter:object
    realization:dict
    @property
    def kernel(self):return self.result.kernel
    def revalidate(self):
        if _record(self.products,self.source_identity)!=self.source_record:
            raise ValueError('original shared RHS source/attributes changed; replan')
        self.result.revalidate()


def build_rhs_epochs(products,resources,storage,adapter,*,source_identity,
        external_buffers_disjoint=False,mode='epochs',output_readouts=None,
        readout_qualification_identities=None,static_budget=512):
    """Return a checked complete invocation; never emits on refusal.

    Members must be independent original source products sharing the same
    immutable RHS SSA identity. Caller owns task construction before planning.
    The scope is one actual source task, retaining each exact primary operation.
    """
    if mode not in ('epochs','drained'):raise ValueError('unknown shared RHS completion mode')
    if type(products) is not tuple or not 2<=len(products)<=128 or any(type(p) is not RHSProduct for p in products):
        raise ValueError('shared RHS requires two to128 typed original products')
    if not isinstance(source_identity,str) or len(source_identity)!=64 or any(c not in '0123456789abcdef' for c in source_identity):
        raise ValueError('shared RHS requires complete original source digest')
    if external_buffers_disjoint is not True:raise ValueError('shared RHS requires original complete external buffers disjoint')
    first=products[0];outputs={p.output for p in products}
    if len(outputs)!=len(products):raise ValueError('shared RHS outputs must remain distinct')
    if len({p.attrs.get('source_task_index') for p in products})>1:
        raise ValueError('coalesce the actual shared source task before planning')
    readouts=(None,)*len(products) if output_readouts is None else output_readouts
    if type(readouts) is not tuple or len(readouts)!=len(products):raise ValueError('exact member readout roster required')
    for p,r in zip(products,readouts):
        if (p.rhs,p.k,p.n)!=(first.rhs,first.k,first.n):raise ValueError('shared RHS must have identical original immutable RHS SSA/K/N')
        if p.lhs in outputs or p.rhs in outputs or outputs.intersection(p.dependencies):raise ValueError('dependent source products cannot share this independent RHS region')
        if p.bias is not None or p.rhs_transposed or not p.zero_seed:raise ValueError('shared RHS requires exact zero seeds and ordinary unbiased operands')
        if p.attrs.get('source_task_index') is not None and p.attrs.get('source_primary_op_index') is None:
            raise ValueError('shared RHS needs exact original contraction attribution')
        stages=p.attrs.get('epilogue',[])
        if r is None:
            if p.attrs.get('output_dtype','i32')!='i32' or stages:
                raise ValueError('narrow shared RHS member needs independently qualified exact original scalar readout')
        else:
            from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
            from merlin.llvmlower.integer_readout import emit_exact_scalar_ir
            if type(r) is not DenseIntegerReadout or p.attrs.get('output_dtype')!='i8' or stages not in (['relu'],['acc_scale'],['acc_scale','relu']) or p.attrs.get('acc_scale_by_column') is not None:
                raise ValueError('only exact direct scalar i8 readout admitted')
            scalar,_=emit_exact_scalar_ir([p.attrs['acc_scale']] if 'acc_scale' in stages else [],relu='relu' in stages)
            if r.scalar_ir!=scalar:raise ValueError('qualified scalar path differs from original member stages')
    edge=storage.lhs.unit_bytes
    problems=tuple(DenseProblem(p.m,p.k,p.n,edge,edge,edge,1,1,4) for p in products)
    resources=replace(resources,max_k_blocks=problems[0].tiles[1])
    # Members close sequentially; the checked library permits shared A/ACC rows.
    storages=(storage,)*len(products)
    owner='immutable-source-rhs-full-completion-v1'
    qualified=replace(adapter,qualification_identity=owner)
    reuse=DenseRHSReuseLaw(qualified.configuration_identity,qualified.numerical_policy_identity,owner,owner,
        rhs_unchanged_by_member_operations=True,independent_member_interleaving=True,
        suppressed_loads_no_faults=True,suppressed_loads_no_extra_observations=True)
    from dense_epochs import selected_law
    law,table,entries,identity=selected_law(qualified)
    if mode=='epochs' and law.unknown():raise ValueError('shared RHS lacks selected ordering premises: '+repr(law.unknown()))
    # Issued counts are bounded compiler budgets; pending capacities come only
    # from selected RTL queue entries. No speculative overlap permission added.
    admission=DenseRetainedReductionAdmission(qualified.configuration_identity,
        qualified.numerical_policy_identity,identity,4096,4096,
        max(1,entries['execute']//2),entries['load'],pending_readouts=entries['store'])
    member=_MemberAdapter(qualified,problems,readouts)
    result=schedule_dense_shared_rhs(problems,resources,storages,qualified,
        encode_member=member.encode,project_member=member.project,source_identity=source_identity,
        name='gemmini_shared_rhs_epochs',dtypes=('i8','i8','i32'),static_budget=static_budget,
        external_buffers_disjoint=True,retention_law=reuse,
        epoch_admission=admission if mode=='epochs' else None,epoch_law=law if mode=='epochs' else None,
        output_readouts=readouts,readout_qualification_identities=readout_qualification_identities,
        verification_timeout_ms=10000)
    if result is None:raise ValueError('complete RHS plus full-K LHS/private output group exceeds selected capacity/domain')
    result.revalidate();record=_record(products,source_identity)
    info={'mechanism':'shared_rhs_epochs','mode':mode,'timing':'UNKNOWN',
        'source_record':json.loads(record),'selected_law':table,'ordering_identity':identity,
        'reuse_law':reuse.record(),'schedule_ir':result.plan_text,'actual_kernel':result.kernel.text(),
        'checked_realization':result.realization,'blocks':[asdict(b) for b in result.blocks]}
    return RHSGroup(result,products,source_identity,record,member,info)


def emit_rhs_epochs(builder,plan,pointers,*,index_unroll='off'):
    """Consume actual checked indexed effects through the existing ISA builder.

    Pointer roster: rhs,lhs0...,output0.... Logical row/column operands reconstruct
    exactly the Ptr formulas independently checked by the library. Real complete
    calls are emitted solely at the library's explicit completion instructions.
    """
    from dense import _IndexBridge
    from dense_residency import invalidate_residency
    count=len(plan.products);expected={'rhs',*(f'lhs{i}' for i in range(count)),*(f'output{i}' for i in range(count))}
    if set(pointers)!=expected or len(set(pointers.values()))!=len(pointers):raise ValueError('actual shared RHS pointer roster disagrees with original disjoint source buffers')
    plan.revalidate();invalidate_residency(builder,reason='shared RHS epoch group acquires complete private state')
    def scope(p):return {'task_index':p.attrs.get('source_task_index'),'source_op_index':p.attrs.get('source_primary_op_index'),'region_id':p.attrs.get('prov.region_id')}
    first=plan.products[0]
    with builder.task_scope(**scope(first)):
        builder.config_ex(dataflow=1);builder.config_ld(first.k,id=0);builder.config_ld(first.n,id=1)
        def lower(name,a):
            member=a['_member'];member=None if member==-1 else member
            p=plan.products[0 if member is None else member]
            r=None if member is None else plan.member_adapter.readouts[member]
            with builder.task_scope(**scope(p)):
                if name in ('lhs_load','weight_load'):
                    weight=name=='weight_load';stride=p.n if weight else p.k
                    ptr=pointers['rhs' if weight else 'lhs'+str(member)]
                    builder.mvin(builder.ptr_offset(ptr,builder.add(builder.mul(a['r'],stride),a['c'])),a['local'],cols=a['cols'],rows=a['rows'],id=1 if weight else 0)
                elif name=='preload':
                    acc=builder.or_(builder.or_(a['acc'],0x80000000),builder.shl(a['continue'],30))
                    builder.preload(a['weight'],acc,b_cols=a['cols'],b_rows=a['depth'],c_cols=a['cols'],c_rows=a['rows'])
                elif name=='compute':builder.compute(a['lhs'],0xffffffff,a_cols=a['depth'],a_rows=a['rows'])
                elif name=='store' or r is not None and name==r.instruction_name:
                    width=4 if r is None else 1
                    builder.config_st(p.n*width,activation=int(r is not None and 'relu' in p.attrs.get('epilogue',[])),scale=p.attrs.get('acc_scale',1.0) if r is not None else 1.0)
                    offset=builder.mul(builder.add(builder.mul(a['r'],p.n),a['c']),width)
                    builder.mvout(builder.ptr_offset(pointers['output'+str(member)],offset),builder.or_(a['local'],0xa0000000 if r is None else 0x80000000),cols=a['cols'],rows=a['rows'])
                else:raise ValueError('unknown actual shared RHS effect '+name)
        bridge=_IndexBridge(builder,index_unroll)
        receipt=plan.result.emit_index_program(bridge,lower,complete_epoch=builder.fence)
    plan.revalidate()
    report={**plan.realization,'emission':{**receipt,'index_unroll':bridge.receipt()}}
    if not hasattr(builder,'dense_rhs_epoch_receipts'):builder.dense_rhs_epoch_receipts=[]
    builder.dense_rhs_epoch_receipts.append(report)
    return report
