"""Checked full-K shared-LHS sibling products, with every output retained.

The parent supplies actual source SSA identities and complete disjoint logical
buffers. Equal shapes or values never establish a group. A product consuming
any group result is refused. The library owns common-LHS placement, separate
member RHS/ACC windows, increasing reductions, complete independent outputs,
and the actual pointer/effect checks. Every target data effect remains fully
synchronous through the common dense terminal encoder; this is no overlap law.

Default admission is raw i32, exact zero seeds, ordinary RHS. Optional direct
scalar i8 readout additionally requires original per-member selected dense
contracts and independent readout qualification; absent declarations refuse.
Bias/per-column/pool exits retain the incumbent and original stage order. In a sibling-feed graph, independent first
products may form a group only when their exits are admitted; downstream
residuals and products remain separate original operations.

Integration: build_siblings(products,resources,storage,adapter,
source_identity=source_sha,external_buffers_disjoint=True) returns a checked
owner. Emit owner.emit_index_program(control,lower_member). The callback is
lower_member(member_or_None, operation_name, indices, external), where external
is None or (logical_buffer, actual_byte_offset). Logical buffers are lhs,
rhs0...,output0.... The callback binds those offsets to original ABI pointers,
uses each member's K/N for CONFIG_LD/ST, and retains all ordinary full fences.
No Ptr is discarded: its checked offset is consumed as an actual index operand.
Emitter invalidates other private scratchpad owners before this route.
Attributed products must already belong to one actual source task, constructed
from the accepted independent source group. Per-member primary source operation
and region remain intact. Shared counted control and LHS loads belong to the
first member; outer builder scope is restored. Distinct uncoalesced tasks refuse.
"""
from dataclasses import dataclass,field,replace
import json
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.block_schedule import Load,Store
from merlin.compile.scheduling.compact_dense_group import schedule_compact_dense_group
from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Const,Loop,Ptr,as_expr
from merlin.transforms.dense_planning import DenseProblem

KNOB_DECLARATION={'dense_sibling_schedule':{
    'values':['off','shared_lhs','shared_lhs_readout'],
    'mechanism':'checked compact_dense_group with full-K shared activation and independent member windows',
    'note':'Opt-in independent source SSA siblings, exact zero seeds. shared_lhs is raw i32; shared_lhs_readout additionally requires original selected per-member readout declarations. Capacity/alias/dependency/epilogue refusal preserves original products. No cycle qualification; off incumbent.'}}


def setting():
    return knob('dense_sibling_schedule',default='off',values=('off','shared_lhs','shared_lhs_readout'),
        root=Path(__file__).resolve().parent)


@dataclass(frozen=True)
class SiblingProduct:
    lhs: str
    rhs: str
    output: str
    m: int
    k: int
    n: int
    attrs: dict=field(default_factory=dict)
    dependencies: tuple[str,...]=()
    zero_seed: bool=True
    rhs_transposed: bool=False
    bias: str|None=None


def _source_record(products,source_identity):
    return json.dumps({'source_identity':source_identity,
        'products':[vars(p) for p in products]},sort_keys=True,separators=(',',':'))


class _MemberAdapter:
    def __init__(self,base,problems,readouts=None):
        self.base=base;self.problems=problems
        self.readouts=(None,)*len(problems) if readouts is None else readouts
    def encode(self,member,op):
        call=self.base.encode(op)
        readout=None if member is None else self.readouts[member]
        if isinstance(op,Store) and readout is not None:
            if op.readout!=readout.identity:raise ValueError('actual member store lost its exact readout identity')
            call=replace(call,instr=readout.instruction_name)
        extra=[('_member',Const(-1 if member is None else member))]
        p=self.problems[0 if member is None else member]
        if isinstance(op,Load):
            tensor,stride=('lhs',p.k) if op.role=='lhs' else ('rhs'+str(member),p.n)
            extra.append(('_external',Ptr(tensor,as_expr(op.dram_row)*stride+op.dram_col)))
        elif isinstance(op,Store):
            extra.append(('_external',Ptr('output'+str(member),
                (as_expr(op.dram_row)*p.n+op.dram_col)*(4 if readout is None else 1))))
        return replace(call,args=call.args+tuple(extra))
    def project(self,call):
        values=dict(call.args);member=values.get('_member')
        if type(member) is not Const or not -1<=member.value<len(self.problems):
            raise ValueError('actual sibling instruction has no constant original member')
        readout=None if member.value==-1 else self.readouts[member.value]
        narrowed=readout is not None and call.instr==readout.instruction_name
        op=self.base.project(replace(call,instr='store' if narrowed else call.instr,
            args=tuple((k,v) for k,v in call.args if not k.startswith('_'))))
        if narrowed:op=replace(op,readout=readout.identity)
        return (None if member.value==-1 else member.value),op


@dataclass
class SiblingSchedule:
    result: object
    products: tuple[SiblingProduct,...]
    source_identity: str
    source_record: str
    member_adapter: object
    realization: dict

    @property
    def kernel(self):return self.result.kernel
    @property
    def choice(self):return self.result.choice
    def revalidate(self):
        if _source_record(self.products,self.source_identity)!=self.source_record:
            raise ValueError('sibling source contracts changed after checking')
        self.result.revalidate()
    def emit_index_program(self,control,lower_member):
        """Consume actual checked pointer expressions through the common encoder."""
        self.revalidate()
        def rewrite(body):
            answer=[]
            for item in body:
                if isinstance(item,Loop):item=replace(item,body=tuple(rewrite(item.body)))
                else:
                    item=replace(item,args=tuple(
                        ('_external_offset',value.offset) if type(value) is Ptr else (name,value)
                        for name,value in item.args))
                answer.append(item)
            return answer
        emitted=replace(self.kernel,body=tuple(rewrite(self.kernel.body)))
        # Tensor names remain tied to the checked per-member pointer roster;
        # the exact byte expression, including logical tails/strides, is emitted.
        def lower(name,values):
            member=values['_member']
            if type(member) is not int:raise ValueError('sibling member must remain a literal')
            member=None if member==-1 else member
            external=None
            if '_external_offset' in values:
                tensor='lhs' if member is None else ('output' if name=='store' or (self.member_adapter.readouts[member] is not None and name==self.member_adapter.readouts[member].instruction_name) else 'rhs')+str(member)
                external=(tensor,values['_external_offset'])
            lower_member(member,'store' if member is not None and self.member_adapter.readouts[member] is not None and name==self.member_adapter.readouts[member].instruction_name else name,{k:v for k,v in values.items() if not k.startswith('_')},external)
        receipt=emit_index_program(emitted,control,lower)
        self.revalidate()
        return {**receipt,'checked_group_kernel_sha256':self.kernel.digest(),
            'pointer_transport':'actual original per-member logical buffer and checked byte offset',
            'completion':'ordinary full completion after each terminal data effect'}


def build_siblings(products,resources,storage,adapter,*,source_identity,
                   external_buffers_disjoint=False,static_budget=256,
                   allow_readout=False,selected_contracts=None,readout_qualification_identities=None):
    """Bind a bounded source-independent group to the existing checked owner.

    The caller obtains identities from one source SSA graph, authorizes only
    immutable ordinary tensor reads and excludes intervening observers/effects.
    This interface is a source contract, not grouping by a case name or shape.
    """
    if (type(products) is not tuple or not 2<=len(products)<=8
            or any(type(p) is not SiblingProduct for p in products)):
        raise ValueError('siblings require two to eight typed original products')
    if (not isinstance(source_identity,str) or len(source_identity)!=64
            or any(c not in '0123456789abcdef' for c in source_identity)):
        raise ValueError('siblings require the current complete source digest')
    if external_buffers_disjoint is not True:
        raise ValueError('siblings require explicit original external buffer nonaliasing')
    first=products[0]
    outputs={p.output for p in products}
    if len(outputs)!=len(products):raise ValueError('sibling result identities must be distinct')
    has_readout=any(p.attrs.get('output_dtype','i32')=='i8' for p in products)
    for p in products:
        if (any(type(x) is not str or not x for x in (p.lhs,p.rhs,p.output))
                or (p.lhs,p.m,p.k)!=(first.lhs,first.m,first.k)):
            raise ValueError('siblings require identical original LHS SSA identity and M/K')
        if p.lhs in outputs or p.rhs in outputs or outputs.intersection(p.dependencies):
            raise ValueError('dependent products cannot be grouped as siblings')
        if not p.zero_seed or p.rhs_transposed or p.bias is not None:
            raise ValueError('siblings require exact zero seeds and ordinary unbiased RHS')
        dtype=p.attrs.get('output_dtype','i32');stages=p.attrs.get('epilogue',[])
        if dtype=='i32' and not stages:continue
        if (not allow_readout or dtype!='i8' or stages not in (['acc_scale'],['acc_scale','relu'])
                or p.attrs.get('acc_scale_by_column') is not None):
            raise ValueError('sibling readout requires opt-in direct scalar acc_scale with optional ReLU')
        import math
        scale=p.attrs.get('acc_scale')
        if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:
            raise ValueError('sibling readout requires exact finite positive original scalar scale')
    readouts=(None,)*len(products)
    if has_readout:
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.llvmlower.integer_readout import emit_exact_scalar_ir
        from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
        if (type(selected_contracts) is not tuple or len(selected_contracts)!=len(products)
                or any(type(c) is not SelectedSchedulingContracts for c in selected_contracts)):
            raise ValueError('shared sibling readout requires original per-member selected dense/readout declarations; selected public software has no dense extension')
        readouts=[]
        for p,contract in zip(products,selected_contracts):
            contract.revalidate()
            if contract.dense is None:raise ValueError('original selected software lacks explicit dense scheduling declaration')
            readout=contract.dense.output_readout if p.attrs.get('output_dtype','i32')=='i8' else None
            if readout is not None:
                if type(readout) is not DenseIntegerReadout:
                    raise ValueError('sibling readout requires one exact scalar path per original member')
                scalar,_=emit_exact_scalar_ir([p.attrs['acc_scale']],relu='relu' in p.attrs.get('epilogue',[]))
                if readout.scalar_ir!=scalar:raise ValueError('original selected readout differs from exact original source scale/activation')
            if p.attrs.get('output_dtype','i32')=='i8' and readout is None:
                raise ValueError('original selected software lacks the member i8 readout declaration')
            readouts.append(readout)
        readouts=tuple(readouts)
    edge=storage.lhs.unit_bytes
    count=len(products)
    rhs_rows=(storage.rhs.capacity//(count*edge))*edge
    acc_rows=(storage.accumulator.capacity//(count*edge))*edge
    if min(rhs_rows,acc_rows)<edge:raise ValueError('member private windows do not fit one primitive tile')
    storages=tuple(replace(storage,
        rhs=replace(storage.rhs,name='sibling_rhs'+str(i),base=storage.rhs.base+i*rhs_rows,capacity=rhs_rows),
        accumulator=replace(storage.accumulator,name='sibling_acc'+str(i),
            base=storage.accumulator.base+i*acc_rows,capacity=acc_rows)) for i in range(count))
    problems=tuple(DenseProblem(p.m,p.k,p.n,edge,edge,edge,1,1,4) for p in products)
    # Full K is a required retained object, never a source reduction reorder.
    tasks=tuple(p.attrs.get('source_task_index') for p in products)
    if len(set(tasks))>1:
        raise ValueError('sibling emission requires one actual source task owner; coalesce independent tasks before planning')
    if any(p.attrs.get('source_task_index') is not None
           and p.attrs.get('source_primary_op_index') is None for p in products):
        raise ValueError('task-attributed siblings require the exact original contraction index')
    # Source task construction must already own the complete shared device
    # operation. Different members retain their exact original primary-op and
    # region identities, while real counted control belongs to that shared task.
    limits=replace(resources,max_k_blocks=problems[0].tiles[1])
    identity='gemmini-synchronous-shared-lhs-private-rows-v1'
    qualified=replace(adapter,qualification_identity=identity)
    law=DenseLHSReuseLaw(qualified.configuration_identity,qualified.numerical_policy_identity,
        identity,identity,
        lhs_unchanged_by_member_operations=True,independent_member_interleaving=True,
        suppressed_loads_no_faults=True,suppressed_loads_no_extra_observations=True)
    # These synchronous implementation premises are backed by full completion,
    # disjoint checked physical roles, independent source SSA products, valid
    # logical ABI buffers and ordinary immutable tensor reads. There is no
    # asynchronous queue, array-retention or copy-overlap qualification here.
    member_adapter=_MemberAdapter(qualified,problems,readouts)
    result=schedule_compact_dense_group(problems,limits,storages,qualified,
        encode_member=member_adapter.encode,project_member=member_adapter.project,
        retention_law=law,name='gemmini_shared_lhs_siblings',dtypes=('i8','i8','i32'),
        static_budget=static_budget,external_buffers_disjoint=True,verification_timeout_ms=5000,
        selected_contracts=selected_contracts,output_readouts=readouts,
        readout_qualification_identities=readout_qualification_identities)
    if result is None:raise ValueError('complete shared LHS and separate full-K member windows exceed admitted capacity')
    result.revalidate()
    source_record=_source_record(products,source_identity)
    info={'mechanism':'compact_dense_group full-K shared-LHS',
        'performance_status':'experimental-unmeasured','source_identity':source_identity,
        'source_products':json.loads(source_record)['products'],
        'task_order':'one actual source task; original primary operation and region preserved per member',
        'member_windows':[{'rhs_base':s.rhs.base,'rhs_rows':s.rhs.capacity,
            'acc_base':s.accumulator.base,'acc_rows':s.accumulator.capacity} for s in storages],
        'source_contract':'independent immutable SSA products; exact zeros; disjoint complete outputs; no intervening observer',
        'checked_kernel':result.kernel.text(),'checked_realization':result.realization,
        'retention_qualification':law.record()}
    return SiblingSchedule(result,products,source_identity,source_record,member_adapter,info)


def emit_siblings(builder,plan,pointers,*,index_unroll=None):
    """Bind actual original pointers and use the ordinary Gemmini ISA encoder.

    ``pointers`` contains lhs,rhs0...,output0.... Explicit group pointer offsets
    drive MVIN/MVOUT directly. Per-member transfer configuration is reissued
    inside the actual counted body, so entering another M panel cannot inherit
    a sibling's different N stride. Each data effect keeps the dense incumbent's
    full completion and no target-specific instruction encoder is duplicated.
    """
    from dense import _IndexBridge
    from dense_residency import invalidate_residency
    expected={'lhs',*(f'rhs{i}' for i in range(len(plan.products))),
        *(f'output{i}' for i in range(len(plan.products)))}
    if set(pointers)!=expected:raise ValueError('sibling pointer roster must match every original input/result')
    if len(set(pointers.values()))!=len(pointers):
        raise ValueError('sibling explicit disjoint-buffer declaration disagrees with actual pointers')
    plan.revalidate()
    invalidate_residency(builder,reason='shared-LHS sibling private RHS windows overwrite other dense owners')
    def scope(product):
        attrs=product.attrs
        return {'task_index':attrs.get('source_task_index'),
            'source_op_index':attrs.get('source_primary_op_index'),
            'region_id':attrs.get('prov.region_id')}
    with builder.task_scope(**scope(plan.products[0])):
        builder.config_ex(dataflow=1)
        builder.config_ld(plan.products[0].k,id=0)
        def lower(member,name,a,external):
            product=plan.products[0 if member is None else member]
            with builder.task_scope(**scope(product)):
                ptr=None if external is None else builder.ptr_offset(pointers[external[0]],external[1])
                if name=='lhs_load':
                    builder.mvin(ptr,a['local'],cols=a['cols'],rows=a['rows'],id=0)
                elif name=='weight_load':
                    builder.config_ld(plan.products[member].n,id=1)
                    builder.mvin(ptr,a['local'],cols=a['cols'],rows=a['rows'],id=1)
                elif name=='preload':
                    acc=builder.or_(builder.or_(a['acc'],0x80000000),builder.shl(a['continue'],30))
                    builder.preload(a['weight'],acc,b_cols=a['cols'],b_rows=a['depth'],
                        c_cols=a['cols'],c_rows=a['rows'])
                    return
                elif name=='compute':
                    builder.compute(a['lhs'],0xffffffff,a_cols=a['depth'],a_rows=a['rows'])
                elif name=='store':
                    product=plan.products[member]
                    narrow=plan.member_adapter.readouts[member] is not None
                    builder.config_st(product.n*(1 if narrow else 4),
                        activation=int(narrow and 'relu' in product.attrs.get('epilogue',[])),
                        scale=product.attrs['acc_scale'] if narrow else 1.0)
                    builder.mvout(ptr,builder.or_(a['local'],0x80000000 if narrow else 0xa0000000),cols=a['cols'],rows=a['rows'])
                else:raise ValueError('unsupported actual sibling effect '+name)
                builder.fence()
        control=_IndexBridge(builder,index_unroll)
        receipt=plan.emit_index_program(control,lower)
    receipt['index_unroll']=control.receipt()
    report={**plan.realization,'emission':receipt}
    if not hasattr(builder,'dense_sibling_receipts'):builder.dense_sibling_receipts=[]
    builder.dense_sibling_receipts.append(report)
    return report
