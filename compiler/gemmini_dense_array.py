"""Checked full-K dense panels with selected WS array-weight persistence.

Only the shared owner's pure K/N/M compute runs change to first/remaining M.
The original fresh-read stream supplies the complete source and epoch proof;
the final emitted stream retains B with the common encoder's funct5 command.
This optional candidate carries no timing or execution qualification.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

from merlin.sched.ir import Call, as_expr
from merlin.compile.scheduling.block_schedule import Preload, Compute
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, emit_compact_dense
from merlin.compile.scheduling.dense_array_retention import (
    DenseArrayRetentionAdmission, propose_dense_array_retention, verify_dense_array_retention)
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN, DenseRetainedReductionAdmission, dense_retained_operand_hoists, dense_retained_compute_order)
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan, dense_schedule_text, parse_dense_schedule_plan
from merlin.transforms.dense_planning import DenseResources, dense_blocks
from gemmini_dense_readout import ReadoutUnavailable

ISA_SOURCE = 'https://github.com/ucb-bar/gemmini#re-use-previous-preloads'


def array_adapter(original):
    """Extend the actual reversible boundary; every original call is unchanged."""
    def call(name, names, values):
        return Call(name, tuple(zip(names, map(as_expr, values))))
    def encode(op):
        if type(op) is Preload and op.weight_row is None and op.weight_block is None:
            return call('gemmini.preload.array', ('acc','continue','rows','cols','depth'),
                (op.accumulator_row,op.accumulate,op.rows,op.cols,op.weight_rows))
        if type(op) is Compute and op.fresh_weights is False:
            return call('gemmini.compute.array', ('lhs','block0','block1','rows','depth'),
                (op.input_row,*op.input_block,op.rows,op.cols))
        return original.encode(op)
    def project(c):
        a=dict(c.args)
        if c.instr=='gemmini.preload.array':
            return Preload(None,None,a['acc'],a['continue'],a['rows'],a['cols'],a['depth'])
        if c.instr=='gemmini.compute.array':
            return Compute(a['lhs'],(a['block0'],a['block1']),a['rows'],a['depth'],False)
        return original.project(c)
    return replace(original,encode=encode,project=project)


def _selected_array_law(adapter, facts_path, receipt):
    from gemmini_conv_array import _names
    facts_path=Path(facts_path)
    facts=facts_path.read_bytes()
    if adapter.configuration_identity!='gemmini-rtl:'+hashlib.sha256(facts).hexdigest():
        raise ReadoutUnavailable('array adapter differs from selected hardware facts')
    if not any(n.get('4')=='COMPUTE_AND_FLIP_CMD' and n.get('5')=='COMPUTE_AND_STAY_CMD'
            and n.get('6')=='PRELOAD_CMD' for n in _names(json.loads(facts))):
        raise ReadoutUnavailable('selected decode lacks fresh/stay/preload endpoints')
    pin=json.loads((facts_path.parent/'COMMAND_ORDERING_SELECTION.json').read_text())
    order=pin['ordering']
    digest=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest()
    if digest!=receipt['ordering_sha256'] or pin['base_facts']['sha256']!=hashlib.sha256(facts).hexdigest():
        raise ReadoutUnavailable('array ordering and base facts identity differ')
    for name,funct in [('COMPUTE_PRELOADED.a_spad',4),('COMPUTE_ACCUMULATE.a_spad',5)]:
        if (order.get('accesses',{}).get(name,{}).get('funct')!=funct
                or order['accesses'][name].get('queue')!='execute'
                or order.get('extents',{}).get(name,{}).get('status')!='derived'):
            raise ReadoutUnavailable('selected array source extent or ordering is unknown')
    if order.get('completion_order',{}).get('execute',{}).get('order')!='in_order':
        raise ReadoutUnavailable('selected fresh/stay execute completion order is unknown')
    return DenseArrayRetentionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-ws-funct4-captures-B-funct5-retains-B-output-descriptor-only-v1')


@dataclass(frozen=True)
class DenseArrayResult:
    original: object
    proposal: object
    adapter: object
    realization: dict
    facts_path: object

    @property
    def kernel(self): return self.proposal.kernel

    def revalidate(self):
        self.original.revalidate()
        self.proposal.revalidate(self.adapter,self.proposal.admission)
        if _selected_array_law(self.adapter,self.facts_path,self.realization)!=self.proposal.admission:
            raise ValueError('selected array law changed after source verification')
        if self.original.kernel!=self.proposal.checking_kernel:
            raise ValueError('array result no longer has the original checked source projection')

    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        self.revalidate()
        result=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return result

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate()
        return materialize_scf_commands(self.kernel)


def plan_array(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False,
               outer_order=('m','n')):
    """Return (checked actual result, receipt, retained adapter), or decline.

    Preserve incumbent M/N blocks and source/readout identity. Full original K
    must fit both assigned scratch windows. No empirical shape threshold or
    execution evidence is inferred from capacity or the source proof.
    """
    from gemmini_dense import seed_source
    from gemmini_dense_pipeline import ordering_premises
    if outer_order!=('m','n') or problem.tiles[0]<2:
        raise ReadoutUnavailable('array retention requires repeated M and original M/N outer order')
    incumbent.revalidate()
    if (bool(problem.bias)!=bool(bias or initial)
            or (problem.lhs_bytes,problem.rhs_bytes,problem.accumulator_bytes)!=(1,1,4)):
        raise ReadoutUnavailable('array source dtype or original initializer contract differs')
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    old=incumbent.choice.block
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=old.m,max_k_blocks=problem.tiles[1],max_n_blocks=old.n,
        partial_output_tiles=True,partial_output_blocks=True)
    block=next((b for b in dense_blocks(problem,resources,k_limit=lambda m,n:storage.k_limit(problem,resources,m,n))
        if (b.m,b.k,b.n)==(old.m,problem.tiles[1],old.n)),None)
    if block is None or block.m<2:
        raise ReadoutUnavailable('incumbent repeated M/N block cannot retain complete K in assigned stores')
    if max(block.m*block.k*block.n,(block.m+block.n)*block.k)>4096:
        raise ReadoutUnavailable('array panel exceeds original finite compiler issue budget')
    flags,pending,receipt=ordering_premises(facts_path)
    required=('ordered_accumulator_writes','accumulator_result_visibility','ordered_configuration_capture',
        'finite_queue_progress','complete_device_drain','exclusive_private_state')
    if not all(flags[key] for key in required):
        raise ReadoutUnavailable('selected retained source/epoch premises are incomplete')
    retained_adapter=array_adapter(adapter)
    array_law=_selected_array_law(retained_adapter,facts_path,receipt)
    identity='selected-command-ordering:'+receipt['ordering_sha256']
    law=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':retained-read-only-operands',4096,4096,pending['execute']//2,pending['load'],
        max_issued_readouts=block.m*block.n,pending_readouts=pending['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    initializer=seed_source(problem.m,problem.n,bias=bias,initial=initial)
    source_identity=hashlib.sha256(json.dumps(dict(source='row-major-source-ordered-dense',
        problem=asdict(problem),initializer=None if initializer is None else asdict(initializer),
        numerical_policy=adapter.numerical_policy_identity),sort_keys=True).encode()).hexdigest()
    raw=emit_compact_dense(problem,block,storage,retained_adapter,name='dense_array',
        dtypes=('i8','i8','i32'),outer_order=outer_order,bias_source=initializer)
    transforms=dense_retained_operand_hoists(raw,retained_adapter,role='lhs')
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),
        retained_adapter,order=('k','n','m'))
    plan=make_dense_schedule_plan(problem,resources,storage,retained_adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=outer_order,loop_transforms=transforms,pipeline_kind='retained_full_k',
        pipeline_prefetch=False,bias_source=initializer)
    text=dense_schedule_text(plan)
    original=schedule_dense_pipeline(problem,resources,storage,retained_adapter,name='dense_array',
        source_identity=source_identity,dtypes=('i8','i8','i32'),static_budget=256,
        external_buffers_disjoint=True,schedule_plan=parse_dense_schedule_plan(text),
        pipeline_admission=law,batch_readouts=True,bias_source=initializer,timeout_ms=60000)
    proposal=propose_dense_array_retention(original.kernel,retained_adapter,array_law)
    proof=verify_dense_array_retention(proposal,problem,block,resources,storage,retained_adapter,law,
        static_budget=256,timeout_ms=60000,bias_source=initializer)
    receipt.update(mode='array_epochs',library='retained_full_k/compute_order/dense_array_retention',
        source_identity=source_identity,block=(block.m,block.k,block.n),schedule_ir=text,
        original_incumbent_sha256=incumbent.kernel.digest(),source_schedule_sha256=original.kernel.digest(),
        transformed_schedule_sha256=proposal.kernel.digest(),array_proof=proof,array_isa_source=ISA_SOURCE,
        source_and_epochs=original.realization,qualification_identity=None)
    result=DenseArrayResult(original,proposal,retained_adapter,receipt,Path(facts_path))
    result.revalidate()
    return result,receipt,retained_adapter


def wrap_terminal(fn,lower,problem):
    """Delegate all original effects; lower only explicit retained pair effects."""
    import gemmini_encode as E
    def value(x): return fn.const(x) if isinstance(x,int) else x
    def add(x,y): return fn.add_i(value(x),value(y))
    def mul(x,y): return fn.mul_i(value(x),value(y))
    def wrapped(name,a):
        if name=='gemmini.preload.array':
            return E.preload(fn,E.GARBAGE,add(add(a['acc'],E.ACC),mul(a['continue'],E.ACCUMULATE)),
                a['depth'],a['cols'],a['rows'],a['cols'])
        if name=='gemmini.compute.array':
            return E.compute(fn,a['lhs'],E.GARBAGE,a['rows'],a['depth'],
                problem.tile_k,problem.tile_n,accumulate=True)
        return lower(name,a)
    return wrapped
