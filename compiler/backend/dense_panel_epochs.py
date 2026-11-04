"""Optional fact-backed two-window dense completion epochs.

The shared epoch pass consumes its own checked two-window source realization.
It never applies a two-window proof to the incumbent retained/capacity kernel.
The selected alternative changes staging; the same primitive encoder and
real full-device completion callback lower the checked final commands.
"""
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.transforms.dense_planning import DenseProblem,DenseResources,dense_blocks
from . import isa

# Selected public RTL ordering facts, not facts inferred from candidate code.
# COMMAND_ORDERING_SELECTION.json ordering_sha256 identifies their extraction.
ORDERING_ID='c339790914b49e5bd2ae131e105257e01e99c0c096ac7cb2e8476340a15ca963'
CROSS_QUEUE=(('COMPUTE_PRELOADED.a_spad', 'MVIN.spad_addr', 'scratchpad', 'completion'), ('COMPUTE_PRELOADED.a_spad', 'MVIN2.spad_addr', 'scratchpad', 'completion'), ('MVIN.spad_addr', 'COMPUTE_PRELOADED.a_spad', 'scratchpad', 'completion'), ('MVIN.spad_addr', 'PRELOAD.weight_spad', 'scratchpad', 'completion'), ('MVIN2.spad_addr', 'COMPUTE_PRELOADED.a_spad', 'scratchpad', 'completion'), ('MVIN2.spad_addr', 'PRELOAD.weight_spad', 'scratchpad', 'completion'), ('PRELOAD.weight_spad', 'MVIN.spad_addr', 'scratchpad', 'completion'), ('PRELOAD.weight_spad', 'MVIN2.spad_addr', 'scratchpad', 'completion'), ('PRELOAD.c_addr', 'MVOUT.acc_addr', 'accumulator', 'completion'))
QUEUE_FACTS={'execute': {'release': 'issue', 'completion': 'in_order', 'dispatch_entries': 16}, 'load': {'release': 'issue', 'completion': 'out_of_order', 'dispatch_entries': 8}, 'store': {'release': 'completion', 'completion': 'in_order', 'dispatch_entries': 4}}
CONFIG_QUEUES={'0': 'execute', '1': 'load', '2': 'store', '3': 'store'}


def epoch_law(adapter):
    """Bind every shared-law flag to the selected original target fact."""
    waits={(a,b,space):mode for a,b,space,mode in CROSS_QUEUE}
    loads=('MVIN.spad_addr','MVIN2.spad_addr')
    readers=('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad')
    flags=dict(
        local_read_after_write=all(waits.get((w,r,'scratchpad'))=='completion' for w in loads for r in readers),
        local_write_after_read=all(waits.get((r,w,'scratchpad'))=='completion' for w in loads for r in readers),
        ordered_accumulator_writes=QUEUE_FACTS['execute']['completion']=='in_order',
        accumulator_result_visibility=waits.get(('PRELOAD.c_addr','MVOUT.acc_addr','accumulator'))=='completion',
        ordered_configuration_capture=CONFIG_QUEUES.get('0')=='execute' and CONFIG_QUEUES.get('1')=='load'
            and CONFIG_QUEUES.get('2')=='store' and all(QUEUE_FACTS[q]['release'] in ('issue','completion') for q in QUEUE_FACTS),
        finite_queue_progress=QUEUE_FACTS['execute']['dispatch_entries']>=2 and QUEUE_FACTS['load']['dispatch_entries']>=1,
        complete_device_drain=True,exclusive_private_state=True)
    # Program premises: lower_dense emits the same real fence callback and
    # owns both assigned windows/ACC until the final completion returns.
    return DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-rtl-panel-epochs:'+ORDERING_ID,**flags)


def resolve_multi_m_epochs(selection=None):
    if selection is None:
        default=knob('DENSE_MULTI_M_EPOCHS',default='off',values=('off','full_m'),root=Path(isa.__file__).resolve().parents[1])
        selection=os.environ.get('DENSE_MULTI_M_EPOCHS',default)
    if selection not in ('off','full_m'):raise ValueError('DENSE_MULTI_M_EPOCHS must be off or full_m')
    return selection


def panel_epoch_region(m,k,n,*,multi_m_epochs='off'):
    """Coarse profitability hypothesis; source/capacity legality stays separate.

    Require sixteen complete K tiles and eight complete N tiles so a long
    reduction and a full output group can amortize repeated LHS staging.
    Explicit 'on' still exposes the broader legal family for paired trials.
    """
    return 0<m<=(4*isa.DIM if resolve_multi_m_epochs(multi_m_epochs)=='full_m' else isa.DIM) and k>=16*isa.DIM and n>=8*isa.DIM


def panel_epoch_schedule(m,k,n,bias=False,column_scales=False,n_tiles=8,completion_epochs=True,*,lhs_retention=None,issue_order=None,multi_m_epochs=None):
    selected=lhs_retention
    if selected is None:
        default=knob('DENSE_EPOCH_LHS_RETAIN',default='regional',values=('off','on','regional'),root=Path(isa.__file__).resolve().parents[1])
        selected=os.environ.get('DENSE_EPOCH_LHS_RETAIN',default)
    if selected not in ('off','on','regional'):raise ValueError('DENSE_EPOCH_LHS_RETAIN must be off, on or regional')
    from .dense_issue_order import resolve_issue_order
    order=resolve_issue_order(issue_order)
    return _panel_epoch_schedule(m,k,n,bias,column_scales,n_tiles,completion_epochs,selected,order,resolve_multi_m_epochs(multi_m_epochs))


@lru_cache(maxsize=256)
def _panel_epoch_schedule(m,k,n,bias,column_scales,n_tiles,completion_epochs,lhs_retention,issue_order="current",multi_m_epochs="off"):
    """A checked alternative for one-M-tile products, common output scaling."""
    if type(n_tiles) is not int or n_tiles not in (4,8,16,32):
        raise ValueError('dense epoch N tiles must be 4, 8, 16 or 32')
    if type(completion_epochs) is not bool:raise ValueError('dense epoch completion must be boolean')
    if column_scales:return None,'panel epochs require common output scale'
    # The opt-in broadens only regional full-M ownership; explicit on stays broad.
    full_m_region=multi_m_epochs=='full_m' and lhs_retention=='regional'
    multi_m=isa.DIM<m<=4*isa.DIM and (lhs_retention=='on' or full_m_region)
    if (m>isa.DIM and not multi_m) or m<=0 or k<=isa.DIM or n<=isa.DIM:
        return None,'panel epochs require one M tile or explicit resident on up to four M tiles, and multiple K/N tiles'
    if multi_m and issue_order!='current':
        return None,'multi-M resident epochs currently require the current shared issue order'
    from .dense import _adapter
    adapter=_adapter();dim=isa.DIM
    p=DenseProblem(m,k,n,dim,dim,dim,1,1,4,bias=bias)
    s=DenseLoopStorage(DenseMemoryView('lhs','spad',dim,0,isa.SPAD_ROWS//2),
        DenseMemoryView('rhs','spad',dim,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),
        DenseMemoryView('acc','acc',4*dim,0,isa.ACC_ROWS))
    r=DenseResources(isa.SPAD_ROWS*dim,isa.ACC_ROWS*4*dim,max_m_blocks=min(4,p.tiles[0]) if multi_m else 1,max_n_blocks=n_tiles,max_k_blocks=1,
        partial_output_tiles=True,partial_output_blocks=True)
    choices=[b for b in dense_blocks(p,r) if b.k==1 and s.admits(p,b,r)
        and 2*b.m*dim<=s.lhs.capacity and 2*b.n*dim<=s.rhs.capacity
        # Full-K retained LHS must fit before selecting a blocked multi-M source.
        # Only exact counted M groups are admitted by this bounded adapter.
        and (not multi_m or (b.m*p.tiles[1]*dim<=s.lhs.capacity and p.tiles[0]%b.m==0))]
    if not choices:return None,'panel epochs require two capacity-legal operand windows'
    block=min(choices,key=lambda b:(b.loads,-b.m*b.n,b.m,b.n))
    if multi_m and lhs_retention=='regional' and block.m!=p.tiles[0]:
        return None,'full-M regional epochs require one complete original M group'
    # Frozen profitability predicate: require an actual saved LHS reload.
    # Complete resident capacity/source/ordering checks remain independent.
    if lhs_retention=='regional' and not (panel_epoch_region(m,k,n,multi_m_epochs=multi_m_epochs) and p.tiles[2]>block.n):
        if multi_m:return None,'full-M regional epochs require the original long-K/output-reuse region'
        return _panel_epoch_schedule(m,k,n,bias,column_scales,n_tiles,completion_epochs,'off','current',multi_m_epochs)
    law=epoch_law(adapter)
    if law.unknown():return None,'selected panel epoch facts UNKNOWN: '+', '.join(law.unknown())
    # Issued work may exceed pending queue capacity. The fact-backed ready/valid
    # queues backpressure real submissions; the shared pass bounds whole epochs.
    admission=DensePanelOverlapAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        law.qualification_identity+':disjoint-owned-windows',p.tiles[1]*block.m*block.n,
        p.tiles[1]*(block.m+block.n),2,8,max_issued_readouts=block.m*block.n,
        pending_readouts=1,readout_configuration='ordered_snapshot_before_store_issue')
    source=sha256(json.dumps([m,k,n,bias,'i8-i8-i32','two-window-epochs']).encode()).hexdigest()
    bias_source=DenseBiasSource(n,'i32') if bias else None
    plan=make_dense_schedule_plan(p,r,s,adapter,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=256,block_tiles=(block.m,block.k,block.n),pipeline_kind='two_window',
        pipeline_prefetch=True,bias_source=bias_source)
    if lhs_retention in ('on','regional'):
        from .dense_lhs_retention import retained_lhs_epoch_schedule
        try:
            result=retained_lhs_epoch_schedule(p,block,r,s,adapter,admission,law,source,completion_epochs=completion_epochs,issue_order=issue_order)
        except ValueError as refusal:
            if multi_m:return None,'multi-M resident source refused: '+str(refusal)
            # A refused optional resident source retains the original checked
            # epoch realization, with exactly the normal source/proof path.
            result,reason=_panel_epoch_schedule(m,k,n,bias,column_scales,n_tiles,completion_epochs,'off','current',multi_m_epochs)
            if result is None:return result,reason
            from dataclasses import replace
            return replace(result,realization={**result.realization,'resident_lhs_refusal':str(refusal)}),None
        result.realization['selected_panel_epoch_adapter']=dict(ordering_identity=ORDERING_ID,
            block_tiles=[block.m,block.k,block.n],max_n_tiles=n_tiles,completion_epochs=completion_epochs,
            issued_pairs=admission.max_issued_compute_pairs,issued_loads=admission.max_issued_loads,
            pending_pairs=2,pending_loads=8,staging='complete immutable LHS; two alternating RHS K-slice windows',
            program_premises=['same real full-device fence callback','exclusive assigned operand/accumulator state'],
            law_flags={flag:getattr(law,flag) for flag in law.__dataclass_fields__ if type(getattr(law,flag)) is bool})
        result.revalidate();return result,None
    try:
        result=schedule_dense_pipeline(p,r,s,adapter,name='dense_panel_epochs',dtypes=('i8','i8','i32'),
            source_identity=source,external_buffers_disjoint=True,schedule_plan=plan,pipeline_admission=admission,
            bias_source=bias_source,readout_mode='separate',batch_readouts=True,epoch_law=law if completion_epochs else None,static_budget=256,timeout_ms=30000)
    except ModuleNotFoundError as exc:
        if exc.name!='z3':raise
        from .dense_panel_epochs_finite import finite_panel_epoch_schedule
        try:
            result=finite_panel_epoch_schedule(p,block,r,s,adapter,admission,law,plan,source,completion_epochs=completion_epochs)
        except ValueError as refusal:
            return None,'finite panel epoch checker refused: '+str(refusal)
    result.realization['selected_panel_epoch_adapter']=dict(ordering_identity=ORDERING_ID,
        block_tiles=[block.m,block.k,block.n],max_n_tiles=n_tiles,completion_epochs=completion_epochs,issued_pairs=admission.max_issued_compute_pairs,
        issued_loads=admission.max_issued_loads,pending_pairs=2,pending_loads=8,
        law_flags={flag:getattr(law,flag) for flag in law.__dataclass_fields__ if type(getattr(law,flag)) is bool},
        staging='two alternating K-slice windows; LHS reloads at each output group',
        program_premises=['same real full-device fence callback','exclusive assigned operand/accumulator state'])
    result.revalidate();return result,None


panel_epoch_schedule.cache_clear=_panel_epoch_schedule.cache_clear
panel_epoch_schedule.cache_info=_panel_epoch_schedule.cache_info
