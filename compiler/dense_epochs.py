"""Fact-backed, opt-in completion epochs for the shared two-window dense path.

This module does not infer independence from source shapes. The selected
ReservationStation ordering probes bind every RAW/WAR pair and ordered ACC
update; selected finite dispatch queues bind backpressure. The library proves
that merged steps consume loads before their slots are overwritten. Only its
remaining explicit drains call the terminal full-device fence. No counts are a
latency claim, and the incumbent remains synchronous compact_dense.
"""
import json
from pathlib import Path
from hashlib import sha256
from dataclasses import replace, asdict
from merlin.transforms.dense_planning import dense_blocks
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan,dense_schedule_text,parse_dense_schedule_plan

# Merge this declaration into TUNING_KNOBS.json; default remains off.
KNOB_DECLARATION={'dense_completion_epochs':{
    'values':['off','two_window_drained','two_window_epochs'],
    'mechanism':'dense_panel_epochs with selected RTL dependency ordering',
    'note':'Same checked two-window schedule; epochs removes only fact-ordered K-step drains. off preserves compact incumbent.'}}
READOUT_KNOB_DECLARATION={'dense_readout_schedule':{
    'values':['off','segmented_drained','segmented_ready'],
    'mechanism':'dense_segmented_pipeline and dense_ready_readout',
    'note':'Same full-LHS/two-RHS plan; ready consumes causal ACC completion and store snapshot facts. Capacity or unsupported numerical boundaries retain incumbent. off remains default.'}}


def selected_law(adapter):
    root=Path(__file__).resolve().parent
    candidates=(root/'command_ordering.json',root.parent/'COMMAND_ORDERING_SELECTION.json')
    selected=next((p for p in candidates if p.exists()),None)
    if selected is None: raise ValueError('dense epochs require packaged selected command ordering facts')
    raw=selected.read_bytes(); selection=json.loads(raw); r=selection['ordering']
    ordering_hash=sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
    if ordering_hash!=selection.get('ordering_sha256') or r.get('status')!='derived':
        raise ValueError('dense ordering selection digest/status is not intact')
    fact_bytes=(root/'facts.json').read_bytes(); facts=json.loads(fact_bytes)['facts']
    if sha256(fact_bytes).hexdigest()!=selection['base_facts']['sha256']:
        raise ValueError('dense ordering selection names different selected facts bytes')
    cross={(x['older'],x['younger'],x['space']):x for x in r.get('cross_queue',[])}
    loads=('MVIN.spad_addr','MVIN2.spad_addr')
    readers=('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad')
    table={}
    def wait(older,younger,space):
        row=cross.get((older,younger,space))
        # tracked=None is usable only for an unconditional completion wait;
        # a dependency inferred from an untracked overlap alone remains UNKNOWN.
        okay=bool(row and row.get('overlap')=='completion' and
            (row.get('tracked') is True or row.get('disjoint')=='completion'))
        return okay,row
    def pairs(name,items):
        rows=[wait(*item) for item in items]
        table[name]={'established':all(ok for ok,_ in rows),'evidence':[row for _,row in rows]}
    pairs('local_read_after_write',[(l,c,'scratchpad') for l in loads for c in readers])
    pairs('local_write_after_read',[(c,l,'scratchpad') for l in loads for c in readers])
    pairs('accumulator_result_visibility',[('PRELOAD.c_addr','MVOUT.acc_addr','accumulator')])
    order=r.get('completion_order',{}); same=r.get('same_queue',{})
    table['ordered_accumulator_writes']={'established':order.get('execute',{}).get('order')=='in_order',
        'evidence':order.get('execute')}
    # Configuration is immutable throughout the compute stream. Store queue
    # release on completion additionally protects per-column scale updates.
    config={int(code):queue for item in r['funct_queue'] for code,queue in item.get('by_code',{}).items()}
    configuration_ok=all(config.get(code)==queue and same.get(queue,{}).get('release') in ('issue','completion')
        for code,queue in ((0,'execute'),(1,'load'),(2,'store')))
    table['ordered_configuration_capture']={'established':configuration_ok,'evidence':{'config_queue':config,'same_queue':same}}
    classes=facts['timing_structure']['command_classes']['classes']
    entries={name:classes[name]['entries'] for name in ('execute','load','store')}
    finite=all(v.get('status')=='derived' and type(v.get('value')) is int and v['value']>0 for v in entries.values())
    finite=finite and entries['execute']['value']>=2
    table['finite_queue_progress']={'established':finite,'evidence':entries}
    table['complete_device_drain']={'established':True,'evidence':'same terminal Gemmini full-device fence as incumbent'}
    table['exclusive_private_state']={'established':True,'evidence':'disjoint assigned LHS/RHS banks and private ACC output block through closing fence'}
    identity='selected-rtl-dense-epochs:'+ordering_hash
    law=DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,identity,
        **{name:item['established'] for name,item in table.items()})
    return law,table,{name:value['value'] for name,value in entries.items()},identity


def build_epoch_schedule(problem,resources,storage,adapter,options,*,bias_source=None):
    """Plan and consume the actual library two-window schedule, never move waits by hand."""
    law,table,entries,identity=selected_law(adapter)
    # Each reduction step has exactly two owned slots; blocks retain the same
    # bounded M/N candidates as the panel knob, intersected with physical stores.
    r=replace(resources,max_k_blocks=1)
    legal=[block for block in dense_blocks(problem,r)
        if block.k==1 and 2*block.m*problem.tile_m<=storage.lhs.capacity
        and 2*block.n*problem.tile_k<=storage.rhs.capacity and storage.admits(problem,block,r)]
    if not legal: raise ValueError('no requested dense block fits two disjoint operand windows')
    block=min(legal,key=lambda b:(b.requested_operand_bytes,-b.m*b.n,b.m,b.n))
    # These are compiler issue budgets, not fabricated pending queue capacities.
    budget=4096
    admission=DensePanelOverlapAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity,budget,budget,max(1,entries['execute']//2),entries['load'],
        max_issued_readouts=block.m*block.n,pending_readouts=entries['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    source_identity=sha256(json.dumps({'problem':asdict(problem),'dtype':['i8','i8','i32'],
        'bias':None if bias_source is None else asdict(bias_source)},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(problem,r,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,1,block.n),
        outer_order=tuple(options['outer_order']),pipeline_prefetch=True,pipeline_kind='two_window',bias_source=bias_source)
    result=schedule_dense_pipeline(problem,r,storage,adapter,name='gemmini_dense_epochs',
        dtypes=('i8','i8','i32'),source_identity=source_identity,external_buffers_disjoint=True,
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),pipeline_admission=admission,
        batch_readouts=True,static_budget=256,timeout_ms=5000,bias_source=bias_source,
        epoch_law=law if options['completion_epochs']=='two_window_epochs' else None)
    result.revalidate()
    return result,{'block':asdict(block),'law_flags':table,'unknown_premises':list(law.unknown()),
        'ordering_identity':identity,'selected_path':options['completion_epochs'],
        'schedule_ir':dense_schedule_text(plan),'after_kernel':result.kernel.text(),
        'batch_readouts':True,'actual_maxima':{'m':r.max_m_blocks,'k':r.max_k_blocks,'n':r.max_n_blocks},
        'two_window_storage':{'lhs_rows':2*block.m*problem.tile_m,
            'rhs_rows':2*block.n*problem.tile_k,'accumulator_rows':block.m*block.n*problem.tile_m,
            'operand_bytes':2*(block.m*problem.tile_m*storage.lhs.unit_bytes+
                block.n*problem.tile_k*storage.rhs.unit_bytes),
            'accumulator_bytes':block.m*block.n*problem.tile_m*storage.accumulator.unit_bytes}}


def build_readout_schedule(problem,resources,storage,adapter,options,*,bias_source=None,timeout_ms=5000):
    """Consume a separate segmented plan and its optional causal ready readout.

    The paired controls retain complete LHS panels and stage RHS segments into
    two disjoint slots. Ready readout changes only the checked final-K frontier;
    it is not the two-window epoch drain-removal mechanism. Selected scratchpad
    RAW/WAR, execute completion order and ACC producer/readout waits establish
    the combined execution law. Every completed epoch still uses the original
    full-device fence. Counts are compiler issue bounds or selected queue entries,
    never a timing result. The full typed decision and before/after counted IR
    are returned so an A/B run can inspect actual transformations.
    """
    from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission,DenseReadyReadoutAdmission
    from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedAdmission,dense_segmented_blocks
    law,table,entries,identity=selected_law(adapter)
    if law.unknown():
        raise ValueError('segmented ready schedule lacks selected ordering premises: '+repr(law.unknown()))
    if problem.tiles[1]<2:
        raise ValueError('segmented readout requires at least two primitive K tiles')
    resources=replace(resources,max_k_blocks=min(resources.max_k_blocks,problem.tiles[1]-1))
    candidates=list(dense_segmented_blocks(problem,resources,storage))
    if not candidates:
        raise ValueError('full LHS plus two RHS segments do not fit requested segmented domain')
    block=min(candidates,key=lambda b:(b.requested_operand_bytes,-b.m*b.n,-b.k,b.m,b.n))
    budget=4096
    common=dict(max_issued_readouts=block.m*block.n,pending_readouts=entries['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    ids=(adapter.configuration_identity,adapter.numerical_policy_identity)
    pending=(max(1,entries['execute']//2),entries['load'])
    overlap=DensePanelOverlapAdmission(*ids,identity+':disjoint-future-operand-loads',budget,budget,*pending,**common)
    reduction=DenseRetainedReductionAdmission(*ids,identity+':ordered-accumulator-commits',budget,budget,*pending,**common)
    admission=DenseSegmentedAdmission(overlap,reduction,identity+':combined-selected-raw-war-acc-config-queues')
    src=sha256(json.dumps({'problem':asdict(problem),'dtype':['i8','i8','i32'],
        'bias':None if bias_source is None else asdict(bias_source)},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=src,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=('m','n'),pipeline_prefetch=True,pipeline_kind='retained_lhs_segmented_k',bias_source=bias_source)
    text=dense_schedule_text(plan)
    kwargs=dict(name='gemmini_dense_readout',dtypes=('i8','i8','i32'),source_identity=src,
        external_buffers_disjoint=True,schedule_plan=parse_dense_schedule_plan(text),
        pipeline_admission=admission,batch_readouts=True,static_budget=256,timeout_ms=timeout_ms,bias_source=bias_source)
    baseline=schedule_dense_pipeline(problem,resources,storage,adapter,**kwargs)
    result=baseline
    if options['readout_schedule']=='segmented_ready':
        readout=DenseReadyReadoutAdmission(*ids,identity+':causal-acc-producer-completion',
            identity+':ordered-store-configuration-capture',block.m*block.n,entries['store'])
        result=schedule_dense_pipeline(problem,resources,storage,adapter,**kwargs,
            readout_mode='ready',readout_admission=readout)
    result.revalidate()
    return result,{'block':asdict(block),'law_flags':table,'unknown_premises':list(law.unknown()),
        'ordering_identity':identity,'selected_path':options['readout_schedule'],
        'schedule_ir':text,'before_kernel':baseline.kernel.text(),'after_kernel':result.kernel.text(),
        'before_realization':baseline.realization,'after_realization':result.realization}
