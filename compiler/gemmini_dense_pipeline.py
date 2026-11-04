"""Fact-bound alternatives to the compact dense incumbent.

Every alternative consumes an explicit checked library schedule decision. This
module does not infer a speed ordering. The auto policy chooses between compact
and epochs only in regions supported by paired execution evidence.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView, DenseBiasSource
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan, dense_schedule_text, parse_dense_schedule_plan
from merlin.transforms.dense_planning import DenseResources


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def ordering_premises(facts_path, ordering_path=None):
    """Bind the selected ISA operand mapping and derive each epoch-law premise.

    UNKNOWN dependencies retain their drains. The program premises refer to the
    original full completion callback and invocation-private on-chip windows.
    Queue occupancy is taken from selected RTL, not an authored hardware limit.
    """
    facts_path=Path(facts_path)
    ordering_path=Path(ordering_path) if ordering_path else facts_path.parent/'COMMAND_ORDERING_SELECTION.json'
    payload=ordering_path.read_bytes(); selection=json.loads(payload)
    facts_bytes=facts_path.read_bytes(); facts=json.loads(facts_bytes)
    if selection.get('base_facts',{}).get('sha256')!=_hash(facts_bytes):
        raise ValueError('dense ordering selection binds different facts')
    record=selection['ordering']
    digest=_hash(json.dumps(record,sort_keys=True).encode())
    if selection.get('ordering_sha256')!=digest or record.get('status')!='derived':
        raise ValueError('dense requires intact derived command ordering')
    accesses=record['accesses']; extents=record['extents']
    # Bind precisely the RoCC funct and local-address slots used by gemmini_encode.
    slots={
        'MVIN.spad_addr':(2,'rs2','write','load'),
        'MVIN2.spad_addr':(1,'rs2','write','load'),
        'PRELOAD.weight_spad':(6,'rs1','read','execute'),
        'PRELOAD.c_addr':(6,'rs2','write','execute'),
        'COMPUTE_PRELOADED.a_spad':(4,'rs1','read','execute'),
        'MVOUT.acc_addr':(3,'rs2','read','store')}
    for key,(funct,operand,direction,queue) in slots.items():
        a=accesses.get(key,{})
        if (a.get('funct'),a.get('operand'),a.get('direction'),a.get('queue'),a.get('offset'),a.get('width'))!=(funct,operand,direction,queue,0,32):
            raise ValueError('dense actual encoder operand is unbound by selected ordering: '+key)
        if extents.get(key,{}).get('status')!='derived':
            raise ValueError('dense extent unavailable: '+key)
    cross={(a['older'],a['younger'],a['space']):a for a in record['cross_queue']}
    same=record['same_queue'];completion=record['completion_order']
    queue_classes=facts['facts']['timing_structure']['command_classes']['classes']
    evidence={}
    def waits(pairs):
        rows=[cross.get(p,{}) for p in pairs]
        return bool(rows) and all(r.get('overlap')=='completion' for r in rows)
    writes=('MVIN.spad_addr','MVIN2.spad_addr')
    reads=('PRELOAD.weight_spad','COMPUTE_PRELOADED.a_spad')
    evidence['local_read_after_write']=[cross.get((w,r,'scratchpad')) for w in writes for r in reads]
    evidence['local_write_after_read']=[cross.get((r,w,'scratchpad')) for w in writes for r in reads]
    flags={
        'local_read_after_write':waits([(w,r,'scratchpad') for w in writes for r in reads]),
        'local_write_after_read':waits([(r,w,'scratchpad') for w in writes for r in reads]),
        'ordered_accumulator_writes':completion.get('execute',{}).get('order')=='in_order' or same.get('execute',{}).get('release')=='completion',
        'accumulator_result_visibility':waits([('PRELOAD.c_addr','MVOUT.acc_addr','accumulator')]),
        'ordered_configuration_capture':True,
        'finite_queue_progress':True,
        'complete_device_drain':True,
        'exclusive_private_state':True}
    by_code={int(code):queue for item in record['funct_queue'] for code,queue in item.get('by_code',{}).items()}
    for key in reads+('PRELOAD.c_addr','MVOUT.acc_addr'):
        scale=extents[key].get('scale')
        if scale is not None and (by_code.get(scale['code'])!=accesses[key]['queue']
                or same.get(accesses[key]['queue'],{}).get('release') not in ('issue','completion')):
            flags['ordered_configuration_capture']=False
    # Configuration capture for the constant-stride mvin streams and per-store scale.
    for code,queue in ((0,'execute'),(1,'load'),(2,'store')):
        if by_code.get(code)!=queue or same.get(queue,{}).get('release') not in ('issue','completion'):
            flags['ordered_configuration_capture']=False
    pending={}
    for queue,minimum in (('execute',2),('load',1),('store',1)):
        entry=queue_classes.get(queue,{}).get('entries',{})
        if entry.get('status')!='derived' or type(entry.get('value')) is not int or entry['value']<minimum:
            raise ValueError('dense finite issue capacity is not derived for '+queue)
        pending[queue]=entry['value']
    evidence.update(ordered_accumulator_writes=completion.get('execute'),
        accumulator_result_visibility=cross.get(('PRELOAD.c_addr','MVOUT.acc_addr','accumulator')),
        ordered_configuration_capture={'funct_queue':record['funct_queue'],'same_queue':same},
        finite_queue_progress={k:queue_classes[k]['entries'] for k in pending},
        complete_device_drain='same E.fence callback as compact route, and a final invocation fence',
        exclusive_private_state='same invocation-private allocation as compact; only its assigned operand rows are split into two windows')
    return flags,pending,{'selection_sha256':_hash(payload),'ordering_sha256':digest,
        'flags':flags,'basis':evidence,'execution_qualification':'candidate; requires matched target execution'}


def plan_pipeline(incumbent,adapter,problem,geometry,facts_path,*,mode='panel_epochs',bias=False,initial=False,outer_order=('m','n')):
    if mode not in ('panel_drained','panel_epochs'):
        raise ValueError('unknown dense pipeline alternative '+str(mode))
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    block=incumbent.choice.block
    if block.k!=1 or 2*block.m*edge>split or 2*block.n*edge>operand_rows-split:
        raise ValueError('dense two-window alternative does not fit the incumbent block')
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=block.m,max_k_blocks=1,max_n_blocks=block.n,
        partial_output_tiles=True,partial_output_blocks=True)
    flags,pending,receipt=ordering_premises(facts_path)
    # The bound declarations describe the candidate adapter and the same source-
    # qualified drain. They are premises to be tested, never a claimed machine PASS.
    admission=DensePanelOverlapAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-command-ordering:'+receipt['ordering_sha256']+':gemmini-encoder-v1',
        4096,4096,pending['execute']//2,pending['load'],
        max_issued_readouts=block.m*block.n,pending_readouts=pending['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    epoch_law=DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        'selected-command-ordering:'+receipt['ordering_sha256'],**flags) if mode=='panel_epochs' else None
    from gemmini_dense import seed_source
    bias_source=seed_source(problem.m,problem.n,bias=bias,initial=initial)
    source_identity=_hash(json.dumps({'source':'row-major-source-ordered-dense',
        'problem':asdict(problem),'initializer':None if bias_source is None else asdict(bias_source),
        'numerical_policy':adapter.numerical_policy_identity},sort_keys=True).encode())
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=outer_order,pipeline_prefetch=True,bias_source=bias_source)
    result=schedule_dense_pipeline(problem,resources,storage,adapter,name='dense_'+mode,
        source_identity=source_identity,dtypes=('i8','i8','i32'),static_budget=256,
        external_buffers_disjoint=True,schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
        pipeline_admission=admission,batch_readouts=True,bias_source=bias_source,
        epoch_law=epoch_law,timeout_ms=60000)
    result.revalidate()
    receipt.update(mode=mode,source_identity=source_identity,block=(block.m,block.k,block.n),
        schedule_ir=dense_schedule_text(plan))
    return result,receipt
