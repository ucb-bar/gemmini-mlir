"""Optional two-slot input staging through the original full-K ready owner.

Capacity choices are proposals, not execution laws. Selected ordering facts bind
all mixed-epoch accesses; the original consumer checks actual relocated rows,
source products, final-K stores, ownership drains and finite progress.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob


def _mixed_target_binding(adapter, admission):
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or ordering['status']!='derived'):
        raise ValueError('input pipeline lacks selected original ordering facts')
    cross={(x['older'],x['younger'],x['space']):x for x in ordering['cross_queue']}
    required=[]
    # LOAD precedes current compute in the mixed epoch. The selected owner is
    # conservative: even disjoint LOAD->COMPUTE waits for completion. Keep it.
    for compute in ('COMPUTE_PRELOADED.a_spad','COMPUTE_ACCUMULATE.a_spad'):
        row=cross[('MVIN.spad_addr',compute,'scratchpad')]
        if row['overlap']!='completion' or row['disjoint']!='completion':
            raise ValueError('input pipeline lacks complete LOAD to actual LHS reader ordering')
        required.append(row)
        row=cross[(compute,'MVIN.spad_addr','scratchpad')]
        if row['overlap']!='completion' or row['tracked'] is not True:
            raise ValueError('input pipeline lacks actual LHS read to slot reuse ordering')
        required.append(row)
    for older,younger,space in (
        ('MVIN.spad_addr','MVOUT.acc_addr','scratchpad'),
        ('MVOUT.acc_addr','MVIN.spad_addr','scratchpad'),
        ('PRELOAD.c_addr','MVOUT.acc_addr','accumulator'),
        ('MVOUT.acc_addr','PRELOAD.c_addr','accumulator')):
        row=cross[(older,younger,space)]
        if row['overlap']!='completion' or row['tracked'] is not True:
            raise ValueError('input pipeline lacks selected load/readout/ACC reuse dependency')
        required.append(row)
    for lane in ('execute','store'):
        if ordering['completion_order'][lane]['order']!='in_order':
            raise ValueError('input pipeline lacks ordered compute/store retirement')
    if ordering['same_queue']['store']['release']!='completion':
        raise ValueError('input pipeline lacks store completion before ownership release')
    for lane in ('load','execute','store'):
        entry=record['finite_entries'][lane]
        if entry['status']!='derived' or type(entry['value']) is not int or entry['value']<1:
            raise ValueError('input pipeline lacks finite mixed-queue progress')
    names=('MVIN.spad_addr','COMPUTE_PRELOADED.a_spad','COMPUTE_ACCUMULATE.a_spad',
           'PRELOAD.weight_spad','PRELOAD.c_addr','MVOUT.acc_addr')
    if any(ordering['extents'][n]['status']!='derived' for n in names):
        raise ValueError('input pipeline lacks complete selected access extents')
    identity='selected-disjoint-input-causal-readout:'+digest
    return replace(admission,disjoint_load_qualification_identity=identity),dict(
        ordering_sha256=digest,dependencies=required,
        extents={n:ordering['extents'][n] for n in names},finite_entries=record['finite_entries'],
        configuration='same uniform CONFIG_ST and untransposed primitive dense configuration; no reconfiguration in mixed epochs',
        conservative_order='selected LOAD to COMPUTE disjoint completion wait retained; no overlap inferred',
        ownership='original full drain closes all current reads, future loads and pending stores before reuse')


def maybe_pipeline_input(ready,problem,block,resources,storage,adapter,*,plan,pipeline_admission,source_identity,residency,selection=None):
    if selection is None:
        selection=knob('DENSE_INPUT_PIPELINE',default='off',values=('off','two_slot'),
                       root=Path(__file__).resolve().parents[1])
    if selection=='off':return ready
    from merlin.compile.scheduling.dense_ready_readout import dense_ready_input_choices,schedule_dense_pipeline
    choices=dense_ready_input_choices(problem,resources,storage)
    placement=next((c for c in choices if c.block==block),None)
    if problem.bias or placement is None:
        ready.realization['input_pipeline_selection']=dict(selection=selection,applied=False,
            reason='requires unbiased invariant RHS and two owned LHS slots on the incumbent full-K block')
        return ready
    admission,binding=_mixed_target_binding(adapter,ready.admission)
    # This is an independent invariant-RHS alternative, even when the incumbent
    # amortized residency selector preferred LHS. Keep its exact selected block.
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_operand_hoists,dense_retained_compute_order
    from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    raw=emit_compact_dense(problem,block,storage,adapter,name='dense_retained',dtypes=('i8','i8','i32'),outer_order=('n','m'))
    transforms=dense_retained_operand_hoists(raw,adapter,role='weight')
    transforms+=dense_retained_compute_order(propose_loop_transforms(raw,transforms),adapter,order=('k','n','m'))
    plan=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(block.m,block.k,block.n),
        outer_order=('n','m'),loop_transforms=transforms,pipeline_kind='retained_full_k',pipeline_prefetch=False)
    result=schedule_dense_pipeline(problem,resources,storage,adapter,name='dense_retained',
        dtypes=('i8','i8','i32'),source_identity=source_identity,external_buffers_disjoint=True,
        schedule_plan=plan,pipeline_admission=pipeline_admission,readout_mode='ready',
        readout_admission=admission,input_pipeline_placement=placement,
        static_budget=128,timeout_ms=30000)
    # The common terminal encoder consumes this selected uniform-ready binding.
    # Reconstructing the checked library result must not drop that existing
    # configuration contract and silently reintroduce per-output reload fences.
    result.realization['ready_readout_selected']=True
    result.realization['ready_readout_target_binding']=dict(ready.realization['ready_readout_target_binding'])
    result.realization['input_pipeline_residency']=dict(residency,role='weight',outer_order=('n','m'),selected='input_pipeline_invariant_rhs',incumbent_role=residency['role'])
    result.realization['input_pipeline_selection']=dict(selection=selection,applied=True,
        placement=asdict(placement),target_binding=binding,
        baseline_kernel_sha256=ready.kernel.digest(),actual_kernel_sha256=result.kernel.digest(),
        source='same source product and output/initializer semantics; original owner revalidates actual transformed kernel')
    result.revalidate()
    return result
