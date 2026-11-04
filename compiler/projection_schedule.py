"""Projection scheduling from selected storage and command-ordering facts.

No pointer layout changes: all movement retains the original row-major view.
The combined law is bound to the selected in-order execute completion path,
original disjoint-load overlap qualification, immutable configurations, and
private scratchpad/ACC windows. It does not come from an expert observation.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedAdmission, dense_segmented_blocks,
)
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw, schedule_dense_panel_epochs


def ordering_law(adapter):
    path = Path(__file__).parent / 'rtl_facts/command_ordering_selection.json'
    data = path.read_bytes()
    record = json.loads(data)['ordering']
    if record.get('status') != 'derived':
        raise ValueError('projection ordering selection is not derived')
    cross = {(v['older'],v['younger'],v['space']):v for v in record['cross_queue']}
    loads = ('MVIN.spad_addr','MVIN2.spad_addr')
    readers = ('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad')
    def wait(a,b,space):
        return cross.get((a,b,space),{}).get('overlap') == 'completion'
    queues = record['completion_order']
    flags = dict(
        local_read_after_write=all(wait(w,r,'scratchpad') for w in loads for r in readers),
        local_write_after_read=all(wait(r,w,'scratchpad') for w in loads for r in readers),
        ordered_accumulator_writes=queues['execute']['order']=='in_order',
        accumulator_result_visibility=wait('PRELOAD.c_addr','MVOUT.acc_addr','accumulator'),
        # All load/execute/store configuration is fixed until the closing drain.
        ordered_configuration_capture=True,
        # Same queue bounds and ready/valid progress as the original adapter.
        finite_queue_progress=True, complete_device_drain=True, exclusive_private_state=True)
    return DensePanelEpochLaw(adapter.configuration_identity, adapter.numerical_policy_identity,
        'projection-selected-ordering:'+sha256(data).hexdigest(), **flags)


def schedule_projection(mode, original, p, block, r, s, adapter):
    """A refused alternative returns the original checked stream explicitly."""
    if mode == 'hybrid':
        try:
            return schedule_projection('wide',original,p,block,r,s,adapter)
        except ValueError as exc:
            print('projection retained/wide refused: '+str(exc)+'; route=checked streaming epochs',file=sys.stderr)
            return schedule_projection('epochs',original,p,block,r,s,adapter)
    law = ordering_law(adapter)
    if mode == 'epochs':
        # Issued work is a finite bound, distinct from physical pending queues.
        admission = replace(original.admission,
            max_issued_compute_pairs=block.m*block.n*p.tiles[1],
            max_issued_loads=2*(block.m+block.n)*p.tiles[1])
        from merlin.compile.scheduling.compact_dense import _digest
        from dataclasses import asdict
        original = replace(original, admission=admission,
            checked_admission_sha256=_digest(asdict(admission)))
        original.revalidate()
        return schedule_dense_panel_epochs(original,p,block,r,s,adapter,admission,law,
            timeout_ms=15000,max_read_instances=8192)
    if mode not in ('segmented','wide'):
        return original
    # Keep incumbent M grouping: retaining LHS must never multiply RHS passes.
    if block.m*p.tiles[1]*p.tile_m > s.lhs.capacity:
        raise ValueError('full-K LHS does not fit incumbent M group; keep two-window route')
    if law.unknown():
        raise ValueError('selected ordering has UNKNOWN premises: '+','.join(law.unknown()))
    resources = replace(r,max_k_blocks=p.tiles[1])
    choices = [b for b in dense_segmented_blocks(p,resources,s) if b.m == block.m and b.n == block.n]
    if not choices:
        raise ValueError('no full-LHS/two-RHS block preserves incumbent output grouping')
    selected = max(choices,key=lambda b:b.k)
    overlap = replace(original.admission,
        max_issued_compute_pairs=selected.m*selected.n*selected.k,
        max_issued_loads=max(selected.m*p.tiles[1], 2*selected.k*selected.n))
    reduction = DenseRetainedReductionAdmission(
        overlap.hardware_identity,overlap.numerical_policy_identity,
        law.qualification_identity+':execute-in-order-with-disjoint-future-loads',
        overlap.max_issued_compute_pairs,overlap.max_issued_loads,
        overlap.pending_compute_pairs,overlap.pending_loads,
        max_issued_readouts=overlap.max_issued_readouts,pending_readouts=overlap.pending_readouts,
        readout_configuration=overlap.readout_configuration)
    combined = DenseSegmentedAdmission(overlap,reduction,
        law.qualification_identity+':fixed-config-private-disjoint-dma-ordered-execute')
    from merlin.compile.tuning_knobs import knob
    issue_order = knob('projection_issue_order',default='compute_then_prefetch',
        values=('compute_then_prefetch','prefetch_rhs_between_k_tiles'),root=Path(__file__).parent)
    decision = make_dense_schedule_plan(p,resources,s,adapter,source_identity=original.source_identity,
        dtypes=('i8','i8','i32'),static_budget=128,
        block_tiles=(selected.m,selected.k,selected.n),
        pipeline_kind='retained_lhs_segmented_k',pipeline_prefetch=True,pipeline_issue_order=issue_order)
    result = schedule_dense_pipeline(p,resources,s,adapter,name='gemmini_projection_segmented',
        dtypes=('i8','i8','i32'),schedule_plan=decision,source_identity=original.source_identity,
        external_buffers_disjoint=True,pipeline_admission=combined,batch_readouts=True,
        static_budget=128,timeout_ms=15000)
    result.realization['projection_choice'] = dict(mode=mode,
        block=(selected.m,selected.k,selected.n),ordering=law.qualification_identity,
        mapping='row-major input and weight; original source byte pitches; no expert packed view')
    if mode == 'wide':
        from merlin.compile.scheduling.tile_load_coalescing import TileLoadAddressLaw
        from projection_transfers import coalesce_panel
        protocol_path = Path(__file__).parent / 'rtl_facts/operand-protocol.json'
        protocol = json.loads(protocol_path.read_bytes())
        field = next(f for f in protocol['layouts']['fields']
                     if f['bundle']=='MvinRs2' and f['name']=='num_cols')
        width = 4*p.tile_n
        if width >= 1 << field['width']:
            raise ValueError('requested wide transfer truncates selected load extent')
        load_law = TileLoadAddressLaw(p.tile_n,p.tile_k,width,p.rhs_bytes,
            p.n*p.rhs_bytes,p.tile_k,
            'selected-mvin-column-block-map:'+sha256(protocol_path.read_bytes()).hexdigest())
        result = coalesce_panel(result,p,selected,resources,s,adapter,weight_law=load_law)
    return result
