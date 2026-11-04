"""Selected WS array persistence bound to the shared convolution M grouping.

Gemmini's documented WS funct5 reuses previous B, while funct4 captures B.
The selected decode/ordering record must independently match those endpoints.
The shared group owner proves identical K/N RHS across a pure bounded M run;
no load, configuration, source K reorder, or completion is authored here.
"""
from dataclasses import replace,asdict
import hashlib,json
from pathlib import Path
from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhases,ConvolutionRhsPhaseLaw,plan_convolution_schedule,
    plan_convolution_storage_alternative)
from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission
from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw

ISA_SOURCE='https://github.com/ucb-bar/gemmini#re-use-previous-preloads'

def _names(value):
    if isinstance(value,dict):
        yield value.get('names',{})
        for child in value.values():yield from _names(child)
    elif isinstance(value,list):
        for child in value:yield from _names(child)

def admit_regional_multirow(source, *, policy, retention_mode):
    """Restrict auto to the incumbent rolling region and one M tile per row.

    This admission precedes the shared storage/order proofs below. It carries
    one public paired timing point; no empirical crossover is asserted.
    """
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    p=source.plan; h,w=p.window.output_spatial
    resources=dict(incumbent_regional_rolling=int(policy=='regional' and
        retention_mode=='rolling_rhs'), output_rows=h,
        width_fits_primitive=int(w<=p.primitive))
    facts=PolicyFacts(AnalysisStamp(source.identity,
        PipelineContext(resources=str(p.primitive)),'convolution_source'),
        resources=tuple(sorted(resources.items())))
    guards=(ResourcePredicate('incumbent_regional_rolling',1),
        ResourcePredicate('output_rows',2),ResourcePredicate('width_fits_primitive',1))
    admitted=all(g.match(facts) is True for g in guards)
    return admitted,dict(policy='regional-multirow-v1',resources=resources,
        output_width=w,primitive=p.primitive,source_identity=source.identity,
        guards=[dict(resource=g.resource,at_least=g.at_least) for g in guards],
        admitted=admitted,public_cycle_points=1,crossover_validated=False)

def admit_regional_wide_rows(source, *, policy, retention_mode):
    """Conservative two-M-tile row region; the actual shared owner still proves it.

    Capacity here uses the existing dense RHS physical-slot planner. It grants
    no source/order permission: callers must consume plan_array_retention with
    multirow=True, complete unphased RHS and a checked multirow output group.
    """
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    from gemmini_conv_policy import prepare_rolling_input
    p=source.plan;h,w=p.window.output_spatial
    columns=len(source.column_intervals)
    rows_bound=_multirow_bound(p,columns)
    _,ring=prepare_rolling_input(source)
    rhs_fit=None;rhs_base=None
    if ring is not None and rows_bound>=2:
        slots=ring.slots+(rows_bound-1)*p.window.stride[0]
        end=ring.base+slots*ring.rows_per_source_row
        rhs_base=(end+p.channel_tile-1)//p.channel_tile*p.channel_tile
        kh,kw=p.window.kernel
        chunks=kh*kw*((p.window.input_shape[-1]+p.channel_tile-1)//p.channel_tile)
        if rhs_base<p.operand_rows:
            rhs_fit=plan_dense_rhs_residency(1,chunks*p.channel_tile,
                columns*p.channel_tile,block=p.channel_tile,
                operand_rows=p.operand_rows,accumulator_rows=p.accumulator_rows,
                rhs_base=rhs_base,max_static_rhs_tiles=p.operand_rows//p.channel_tile)
    resources=dict(incumbent_regional_rolling=int(policy=='regional' and retention_mode=='rolling_rhs'),
        two_primitive_width=int(p.primitive<w<=2*p.primitive),output_rows=h,
        grouped_output_rows=rows_bound,complete_unphased_rhs_capacity=int(rhs_fit is not None))
    facts=PolicyFacts(AnalysisStamp(source.identity,
        PipelineContext(resources=str((p.primitive,p.operand_rows,p.accumulator_rows))),
        'convolution_source'),resources=tuple(sorted(resources.items())))
    guards=(ResourcePredicate('incumbent_regional_rolling',1),
        ResourcePredicate('two_primitive_width',1),ResourcePredicate('output_rows',2),
        ResourcePredicate('grouped_output_rows',2),ResourcePredicate('complete_unphased_rhs_capacity',1))
    admitted=all(g.match(facts) is True for g in guards)
    return admitted,dict(policy='regional-wide-two-M-tiles-v1',resources=resources,
        output_width=w,primitive=p.primitive,source_identity=source.identity,
        operand_rows=p.operand_rows,accumulator_rows=p.accumulator_rows,
        complete_rhs_rows=None if rhs_fit is None else rhs_fit.rhs_rows,rhs_base=rhs_base,
        guards=[dict(resource=g.resource,at_least=g.at_least) for g in guards],
        admitted=admitted,public_cycle_points=1,crossover_validated=False,
        final_admission='selected source/order laws, complete unphased RHS, actual shared joint storage and >=2 original output rows remain mandatory')

def _multirow_bound(p,columns):
    h,w=p.window.output_spatial
    per_row=(w+p.row_tile-1)//p.row_tile
    return min(h,16,max(1,16//per_row),
        max(1,(p.accumulator_rows-p.accumulator_row)//(per_row*columns*p.primitive)))

def _phased_spatial_bound(p, input_rows):
    """Bound an explicit alternative by one complete K panel and live ACC.

    Original column intervals are independent. Narrower full-K phases trade
    repeated input traversals for a larger spatial run retaining array weights.
    This is a capacity proposal, never a timing-based default selection.
    """
    h,w=p.window.output_spatial
    per_row=(w+p.row_tile-1)//p.row_tile
    kh,kw=p.window.kernel
    reduction_rows=kh*kw*((p.window.input_shape[-1]+p.channel_tile-1)//p.channel_tile)*p.channel_tile
    span=(kh-1)*p.window.dilation[0]+1
    maximum=min(h,16,max(1,16//per_row))
    for rows in range(maximum,1,-1):
        slots=span+(rows-1)*p.window.stride[0]
        end=input_rows.base+slots*input_rows.rows_per_source_row
        rhs_base=(end+p.channel_tile-1)//p.channel_tile*p.channel_tile
        operand_columns=(p.operand_rows-rhs_base)//reduction_rows
        acc_columns=(p.accumulator_rows-p.accumulator_row)//(rows*per_row*p.primitive)
        columns=min(operand_columns,acc_columns)
        if columns>=1:
            return rows,columns
    return 1,None

def plan_array_retention(schedule,rhs,input_rows, *, multirow=False):
    """Return actual checked owners/kwargs, or an explicit unchanged refusal."""
    refused=lambda reason:(schedule,rhs,{},dict(selected=False,reason=reason))
    if rhs is None or input_rows is None or schedule.epoch_law is None:
        return refused('requires original rolling input, complete resident RHS and compute epochs')
    root=Path(__file__).resolve().parent
    facts=(root/'data/rtl_facts.json').read_bytes()
    pin=json.loads((root/'data/COMMAND_ORDERING_SELECTION.json').read_text())
    order=pin['ordering'];digest=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest()
    if (pin['base_facts']['sha256']!=hashlib.sha256(facts).hexdigest()
            or digest!=pin['ordering_sha256'] or digest!=schedule.epoch_law.hardware_identity):
        return refused('selected ordering/facts identity differs')
    if not any(n.get('4')=='COMPUTE_AND_FLIP_CMD' and n.get('5')=='COMPUTE_AND_STAY_CMD'
               and n.get('6')=='PRELOAD_CMD' for n in _names(json.loads(facts))):
        return refused('selected decode lacks documented fresh/stay/preload endpoints')
    accesses=order.get('accesses',{});extents=order.get('extents',{})
    for name,funct in [('COMPUTE_PRELOADED.a_spad',4),('COMPUTE_ACCUMULATE.a_spad',5)]:
        if (accesses.get(name,{}).get('funct')!=funct
                or accesses[name].get('queue')!='execute'
                or extents.get(name,{}).get('status')!='derived'):
            return refused('selected fresh/stay source extent or execute ordering unknown')
    if order.get('completion_order',{}).get('execute',{}).get('order')!='in_order':
        return refused('fresh/stay execute order unknown')
    # A short original row keeps its complete mask. Select two neighboring
    # source rows only after assigning the complete delayed-read input union.
    # The shared joint storage planner then places complete-K RHS phases after
    # this new actual footprint; no ring slot or RHS address is inferred.
    p=schedule.source.plan
    oh,ow=p.window.output_spatial
    rows_bound=2 if ow<=p.primitive and oh>1 else 1
    rhs_column_blocks=None
    from merlin.compile.tuning_knobs import knob
    mode=knob('conv_array_retention',default='off',
        values=('off','on','auto','multirow'),root=root)
    # Auto reaches this owner only through the caller's existing regional
    # admission. Preserve complete resident RHS traversal sharing, and expand
    # spatial reuse only when the original invocation already requires phases.
    phased_auto=mode=='auto' and type(rhs) is ConvolutionRhsPhases
    if multirow or phased_auto:
        # Bound neighboring original rows by the shared M-block limit and
        # full original N ACC demand. Joint planner still checks each phase.
        columns=len(schedule.source.column_intervals)
        rows_bound=_multirow_bound(p,columns)
        if mode in ('auto','multirow') and type(rhs) is ConvolutionRhsPhases:
            rows_bound,rhs_column_blocks=_phased_spatial_bound(p,input_rows)
    retained_input=input_rows
    storage=None
    if rows_bound>1:
        span=(p.window.kernel[0]-1)*p.window.dilation[0]+1
        slots=span+(rows_bound-1)*p.window.stride[0]
        end=input_rows.base+slots*input_rows.rows_per_source_row
        if end+p.channel_tile>p.operand_rows:
            return refused('neighboring-row input union leaves no complete RHS tile')
        retained_input=replace(input_rows,slots=slots,
            private_rows=end-input_rows.private_first)
    law=replace(schedule.retention_law,independent_output_rows=True,no_intermediate_observers=True)
    issue=schedule.issue_law
    if rows_bound>1:
        issue=replace(issue,load_issue_budget=max(issue.load_issue_budget,
            retained_input.stage_command_bound(retained_input.slots),
            p.row_tile+1+rows_bound*((ow+p.row_tile-1)//p.row_tile)*len(schedule.source.column_intervals)*int(p.bias)))
    updated=plan_convolution_schedule(schedule.source,schedule.choice,retention_law=law,
        issue_law=issue,epoch_law=schedule.epoch_law)
    if rows_bound>1:
        rhs_law=rhs.phases[0].law if type(rhs) is ConvolutionRhsPhases else rhs.law
        phase_law=(rhs.law if type(rhs) is ConvolutionRhsPhases else
            ConvolutionRhsPhaseLaw(schedule.source.identity,
                'original-disjoint-output-columns-no-intermediate-observers-v1',True,True,True,True))
        storage=plan_convolution_storage_alternative(updated,retained_input,
            operand_rows=p.operand_rows,rhs_law=rhs_law,rhs_phase_law=phase_law,
            max_row_blocks=16,max_output_rows=rows_bound,
            rhs_column_blocks=rhs_column_blocks)
        updated_rhs=storage.rhs;group=storage.output_group
    else:
        if type(rhs) is ConvolutionRhsPhases:
            updated_rhs=replace(rhs,schedule=updated,
                phases=tuple(replace(phase,schedule=updated) for phase in rhs.phases))
        else:updated_rhs=replace(rhs,schedule=updated)
        group=plan_rolling_convolution_output_group(updated,updated_rhs,retained_input,max_row_blocks=16)
    if group is None:return refused('no capacity-legal reusable spatial M run')
    groups=group if isinstance(group,tuple) else (group,)
    epoch=updated.epoch_law
    array=DenseArrayRetentionAdmission(epoch.hardware_identity,epoch.numerical_policy_identity,
        'selected-ws-funct4-captures-B-funct5-retains-B-output-descriptor-only-v1')
    classes=json.loads(facts)['facts']['timing_structure']['command_classes']['classes']
    entries=classes['store']['entries']
    if entries.get('status')!='derived' or entries.get('value',0)<1:
        return refused('finite selected store progress unknown')
    maximum=max(g.row_blocks*g.panel.column_blocks for g in groups if g is not None)
    readout=ConvolutionReadoutEpochLaw(epoch.hardware_identity,epoch.numerical_policy_identity,
        updated.source.identity,'selected-original-ACC-RAW-and-config-capture-private-M-group-v1',
        maximum,entries['value'],epoch.local_read_after_write,epoch.accumulator_result_visibility,
        epoch.ordered_configuration_capture,epoch.finite_queue_pair_progress,
        epoch.complete_device_drain,epoch.exclusive_private_state)
    return updated,updated_rhs,dict(output_group=group,array_law=array,readout_law=readout,**(dict(input_rows=retained_input) if rows_bound>1 else {})),dict(
        selected=True,requested_multirow=multirow,requested_output_rows=rows_bound,
        automatic_phased_spatial=phased_auto,
        requested_rhs_column_blocks=rhs_column_blocks,
        isa_source=ISA_SOURCE,selected_ordering_sha256=digest,
        array_law=asdict(array),readout_law=asdict(readout),
        group=[None if g is None else asdict(g) for g in groups],
        input_rows_before=input_rows.record(),input_rows_after=retained_input.record(),
        joint_storage=None if storage is None else storage.summary(),
        source_contract='independent original output rows; no intermediate observers; each K order unchanged',
        target_execution_verified=False,timing=None)
