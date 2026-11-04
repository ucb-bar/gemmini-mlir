"""Opt-in source-general full-K operand retention for raw integer contractions.

The library already expresses both families: LHS outside independent N groups,
or RHS outside independent M groups. This adapter binds that previously unused
retained_full_k decision to the same selected ordered accumulator/store facts,
private bank geometry and terminal encoder as dense epochs. It constructs and
consumes actual invariant-hoist transformations on the original compact Kernel;
no flattened command recipe, new execution law, array residency or timing grant
is supplied. Full/tail groups keep the library's exact source reduction order.

The optional route widens only the K planning limit to the complete source K.
M/N proposals are bounded by four primitive tiles and the original private
assignments, with explicit axis overrides preserved. The library ranks legal
retained choices by requested source traffic (not predicted latency). A choice
must actually hoist an operand across at least two independent output groups.
Bias, scale, and per-column output use the existing incumbent. Transposed RHS
uses the original pure tile transport with an explicit completion after every
MVIN, so its host staging tile is never overwritten while DMA can still read it.
"""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import json

KNOB_DECLARATION={'attention_retention':{
    'values':['off','full_k_operand','geometry_supported','resource_phased'],
    'mechanism':'checked retained_full_k dense decision with actual invariant operand hoists across independent output groups',
    'note':'Default off. Raw unbiased i8xi8->i32 only; transpose requires completed pure tile staging; complete K must fit both private operand assignments. Bounded M/N proposals, exact tails, selected ordered completion, common terminal encoder. Traffic is not timing evidence.'}}


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('attention_retention',default='off',values=('off','full_k_operand','geometry_supported','resource_phased'),
                root=Path(__file__).resolve().parent)



def select_retention(mode, problem, *, rhs_transposed=False, wide_route_selected=False):
    """Experimental performance coverage; independent owner proves legality.

    Complete modeled training supports ordinary full-height M panels, while
    ordinary M1 decode regresses. The resource_phased experiment compares
    bounded RHS phases when a second full panel cannot fit its assigned bank;
    this resource class has no measured timing qualification. The selected M
    edge is a hypothesis needing M=edge-1/edge and independent timed holdouts.
    Transposed transport has no paired timing support and stays on incumbent.
    Explicit full_k_operand retains its original unfiltered meaning.
    """
    from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    from merlin.transforms.shape_policy import ShapeScheduleClass
    if mode not in ('off','full_k_operand','geometry_supported','resource_phased'):
        raise ValueError('unknown attention operand retention schedule')
    shape=((problem.m,problem.k,problem.n),)
    source=sha256(json.dumps(shape).encode()).hexdigest()
    edge=problem.tile_m
    if type(edge) is not int or edge<=0:
        raise ValueError('attention coverage needs positive selected M edge')
    facts=PolicyFacts(AnalysisStamp(source,PipelineContext(),
        'attention-retention-source-geometry'),shapes=shape)
    group=ShapeScheduleClass('ordinary_full_height_m',
        (ShapePredicate(0,0,at_least=edge),),'off')
    matches=tuple(p.match(facts) for p in group.predicates)
    matched=all(v is True for v in matches)
    selected=(mode if mode not in ('geometry_supported','resource_phased') else
        ('resource_phased' if mode=='resource_phased' else 'full_k_operand')
        if matched and not rhs_transposed and not wide_route_selected else 'off')
    return {'schema':'gemmini.attention_retention_geometry.v1','requested':mode,
        'selected':selected,'geometry_class':group.name if matched else None,
        'source_shape':[problem.m,problem.k,problem.n],
        'class_predicates':[asdict(p) for p in group.predicates],
        'class_matches':list(matches),'ordinary_rhs_guard':not rhs_transposed,
        'selected_m_edge':edge,'k_performance_cutoff':None,
        'preferred_wide_route_guard':not wide_route_selected,
        'status':'experimental coverage; no promotion or heldout qualification',
        'missing_timing':'M=edge-1/edge cutoff and independent inside-class holdouts',
        'legality':'unchanged full-K capacity/source/effects/order/readout owner',
        'actual_schedule_applied':False}


def rank_completed_pair(incumbent, retained):
    """Screen a caller-pinned same-domain complete pair with shared selector.

    Inputs are original ObservedMetrics, with cycles and complete provider/scope
    custody retained by the caller. No case names or timings enter compilation.
    Missing metrics retain the shared selector's explicit unknown semantics.
    Ranking never qualifies the experimental geometry class for promotion.
    """
    from merlin.transforms.pass_exploration import rank_performance_variants
    return rank_performance_variants((('off',incumbent),('full_k_operand',retained)))


def build_retained_schedule(problem,resources,storage,adapter,options=None,*,attrs=None,
                            rhs_transposed=False,bias_source=None,transpose_staging_completed=False,timeout_ms=5000,
                            mode='full_k_operand'):
    """Return (checked actual library result, JSON-safe consumed-plan record).

    The ordinary dense callback retains logical pointer strides K/N; no source
    layout transformation is performed here. Row-per-key transposed RHS uses
    the common encoder's original pure transpose staging and requires its
    explicit per-MVIN full completion before any staging overwrite.
    Refusal emits nothing; caller retains its exact previously selected route.
    """
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import (
        dense_retained_choices,dense_retained_operand_hoists,DenseRetainedReductionAdmission)
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan,dense_schedule_text,parse_dense_schedule_plan
    from dense_epochs import selected_law
    attrs=attrs or {};options=dict(options or {})
    if mode not in ('full_k_operand','resource_phased'):
        raise ValueError('unknown concrete attention retention construction mode')
    if mode=='resource_phased' and (rhs_transposed or problem.m<problem.tile_m):
        raise ValueError('resource-phased experimental coverage requires ordinary full-height M panels')
    if attrs.get('output_dtype','i32')!='i32' or attrs.get('epilogue',[]) or bias_source is not None or problem.bias:
        raise ValueError('full-K operand retention currently admits unbiased raw i32 source exits only')
    if attrs.get('acc_scale_by_column') is not None:
        raise ValueError('full-K operand retention requires raw output without per-column snapshots')
    if rhs_transposed and transpose_staging_completed is not True:
        raise ValueError('retained transposed RHS requires full completion after each staged MVIN before host tile overwrite')
    if not problem.tile_m==problem.tile_k==problem.tile_n:
        raise ValueError('full-K operand retention requires the existing square primitive encoder')
    if not all(value>0 for value in (problem.m,problem.k,problem.n)):
        raise ValueError('full-K operand retention needs positive source axes')
    maxima={'m':min(4,problem.tiles[0]),'k':problem.tiles[1],'n':min(4,problem.tiles[2])}
    for axis,explicit in options.get('axis_overrides',{}).items():
        if explicit and axis in maxima:maxima[axis]=int(options.get('max_'+axis+'_blocks',getattr(resources,'max_'+axis+'_blocks')))
    limits=replace(resources,**{'max_'+axis+'_blocks':value for axis,value in maxima.items()})
    choices=dense_retained_choices(problem,limits,storage)
    selected=None
    for choice in choices:
        groups=problem.tiles[2]//choice.block.n if choice.role=='lhs' else problem.tiles[0]//choice.block.m
        if groups<2:continue
        raw=emit_compact_dense(problem,choice.block,storage,adapter,name='gemmini_full_k_retention',
            dtypes=('i8','i8','i32'),outer_order=choice.outer_order)
        hoists=dense_retained_operand_hoists(raw,adapter,role=choice.role)
        if hoists:selected=(choice,raw,hoists);break
    if selected is None:
        raise ValueError('no complete-K operand panel fits original assigned stores with an actual reusable output-group axis')
    choice,raw,hoists=selected;block=choice.block
    phase_policy = _phase_policy(problem,resources,storage,choice)
    if mode=='resource_phased' and phase_policy['matched']:
        # Existing combined owner stages complete LHS once across N groups,
        # alternates two bounded RHS K segments and checks every actual
        # reduction, load/store, interval and completion. It never slices the
        # source into partial products or changes the increasing K order.
        from dense_epochs import build_readout_schedule
        phase_limits=replace(limits,max_k_blocks=problem.tiles[1]-1)
        result,info=build_readout_schedule(problem,phase_limits,storage,adapter,
            dict(options,readout_schedule='segmented_ready'),timeout_ms=timeout_ms)
        result.revalidate()
        info.update(selected_path='resource_phased',retained_role='lhs',
            phase_policy=phase_policy,planning_maxima={**maxima,'k':phase_limits.max_k_blocks},
            source_shape=[problem.m,problem.k,problem.n],source_identity=
                sha256(json.dumps({'problem':asdict(problem),'dtypes':['i8','i8','i32']},sort_keys=True).encode()).hexdigest(),
            checked_kernel_sha256=result.kernel.digest(),
            checked_library_sources=list(result.original.checked_library_sources),
            checked_ready_owner_sha256=result.owner_sha256,
            private_storage=asdict(storage),actual_resources=asdict(phase_limits),
            completion='existing checked retained-LHS/two-RHS phases and causal ready readout',
            performance_status='unmeasured experiment; capacity class does not establish a latency crossover')
        return result,info
    law,flags,queues,identity=selected_law(adapter)
    if law.unknown():raise ValueError('full-K retention lacks selected ordering premises: '+repr(law.unknown()))
    # Issued maxima bound one source-checked completed epoch; pending entries
    # come directly from selected queues, independent of source shape.
    budget=4096
    admission=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':read-only-full-k-retention',budget,budget,max(1,queues['execute']//2),queues['load'],
        max_issued_readouts=block.m*block.n,pending_readouts=queues['store'],
        readout_configuration='ordered_snapshot_before_store_issue')
    source_identity=sha256(json.dumps({'problem':asdict(problem),'dtypes':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
    plan=make_dense_schedule_plan(problem,limits,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=256,block_tiles=(block.m,block.k,block.n),
        outer_order=choice.outer_order,loop_transforms=hoists,pipeline_kind='retained_full_k',pipeline_prefetch=False)
    text=dense_schedule_text(plan)
    result=schedule_dense_pipeline(problem,limits,storage,adapter,name='gemmini_full_k_retention',
        dtypes=('i8','i8','i32'),source_identity=source_identity,external_buffers_disjoint=True,
        schedule_plan=parse_dense_schedule_plan(text),pipeline_admission=admission,
        batch_readouts=True,static_budget=256,timeout_ms=timeout_ms)
    result.revalidate()
    return result,{'selected_path':'full_k_operand','requested_mode':mode,'phase_policy':phase_policy,'retained_role':choice.role,'block':asdict(block),
        'choice':asdict(choice),'planning_maxima':maxima,'source_shape':[problem.m,problem.k,problem.n],
        'schedule_ir':text,'before_kernel':raw.text(),'after_kernel':result.kernel.text(),
        'source_identity':source_identity,'actual_hoists':[asdict(h) for h in hoists],
        'checked_kernel_sha256':result.checked_kernel_sha256,'checked_library_sources':list(result.checked_library_sources),
        'ordering_identity':identity,'law_flags':flags,'selected_queues':queues,
        'private_storage':asdict(storage),'actual_resources':asdict(limits),
        'input_pointer_strides':{'lhs_bytes':problem.k,'rhs_bytes':problem.k if rhs_transposed else problem.n,'output_bytes':problem.n*4},
        'rhs_view':'row-per-key transposed by original pure layout transport' if rhs_transposed else 'ordinary row-major',
        'transpose_completion':'full fence after each staged MVIN before host overwrite' if rhs_transposed else None,
        'completion':'actual checked retained epochs and batched readouts use terminal full-device fences',
        'performance_status':'unmeasured experiment; traffic ranking is not complete-call latency'}


def _phase_policy(problem,resources,storage,choice):
    """Resource-relative experimental class, never a legality permission.

    Full-K traffic minimization ignores the size of its serialized RHS burst.
    When another complete RHS panel cannot coexist in the originally assigned
    bank/byte budget, compare bounded two-slot K phases through the existing
    segmented owner. This can change overlap and transfer locality; it is not
    evidence of a performance win. The complete-LHS assignment stays fixed.
    """
    from merlin.transforms.pipeline_policy import PolicyFacts,ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
    from merlin.transforms.shape_policy import ShapeScheduleClass
    block=choice.block
    lhs_rows=block.m*problem.tiles[1]*problem.tile_m
    rhs_rows=block.n*problem.tiles[1]*problem.tile_k
    shortfall=max(0,(2*rhs_rows-storage.rhs.capacity)*storage.rhs.unit_bytes,
        lhs_rows*storage.lhs.unit_bytes+2*rhs_rows*storage.rhs.unit_bytes-resources.operand_bytes)
    resource_values=(('second_full_rhs_slot_shortfall_bytes',shortfall),)
    resource_identity=sha256(json.dumps({'resources':asdict(resources),'storage':asdict(storage)},sort_keys=True).encode()).hexdigest()
    source_identity=sha256(json.dumps(asdict(problem),sort_keys=True).encode()).hexdigest()
    facts=PolicyFacts(AnalysisStamp(source_identity,PipelineContext(resources=resource_identity),
        'attention-retention-phase-geometry'),shapes=((problem.m,problem.k,problem.n),),resources=resource_values)
    group=ShapeScheduleClass('full_rhs_panel_has_no_second_slot',
        (ResourcePredicate('second_full_rhs_slot_shortfall_bytes',at_least=1),),'full_k_operand')
    matches=tuple(predicate.match(facts) for predicate in group.predicates)
    eligible_family=choice.role=='lhs' and problem.tiles[1]>1
    return {'schema':'gemmini.attention_retention_phases.v1','geometry_class':group.name,
        'matched':eligible_family and all(value is True for value in matches),
        'family_guard':eligible_family,'predicates':[asdict(p) for p in group.predicates],
        'predicate_matches':list(matches),'resources':dict(resource_values),
        'resource_identity':resource_identity,'complete_lhs_rows':lhs_rows,
        'complete_rhs_rows':rhs_rows,'rhs_private_rows':storage.rhs.capacity,
        'original_operand_bytes':resources.operand_bytes,'full_k_block':asdict(block),
        'performance_cutoff':'no fixed K threshold; actual padded panel and owned resource extents',
        'timing':'UNKNOWN; paired same-source timing and reserved neighbors required',
        'legality':'separate unchanged selected scheduling/source/storage/completion owner'}
