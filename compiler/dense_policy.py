"""Declared dense scheduling experiments; no timing-based selection at compile time.

All profiles use the same compact_dense scheduler and terminal Gemmini encoder.
primitive1 is the incumbent (one M/K/N primitive tile). panel2/panel4 admit
larger capacity-checked groups: an LHS panel can serve multiple N tiles, an RHS
panel multiple M tiles, and multiple K slices coexist in distinct operand rows.
The hypothesis is reduced operand reload traffic; larger live panels can instead
increase command latency or reduce useful overlap. Neither counts nor capacity
establish speedup. mn/nm exchanges independent output-group axes without changing
source reduction order. Scalar/per-column readout and public output boundaries
remain part of every region. All alternatives retain synchronous completion.
The independent readout_schedule experiment selects a full-LHS/two-RHS
segmented plan. Its drained control and causal ready consumer share physical
placement; ready places original stores after their final-K producers inside
the compute epoch. The hypothesis is lower submission/completion overhead for
small outputs and less readout serialization on large groups. Selected ACC
visibility and queue configuration facts qualify it; capacity, per-column
numerical snapshots and transpose staging have separate admission/refusal.
Completion epochs and segmented readout are mutually exclusive experiments.
The optional geometry_epochs policy applies two-window epochs to multi-slice
K on qualified scalar-readout layouts. geometry_panel4_epochs additionally
offers four M/N blocks to the same checked plan for outputs at least four
array widths wide. Decode/matrix and narrow/general/wide geometry classes are
reported for held-out testing. The hypotheses are amortized K completion and
less operand reload traffic within a larger output group; no new target law or
empirical promotion follows from the shape predicates. Explicit route choices
and axis caps override these experimental policies; the policy default is off.
validated_panel_epochs keeps the same recipe behind a conservative N<=2048
coverage guard. That upper bound limits promotion to the measured coverage;
it is not an inferred performance crossover. The experimental policies remain
available unchanged, including their behavior beyond this guard.
Maxpool uses a separately checked gathered-window composition over a private i32
device spill: the original scalar path precedes hardware max for each value.
This coverage route is intentionally independent of panel profile selection;
its extra full-width transfer is visible in the dense pooling receipt.

``current_options`` reads the existing merlin.compile.tuning_knobs API.
``regional_legality`` separately checks storage and source-independent dimensions;
compact_dense then verifies the actual emitted indexed accesses and reduction.
The optional wide_panel4 profile tests a geometry-only boundary: four complete
array widths of output columns. Three provisional paired observations support
testing a width separator (two wide improvements and one narrow regression),
but do not locate its crossover. The four-width boundary is an unassessed
hypothesis; exact boundary neighbors, held-out widths and M/K interactions must
pass before promotion. No capsule identities or runtime timing participate in
compilation. The original, scalability and connected model panels must pass
paired numeric/cycle checks before an alternative becomes incumbent.
"""
import hashlib
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext

PROFILES={'primitive1':1,'panel2':2,'panel4':4}
GEOMETRY_PROFILE='wide_panel4'
# The coordinator merges this declaration into TUNING_KNOBS.json independently.
PROFILE_KNOB_DECLARATION={
    'values':[*PROFILES,GEOMETRY_PROFILE],
    'mechanism':'compact_dense capacity-checked panels with optional output-width predicate',
    'note':'primitive1 remains incumbent; wide_panel4 is an unassessed opt-in N >= 4*array_edge hypothesis requiring boundary and held-out paired cycles.'}
SCHEDULE_POLICY_KNOB_DECLARATION={'dense_schedule_policy':{
    'values':['off','geometry_epochs','geometry_panel4_epochs','validated_panel_epochs'],
    'mechanism':'source geometry predicates over dense_panel_epochs and checked panel capacity',
    'note':'Experimental policies remain unchanged. validated_panel_epochs uses the same panel4/epoch plan only for K>edge and N<=2048 on scalar nontransposed layouts; N bound is a measured-coverage guard, not a profitable crossover. Explicit alternatives/axis caps take precedence. off stays incumbent.'}}

VALIDATED_PANEL_MAX_N=2048
# Evidence references are immutable provenance, never inputs to shape selection.
VALIDATED_PANEL_EVIDENCE={
    'status':'paired-panel-supported-with-coverage-guard',
    'candidate':'policy007 geometry_panel4_epochs',
    'source_manifest':'artifacts/policy007_source_manifest.json',
    'candidate_submission_sha256':'72debf78c8509bc31b5d1fdc71900e7a1fe96d79da3d50390a50f86b75e1ebe8',
    'paired_baseline_submission_sha256':'4f474c05fc2fdc5fed92008a1ad59d74d9e33f077a68915781e1422c95cd85de',
    'paired_training':{
        'candidate_request':'2_1793272585849759295_88bc6c667099a1bbd0717f663d1ecb63',
        'baseline_request':'2_1793272587176398966_0c6cb782ce8d8b301b2157516088fc52'},
    'held_out':{
        'candidate_request':'2_1793273209429690905_7d9854a59610f28d01ce2ce13fde1d56',
        'baseline_request':'2_1793273209469385392_6ff5b4fecfcf4a00b1c6f247f806600e',
        'result':'five complete improvements; width32000 complete UNKNOWN'},
    'measured_representative':{'case':'SY_kdepthfitsdouble',
        'baseline_barrier_cycles':47752,'candidate_barrier_cycles':14514,
        'candidate_request':'2_1793272750158493981_5a2cd022ed4c18b8b2075e10cff2aab5',
        'baseline_request':'70_1793272103040788977_2cd87358c052f8a8444ebf6746494f74',
        'baseline_submission_sha256':'f8fef72ae9c983ec9fd4d7ed71ee9bc9a2e8645961cbbdbccd6b0adfb98cd1d7',
        'evidence':'measured gSIM, both certified'},
    'coverage_boundary':'N<=2048 bounds promotion coverage; no crossover claim',
}


def geometry_profile(*,m,k,n,edge):
    """Evaluate the experimental width predicate; this grants no legality.

    Facts describe only the current contraction geometry. Unknown hardware,
    numeric and execution identities remain unknown: selecting a capacity
    maximum cannot authorize a transform or bypass regional verification.
    """
    if any(type(v) is not int or v <= 0 for v in (m,k,n,edge)):
        raise ValueError('dense geometry policy requires positive M, K, N and array edge')
    shapes=((m,n),(m,k),(k,n))
    source=hashlib.sha256(json.dumps(shapes,separators=(',',':')).encode()).hexdigest()
    facts=PolicyFacts(AnalysisStamp(source,PipelineContext(),'dense-geometry'),shapes=shapes)
    guard=ShapePredicate(operand=0,axis=1,at_least=4*edge)
    matched=guard.match(facts)
    selected='panel4' if matched is True else 'primitive1'
    return selected,{'status':'experimental-unassessed','selected':selected,
        'fallback':'primitive1','predicate':{'operand':0,'axis':1,'at_least':4*edge},
        'matched':matched,'source_shapes':shapes,
        'required_validation':'paired boundary neighbors, held-out widths and M/K interactions'}


def current_options(*,m=None,k=None,n=None,edge=16):
    root=Path(__file__).resolve().parent
    profile=knob('dense_block_profile',default='primitive1',
        values=(*PROFILES,GEOMETRY_PROFILE),root=root)
    selected=profile
    selection=None
    if profile==GEOMETRY_PROFILE:
        selected,selection=geometry_profile(m=m,k=k,n=n,edge=edge)
    base=PROFILES[selected]
    result={'profile':profile, 'completion_epochs':knob('dense_completion_epochs',
        default='off',values=('off','two_window_drained','two_window_epochs'),root=root)}
    result['readout_schedule']=knob('dense_readout_schedule',default='off',
        values=('off','segmented_drained','segmented_ready'),root=root)
    result['schedule_policy']=knob('dense_schedule_policy',default='off',
        values=('off','geometry_epochs','geometry_panel4_epochs','validated_panel_epochs'),root=root)
    if selection is not None:
        result['geometry_selection']=selection
    result['axis_overrides']={}
    for axis in ('m','k','n'):
        limit=knob('dense_'+axis+'_blocks',default=0,values=(0,1,2,4),root=root)
        result['max_'+axis+'_blocks']=limit or base
        result['axis_overrides'][axis]=bool(limit)
    order=knob('dense_outer_order',default='mn',values=('mn','nm'),root=root)
    result['outer_order']=tuple(order)
    return result


def apply_geometry_schedule(options,*,m,k,n,edge,tile_n,rhs_transposed=False):
    """Select an experimental recipe, then leave legality to its checked owner.

    Multi-slice reduction is a source feature, not a speed guarantee. Coarse
    decode/wide/narrow classes describe the next paired panel; they do not key
    on workload identity. Three observed reductions motivate testing epochs
    across these classes. Boundary and held-out evidence remains incomplete.
    Explicit mechanisms and axis caps retain precedence over this policy.
    """
    choice=options.get('schedule_policy','off')
    if choice=='off':
        return options
    if choice not in ('geometry_epochs','geometry_panel4_epochs','validated_panel_epochs'):
        raise ValueError('unknown dense geometry schedule policy '+repr(choice))
    shapes=((m,k,n),)
    stamp=AnalysisStamp(hashlib.sha256(json.dumps(shapes).encode()).hexdigest(),PipelineContext(),'dense-geometry')
    facts=PolicyFacts(stamp,shapes=shapes)
    multiple_k=ShapePredicate(0,1,at_least=edge+1).match(facts) is True
    decode=ShapePredicate(0,0,at_most=edge).match(facts) is True
    narrow=ShapePredicate(0,2,at_most=edge).match(facts) is True
    wide=ShapePredicate(0,2,at_least=4*edge).match(facts) is True
    supported_width=ShapePredicate(0,2,at_most=VALIDATED_PANEL_MAX_N).match(facts) is True
    receipt={'status':'experimental-unassessed','policy':choice,
        'class':('decode' if decode else 'matrix')+'_'+('narrow' if narrow else 'wide' if wide else 'general'),
        'modifiers':{'ragged':any(d%edge for d in (m,k,n)),
            'transposed_rhs':bool(rhs_transposed),'per_column_readout':tile_n!=edge},
        'multi_slice_k':multiple_k,'source_shape':[m,k,n],
        'performance_predicates':{'multi_slice_k_min':edge+1,'wide_n_min':4*edge},
        'required_validation':'held-out M/K interactions, K=edge/edge+1 and N=4*edge neighbors',
        'legality':'separate regional capacity and selected ordering checks'}
    options=dict(options)
    options['schedule_selection']=receipt
    if choice=='validated_panel_epochs':
        receipt.update(status=VALIDATED_PANEL_EVIDENCE['status'],
            validation=dict(VALIDATED_PANEL_EVIDENCE),
            coverage_guard={'operand':0,'axis':2,'at_most':VALIDATED_PANEL_MAX_N,
                'matched':supported_width,'purpose':'timed coverage, not profitable crossover'},
            required_validation='complete timing beyond N=2048; boundary neighbors remain unmeasured')
    if options.get('completion_epochs','off')!='off' or options.get('readout_schedule','off')!='off':
        receipt['selection']='preserved-explicit-alternative'
    elif rhs_transposed or tile_n!=edge:
        receipt['selection']='incumbent'
        receipt['refusal']='source staging or numerical snapshot needs independent admission'
    elif not multiple_k:
        receipt['selection']='incumbent'
        receipt['refusal']='single primitive K slice has no interior K drains'
    elif choice=='validated_panel_epochs' and not supported_width:
        receipt['selection']='incumbent'
        receipt['refusal']='output width exceeds validated timing coverage'
    else:
        options['completion_epochs']='two_window_epochs'
        receipt['selection']='two_window_epochs'
        if choice in ('geometry_panel4_epochs','validated_panel_epochs') and wide:
            overrides=options.get('axis_overrides',{})
            for axis in ('m','n'):
                if not overrides.get(axis,False): options['max_'+axis+'_blocks']=4
            receipt['panel_maxima']={axis:options['max_'+axis+'_blocks'] for axis in ('m','n')}
    return options


def regional_legality(options,*,edge,bank_rows,operand_bytes,accumulator_bytes,tile_n):
    """Check the largest requested live group independent of scheduling preference.

    Returns the precise simultaneous physical byte/row bounds. A profile only
    permits blocks up to these maxima; the shared planner also intersects shape
    tails and ranks its admitted blocks by its declared static planning cost.
    """
    m,k,n=(int(options['max_'+a+'_blocks']) for a in ('m','k','n'))
    if any(v not in (1,2,4) for v in (m,k,n)):
        raise ValueError('dense block maxima require 1, 2 or 4')
    if tuple(options['outer_order']) not in (('m','n'),('n','m')):
        raise ValueError('dense output group ordering must be mn or nm')
    lhs_rows=m*k*edge; rhs_rows=k*n*edge; accumulator_rows=m*n*edge
    operand_footprint=(lhs_rows+rhs_rows)*edge
    accumulator_footprint=accumulator_rows*edge*4
    bounds={'lhs_rows':lhs_rows,'rhs_rows':rhs_rows,'accumulator_rows':accumulator_rows,
        'operand_physical_bytes':operand_footprint,'accumulator_physical_bytes':accumulator_footprint,
        'lhs_assigned_rows':bank_rows,'rhs_assigned_rows':bank_rows,
        'operand_available_bytes':operand_bytes,'accumulator_available_bytes':accumulator_bytes,
        'logical_output_tile_columns':tile_n,'completion':'synchronous-per-operation'}
    if lhs_rows>bank_rows or rhs_rows>bank_rows or operand_footprint>operand_bytes or accumulator_footprint>accumulator_bytes:
        raise ValueError('requested dense profile exceeds selected assigned stores: '+repr(bounds))
    return bounds
