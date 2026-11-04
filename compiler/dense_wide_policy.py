"""Experimental geometry coverage for the existing checked wide dense route.

Complete paired QKV/GU observations support the provisionally selected bounded matrix class. No empirical crossover beyond those observations is established.
Independent inside-class M/K/N timing holdouts remain unavailable in the selected
public roster. Counts establish application, not speed.
The classes describe decode and short-matrix wide projections. Resource fit
and every numerical, alias, lifetime and ordering obligation remain with the
original dense_wide builder and common terminal effect encoder.

Integration may use select_wide_schedule to choose the existing route before
its current build hook, or build_selected_wide to obtain that actual checked
owner. SelectedWide.emit_index_program forwards the same callbacks and marks
application only after consuming the checked indexed program. Explicit off and
retained_lhs_wide keep their original meanings. No profiling occurs at compile
time. The supported_geometry mode remains opt-in. The matrix_coalesced_geometry mode is provisionally selected with the paired support and limitations below. The latter keeps M1 on the incumbent and consumes the original
checked RHS band-coalescing owner on a bounded one-array-row matrix class. It is a new experiment,
not a promotion of either mechanism.
"""
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import json
import math

from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate, ResourcePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from merlin.transforms.shape_policy import ShapeScheduleClass
from dense_wide import build_wide_schedule

KNOB_DECLARATION = {'dense_wide_schedule': {
    'values': ['off', 'retained_lhs_wide', 'supported_geometry', 'matrix_coalesced_geometry'],
    'mechanism': 'source/resource geometry coverage over checked retained-LHS segmented ready readout',
    'note': 'matrix_coalesced_geometry is provisionally selected. It offers bounded M>1 wide scalar projections through actual checked coalesced RHS emission. Coverage guards remain experimental; paired holdouts required.'}}

EVIDENCE = {
    'status': 'one complete modeled training improvement; heldouts pending',
    'candidate_request': '3_1793281090323871074_4c144b45139b4d97ebf6a9295a8fb8d0',
    'baseline_request': '3_1793281090323444060_8619cc77e3ae401db8e01e53e68ee39b',
    'complete_training': {'m':16,'k':2048,'n':2560,'before_cycles':6930312,'after_cycles':1623056},
    'unknown_training': {'m':16,'k':2048,'n':11264,'timing':'UNKNOWN',
        'candidate_commands':271173,'candidate_fences':719,
        'baseline_commands':361157,'baseline_fences':271043},
    'qualification': 'Spike model; no RTL timing claim, no inferred win from command counts',
    'planned_holdouts': {'m':[8,32],'k':[1024,4096],'n':[3072,4096]}}


def select_wide_schedule(mode, problem, resources, storage, *, attrs=None,
                         rhs_transposed=False, options=None):
    """Return JSON-safe route preference, never an execution qualification.

    Resource predicates check necessary private-row footprints only;
    building and consuming the actual capacity/source-verified owner is required.
    Other explicitly selected completion/readout routes take precedence over
    supported_geometry. Axis overrides stay authoritative in the actual builder.
    """
    if mode not in ('off','retained_lhs_wide','supported_geometry','matrix_coalesced_geometry'):
        raise ValueError('unknown wide geometry policy mode')
    attrs=dict(attrs or {}); options=dict(options or {})
    edge=problem.tile_k
    if any(type(v) is not int or v<=0 for v in
           (problem.m,problem.k,problem.n,edge,problem.tile_m,problem.tile_n)):
        raise ValueError('wide geometry policy requires positive static source extents')
    shape=((problem.m,problem.k,problem.n),)
    source=sha256(json.dumps(shape).encode()).hexdigest()
    private_rows=(('lhs_private_rows',storage.lhs.capacity),
        ('rhs_private_rows',storage.rhs.capacity),
        ('acc_private_rows',storage.accumulator.capacity))
    resource_identity=sha256(json.dumps(private_rows).encode()).hexdigest()
    facts=PolicyFacts(AnalysisStamp(source,PipelineContext(resources=resource_identity),
        'wide-dense-source-geometry'),shapes=shape,resources=private_rows)
    # Necessary one-M-tile footprints only; simultaneous disjoint segments,
    # all actual panels, primitive tails and budgets are checked by the planner.
    retained_rows=problem.tiles[1]*edge
    common=(ShapePredicate(0,1,at_least=edge+1),ShapePredicate(0,2,at_least=2049),
        ResourcePredicate('lhs_private_rows',retained_rows),
        ResourcePredicate('rhs_private_rows',2*edge),ResourcePredicate('acc_private_rows',edge))
    if mode=='matrix_coalesced_geometry':
        # Conservative experiment coverage, not a learned profitable threshold.
        # Four selected RHS-row capacities bounds the logical output extent;
        # this does not claim a complete RHS fits or infer byte capacity from N.
        classes=(ShapeScheduleClass('bounded_wide_matrix_rhs_bands',common+(
            ShapePredicate(0,0,at_least=edge,at_most=edge),
            ShapePredicate(0,2,at_most=4*storage.rhs.capacity),),'off'),)
    else:
        classes=(ShapeScheduleClass('wide_decode',common+(ShapePredicate(0,0,at_least=1,at_most=edge),),'off'),
            ShapeScheduleClass('wide_short_matrix',common+(ShapePredicate(0,0,at_least=edge+1,at_most=4*edge),),'off'))
    checks=[(c,tuple(p.match(facts) for p in c.predicates)) for c in classes]
    matched=next((c.name for c,values in checks if all(v is True for v in values)),None)
    stages=attrs.get('epilogue',[])
    raw=attrs.get('output_dtype','i32')=='i32' and not stages
    scale=attrs.get('acc_scale',1.0)
    scalar_i8=(attrs.get('output_dtype')=='i8' and stages in (['acc_scale'],['acc_scale','relu'])
        and isinstance(scale,(int,float)) and not isinstance(scale,bool) and math.isfinite(scale) and scale>0)
    numerical=(raw or scalar_i8) and not problem.bias and attrs.get('acc_scale_by_column') is None
    ordinary=(not rhs_transposed and problem.tile_m==edge==problem.tile_n
        and (problem.lhs_bytes,problem.rhs_bytes,problem.accumulator_bytes)==(1,1,4))
    explicit_other=any(options.get(key,'off')!='off' for key in ('completion_epochs','readout_schedule'))
    aligned=(all(ShapePredicate(0,axis,multiple_of=edge).match(facts) is True for axis in (1,2))
        if mode=='matrix_coalesced_geometry' else True)
    policy_mode=mode in ('supported_geometry','matrix_coalesced_geometry')
    selected=mode if not policy_mode else (
        'retained_lhs_wide' if matched and numerical and ordinary and aligned and not explicit_other else 'off')
    selected_loads='coalesced4' if mode=='matrix_coalesced_geometry' and selected!='off' else 'unchanged'
    # Do not invoke the solver, duplicate a plan, or suppress a concrete refusal.
    return {'schema':'gemmini.wide_geometry_policy.v1','requested':mode,'selected':selected,
        **({'selected_wide_loads':selected_loads} if mode=='matrix_coalesced_geometry' else {}),
        'geometry_class':matched,'source_shape':[problem.m,problem.k,problem.n],
        'class_checks':[{'class':c.name,'predicates':[asdict(p) for p in c.predicates],
            'matches':list(values)} for c,values in checks],
        'source_numerical_guard':numerical,'ordinary_layout_guard':ordinary,
        'explicit_other_route_precedence':explicit_other,
        **({'source_alignment_guard':aligned} if mode=='matrix_coalesced_geometry' else {}),
        'necessary_retained_lhs_rows':retained_rows,
        'status':'experimental coverage; complete paired heldouts required',
        'actual_schedule_applied':False,
        'legality':'original capacity, source/effect, alias, numerical and selected ordering checks remain authoritative'}


@dataclass
class SelectedWide:
    owner: object
    info: dict
    policy: dict

    @property
    def kernel(self): return self.owner.kernel

    @property
    def realization(self): return self.owner.realization

    def revalidate(self):
        self.owner.revalidate()
        return self

    def emit_index_program(self, *args, **kwargs):
        self.revalidate()
        result=self.owner.emit_index_program(*args, **kwargs)
        self.owner.revalidate()
        self.policy['actual_schedule_applied']=True
        return result


def build_selected_wide(mode,problem,resources,storage,adapter,options=None,*,attrs=None,
                        bias_source=None,rhs_transposed=False):
    """Return (SelectedWide or None, decision); failures retain exact diagnostics.

    Explicit retained_lhs_wide bypasses coverage only. It does not bypass any
    original proof. The caller uses its existing exact incumbent on None.
    """
    decision=select_wide_schedule(mode,problem,resources,storage,attrs=attrs,
        rhs_transposed=rhs_transposed,options=options)
    if decision['selected']=='off':return None,decision
    try:
        owner,info=build_wide_schedule(problem,resources,storage,adapter,options,
            bias_source=bias_source,rhs_transposed=rhs_transposed)
    except ValueError as error:
        return None,{**decision,'selected':'off','schedule_refusal':str(error)}
    if decision.get('selected_wide_loads')=='coalesced4':
        # dense_wide may already consume an explicit global coalesced choice.
        # Never wrap an exact checked owner twice. Otherwise use its ACTUAL
        # realized limits and block, not the caller's smaller incumbent limits.
        if 'wide_load_coalescing' not in info:
            from dense_wide_loads import maybe_coalesce
            maxima=info['planning_maxima']
            limits=replace(resources,**{'max_'+axis+'_blocks':maxima[axis] for axis in ('m','k','n')})
            try:
                owner,info=maybe_coalesce(owner,problem,limits,storage,adapter,info,mode='coalesced4')
            except ValueError as error:
                return None,{**decision,'selected':'off','schedule_refusal':str(error)}
        decision={**decision,'coalescing_checked':True,
            'checked_coalesced_kernel_sha256':owner.kernel.digest()}
    decision={**decision,'checked_schedule_built':True}
    return SelectedWide(owner,info,decision),decision


# Frozen development panel: provenance only, never consulted by selection.
# No unlisted source is submitted to a broker. Untimed geometry neighbors may
# be checked locally; exact admitted public sources are the paired-test panel.
MATRIX_COALESCING_EVIDENCE = {
    'status':'experimental training support only; no validated geometry cutoff',
    'artifact':'artifacts/experiments/wide_coalescing036/evidence.json',
    'program_pins':'artifacts/experiments/wide_coalescing036/program_pins.json',
    'complete_modeled_training':[
        {'m':16,'k':2048,'n':2560,'wide_cycles':1623056,'coalesced_cycles':1164350},
        {'m':16,'k':2048,'n':11264,'wide_cycles':6952750,'coalesced_cycles':4915287}],
    'negative_measured_witness':{'before_cycles':5020,'after_cycles':5124,
        'qualification':'small explicit-ready witness; geometry class remains experimental and narrow'},
    'qualification':'complete PROVISIONAL Spike same-model comparisons; no RTL win claim',
    'hypothesis':'full selected hardware M edge plus retained complete K reuses LHS across wide output groups; checked RHS bands amortize actual load issue while preserving all completions',
    'direct_incumbent_comparison':{'artifact':'artifacts/experiments/matrix_policy038/evidence.json','qkv_before':6930312,'qkv_after':1164350,'gu_before':31760552,'gu_after':4915287,'negative_N2048_controls':'unchanged'},
    'missing':'independent inside-class M/K/N cutoff timing coverage'}

FROZEN_MATRIX_PANEL = {
    'schema':'gemmini.wide_matrix_policy_panel.v1',
    'selection_inputs':'only source geometry/resources/numerical views, never these names',
    'training':[
        {'source':'TL_mm_qkv_fused_m16k2048n2560_i8.mlir','shape':[16,2048,2560]},
        {'source':'TL_mm_gu_fused_m16k2048n11264_i8.mlir','shape':[16,2048,11264]}],
    'admitted_public_guards':[
        {'source':'TL_mm_o_m16k2048n2048_raw.mlir','shape':[16,2048,2048],'reason':'same M/K at N boundary'},
        {'source':'TL_mm_down_m16k5632n2048_raw.mlir','shape':[16,5632,2048],'reason':'width and full-LHS capacity boundary'},
        {'source':'JT_tiny_decode_m1k2048n32000_raw.mlir','shape':[1,2048,32000],'reason':'M1 and output coverage boundary; complete timing UNKNOWN'},
        {'source':'CH_sibling_feed_tiny_llama_decode_m1k2048h5728n2048.mlir','shape':[1,2048,5728],'reason':'M1 connected scaled exits; complete timing UNKNOWN'}],
    'local_only_untimed_neighbors':{
        'm':[8,15,16,17,32],'k':[1024,2048,4096,4112],'n':[2048,2064,2560,3072,4096,16384,16400],
        'broker_eligible':False,'purpose':'predicate/capacity/numerical boundaries, not empirical timing evidence'},
    'holdout_limitation':'public inventory has no independent admitted M>1 wide source beyond two training projections'}
