"""Experimental geometry coverage for the existing checked wide dense route.

No default or empirical crossover is selected here. The complete paired QKV
observation motivates held-out M/K/N sweeps; the larger GU observation remains
UNKNOWN despite actual wide emission. Counts establish application, not speed.
The classes describe decode and short-matrix wide projections. Resource fit
and every numerical, alias, lifetime and ordering obligation remain with the
original dense_wide builder and common terminal effect encoder.

Integration may use select_wide_schedule to choose the existing route before
its current build hook, or build_selected_wide to obtain that actual checked
owner. SelectedWide.emit_index_program forwards the same callbacks and marks
application only after consuming the checked indexed program. Explicit off and
retained_lhs_wide keep their original meanings. No profiling occurs at compile
time. The new supported_geometry mode is default-off pending paired holdouts.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math

from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate, ResourcePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from merlin.transforms.shape_policy import ShapeScheduleClass
from dense_wide import build_wide_schedule

KNOB_DECLARATION = {'dense_wide_schedule': {
    'values': ['off', 'retained_lhs_wide', 'supported_geometry'],
    'mechanism': 'source/resource geometry coverage over checked retained-LHS segmented ready readout',
    'note': 'off stays default; supported_geometry offers N>2048,K>edge,M<=4edge ordinary scalar projections only when original capacity planning admits. Experimental holdouts pending; no cycle claim for UNKNOWN GU.'}}

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
    if mode not in ('off','retained_lhs_wide','supported_geometry'):
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
    selected=mode if mode!='supported_geometry' else (
        'retained_lhs_wide' if matched and numerical and ordinary and not explicit_other else 'off')
    # Do not invoke the solver, duplicate a plan, or suppress a concrete refusal.
    return {'schema':'gemmini.wide_geometry_policy.v1','requested':mode,'selected':selected,
        'geometry_class':matched,'source_shape':[problem.m,problem.k,problem.n],
        'class_checks':[{'class':c.name,'predicates':[asdict(p) for p in c.predicates],
            'matches':list(values)} for c,values in checks],
        'source_numerical_guard':numerical,'ordinary_layout_guard':ordinary,
        'explicit_other_route_precedence':explicit_other,
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
    decision={**decision,'checked_schedule_built':True}
    return SelectedWide(owner,info,decision),decision
