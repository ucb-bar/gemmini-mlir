"""Default-off bounded-output M1 segmented/ready projection adapter.

Reuses the original dense_epochs selected-law builder and exact ready owner.
The missing behavior is a capacity-derived redivision of the ORIGINAL adjacent
private LHS/RHS union, allowing a complete one-tile LHS panel longer than the
incumbent half-bank. No extra storage, source alias, execution or numeric grant.
The original owner checks its complete source/effects, disjoint live windows,
ordered reductions/readout and every final address-only coalescing proposal.
Counts are not latency evidence. Explicit modes remain experimental.
geometry_supported is opt-in: ordinary M1 scalar/raw projections spanning more
than one capacity-bounded N group. It excludes the initial single-group
regression without fitting an arbitrary output-width cutoff. The two-group
boundary and ragged neighbors have no independent timing validation yet.
Existing explicit modes, source numeric guards and owner proofs are unchanged.
"""
from dataclasses import replace,asdict
from pathlib import Path

KNOB_DECLARATION={'decode_projection':{
 'values':['off','raw_segmented','scalar_segmented','geometry_supported'],
 'mechanism':'capacity-repartitioned retained-LHS segmented ready readout with checked RHS bands',
 'note':'Default off. M1, edge<K, N<=2048, ordinary square views. raw_segmented raw unbiased i32 only; scalar_segmented additionally admits existing positive scalar i8 readout/ReLU guard. Original checked owner and common encoder, no new numeric law. geometry_supported additionally requires multiple resource-bounded output groups and preserves explicit alternative/axis settings; experimental off by default.'}}


def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('decode_projection',default='off',values=('off','raw_segmented','scalar_segmented','geometry_supported'),root=Path(__file__).resolve().parent)


def select_projection(mode,problem,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate,ResourcePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 from hashlib import sha256
 import json
 from dense_wide_policy import select_wide_schedule
 if mode not in ('off','raw_segmented','scalar_segmented','geometry_supported'):raise ValueError('unknown decode projection mode')
 attrs=dict(attrs or {});options=dict(options or {})
 shape=((problem.m,problem.k,problem.n),)
 acc_rows=storage.accumulator.capacity
 resource_facts=(('acc_private_rows',acc_rows),)
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),
  PipelineContext(resources=sha256(json.dumps(resource_facts).encode()).hexdigest()),
  'decode-source-geometry'),shapes=shape,resources=resource_facts)
 group=ShapeScheduleClass('bounded_output_row_projection',(
  ShapePredicate(0,0,at_least=1,at_most=1),ShapePredicate(0,1,at_least=problem.tile_k+1),
  ShapePredicate(0,2,at_least=1,at_most=2048)),'off')
 # This is the existing planner's N maximum, independent of source identity.
 # More than one such group creates actual cross-output LHS reuse; a single
 # group has no such saving and regressed in the initial completed panel.
 group_n_tiles=min(64,acc_rows//problem.tile_m)
 group_columns=group_n_tiles*problem.tile_n
 if mode=='geometry_supported':
  group=ShapeScheduleClass('bounded_row_projection_multiple_output_groups',
   group.predicates+(ShapePredicate(0,2,at_least=group_columns+1),
    ResourcePredicate('acc_private_rows',problem.tile_m)), 'off')
 matches=tuple(p.match(facts) for p in group.predicates)
 overrides=any(options.get('axis_overrides',{}).values())
 selection=options.get('schedule_selection',{}).get('selection')
 explicit_other=(options.get('readout_schedule','off')!='off' or
  (options.get('completion_epochs','off')!='off' and
   selection not in ('two_window_epochs',)))
 preserve_explicit=(mode=='geometry_supported' and (overrides or explicit_other))
 # Exactly the pre-existing positive scalar/ReLU numeric admission; it does
 # not establish the experiment's N>2048 geometry, which is not reused here.
 numeric=select_wide_schedule('retained_lhs_wide',problem,resources,storage,
  attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 raw=attrs.get('output_dtype','i32')=='i32' and not attrs.get('epilogue',[])
 admitted=(all(v is True for v in matches) and numeric['ordinary_layout_guard']
  and numeric['source_numerical_guard'] and (raw or mode in ('scalar_segmented','geometry_supported')) and not preserve_explicit)
 return {'schema':'gemmini.decode_projection_selection.v1','requested':mode,
  'selected':mode if mode!='off' and admitted else 'off','source_shape':[problem.m,problem.k,problem.n],
  'geometry_class':group.name,'predicates':[asdict(p) for p in group.predicates],'matches':list(matches),
  'ordinary_layout_guard':numeric['ordinary_layout_guard'],'source_numerical_guard':numeric['source_numerical_guard'],
  'raw_output':raw,'numeric_contract':'unchanged dense_wide positive scalar i8/RNE/ReLU guard and common terminal readout',
  'actual_schedule_applied':False,'performance_status':'experimental; two-group reuse hypothesis, timing neighbors unvalidated',
  'group_n_tiles':group_n_tiles,'group_columns':group_columns,
  'explicit_alternative_preserved':preserve_explicit,
  'policy_role':'preference only; original capacity/source/numeric/completion owner remains authoritative'}


def repartition_private(problem,resources,storage):
 """Subdivide only the proven original adjacent equally addressed private union."""
 a,w=storage.lhs,storage.rhs
 if (a.backing!=w.backing or a.unit_bytes!=w.unit_bytes or a.base+a.capacity!=w.base
     or a.base<0 or storage.accumulator is None):
  raise ValueError('decode repartition requires original adjacent disjoint equal-unit private operand views')
 total=a.capacity+w.capacity
 if total*a.unit_bytes>resources.operand_bytes:
  raise ValueError('original private union exceeds selected operand byte capacity')
 lhs_rows=problem.tiles[1]*problem.tile_m
 if lhs_rows%w.alignment or a.base%a.alignment or total-lhs_rows<2*problem.tile_k:
  raise ValueError('complete padded LHS leaves no aligned space for two original checked RHS segments')
 result=replace(storage,lhs=replace(a,capacity=lhs_rows),rhs=replace(w,base=a.base+lhs_rows,capacity=total-lhs_rows))
 return result,{'original':asdict(storage),'repartitioned':asdict(result),
  'union_base':a.base,'union_rows':total,'union_bytes':total*a.unit_bytes,
  'lhs_padded_rows':lhs_rows,'rhs_double_buffer_rows':total-lhs_rows,
  'ownership':'same exclusive private union as incumbent; no new allocation or external tensor alias assumption'}


def build_projection(problem,resources,storage,adapter,options=None,*,mode='raw_segmented',attrs=None,rhs_transposed=False):
 """Return checked actual ready/coalesced owner and consumed decision record.

 No target commands emitted on refusal. Caller preserves exact incumbent.
 The original full source K order and all causal completion drains remain.
 """
 from dense_epochs import build_readout_schedule
 from dense_wide_loads import maybe_coalesce
 from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
 options=dict(options or {})
 decision=select_projection(mode,problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('source geometry/layout/numeric exit outside explicit decode projection mode')
 assigned,ownership=repartition_private(problem,resources,storage)
 maxima={'m':1,'k':problem.tiles[1]-1,'n':min(64,problem.tiles[2],storage.accumulator.capacity//problem.tile_m)}
 for axis,explicit in options.get('axis_overrides',{}).items():
  if explicit and axis in maxima:maxima[axis]=int(options.get('max_'+axis+'_blocks',getattr(resources,'max_'+axis+'_blocks')))
 limits=replace(resources,**{'max_'+a+'_blocks':v for a,v in maxima.items()})
 choices=tuple(dense_segmented_blocks(problem,limits,assigned))
 if not choices:raise ValueError('no checked complete-LHS/two-RHS-segment block fits repartitioned original union')
 owner,info=build_readout_schedule(problem,limits,assigned,adapter,dict(options,readout_schedule='segmented_ready'))
 owner.revalidate()
 if mode=='geometry_supported' and problem.tiles[2]<=info['block']['n']:
  raise ValueError('selected checked block offers no cross-output-group LHS reuse')
 info=dict(info,selected_path='decode_segmented_ready',planning_maxima=maxima,
  candidate_count=len(choices),decode_projection=decision,private_repartition=ownership,
  mechanism='existing retained_lhs_segmented_k plus causal ready readout and exact-owner RHS bands',
  terminal_encoding='unchanged dense callback and configured source scalar readout')
 owner,info=maybe_coalesce(owner,problem,limits,assigned,adapter,info,mode='coalesced4')
 owner.revalidate()
 return owner,info
