"""Default-off multirow preference for the existing retained-LHS ready owner.

This is a missing conditional adapter, not a new schedule. The public segmented
owner already stages complete K per M block outside its N-group loop, retains
it until all N groups finish, and reuses private storage only at its checked
completion boundaries. The global readout flag also changes M1/M16 products;
this wrapper leaves those, quantized exits and explicit alternatives alone.
No new allocation, numerical law, source stamp or qualification is constructed.
The selected complete source M/K/N, original banks, pending queues, storage,
all reductions and ready completion remain the existing owner's responsibility.
No timing claim: larger retained panels change traffic and serialization costs.
"""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import json

KNOB_DECLARATION={'multirow_lhs':{'values':['off','segmented_ready'],
 'mechanism':'source/resource class selecting original full-LHS segmented causal ready schedule for multiple M tiles',
 'note':'Default off. Ordinary unbiased raw i8xi8->i32 only, M>edge, multiple bounded N groups and complete padded LHS M tile fits original bank. Existing owner and common encoder; preserves explicit axis/readout alternatives. No timing qualification.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('multirow_lhs',default='off',values=('off','segmented_ready'),root=Path(__file__).resolve().parent)

def select(mode,p,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate,ResourcePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','segmented_ready'):raise ValueError('unknown multirow LHS mode')
 attrs=dict(attrs or {});options=dict(options or {});shape=(p.m,p.k,p.n)
 lhs_rows=p.tiles[1]*p.tile_m
 maximum_n=min(4,p.tiles[2],storage.accumulator.capacity//p.tile_m) if storage.accumulator is not None else 0
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),
  PipelineContext(resources=sha256(json.dumps(asdict(storage),sort_keys=True).encode()).hexdigest()),
  'multirow-lhs-source-resources'),shapes=(shape,),resources=(('lhs_capacity_rows',storage.lhs.capacity),))
 group=ShapeScheduleClass('multiple_m_tiles_multiple_n_groups',(
  ShapePredicate(0,0,at_least=p.tile_m+1),ShapePredicate(0,1,at_least=p.tile_k+1),
  ShapePredicate(0,2,at_least=maximum_n*p.tile_n+1),ResourcePredicate('lhs_capacity_rows',at_least=lhs_rows)),'off')
 matches=[pred.match(facts) for pred in group.predicates]
 raw=attrs.get('output_dtype','i32')=='i32' and not attrs.get('epilogue',[]) and not p.bias and not any(attrs.get(key) is not None for key in ('acc_scale','acc_scale_by_column','acc_scale_per_col','bias','bias_dtype','relu','pool','requant','source_epilogue'))
 ordinary=not rhs_transposed and p.tile_m==p.tile_k==p.tile_n and (p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)==(1,1,4)
 explicit=options.get('wide_schedule','off')!='off' or any(options.get('axis_overrides',{}).values()) or options.get('readout_schedule','off')!='off' or options.get('completion_epochs','off') not in ('off','two_window_epochs')
 # two_window_epochs selected by the incumbent shape policy is the comparison
 # route, not an explicit request that may be silently replaced.
 if options.get('completion_epochs','off')=='two_window_epochs' and options.get('schedule_selection',{}).get('selection')!='two_window_epochs':explicit=True
 admitted=all(x is True for x in matches) and raw and ordinary and maximum_n>0 and not explicit
 return {'schema':'gemmini.multirow_lhs_selection.v1','requested':mode,'selected':mode if mode!='off' and admitted else 'off',
  'source_shape':list(shape),'geometry_class':group.name,'predicates':[asdict(pred) for pred in group.predicates],'matches':matches,
  'raw_i32_guard':raw,'ordinary_layout_guard':ordinary,'explicit_alternative_preserved':explicit,
  'complete_lhs_tile_rows':lhs_rows,'original_lhs_rows':storage.lhs.capacity,'maximum_n_blocks':maximum_n,
  'legality':'separate original typed segmented/ready source, storage, range, reduction and completion owner',
  'actual_schedule_applied':False,'performance_status':'unmeasured experimental class; all complete-call costs remain to be compared'}

def build(p,resources,storage,adapter,options=None,*,mode='segmented_ready',attrs=None,rhs_transposed=False):
 from dense_epochs import build_readout_schedule
 from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside multirow retained-LHS source/resource/explicit-option class: '+json.dumps(decision,sort_keys=True))
 maxima={'m':min(4,p.tiles[0],storage.lhs.capacity//decision['complete_lhs_tile_rows']),
  'k':p.tiles[1]-1,'n':decision['maximum_n_blocks']}
 limits=replace(resources,**{'max_'+axis+'_blocks':value for axis,value in maxima.items()})
 candidates=tuple(dense_segmented_blocks(p,limits,storage))
 if not candidates:raise ValueError('original banks/bytes/ACC do not admit any complete-LHS/two-RHS segmented block')
 owner,info=build_readout_schedule(p,limits,storage,adapter,dict(options or {},readout_schedule='segmented_ready'))
 owner.revalidate()
 if p.tiles[2]<=info['block']['n']:raise ValueError('actual original checked block offers no cross-N LHS reuse')
 return owner,dict(info,selected_path='multirow_lhs_segmented_ready',decode_projection=decision,multirow_lhs=dict(decision,
  actual_owner_type=type(owner).__name__,checked_kernel_sha256=owner.kernel.digest(),planning_maxima=maxima,
  candidate_count=len(candidates),private_storage=asdict(storage),actual_resources=asdict(limits),
  emission='unchanged original checked owner emit_index_program and dense terminal callback',
  costs='complete LHS staged once per checked M block; RHS loads, readouts and original causal drains remain'))
