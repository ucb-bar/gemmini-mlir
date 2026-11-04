"""Default-off M1 retained-input streaming with a checked partial N group.

The existing wide domain uses gcd(64,Ntiles), reducing ragged large projections
to tiny groups. This adapter offers the original owner the actual resource-
bounded full group plus its partial final group. Original selected laws, typed
source, ownership, complete K order and final-K readout are unchanged. Primitive
RHS transfers are deliberate: the existing exact wide-load normal-form checker
currently refuses this domain, and that proof is not bypassed. No cycle claim.
"""
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import json

KNOB_DECLARATION={'sibling_stream':{'values':['off','grouped_tail'],
 'mechanism':'checked retained-LHS segmented ready projection with actual partial final output group',
 'note':'Default off; ordinary M1 ragged output-group scalar/raw projections, complete padded LHS fits existing private view; no new storage or numeric law. Preserves previously selected routes.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('sibling_stream',default='off',values=('off','grouped_tail'),root=Path(__file__).resolve().parent)

def select(mode,problem,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate,ResourcePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 from dense_wide_policy import select_wide_schedule
 if mode not in ('off','grouped_tail'):raise ValueError('unknown grouped-tail streaming mode')
 attrs=dict(attrs or {});options=dict(options or {})
 cap=min(64,storage.accumulator.capacity//problem.tile_m)
 shape=(problem.m,problem.k,problem.n)
 rf=(('lhs_private_rows',storage.lhs.capacity),('acc_private_rows',storage.accumulator.capacity))
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),PipelineContext(resources=sha256(json.dumps(rf).encode()).hexdigest()),'grouped-tail-projection'),shapes=(shape,),resources=rf)
 group=ShapeScheduleClass('row_projection_partial_output_group',(
  ShapePredicate(0,0,at_least=1,at_most=1),ShapePredicate(0,1,at_least=problem.tile_k+1),
  ShapePredicate(0,2,at_least=cap*problem.tile_n+1),
  ResourcePredicate('lhs_private_rows',problem.tiles[1]*problem.tile_m)),'off')
 numeric=select_wide_schedule('retained_lhs_wide',problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 matches=[p.match(facts) for p in group.predicates]
 partial=cap>1 and problem.tiles[2]%cap!=0
 other=any(options.get('axis_overrides',{}).values()) or options.get('readout_schedule','off')!='off' or options.get('completion_epochs','off')!='off' or options.get('wide_schedule','off')!='off'
 admit=all(x is True for x in matches) and partial and numeric['source_numerical_guard'] and numeric['ordinary_layout_guard'] and not other and not problem.bias
 return {'schema':'gemmini.sibling_stream_selection.v1','requested':mode,'selected':mode if mode!='off' and admit else 'off','geometry_class':group.name,'shape':list(shape),'predicates':[asdict(p) for p in group.predicates],'matches':matches,'output_group_tiles':cap,'partial_output_group':partial,'source_numerical_guard':numeric['source_numerical_guard'],'ordinary_layout_guard':numeric['ordinary_layout_guard'],'explicit_alternative_preserved':other,'status':'experimental; no complete-cycle evidence','actual_emission_applied':False}

class StreamOwner:
 def __init__(self,owner,receipt):self.owner,self.receipt=owner,receipt
 @property
 def kernel(self):return self.owner.kernel
 @property
 def realization(self):return self.owner.realization
 def revalidate(self):return self.owner.revalidate()
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  self.revalidate();result=self.owner.emit_index_program(control,lower_effect,complete_epoch=complete_epoch);self.revalidate()
  self.receipt['actual_emission_applied']=True;self.receipt['emitted_kernel_sha256']=self.kernel.digest();result['sibling_stream']=self.receipt
  return result

def build(problem,resources,storage,adapter,options=None,*,mode='grouped_tail',attrs=None,rhs_transposed=False):
 from dense_epochs import build_readout_schedule
 from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks
 options=dict(options or {})
 decision=select(mode,problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside ordinary M1 partial-group source/storage/numeric domain')
 maxima={'m':1,'k':problem.tiles[1]-1,'n':decision['output_group_tiles']}
 limits=replace(resources,**{'max_'+a+'_blocks':v for a,v in maxima.items()})
 choices=tuple(dense_segmented_blocks(problem,limits,storage))
 if not choices:raise ValueError('complete padded LHS plus both original RHS segments do not fit')
 owner,info=build_readout_schedule(problem,limits,storage,adapter,dict(options,readout_schedule='segmented_ready'),timeout_ms=20000)
 owner.revalidate()
 if info['block']['n']<=1 or problem.tiles[2]%info['block']['n']==0:
  raise ValueError('actual checked block has no partial final output group')
 receipt={'schema':'gemmini.sibling_stream.v1','selection':decision,'source_shape':[problem.m,problem.k,problem.n],'planning_maxima':maxima,'block':info['block'],'candidate_count':len(choices),'owner_kernel_sha256':owner.kernel.digest(),'storage':asdict(storage),'verification_timeout_ms':20000,'rhs_loads':'original primitive; no unproved rectangle refinement','numeric_contract':'unchanged dense_wide positive scalar-i8/RNE/ReLU or raw-i32 source guard and common terminal readout','actual_emission_applied':False}
 result=StreamOwner(owner,receipt)
 return result,dict(info,selected_path='grouped_tail_projection',sibling_stream=receipt,decode_projection=decision,mechanism='actual partial output-group domain on original retained-LHS segmented ready owner')
