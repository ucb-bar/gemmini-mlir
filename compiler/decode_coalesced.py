"""Default-off RHS rectangle refinement of the existing dense panel epochs.

Unlike decode_projection this retains the incumbent two-window block, scratch
assignment, K traversal and all fact-selected epoch boundaries. It introduces
no segmented/full-LHS/ready scheduling. The actual wide loads are first admitted
by the original drained panel verifier. Their final epoch stream is decoded
back to all primitive bands and compared, including EVERY completion, against
the original checked epoch stream. A bounded exact stream join composes those
existing owners; no asynchronous law or numeric qualification is introduced.
The wider final kernel, never the checking expansion, reaches the common encoder.
"""
from pathlib import Path
from dataclasses import asdict,replace
from hashlib import sha256
import json

KNOB_DECLARATION={'decode_coalesced':{
 'values':['off','coalesced4'],
 'mechanism':'exact-owner RHS band coalescing on unchanged fact-backed two-window dense epochs',
 'note':'Default off. Ordinary unbiased M1 scalar/raw exits with N within one accumulator group. Same original K order/storage/block/epoch boundaries; actual RHS DMA rectangles at most64 columns. No latency claim.'}}


def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('decode_coalesced',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)


def select(mode,problem,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate,ResourcePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 from dense_wide_policy import select_wide_schedule
 if mode not in ('off','coalesced4'):raise ValueError('unknown narrow decode coalescing mode')
 options=dict(options or {});attrs=dict(attrs or {})
 rows=storage.accumulator.capacity
 resource_facts=(('acc_private_rows',rows),)
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps([problem.m,problem.k,problem.n]).encode()).hexdigest(),
  PipelineContext(resources=sha256(json.dumps(resource_facts).encode()).hexdigest()),'decode-coalesced-source'),
  shapes=((problem.m,problem.k,problem.n),),resources=resource_facts)
 group=ShapeScheduleClass('single_accumulator_group_row_projection',(
  ShapePredicate(0,0,at_least=1,at_most=1),ShapePredicate(0,1,at_least=problem.tile_k+1),
  ShapePredicate(0,2,at_least=2*problem.tile_n,at_most=min(64,rows//problem.tile_m)*problem.tile_n),
  ResourcePredicate('acc_private_rows',problem.tile_m)),'off')
 numeric=select_wide_schedule('retained_lhs_wide',problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 checks=[p.match(facts) for p in group.predicates]
 # Axis controls remain authoritative. An explicitly selected readout or
 # drained alternative is not silently replaced by this independent experiment.
 other=(options.get('readout_schedule','off')!='off' or
  options.get('completion_epochs','off') not in ('off','two_window_epochs'))
 admitted=all(v is True for v in checks) and numeric['source_numerical_guard'] and numeric['ordinary_layout_guard'] and not other
 return {'schema':'gemmini.decode_coalesced_selection.v1','requested':mode,
  'selected':mode if mode!='off' and admitted else 'off','source_shape':[problem.m,problem.k,problem.n],
  'geometry_class':group.name,'predicates':[asdict(p) for p in group.predicates],'matches':checks,
  'source_numerical_guard':numeric['source_numerical_guard'],'ordinary_layout_guard':numeric['ordinary_layout_guard'],
  'other_explicit_route':other,'actual_emission_applied':False,
  'performance_status':'experimental; no paired complete timing yet'}


def exact_epoch_join(original,actual,problem,storage,adapter,codec,*,max_calls=1000000):
 """Decode all bands and prove complete concrete effect/completion equality.

 A finite proof over the actual static source domain, not a sampled oracle.
 Refuse beyond the explicit budget; no truncation, ignored fields or stores.
 The shared dense decoder independently validates every source/local band.
 """
 from merlin.compile.scheduling.panel_tile_loads import expand_dense_panel_tile_loads
 from merlin.sched.ir import instances,concretize
 from itertools import zip_longest
 original.revalidate()
 expanded,projection=expand_dense_panel_tile_loads(actual,problem,storage,adapter,
  project=codec.project,encode_band=codec.band,max_static_nodes=4096)
 count=0
 for left,right in zip_longest(instances(original.kernel),instances(expanded)):
  count+=1
  if count>max_calls:raise ValueError('exact epoch-band composition exceeds concrete effect proof budget')
  if left is None or right is None:raise ValueError('epoch-band composition changes effect count')
  a,ae=left;b,be=right
  if replace(a,args=())!=replace(b,args=()) or concretize(a,ae)!=concretize(b,be):
   raise ValueError('epoch-band composition changes complete decoded effect/completion at index '+str(count-1))
 original.revalidate()
 return {'schema':'gemmini.exact_epoch_band_join.v1','compared_complete_effects':count,
  'maximum_effects':max_calls,'original_epoch_kernel_sha256':original.kernel.digest(),
  'actual_wide_epoch_kernel_sha256':actual.digest(),'expanded_kernel_sha256':expanded.digest(),
  'projection':projection,'proof':'all concretized instruction operands and completions in identical ascending order'}


class CoalescedEpochs:
 """Own the actual wide program and both independently checked prerequisites."""
 def __init__(self,original,drained_wide,kernel,problem,storage,adapter,codec,receipt):
  self.original,self.drained_wide,self.kernel=original,drained_wide,kernel
  self.problem,self.storage,self.adapter,self.codec=problem,storage,adapter,codec
  self.receipt=receipt;self.checked_hash=kernel.digest();self.realization=dict(original.realization,
   kernel_sha256=kernel.digest(),decode_rhs_band_composition=receipt)
 def revalidate(self):
  self.original.revalidate();self.drained_wide.revalidate()
  if self.kernel.digest()!=self.checked_hash:raise ValueError('actual coalesced epoch commands changed after checking')
  proof=exact_epoch_join(self.original,self.kernel,self.problem,self.storage,self.adapter,self.codec)
  if proof!=self.receipt['epoch_join']:raise ValueError('epoch band proof changed after checking')
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  from merlin.compile.scheduling.llvm_program import emit_index_effect_program
  from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
  self.revalidate()
  report=emit_index_effect_program(self.kernel,control,lower_effect,complete_epoch=complete_epoch,completion_instruction=DRAIN)
  self.revalidate();self.receipt['actual_emission_applied']=True
  report['decode_coalesced']=self.receipt
  return report


def build(problem,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False):
 """Return actual checked wide epoch owner and ordinary dense block metadata.

 Parent hook before generic epochs can use this as result,epoch_info,epochs=True.
 Unsupported/failed exact joins raise ValueError before target commands; preserve
 original incumbent. Pass original resources/options, not new storage budgets.
 """
 from dense_epochs import build_epoch_schedule
 from dense_wide_loads import WideRHSAdapter
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
 from merlin.transforms.dense_planning import DenseBlock
 options=dict(options or {})
 decision=select(mode,problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('source/layout/numeric geometry outside narrow decode coalescing mode')
 epoch_options=dict(options,completion_epochs='two_window_epochs',outer_order=options.get('outer_order',('m','n')))
 original,info=build_epoch_schedule(problem,resources,storage,adapter,epoch_options)
 if not hasattr(original,'original'):raise ValueError('coalesced epochs requires original checked two-window epoch owner')
 drained=original.original
 codec=WideRHSAdapter(problem,storage,adapter)
 # Both proposals are pure address-only transformations of their ACTUAL owner;
 # all band effects are separately decoded and admitted, not copied receipts.
 drained_proposal,drained_changes=propose_tile_load_coalescing(drained.kernel,
  project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 checked=admit_panel_tile_loads(drained,drained_proposal,problem,DenseBlock(**info['block']),
  resources,storage,adapter,project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=5000)
 actual,changes=propose_tile_load_coalescing(original.kernel,
  project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual incumbent panel has no coalescible RHS run under requested block controls')
 proof=exact_epoch_join(original,actual,problem,storage,adapter,codec)
 receipt={'schema':'gemmini.coalesced_dense_epochs.v1','selection':decision,
  'actual_emission_applied':False,'changes':changes,'drained_changes':drained_changes,
  'drained_full_verification':checked.realization,'epoch_join':proof,
  'target_addressing_evidence':codec.evidence,
  'preserved':'original block, storage, complete K order, all original epoch drains, selected queue bounds and exact scalar readout'}
 result=CoalescedEpochs(original,checked,actual,problem,storage,adapter,codec,receipt)
 result.revalidate()
 return result,dict(info,selected_path='decode_coalesced_epochs',decode_coalesced=receipt,
  decode_projection=decision,before_kernel=original.kernel.text(),after_kernel=actual.text())
