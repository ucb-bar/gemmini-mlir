"""Default-off RHS coalescing of an unchanged, already selected dense epoch.

Initial experimental coverage is one full M tile, aligned raw i32 contractions
within the existing N<=2048 epoch-policy coverage. This is not a profitable
crossover claim. Admission, storage and full effect/completion equality remain
with the original library owners. No segmented/full-K retention or drain
movement occurs. An incumbent without epochs or adjacent RHS loads refuses.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from decode_coalesced import CoalescedEpochs,exact_epoch_join

KNOB_DECLARATION={'dense_epoch_coalescing':{
 'values':['off','coalesced4'],
 'mechanism':'original checked two-window dense epochs with exact-owner RHS rectangles',
 'note':'Default off; aligned ordinary raw i32 full-height one-M-tile source only. Preserves actual incumbent block, K order, storage and every completion. No performance claim.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('dense_epoch_coalescing',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)

def select(mode,problem,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate,ResourcePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','coalesced4'):raise ValueError('unknown dense epoch coalescing mode')
 attrs=dict(attrs or {});options=dict(options or {})
 shape=(problem.m,problem.k,problem.n);res=(('acc_private_rows',storage.accumulator.capacity),)
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),PipelineContext(resources=sha256(json.dumps(res).encode()).hexdigest()),'dense-epoch-coalescing'),shapes=(shape,),resources=res)
 group=ShapeScheduleClass('aligned_full_height_single_m_tile_epochs',(
  ShapePredicate(0,0,at_least=problem.tile_m,at_most=problem.tile_m),
  ShapePredicate(0,1,at_least=problem.tile_k+1),
  ShapePredicate(0,2,at_least=2*problem.tile_n,at_most=2048),
  ResourcePredicate('acc_private_rows',problem.tile_m)),'off')
 matches=[p.match(facts) for p in group.predicates]
 aligned=all(d%t==0 for d,t in zip(shape,(problem.tile_m,problem.tile_k,problem.tile_n)))
 raw=not problem.bias and attrs.get('output_dtype','i32')=='i32' and not attrs.get('epilogue',[]) and not any(attrs.get(k) is not None for k in ('bias','bias_dtype','acc_scale','acc_scale_per_col','relu','pool','requant'))
 ordinary=not rhs_transposed and problem.lhs_bytes==1 and problem.rhs_bytes==1 and problem.accumulator_bytes==4
 epoch=options.get('completion_epochs','off')=='two_window_epochs' and options.get('readout_schedule','off')=='off'
 admitted=all(x is True for x in matches) and aligned and raw and ordinary and epoch
 return {'schema':'gemmini.dense_epoch_coalescing_selection.v1','requested':mode,'selected':mode if mode!='off' and admitted else 'off','geometry_class':group.name,'source_shape':list(shape),'predicates':[asdict(p) for p in group.predicates],'matches':matches,'aligned':aligned,'raw_i32':raw,'ordinary_layout':ordinary,'incumbent_two_window_epochs':epoch,'status':'experimental; no paired complete timing','actual_emission_applied':False}

class DenseCoalescedEpochs(CoalescedEpochs):
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  report=super().emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  report['dense_epoch_coalescing']=report.pop('decode_coalesced')
  return report

def build(problem,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False):
 from dense_epochs import build_epoch_schedule
 from dense_wide_loads import WideRHSAdapter
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
 from merlin.transforms.dense_planning import DenseBlock
 options=dict(options or {})
 decision=select(mode,problem,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('source/layout/numeric or actual incumbent epoch outside aligned multirow coalescing mode')
 original,info=build_epoch_schedule(problem,resources,storage,adapter,options)
 if not hasattr(original,'original'):raise ValueError('actual incumbent is not checked two-window epochs')
 drained=original.original;codec=WideRHSAdapter(problem,storage,adapter)
 proposal,drained_changes=propose_tile_load_coalescing(drained.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 checked=admit_panel_tile_loads(drained,proposal,problem,DenseBlock(**info['block']),resources,storage,adapter,project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=5000)
 actual,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual incumbent block has no coalescible RHS run')
 proof=exact_epoch_join(original,actual,problem,storage,adapter,codec)
 receipt={'schema':'gemmini.dense_epoch_coalescing.v1','selection':decision,'actual_emission_applied':False,'changes':changes,'drained_changes':drained_changes,'drained_full_verification':checked.realization,'epoch_join':proof,'target_addressing_evidence':codec.evidence,'preserved':'actual incumbent block, storage, complete K order and every completion'}
 result=DenseCoalescedEpochs(original,checked,actual,problem,storage,adapter,codec,receipt);result.revalidate()
 return result,dict(info,selected_path='dense_coalesced_epochs',dense_epoch_coalescing=receipt,decode_projection=decision,before_kernel=original.kernel.text(),after_kernel=actual.text())
