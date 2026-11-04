"""Default-off multirow raw RHS band refinement and separate retention trial.

Primary epoch_bands refines the actual incumbent multirow two-window block,
with identical source, storage, K order, byte traffic, readouts and completions.
Separate rhs_reuse4 uses public schedule_dense_m_panels and its original
retained-reduction law; primitive M panels expose RHS reuse across >=2 full
source M tiles, while partial M families retain their checked RHS reloads.
Both use public admit_panel_tile_loads and a complete concrete effect join.
No storage is repartitioned; no asynchronous, arithmetic or array law is added.
"""
from dataclasses import replace,asdict
from hashlib import sha256
from pathlib import Path
import json
from decode_coalesced import CoalescedEpochs,exact_epoch_join

KNOB_DECLARATION={'multirow_raw':{'values':['off','epoch_bands','rhs_reuse4'],
 'mechanism':'exact RHS rectangles on original multirow two-window epochs; separate checked full-K RHS-retained M panels',
 'note':'Default off. epoch_bands refines actual selected two-window block for M>edge without extra bytes; rhs_reuse4 separately retains complete RHS across primitive M panels. Exact tails/original owners retained; no timing qualification.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('multirow_raw',default='off',values=('off','epoch_bands','rhs_reuse4'),root=Path(__file__).resolve().parent)

def select(mode,p,r,s,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','epoch_bands','rhs_reuse4'):raise ValueError('unknown multirow raw retention mode')
 attrs=dict(attrs or {});options=dict(options or {});shape=(p.m,p.k,p.n)
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),PipelineContext(),'multirow-rhs-panels'),shapes=(shape,))
 group=ShapeScheduleClass('multirow_original_epoch_raw' if mode=='epoch_bands' else 'multiple_complete_m_tiles_raw_projection',(
  ShapePredicate(0,0,at_least=p.tile_m+1 if mode=='epoch_bands' else 2*p.tile_m),ShapePredicate(0,1,at_least=p.tile_k+1),ShapePredicate(0,2,at_least=2*p.tile_n)),'off')
 matches=[x.match(facts) for x in group.predicates]
 raw=attrs.get('output_dtype','i32')=='i32' and not attrs.get('epilogue',[]) and not p.bias and not any(attrs.get(k) is not None for k in ('acc_scale','acc_scale_by_column','acc_scale_per_col','bias','bias_dtype','relu','pool','requant','source_epilogue'))
 ordinary=not rhs_transposed and (p.tile_m,p.tile_k,p.tile_n)==(16,16,16) and (p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)==(1,1,4)
 n_tiles=min(4,p.tiles[2],s.rhs.capacity//(p.tiles[1]*p.tile_k))
 capacity=(options.get('completion_epochs','off')=='two_window_epochs' if mode=='epoch_bands' else n_tiles>=2 and p.tiles[1]*p.tile_m<=s.lhs.capacity)
 other=any(options.get('axis_overrides',{}).values()) or options.get('readout_schedule','off')!='off' or options.get('wide_schedule','off')!='off' or options.get('completion_epochs','off') not in ('off','two_window_epochs')
 admitted=all(x is True for x in matches) and raw and ordinary and capacity and not other
 return {'schema':'gemmini.multirow_raw_selection.v1','requested':mode,'selected':mode if mode!='off' and admitted else 'off','source_shape':list(shape),'geometry_class':group.name,'predicates':[asdict(x) for x in group.predicates],'matches':matches,'raw_i32':raw,'ordinary_layout':ordinary,'capacity_candidate':capacity,'candidate_n_tiles':n_tiles,'explicit_alternative_preserved':other,'status':'experimental source/resource class; no cycle support','actual_emission_applied':False}

class MultirowRaw(CoalescedEpochs):
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  report=super().emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  report['multirow_raw']=report.pop('decode_coalesced')
  self.receipt['emitted_kernel_sha256']=self.kernel.digest()
  return report

def build(p,resources,storage,adapter,options=None,*,mode='epoch_bands',attrs=None,rhs_transposed=False):
 from dense_epochs import selected_law
 from dense_wide_loads import WideRHSAdapter
 from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission,schedule_dense_m_panels
 from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside original raw/multirow/layout/capacity domain')
 if mode=='epoch_bands':return _build_epoch_bands(p,resources,storage,adapter,options,decision)
 limits=replace(resources,max_m_blocks=1,max_k_blocks=p.tiles[1],max_n_blocks=decision['candidate_n_tiles'])
 law,flags,queues,identity=selected_law(adapter)
 if law.unknown():raise ValueError('selected retained-reduction premises unavailable: '+repr(law.unknown()))
 admission=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
  identity+':read-only-full-k-retention',4096,4096,max(1,queues['execute']//2),queues['load'],
  max_issued_readouts=decision['candidate_n_tiles'],pending_readouts=queues['store'],readout_configuration='ordered_snapshot_before_store_issue')
 source=sha256(json.dumps({'problem':asdict(p),'dtype':['i8','i8','i32']},sort_keys=True).encode()).hexdigest()
 attempt=schedule_dense_m_panels(p,limits,storage,adapter,admission,n_tiles=decision['candidate_n_tiles'],m_tiles=1,
  name='gemmini_multirow_rhs_panels',dtypes=('i8','i8','i32'),source_identity=source,external_buffers_disjoint=True,
  static_budget=256,timeout_ms=20000)
 if attempt.result is None:raise ValueError('original M-panel owner refused: '+json.dumps(attempt.feedback,sort_keys=True))
 original=attempt.result;original.revalidate()
 codec=WideRHSAdapter(p,storage,adapter)
 proposal,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual retained owner has no coalescible RHS run')
 # The public consumer automatically selects verify_dense_retained_reduction
 # for this typed original admission; no panel-overlap law is substituted.
 # Read actual checked plan to retain all block accounting fields, not a
 # synthesized resource estimate (the verifier consumes the original plan).
 from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan,read_dense_schedule_plan
 plan=parse_dense_schedule_plan(original.realization['schedule_ir']['text'])
 decision_plan=read_dense_schedule_plan(plan,p,limits,storage,adapter,source_identity=source,dtypes=('i8','i8','i32'),static_budget=256)
 block=decision_plan.block
 checked=admit_panel_tile_loads(original,proposal,p,block,limits,storage,adapter,project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 proof=exact_epoch_join(original,checked.kernel,p,storage,adapter,codec)
 receipt={'schema':'gemmini.multirow_raw.v1','selection':decision,'actual_emission_applied':False,'block':asdict(block),
  'm_panel_owner':attempt.feedback,'selected_law_flags':flags,'changes':changes,'drained_full_verification':checked.realization,
  'epoch_join':proof,'private_storage':asdict(storage),'target_addressing_evidence':codec.evidence,
  'preserved':'all original source reductions, full/tail M-family effects, raw i32 outputs and explicit retained-owner completions',
  'baseline_scope':'original full-K RHS-retained M-panel owner before RHS band refinement; distinct from current two-window incumbent'}
 result=MultirowRaw(original,checked,checked.kernel,p,storage,adapter,codec,receipt);result.revalidate()
 return result,{'selected_path':'multirow_rhs_panels','multirow_raw':receipt,'decode_projection':decision,'block':asdict(block),
  'schedule_ir':original.realization['schedule_ir']['text'],'before_kernel':original.kernel.text(),'after_kernel':checked.kernel.text()}


def _build_epoch_bands(p,resources,storage,adapter,options,decision):
 """Only address refinement of the ACTUAL selected multirow epoch block."""
 from dense_epochs import build_epoch_schedule
 from dense_wide_loads import WideRHSAdapter
 from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.transforms.dense_planning import DenseBlock
 original,info=build_epoch_schedule(p,resources,storage,adapter,dict(options))
 if not hasattr(original,'original'):raise ValueError('requires actual selected two-window epoch owner')
 drained=original.original;codec=WideRHSAdapter(p,storage,adapter)
 proposal,drained_changes=propose_tile_load_coalescing(drained.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 checked=admit_panel_tile_loads(drained,proposal,p,DenseBlock(**info['block']),resources,storage,adapter,
  project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 actual,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual incumbent multirow block has no coalescible RHS run')
 proof=exact_epoch_join(original,actual,p,storage,adapter,codec)
 receipt={'schema':'gemmini.multirow_raw_epoch_bands.v1','selection':decision,'actual_emission_applied':False,
  'block':info['block'],'drained_changes':drained_changes,'changes':changes,'drained_full_verification':checked.realization,
  'epoch_join':proof,'private_storage':asdict(storage),'target_addressing_evidence':codec.evidence,
  'preserved':'actual selected block/K order/private storage/all effects and completions, including every M/K/N tail',
  'baseline_scope':'actual selected multirow two-window epoch owner; only RHS load rectangles change'}
 result=MultirowRaw(original,checked,actual,p,storage,adapter,codec,receipt);result.revalidate()
 return result,dict(info,selected_path='multirow_epoch_bands',multirow_raw=receipt,decode_projection=decision,
  before_kernel=original.kernel.text(),after_kernel=actual.text())
