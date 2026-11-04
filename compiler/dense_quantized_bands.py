"""Default-off scalar-i8 two-window epoch RHS bands, with exact readout cut.

The unchanged raw arithmetic/epoch owner proves source products and completion.
A separate existing declared-path/full-i32 scalar theorem binds the common
terminal CONFIG_ST/MVOUT cut to the original scalar-i8 source. This is NOT a
new selected DenseIntegerReadout qualification. No raw spill/reload is emitted.
The bounded panel4 experiment is distinct from retained-LHS wide scheduling;
within that original checked epoch plan only RHS load rectangles change.
"""
from dataclasses import replace,asdict
from hashlib import sha256
import json,math,struct
from pathlib import Path
from decode_coalesced import CoalescedEpochs,exact_epoch_join

KNOB_DECLARATION={'dense_quantized_bands':{'values':['off','coalesced4'],
 'mechanism':'checked scalar-i8 panel4 two-window epochs plus exact RHS band join',
 'note':'Default off; one full M tile, ordinary unbiased positive scalar i8/RNE with optional ReLU; unchanged terminal numeric cut. Separate panel experiment, no profitable coverage claim.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('dense_quantized_bands',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)

def source_readout(attrs):
 from dense_group_readout import _path_proof
 from column_readout import _numerical_proof
 a=dict(attrs or {})
 if (a.get('output_dtype')!='i8' or a.get('epilogue') not in (['acc_scale'],['acc_scale','relu'])
  or any(a.get(k) is not None for k in ('acc_scale_by_column','acc_scale_per_col','bias','bias_dtype','pool','requant','source_epilogue'))):
  raise ValueError('requires original unbiased scalar i8 acc_scale then optional ReLU only')
 scale=a.get('acc_scale')
 if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:
  raise ValueError('source scalar must be finite and positive')
 try:bits=struct.unpack('<I',struct.pack('<f',scale))[0]
 except OverflowError:raise ValueError('source scalar must fit finite binary32')
 rounded=struct.unpack('<f',struct.pack('<I',bits))[0]
 if not math.isfinite(rounded) or rounded<=0:raise ValueError('source scalar rounds outside positive finite binary32')
 relu='relu' in a['epilogue']
 return {'schema':'gemmini.original_scalar_terminal_cut.v1','scale_bits':bits,'relu':relu,
  'declared_path':_path_proof(),'full_i32_numerical_theorem':_numerical_proof(bits,relu),
  'terminal':'unchanged dense.py CONFIG_ST source scalar/activation, i8 MVOUT flag and byte stride',
  'direct_selected_dense_readout_qualification':False,'raw_materialization':False}

def select(mode,p,r,s,*,attrs=None,rhs_transposed=False,options=None):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','coalesced4'):raise ValueError('unknown quantized epoch bands mode')
 shape=(p.m,p.k,p.n);options=dict(options or {})
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),PipelineContext(),'scalar-epoch-bands'),shapes=(shape,))
 group=ShapeScheduleClass('full_height_scalar_i8_panel',(
  ShapePredicate(0,0,at_least=p.tile_m,at_most=p.tile_m),
  ShapePredicate(0,1,at_least=p.tile_k+1),ShapePredicate(0,2,at_least=2*p.tile_n)),'off')
 matches=[x.match(facts) for x in group.predicates]
 try:readout=source_readout(attrs);numeric=True;reason=None
 except ValueError as ex:readout=None;numeric=False;reason=str(ex)
 ordinary=not rhs_transposed and not p.bias and (p.tile_m,p.tile_k,p.tile_n)==(16,16,16) and (p.lhs_bytes,p.rhs_bytes,p.accumulator_bytes)==(1,1,4)
 other=(any(options.get('axis_overrides',{}).values()) or options.get('readout_schedule','off')!='off'
  or options.get('completion_epochs','off') not in ('off','two_window_epochs')
  or options.get('wide_schedule','off')!='off')
 admitted=all(x is True for x in matches) and ordinary and numeric and not other
 return {'schema':'gemmini.dense_quantized_bands_selection.v1','requested':mode,'selected':mode if mode!='off' and admitted else 'off','source_shape':list(shape),'geometry_class':group.name,'predicates':[asdict(x) for x in group.predicates],'matches':matches,'ordinary_layout':ordinary,'source_readout':readout,'numeric_refusal':reason,'explicit_axis_or_readout_preserved':other,'performance':'experimental, no paired evidence','actual_emission_applied':False}

class QuantizedBands(CoalescedEpochs):
 def revalidate(self):
  if source_readout(self.source_attrs)!=self.receipt['source_readout']:
   raise ValueError('original scalar terminal readout changed after checking')
  return super().revalidate()
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  report=super().emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  report['dense_quantized_bands']=report.pop('decode_coalesced')
  self.receipt['emitted_kernel_sha256']=self.kernel.digest()
  return report

def build(p,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False):
 from dense_epochs import build_epoch_schedule
 from dense_wide_loads import WideRHSAdapter
 from merlin.transforms.dense_planning import DenseBlock
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
 options=dict(options or {});decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside scalar full-height source/layout/numerical domain: '+str(decision['numeric_refusal']))
 # This explicit experiment selects the existing bounded panel4 plan, rather
 # than pretending to refine the different default retained-LHS wide owner.
 limits=replace(resources,max_m_blocks=1,max_k_blocks=1,max_n_blocks=4)
 epoch_options=dict(options,completion_epochs='two_window_epochs',outer_order=tuple(options.get('outer_order',('m','n'))))
 original,info=build_epoch_schedule(p,limits,storage,adapter,epoch_options)
 if not hasattr(original,'original'):raise ValueError('requires original checked two-window epoch owner')
 drained=original.original;codec=WideRHSAdapter(p,storage,adapter)
 proposal,drained_changes=propose_tile_load_coalescing(drained.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 checked=admit_panel_tile_loads(drained,proposal,p,DenseBlock(**info['block']),limits,storage,adapter,
  project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 actual,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual original epoch plan has no coalescible RHS run')
 proof=exact_epoch_join(original,actual,p,storage,adapter,codec)
 receipt={'schema':'gemmini.dense_quantized_bands.v1','selection':decision,'source_readout':decision['source_readout'],
  'block':info['block'],'planning_maxima':{'m':1,'k':1,'n':4},'actual_emission_applied':False,
  'changes':changes,'drained_changes':drained_changes,'drained_full_verification':checked.realization,
  'epoch_join':proof,'target_addressing_evidence':codec.evidence,
  'preserved':'checked panel4 arithmetic/K/storage/drains and original common terminal scalar-i8 numeric cut',
  'baseline_scope':'same checked panel4 epoch stream before RHS refinement; differs from default retained-LHS wide plan'}
 result=QuantizedBands(original,checked,actual,p,storage,adapter,codec,receipt)
 result.source_attrs=json.loads(json.dumps(attrs));result.revalidate()
 return result,dict(info,selected_path='dense_quantized_bands',dense_quantized_bands=receipt,
  decode_projection=decision,before_kernel=original.kernel.text(),after_kernel=actual.text())
