"""Default-off panel4-only ablation of the checked scalar-i8 epoch-bands route.

Uses precisely the frozen source readout guard/theorem and original panel4
owner. No load refinement is constructed or emitted. Existing primitive
loads, original epoch boundaries and common terminal scalar readout remain.
"""
from dataclasses import replace
from pathlib import Path
import json
from dense_quantized_bands import source_readout,select as bands_select

KNOB_DECLARATION={'quantized_plan_ablation':{'values':['off','panel_only'],
 'mechanism':'same checked scalar-i8 panel4 two-window owner without RHS widening',
 'note':'Default off. Ablates panel policy versus dense_quantized_bands coalesced4; original source numeric proof and common terminal encoder unchanged. No timing qualification.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('quantized_plan_ablation',default='off',values=('off','panel_only'),root=Path(__file__).resolve().parent)

def select(mode,p,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 if mode not in ('off','panel_only'):raise ValueError('unknown quantized plan ablation mode')
 decision=bands_select('off' if mode=='off' else 'coalesced4',p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 decision.update(schema='gemmini.quantized_plan_ablation_selection.v1',requested=mode,
  selected='panel_only' if decision['selected']=='coalesced4' else 'off',
  performance='experimental panel-only ablation; no paired evidence')
 return decision

class PanelOnly:
 def __init__(self,original,attrs,receipt):
  self.original=original;self.kernel=original.kernel;self.source_attrs=json.loads(json.dumps(attrs))
  self.receipt=receipt;self.realization=original.realization
 def revalidate(self):
  self.original.revalidate()
  if self.kernel!=self.original.kernel or self.kernel.digest()!=self.receipt['original_panel_kernel_sha256']:
   raise ValueError('panel-only ablation changed original checked epoch kernel')
  if source_readout(self.source_attrs)!=self.receipt['source_readout']:
   raise ValueError('original scalar terminal readout changed after checking')
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  self.revalidate()
  report=self.original.emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  self.revalidate();self.receipt['actual_emission_applied']=True
  self.receipt['emitted_kernel_sha256']=self.kernel.digest()
  return dict(report,quantized_plan_ablation=self.receipt)

def build(p,resources,storage,adapter,options=None,*,mode='panel_only',attrs=None,rhs_transposed=False):
 from dense_epochs import build_epoch_schedule
 options=dict(options or {})
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside frozen scalar full-height source/layout/numerical domain: '+str(decision['numeric_refusal']))
 limits=replace(resources,max_m_blocks=1,max_k_blocks=1,max_n_blocks=4)
 epoch_options=dict(options,completion_epochs='two_window_epochs',outer_order=tuple(options.get('outer_order',('m','n'))))
 original,info=build_epoch_schedule(p,limits,storage,adapter,epoch_options)
 if not hasattr(original,'original'):raise ValueError('requires original checked two-window epoch owner')
 original.revalidate()
 receipt={'schema':'gemmini.quantized_plan_ablation.v1','selection':decision,
  'source_readout':decision['source_readout'],'block':info['block'],
  'planning_maxima':{'m':1,'k':1,'n':4},'actual_emission_applied':False,
  'original_panel_kernel_sha256':original.kernel.digest(),
  'rhs_refinement':'absent; actual checked primitive RHS effects emitted unchanged',
  'preserved':'frozen panel4 arithmetic, K/storage/completions and original scalar-i8 terminal cut',
  'baseline_scope':'exact original owner of dense_quantized_bands075 before RHS refinement'}
 result=PanelOnly(original,attrs,receipt);result.revalidate()
 return result,dict(info,selected_path='quantized_plan_ablation',quantized_plan_ablation=receipt,
  decode_projection=decision,before_kernel=original.kernel.text(),after_kernel=original.kernel.text())
