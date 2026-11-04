"""Default-off exact RHS rectangles on the original multirow retained-LHS owner.

Order: existing source/resource selection -> original segmented+ready owners ->
public same-owner causal-ready band admission -> exact complete concrete
ready-stream join -> unchanged terminal encoder. Only contiguous RHS loads
within original segments widen. No stage, allocation, K order, numeric exit,
ready/last-use edge, completion or source ownership is added or removed.

The exact original DenseReadyReadoutResult and genuine public PanelTileLoadResult
remain validated; no owner is cast, copied into a new provenance or falsely qualified.
The finite complete join refuses past its one-million-effect bound. Counts do
not establish complete-call latency; this independent experiment stays off.
"""
from dataclasses import replace
from pathlib import Path
from decode_coalesced import CoalescedEpochs,exact_epoch_join

KNOB_DECLARATION={'multirow_lhs_bands':{'values':['off','coalesced4'],
 'mechanism':'public same-owner causal-ready RHS rectangle admission plus exact full ready-stream effect/completion join on unchanged multirow retained-LHS owner',
 'note':'Default off. Original multirow_lhs source/layout/override/capacity guards, same block/storage/K/readout/drains; actual RHS columns<=64. No timing qualification.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('multirow_lhs_bands',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)

class MultirowLHSBands(CoalescedEpochs):
 def __init__(self,original,checked,kernel,problem,storage,adapter,codec,receipt):
  if checked.original is not original:raise ValueError('wide owner lost its exact original ready owner')
  super().__init__(original,checked,kernel,problem,storage,adapter,codec,receipt)
  # Keep the actual public refinement report, rather than reconstruct its
  # capability/source provenance from the original primitive report.
  self.realization=dict(checked.realization)
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  report=super().emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  report['multirow_lhs_bands']=report.pop('decode_coalesced')
  self.receipt['emitted_kernel_sha256']=self.kernel.digest()
  return report

def select(mode,p,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from multirow_lhs import select as select_original
 if mode not in ('off','coalesced4'):raise ValueError('unknown multirow LHS band mode')
 decision=select_original('off' if mode=='off' else 'segmented_ready',p,resources,storage,
  attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 return dict(decision,band_mode=mode,selected='coalesced4' if mode!='off' and decision['selected']!='off' else 'off')

def build(p,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False):
 from multirow_lhs import build as build_original
 from dense_wide_loads import WideRHSAdapter
 from merlin.transforms.dense_planning import DenseBlock
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult
 from merlin.compile.scheduling.dense_ready_readout import DenseReadyReadoutResult,admit_ready_readout_tile_loads
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside original multirow LHS source/layout/capacity/explicit-option class')
 original,info=build_original(p,resources,storage,adapter,options,mode='segmented_ready',attrs=attrs,rhs_transposed=rhs_transposed)
 if type(original) is not DenseReadyReadoutResult or type(original.original) is not DenseSegmentedPipelineResult:
  raise ValueError('requires exact original ready and drained segmented owner types')
 original.revalidate();drained=original.original
 maxima=info['multirow_lhs']['planning_maxima']
 limits=replace(resources,**{'max_'+axis+'_blocks':value for axis,value in maxima.items()})
 codec=WideRHSAdapter(p,storage,adapter)
 actual,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,
  encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual original ready stream has no contiguous RHS run')
 checked=admit_ready_readout_tile_loads(original,actual,p,DenseBlock(**info['block']),limits,storage,adapter,
  project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 checked.revalidate()
 proof=exact_epoch_join(original,actual,p,storage,adapter,codec)
 receipt={'schema':'gemmini.multirow_lhs_bands.v1','selection':decision,'actual_emission_applied':False,
  'block':info['block'],'planning_maxima':maxima,'original_ready_kernel_sha256':original.kernel.digest(),
  'original_drained_kernel_sha256':drained.kernel.digest(),'actual_wide_kernel_sha256':actual.digest(),
  'changes':changes,'ready_full_verification':checked.realization,
  'epoch_join':proof,'target_addressing_evidence':codec.evidence,
  'proof_route':'public admit_ready_readout_tile_loads on exact original ready owner + complete concrete ready expansion join',
  'preserved':'original M/K/N, complete LHS retention, same private windows/block/K order/raw readout/all completions and ready/last-use edges',
  'performance_status':'experimental; no cycle claim'}
 result=MultirowLHSBands(original,checked,actual,p,storage,adapter,codec,receipt);result.revalidate()
 return result,dict(info,selected_path='multirow_lhs_bands',multirow_lhs_bands=receipt,
  before_kernel=original.kernel.text(),after_kernel=actual.text())
