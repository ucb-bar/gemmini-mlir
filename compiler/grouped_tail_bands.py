"""Default-off exact RHS-band refinement of checked grouped-tail ready streams.

The ready owner's syntactic wide-load normal form is deliberately NOT used:
first the original drained segmented owner admits the real rectangles with its
unchanged verifier, then every final ready-stream rectangle is expanded and
joined against every original concrete effect and completion. Both owners and
the complete bounded join are revalidated before and after actual emission.
This adapter moves no drain, source axis, compute, readout or storage boundary.
"""
from dataclasses import replace
from decode_coalesced import CoalescedEpochs, exact_epoch_join

KNOB_DECLARATION={'grouped_tail_bands':{'values':['off','coalesced4'],
 'mechanism':'original grouped-tail ready owner plus independently checked segmented RHS rectangles and complete concrete effect join',
 'note':'Default off; same source guards/block/storage/K/readout/drains as sibling_stream grouped_tail. At most four contiguous RHS tiles; exact bounded refusal, no latency claim.'}}

def setting():
 from pathlib import Path
 from merlin.compile.tuning_knobs import knob
 return knob('grouped_tail_bands',default='off',values=('off','coalesced4'),root=Path(__file__).resolve().parent)

class GroupedTailBands(CoalescedEpochs):
 def emit_index_program(self,control,lower_effect,*,complete_epoch):
  result=super().emit_index_program(control,lower_effect,complete_epoch=complete_epoch)
  result['grouped_tail_bands']=result.pop('decode_coalesced')
  self.receipt['emitted_kernel_sha256']=self.kernel.digest()
  return result

def build(problem,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False):
 """Return checked actual wide owner and generic dense metadata; raise to fallback.

 Parent invokes only when opted in, before primitive grouped_tail emission.
 Original source eligibility stays with sibling_stream.build. No retry grants
 permission after a concrete failure; all attempts emit nothing until checked.
 """
 if mode!='coalesced4':raise ValueError('grouped-tail band mode must be explicitly coalesced4')
 from sibling_stream import build as build_stream
 from dense_wide_loads import WideRHSAdapter
 from merlin.transforms.dense_planning import DenseBlock
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 from merlin.compile.scheduling.dense_segmented_pipeline import admit_segmented_tile_loads, DenseSegmentedPipelineResult
 from merlin.compile.scheduling.dense_ready_readout import DenseReadyReadoutResult
 wrapped,info=build_stream(problem,resources,storage,adapter,options,
  mode='grouped_tail',attrs=attrs,rhs_transposed=rhs_transposed)
 original=wrapped.owner
 if type(original) is not DenseReadyReadoutResult or type(original.original) is not DenseSegmentedPipelineResult:
  raise ValueError('requires original checked ready and drained segmented owner types')
 drained=original.original
 maxima=info['sibling_stream']['planning_maxima']
 limits=replace(resources,**{'max_'+a+'_blocks':v for a,v in maxima.items()})
 codec=WideRHSAdapter(problem,storage,adapter)
 proposal,drained_changes=propose_tile_load_coalescing(drained.kernel,
  project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not drained_changes['changes']:raise ValueError('original drained owner has no coalescible RHS run')
 checked=admit_segmented_tile_loads(drained,proposal,problem,DenseBlock(**info['block']),limits,storage,adapter,
  project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 checked.revalidate()
 actual,changes=propose_tile_load_coalescing(original.kernel,
  project=codec.eligible,encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual ready owner has no coalescible RHS run')
 proof=exact_epoch_join(original,actual,problem,storage,adapter,codec)
 receipt={'schema':'gemmini.grouped_tail_bands.v1','actual_emission_applied':False,
  'source_selection':info['sibling_stream']['selection'],'block':info['block'],
  'original_ready_kernel_sha256':original.kernel.digest(),'original_drained_kernel_sha256':drained.kernel.digest(),
  'actual_wide_kernel_sha256':actual.digest(),'changes':changes,'drained_changes':drained_changes,
  'drained_full_verification':checked.realization,'epoch_join':proof,
  'target_addressing_evidence':codec.evidence,
  'preserved':'complete original typed source effects, block/storage/K/readout, every completion in original ascending order',
  'proof_route':'public admit_segmented_tile_loads plus bounded full concrete ready-stream join; refused ready-normal-form result never accepted'}
 result=GroupedTailBands(original,checked,actual,problem,storage,adapter,codec,receipt)
 result.revalidate()
 return result,dict(info,selected_path='grouped_tail_bands',grouped_tail_bands=receipt,
  before_kernel=original.kernel.text(),after_kernel=actual.text())
