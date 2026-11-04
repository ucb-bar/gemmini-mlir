"""Explicit bounded planning alternatives on the original retained-LHS owner.

M2/M4 and N4/N8/N16 are planning maxima, not mandatory physical block sizes or
empirical cutoffs. Original source/layout/explicit-option guards precede actual
capacity planning. The original public segmented+ready owner is constructed
with unchanged physical banks and complete source M/K/N. Its exact same-owner
RHS refinement and exhaustive concrete effect/completion join are consumed by
098's existing MultirowLHSBands terminal. Unsupported cases retain the incumbent.
All alternatives are experimental; default off, no profitability qualification.
"""
from dataclasses import asdict,replace
from pathlib import Path

MODES=('off','m2n4','m2n8','m2n16','m4n4','m4n8','m4n16')
KNOB_DECLARATION={'multirow_lhs_geometry':{'values':list(MODES),
 'mechanism':'bounded M/N planning maxima on the original segmented causal-ready retained-LHS owner plus exact same-owner RHS rectangles',
 'note':'Default off. Original source, private physical capacity, explicit-option and completion guards; M/N maxima are experimental alternatives, not profitability cutoffs.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('multirow_lhs_geometry',default='off',values=MODES,root=Path(__file__).resolve().parent)

def select(mode,p,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from multirow_lhs import select as original_select
 if mode not in MODES:raise ValueError('unknown bounded multirow LHS geometry mode')
 decision=original_select('off' if mode=='off' else 'segmented_ready',p,resources,storage,
  attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 return dict(decision,requested=mode,selected=mode if mode!='off' and decision['selected']!='off' else 'off',
  planning_choice={'m':int(mode[1]),'n':int(mode.split('n')[1])} if mode!='off' else None,
  performance_status='explicit unmeasured planning alternative; legality and complete-call profitability remain separate')

def build(p,resources,storage,adapter,options=None,*,mode='m2n8',attrs=None,rhs_transposed=False):
 from dense_epochs import build_readout_schedule
 from dense_wide_loads import WideRHSAdapter
 from multirow_lhs_bands import MultirowLHSBands
 from decode_coalesced import exact_epoch_join
 from merlin.transforms.dense_planning import DenseBlock
 from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult,dense_segmented_blocks
 from merlin.compile.scheduling.dense_ready_readout import DenseReadyReadoutResult,admit_ready_readout_tile_loads
 from merlin.compile.scheduling.tile_load_coalescing import propose_tile_load_coalescing
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='off':raise ValueError('outside original retained-LHS source/resource/explicit-option class')
 choice=decision['planning_choice']
 maxima={'m':min(choice['m'],p.tiles[0],storage.lhs.capacity//decision['complete_lhs_tile_rows']),
  'k':p.tiles[1]-1,'n':min(choice['n'],p.tiles[2],storage.accumulator.capacity//p.tile_m)}
 limits=replace(resources,**{'max_'+axis+'_blocks':value for axis,value in maxima.items()})
 candidates=tuple(dense_segmented_blocks(p,limits,storage))
 if not candidates:raise ValueError('original physical banks/bytes/ACC admit no bounded complete-LHS/two-RHS block')
 original,info=build_readout_schedule(p,limits,storage,adapter,dict(options or {},readout_schedule='segmented_ready'))
 if type(original) is not DenseReadyReadoutResult or type(original.original) is not DenseSegmentedPipelineResult:
  raise ValueError('requires exact original public ready and segmented owner types')
 original.revalidate()
 if p.tiles[2]<=info['block']['n']:raise ValueError('actual original block offers no cross-N LHS reuse')
 codec=WideRHSAdapter(p,storage,adapter)
 actual,changes=propose_tile_load_coalescing(original.kernel,project=codec.eligible,
  encode=codec.encode,max_tiles=4,coalesce_tail=True)
 if not changes['changes']:raise ValueError('actual original ready stream has no contiguous RHS run')
 checked=admit_ready_readout_tile_loads(original,actual,p,DenseBlock(**info['block']),limits,storage,adapter,
  project=codec.project,encode_band=codec.band,static_budget=256,timeout_ms=20000)
 checked.revalidate()
 proof=exact_epoch_join(original,actual,p,storage,adapter,codec)
 receipt={'schema':'gemmini.multirow_lhs_geometry.v1','selection':decision,'actual_emission_applied':False,
  'block':info['block'],'planning_maxima':maxima,'candidate_count':len(candidates),
  'private_storage':asdict(storage),'actual_resources':asdict(limits),
  'original_ready_kernel_sha256':original.kernel.digest(),'original_drained_kernel_sha256':original.original.kernel.digest(),
  'actual_wide_kernel_sha256':actual.digest(),'changes':changes,'ready_full_verification':checked.realization,
  'epoch_join':proof,'target_addressing_evidence':codec.evidence,
  'proof_route':'original typed segmented+ready owner; public exact same-owner RHS refinement; complete concrete ready expansion join',
  'preserved':'full source axes, original private banks, complete LHS retention, checked K segments, raw readout and every completion',
  'performance_status':'explicit alternative, no timed qualification'}
 owner=MultirowLHSBands(original,checked,actual,p,storage,adapter,codec,receipt);owner.revalidate()
 return owner,dict(info,selected_path='multirow_lhs_geometry',decode_projection=decision,
  multirow_lhs=dict(decision,planning_maxima=maxima,actual_owner_type=type(original).__name__),
  multirow_lhs_bands=receipt,multirow_lhs_geometry=receipt,
  before_kernel=original.kernel.text(),after_kernel=actual.text())

def planning_family(problem,resources,storage,adapter,options=None,*,attrs=None,rhs_transposed=False):
 """Bind real owner construction to the shared bounded exploration interface.

 The caller freezes training/holdout geometry and supplies timing externally.
 No static objective in this family claims or selects target profitability.
 """
 from merlin.transforms.pass_exploration import PassFamily
 from merlin.transforms.pipeline_policy import RegisteredPass
 from merlin.transforms.pipeline_state import PassSpec
 def bind(knobs):
  mode='m%dn%d'%(knobs['m'],knobs['n'])
  def action(state):
   return {'checked_schedule':build(problem,resources,storage,adapter,options,
    mode=mode,attrs=attrs,rhs_transposed=rhs_transposed)}
  return RegisteredPass(PassSpec('multirow_lhs_geometry',0,produces=('checked_schedule',),
   mutates_ir=False,ir_level='dense-problem'),action)
 return PassFamily('multirow_lhs_geometry',bind,(('m',(2,4)),('n',(4,8,16))))
