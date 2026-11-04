"""Default-off bounded regional choice between unchanged098 and102 owners.

The M>=two primitive rows and padded K<=32 primitive depths bound this
provisional performance evidence domain; neither is a hardware legality law.
N<=eight primitive columns chooses098 coalesced4, otherwise102 m4n8.
Original092 source/layout/resource/explicit-option guards and both actual
owners remain authoritative. No new schedule, terminal, stamp, proof budget,
retry or wall-time admission guarantee is introduced. Independent connected
holdout timing and measured-original qualification remain outstanding.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json

KNOB_DECLARATION={'multirow_lhs_policy':{'values':['off','bounded_geometry'],
 'mechanism':'bounded source/resource regional choice of original098 coalesced4 or102 m4n8 retained-LHS RHS-band owners',
 'note':'Default off; original092 eligibility plus M>=2tile_m and padded K<=32tile_k; Ntiles<=8 chooses098, otherwise102 m4n8. Provisional paired model support, independent connected and measured-original validation outstanding; no proof-time guarantee.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('multirow_lhs_policy',default='off',values=('off','bounded_geometry'),root=Path(__file__).resolve().parent)

def select(mode,p,resources,storage,*,attrs=None,rhs_transposed=False,options=None):
 from multirow_lhs import select as original_select
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','bounded_geometry'):raise ValueError('unknown multirow LHS regional policy')
 options=dict(options or {})
 original=original_select('off' if mode=='off' else 'segmented_ready',p,resources,storage,
  attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 shape=(p.m,p.k,p.n)
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps(shape).encode()).hexdigest(),
  PipelineContext(resources=sha256(json.dumps(asdict(storage),sort_keys=True).encode()).hexdigest()),
  'multirow-lhs-bounded-performance-domain'),shapes=(shape,))
 common=(ShapePredicate(0,0,at_least=2*p.tile_m),ShapePredicate(0,1,at_most=32*p.tile_k))
 classes=(ShapeScheduleClass('bounded_reduction_small_n_groups',common+(ShapePredicate(0,2,at_most=8*p.tile_n),),'incumbent'),
  ShapeScheduleClass('bounded_reduction_multiple_n_groups',common+(ShapePredicate(0,2,at_least=8*p.tile_n+1),),'incumbent'))
 matches=[[predicate.match(facts) for predicate in group.predicates] for group in classes]
 explicit=any(options.get(key,'off')!='off' for key in ('multirow_lhs','multirow_lhs_bands','multirow_lhs_geometry'))
 route='incumbent'
 if mode!='off' and original['selected']!='off' and not explicit:
  if all(v is True for v in matches[0]):route='coalesced4'
  elif all(v is True for v in matches[1]):route='m4n8'
 return {'schema':'gemmini.multirow_lhs_policy.v1','requested':mode,'selected':route,
  'source_shape':list(shape),'primitive':[p.tile_m,p.tile_k,p.tile_n],
  'padded_k':p.tiles[1]*p.tile_k,'n_tiles':p.tiles[2],
  'classes':[{'name':g.name,'predicates':[asdict(x) for x in g.predicates],'matches':m} for g,m in zip(classes,matches)],
  'original_eligibility':original,'explicit_lhs_alternative_preserved':explicit,
  'support':{'status':'provisional complete paired Spike-model evidence; no measured-original or independent connected qualification',
   'n4':'M109/K49/N101 complete modeled cycles 15220 -> 14664; unchanged 098 owner',
   'n8':'M141/K49/N149 complete modeled cycles 26978 -> 22577 and M49/K512/N2048 594908 -> 485602; unchanged 102 m4n8 owner',
   'boundaries':'M17 tail regressions motivate multirow lower bound; K<=512 bounds available reduction evidence, not a proven crossover; Ntile8 split has no independent boundary timing',
   'n16':'not selected: PD marginal advantage over n8 insufficient support',
   'references':['artifacts/experiments/integration099/dense_core_evidence/evidence.json','artifacts/experiments/integration104/dense_m4n8_core.collect.json','artifacts/experiments/integration104/dense_m4n8_scal.collect.json']},
  'uncertainty':'local neighbors establish construction only; fresh connected holdout and measured original witnesses outstanding; identical geometry may conservatively refuse wall-time-limited original proofs',
  'actual_schedule_constructed':False}

def build(p,resources,storage,adapter,options=None,*,mode='bounded_geometry',attrs=None,rhs_transposed=False):
 decision=select(mode,p,resources,storage,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 if decision['selected']=='incumbent':raise ValueError('outside bounded multirow LHS policy or explicit alternative preserved')
 if decision['selected']=='coalesced4':
  from multirow_lhs_bands import build as actual_build
 else:
  from multirow_lhs_geometry import build as actual_build
 owner,info=actual_build(p,resources,storage,adapter,options,mode=decision['selected'],attrs=attrs,rhs_transposed=rhs_transposed)
 owner.revalidate()
 decision=dict(decision,actual_schedule_constructed=True,actual_kernel_sha256=owner.kernel.digest(),
  actual_owner_type=type(owner).__name__,actual_block=info['block'])
 return owner,dict(info,multirow_lhs_policy=decision)
