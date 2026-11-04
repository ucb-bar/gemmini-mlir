"""Default-off shared geometry gate on the unchanged scalar-i8 epoch-band owner.

Only one K and two wide N values have completed public timing support. The
128-edge K/N minima freeze a coarse experimental coverage class, not a learned
K crossover. Alignment/layout/numerics and all owner proofs remain separate.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import dense_quantized_bands as original

KNOB_DECLARATION={'quantized_geometry':{'values':['off','large_aligned'],
 'mechanism':'shared full-height/large-aligned geometry gate on checked scalar-i8 panel4 RHS bands',
 'note':'Default off delegates prior explicit behavior. M=edge,K/N>=128edge and edge alignment; original numeric/layout/capacity/override proofs unchanged. Only K2048,N2560/11264 timed; no independent K crossover support.'}}

def geometry_setting():
 from merlin.compile.tuning_knobs import knob
 return knob('quantized_geometry',default='off',values=('off','large_aligned'),root=Path(__file__).parent)

def setting():
 """Drop-in old knob reader: new class activates the existing checked route."""
 mode=geometry_setting()
 if mode not in ('off','large_aligned'):raise ValueError('unknown quantized geometry mode')
 return original.setting() if mode=='off' else 'coalesced4'

def geometry_select(mode,p):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if mode not in ('off','large_aligned'):raise ValueError('unknown quantized geometry mode')
 shape=(p.m,p.k,p.n);edge=p.tile_m
 facts=PolicyFacts(AnalysisStamp(sha256(json.dumps([shape,edge]).encode()).hexdigest(),PipelineContext(),'large-scalar-panel-geometry'),shapes=(shape,))
 group=ShapeScheduleClass('full_height_large_scalar_projection',(
  ShapePredicate(0,0,at_least=edge,at_most=edge),
  ShapePredicate(0,1,at_least=128*edge),ShapePredicate(0,2,at_least=128*edge)),'off')
 alignment=(ShapePredicate(0,1,multiple_of=edge),ShapePredicate(0,2,multiple_of=edge))
 matches=[x.match(facts) for x in group.predicates];aligned=[x.match(facts) for x in alignment]
 return {'schema':'gemmini.quantized_geometry.v1','requested':mode,
  'selected':mode if mode!='off' and all(x is True for x in matches+aligned) else 'off',
  'geometry_class':group.name,'source_shape':list(shape),'selected_edge':edge,
  'predicates':[asdict(x) for x in group.predicates],'matches':matches,
  'alignment_predicates':[asdict(x) for x in alignment],'alignment_matches':aligned,
  'legality':'original scalar source/layout/explicit-override/capacity and complete epoch/refinement owners remain independently required',
  'performance_support':{'observed_shapes':[[16,2048,2560],[16,2048,11264]],
   'status':'two complete provisional modeled comparisons; no independent in-class timing holdout',
   'K_crossover':'UNKNOWN;128-edge threshold is a frozen coarse coverage guard, not an empirically learned crossover',
   'local_neighbor_checks':'construction/numerical evidence only; broader performance UNKNOWN'}}

def select(mode,p,r,s,*,attrs=None,rhs_transposed=False,options=None,geometry_mode=None):
 gm=geometry_setting() if geometry_mode is None else geometry_mode
 if gm=='off':return original.select(mode,p,r,s,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 decision=geometry_select(gm,p)
 if decision['selected']=='off':return {'selected':'off','quantized_geometry':decision,'original_legality':'not constructed; outside geometry'}
 result=original.select(mode,p,r,s,attrs=attrs,rhs_transposed=rhs_transposed,options=options)
 return dict(result,quantized_geometry=decision)

def build(p,resources,storage,adapter,options=None,*,mode='coalesced4',attrs=None,rhs_transposed=False,geometry_mode=None):
 gm=geometry_setting() if geometry_mode is None else geometry_mode
 if gm=='off':return original.build(p,resources,storage,adapter,options,mode=mode,attrs=attrs,rhs_transposed=rhs_transposed)
 decision=geometry_select(gm,p)
 if decision['selected']=='off':raise ValueError('outside large aligned scalar geometry: '+json.dumps(decision,sort_keys=True))
 owner,info=original.build(p,resources,storage,adapter,options,mode=mode,attrs=attrs,rhs_transposed=rhs_transposed)
 owner.revalidate()
 return owner,dict(info,quantized_geometry=dict(decision,checked_kernel_sha256=owner.kernel.digest(),
  consumed_owner_type=type(owner).__name__,emission='unchanged actual original owner/common terminal encoder; see original actual-emission receipt'))
