"""Default-off geometry gate around the unchanged original-row source owner.

All complete source groups must occupy at least half the selected array height,
with edge-aligned K/N >= four edges. This is a performance coverage class only:
source, numerical, ABI, capacity, transport and completion legality stay with
original_row_width.prepare and its checked owners. No source rewriting occurs.
"""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json

KNOB_DECLARATION={'original_row_geometry':{'values':['off','half_array_aligned'],
 'mechanism':'all-group shared shape predicates around checked original-pointer row reuse',
 'note':'Default off preserves explicit original adapter. Half-array rows, aligned K/N>=4edge; original owner independently proves complete source/ABI/capacity. Sole public timing support is one complete source with interacting reciprocal K/N regions; broader performance unknown.'}}

def setting():
 from merlin.compile.tuning_knobs import knob
 return knob('original_row_geometry',default='off',values=('off','half_array_aligned'),root=Path(__file__).parent)

def group_policy(owner,edge):
 from merlin.transforms.pipeline_policy import PolicyFacts,ShapePredicate
 from merlin.transforms.pipeline_state import AnalysisStamp,PipelineContext
 from merlin.transforms.shape_policy import ShapeScheduleClass
 if type(edge) is not int or edge<2 or edge%2:raise ValueError('half-array policy requires selected even positive array edge')
 owner.revalidate();records=[]
 for g in owner.groups:
  shape=(len(g['lhs']),g['k'],g['n'])
  identity=sha256(json.dumps({'shape':shape,'edge':edge},sort_keys=True).encode()).hexdigest()
  facts=PolicyFacts(AnalysisStamp(identity,PipelineContext(),'original-row-source-geometry'),shapes=(shape,))
  group=ShapeScheduleClass('half_array_aligned_original_rows',(
   ShapePredicate(0,0,at_least=edge//2),
   ShapePredicate(0,1,at_least=4*edge),
   ShapePredicate(0,2,at_least=4*edge)),'off')
  alignment=(ShapePredicate(0,1,multiple_of=edge),ShapePredicate(0,2,multiple_of=edge))
  matches=[p.match(facts) for p in group.predicates]
  aligned=[p.match(facts) for p in alignment]
  records.append({'shape':list(shape),'geometry_class':group.name,'predicates':[asdict(p) for p in group.predicates],
   'matches':matches,'alignment_predicates':[asdict(p) for p in alignment],
   'alignment_matches':aligned,'selected':all(x is True for x in matches+aligned),
   'checked_kernel_sha256':g['schedule'].kernel.digest()})
 return {'schema':'gemmini.original_row_geometry.v1','requested':'half_array_aligned',
  'selected':'half_array_aligned' if records and all(g['selected'] for g in records) else 'off',
  'selected_array_edge':edge,'all_complete_source_groups_required':True,'groups':records,
  'legality':'unchanged original complete source/numerical/ABI/storage owners and original transport qualification',
  'performance_support':{'public_complete_call_sources':1,
   'observed_interacting_group_shapes':[[8,64,2048],[8,2048,64]],
   'scope':'sole public complete source; two interacting reciprocal K/N aspect-ratio regions; not independent shape timing',
   'local_neighbor_evidence':'source/owner/transport construction checks only; no timing',
   'broader_class_performance':'UNKNOWN'}}

def prepare(source,cb,*,max_members=8,mode=None):
 """Drop-in prepare import; off delegates exactly, geometry refusal returns None.

 Caller still controls original_row_batching=separate_rows, N reuse and lane
 transport independently. The input cb is never modified by this wrapper.
 """
 from original_row_width import prepare as original_prepare
 mode=setting() if mode is None else mode
 if mode not in ('off','half_array_aligned'):raise ValueError('unknown original row geometry mode')
 prepared=original_prepare(source,cb,max_members=max_members)
 if mode=='off' or prepared is None:return prepared
 out,owner=prepared
 from dense import _geometry
 receipt=group_policy(owner,_geometry()[0])
 if receipt['selected']=='off':return None
 out['params']['original_row_geometry']=receipt
 owner.revalidate()
 return out,owner
