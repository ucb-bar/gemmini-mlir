"""Optional regional choice through the shared current-source pass policy.

Independent scheduling consumers still own all source, storage and completion
checks. This authored opt-in policy has native paired support on two shapes;
it is not an assessed_shape_policy qualification or complete holdout evidence.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, ShapePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
from . import isa
from .dense_pipeline import PANEL_GEOMETRIES
from .dense_weight_panels import weight_panel_schedule

LEVEL='authored_dense_weight_source'
OUTPUT='checked_dense_weight_schedule'


@dataclass(frozen=True)
class DenseWeightSource:
    m: int
    k: int
    n: int
    bias: bool=False
    column_scales: bool=False
    geometry: str='wide'
    storage_split: str='half'
    choice_policy: str='traffic'

    def __post_init__(self):
        if (any(type(value) is not int or value<=0 for value in (self.m,self.k,self.n))
                or any(type(value) is not bool for value in (self.bias,self.column_scales))
                or self.geometry not in PANEL_GEOMETRIES
                or self.storage_split not in ('half','adaptive')
                or self.choice_policy not in ('traffic','amortized_m')):
            raise ValueError('weight regional policy requires exact source and supported resource options')


def weight_regional_policy():
    """Existing predicates select a checked artifact, never grant legality."""
    return PipelinePolicy((
        PipelineVariant('weight_panels',('dense_weight_panels',),(
            ResourcePredicate('m_tiles',128),
            ShapePredicate(1,1,at_most=8*isa.DIM),
            ResourcePredicate('uniform_column_readout',1),
            ResourcePredicate('weight_full_k_legal',1),
            ResourcePredicate('weight_m_tiles',4),
        )),
        PipelineVariant('retained',('dense_weight_retained',)),
    ),fallback='retained')


def _session(source,context):
    def verify(value):
        if type(value) is not DenseWeightSource:
            raise ValueError('weight regional source changed type')
        value.__post_init__()
    return PipelineSession(source,context=context,ir_level=LEVEL,
        fingerprint=lambda value:sha256(json.dumps(asdict(value),sort_keys=True).encode()).hexdigest(),
        verify_ir=verify)


def execute_dense_weight_regional(m,k,n,bias=False,column_scales=False,*,
        geometry='wide',storage_split='half',choice_policy='traffic',context=None,
        retained_planner=None):
    """Return the actual ``(checked_schedule_or_None, refusal, receipt)``.

    No timing lookup, simulation or workload identity enters compilation. The
    fallback returns the original retained refusal for ordinary caller handling.
    Unknown caller resource context selects the unconditional retained path.
    """
    source=DenseWeightSource(m,k,n,bias,column_scales,geometry,storage_split,choice_policy)
    if context is None:
        view=dict(dim=isa.DIM,spad_rows=isa.SPAD_ROWS,acc_rows=isa.ACC_ROWS,
                  geometry=geometry,storage_split=storage_split)
        context=PipelineContext(resources='authored_dense_weight_resource_view:'+sha256(
            json.dumps(view,sort_keys=True).encode()).hexdigest())
    session=_session(source,context)
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner=_retained_schedule
    retained,retained_refusal=retained_planner(m,k,n,bias,column_scales,False,geometry,storage_split)
    if storage_split=='half':
        weight,weight_refusal=weight_panel_schedule(m,k,n,bias,column_scales,
            geometry=PANEL_GEOMETRIES[geometry],choice_policy=choice_policy)
    else:
        weight,weight_refusal=None,'weight regional candidate requires original half private assignments'
    artifacts={'retained':(retained,retained_refusal),'weight_panels':(weight,weight_refusal)}
    identities={key:None if result is None else result.kernel.digest()
                for key,(result,_) in artifacts.items()}

    def validate():
        source.__post_init__()
        expected=[('lhs',(m,k),'i8','read'),('rhs',(k,n),'i8','read')]
        if bias:expected.append(('bias',(n,),'i32','read'))
        expected.append(('output',(m,n),'i32','write'))
        for key,(result,refusal) in artifacts.items():
            if result is None:
                if not refusal or identities[key] is not None:
                    raise ValueError('missing weight policy artifact lost original refusal')
                continue
            result.revalidate()
            if refusal is not None or result.kernel.digest()!=identities[key]:
                raise ValueError('weight regional checked artifact changed')
            if [(arg.name,arg.shape,arg.dtype,arg.access) for arg in result.kernel.args]!=expected:
                raise ValueError('weight regional artifact does not implement this current source ABI')
            if key=='weight_panels':
                receipt=result.realization['weight_panels']
                if tuple(receipt['geometry_bound'])!=PANEL_GEOMETRIES[geometry]:
                    raise ValueError('weight regional candidate changed selected geometry bounds')
            elif result.realization['panel_geometry']['selection']!=geometry:
                raise ValueError('retained weight policy fallback changed selected geometry')
    validate()
    kt=(k+isa.DIM-1)//isa.DIM
    block=None if weight is None else weight.realization['weight_panels']['selected_block_tiles']
    if block is not None and block[1]!=kt:
        raise ValueError('weight regional candidate must actually retain full K')
    resources=dict(m_tiles=(m+isa.DIM-1)//isa.DIM,
        uniform_column_readout=int(not column_scales),weight_full_k_legal=int(block is not None),
        weight_m_tiles=0 if block is None else block[0],dim=isa.DIM,
        spad_rows=isa.SPAD_ROWS,acc_rows=isa.ACC_ROWS)
    facts=PolicyFacts(session.stamp(),shapes=((m,k),(k,n)),
        element_types=('i8','i8'),layouts=('row_major','row_major'),
        operations=frozenset({'dense'}),resources=tuple(sorted(resources.items())))
    registry={}
    for key in artifacts:
        def action(current,key=key):
            if current.module!=source:
                raise ValueError('weight regional registry cannot reuse a different source artifact')
            validate()
            return {OUTPUT:artifacts[key]}
        name='dense_weight_'+('panels' if key=='weight_panels' else 'retained')
        registry[name]=RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),
            mutates_ir=False,ir_level=LEVEL),action)
    selected=execute_policy(session,weight_regional_policy(),registry,facts,output=OUTPUT)
    schedule,refusal=selected.artifact
    receipt=dict(schema='authored_dense_weight_regional.v2',policy=selected.receipt,
        candidate_kernels=identities,candidate_refusals={key:value[1] for key,value in artifacts.items()},
        choice_policy=choice_policy,geometry=geometry,storage_split=storage_split,
        qualification='provisional authored option; native paired support incomplete on holdouts; not assessed_shape_policy qualification')
    return schedule,refusal,receipt


def regional_weight_schedule(m,k,n,bias,column_scales=False,*,geometry='wide',
                             storage_split='half',choice_policy='traffic',context=None,
                             retained_planner=None):
    """Compatibility pair API; preserve policy receipt beside actual artifact."""
    selected,refusal,receipt=execute_dense_weight_regional(m,k,n,bias,column_scales,
        geometry=geometry,storage_split=storage_split,choice_policy=choice_policy,
        context=context,retained_planner=retained_planner)
    if selected is None:return None,refusal
    selected.revalidate()
    return replace(selected,realization=dict(selected.realization,
        regional_weight_selection=receipt)),refusal
