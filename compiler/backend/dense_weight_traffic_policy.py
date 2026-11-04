"""Optional current-artifact weight residency by exact scheduled input traffic.

This authored heuristic preserves the existing narrow weight region and adds
one provisional traffic region. It is neither an assessed shape policy nor
held-out/native qualification; no default dispatcher or timing data is used.
"""
from dataclasses import asdict,replace
from hashlib import sha256
import json
from merlin.transforms.pipeline_policy import (
    PipelinePolicy,PipelineVariant,PolicyFacts,RegisteredPass,
    ResourcePredicate,ShapePredicate,execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec,PipelineContext,PipelineSession
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_verification import _project,_requested_elements
from merlin.compile.scheduling.dense_panel_pipeline import _serial_projection
from . import isa
from .dense_pipeline import PANEL_GEOMETRIES
from .dense_weight_panels import weight_panel_schedule
from .dense_weight_regional import DenseWeightSource

LEVEL='authored_dense_weight_traffic_source'
OUTPUT='checked_dense_weight_traffic_schedule'
MIN_REDUCTION_BPS=500


def scheduled_input_bytes(result):
    """Sum exact actual counted Load extents; never enumerate dynamic instances.

    Shared projected-instruction traffic accounting partitions only equality
    selects in row/column extents. Loop counts multiply each static site's sum.
    Bias seed traffic and output stores are excluded from input operand bytes.
    Unsupported extent forms refuse accounting rather than using a plan score.
    """
    from .dense import _adapter
    result.revalidate()
    arguments={arg.name:arg for arg in result.kernel.args}
    if any(name not in arguments or arguments[name].dtype!='i8' for name in ('lhs','rhs')):
        raise ValueError('weight traffic accounting requires actual i8 input operands')
    totals={'lhs':0,'weight':0}
    sites=_project(_serial_projection(result.kernel),_adapter())
    for site in sites:
        op=site.operation
        if type(op)is Load:
            if op.role not in totals:
                raise ValueError('weight traffic policy encountered unknown input role')
            totals[op.role]+=_requested_elements(site,op.rows,op.cols)
    # Current policy ABI fixes both input operands to signed i8, hence one byte.
    result.revalidate()
    return {'kernel_sha256':result.kernel.digest(),'lhs_bytes':totals['lhs'],
        'rhs_bytes':totals['weight'],'input_bytes':totals['lhs']+totals['weight'],
        'accounting':'actual projected i8 Load rows*cols; exact counted/select sums'}


def weight_traffic_policy(*,require_multiple_m_panels=False):
    """Existing predicates choose actual products; capacity remains independent."""
    common=(ResourcePredicate('uniform_column_readout',1),
        ResourcePredicate('weight_full_k_legal',1),
        ResourcePredicate('weight_m_tiles',1 if require_multiple_m_panels else 4))
    if require_multiple_m_panels:
        # Counted traversal replaces the old block-width proxy in this opt-in
        # policy. Even the old narrow region must demonstrate actual savings.
        common+= (ResourcePredicate('weight_counted_m_panels',2),
            ResourcePredicate('input_reduction_basis_points',MIN_REDUCTION_BPS))
    return PipelinePolicy((
        PipelineVariant('weight_existing_region',('dense_weight_traffic_panels',),(
            ResourcePredicate('m_tiles',128),ShapePredicate(1,1,at_most=8*isa.DIM),*common)),
        PipelineVariant('weight_traffic_region',('dense_weight_traffic_panels',),(
            ResourcePredicate('retained_full_k_legal',1),
            ResourcePredicate('input_reduction_basis_points',MIN_REDUCTION_BPS),*common)),
        PipelineVariant('retained',('dense_weight_traffic_retained',)),
    ),fallback='retained')


def execute_dense_weight_traffic(m,k,n,bias=False,column_scales=False,*,
        geometry='wide',storage_split='half',context=None,retained_planner=None,
        weight_geometry=None,require_multiple_m_panels=False):
    """Return (actual checked artifact or None, original refusal, policy receipt).

    The weight candidate uses the retained geometry unless an explicit resource
    bound is supplied. Counted-panel admission is opt-in; the amortized-M option
    stays separate. Both candidates are current-source checked products,
    and their data/completion commands reach the original terminal encoder.
    Unknown caller resource custody selects the unconditional retained path.
    """
    source=DenseWeightSource(m,k,n,bias,column_scales,geometry,storage_split,'traffic')
    source.__post_init__()
    if type(require_multiple_m_panels)is not bool:
        raise ValueError('weight traffic requires an explicit boolean panel policy')
    weight_bound=PANEL_GEOMETRIES[geometry] if weight_geometry is None else tuple(weight_geometry)
    if len(weight_bound)!=2 or any(type(v)is not int or v<=0 for v in weight_bound):
        raise ValueError('weight traffic geometry must contain two positive integer bounds')
    options={} if weight_geometry is None and not require_multiple_m_panels else {
        'weight_geometry':weight_bound,'require_multiple_m_panels':require_multiple_m_panels}
    if context is None:
        view={'dim':isa.DIM,'spad_rows':isa.SPAD_ROWS,'acc_rows':isa.ACC_ROWS,
            'geometry':geometry,'storage_split':storage_split}
        if options:view['weight_options']=options
        context=PipelineContext(resources='authored_weight_traffic_resource_view:'+sha256(
            json.dumps(view,sort_keys=True).encode()).hexdigest())
    def verify(value):
        if type(value)is not DenseWeightSource:
            raise ValueError('weight traffic source changed type')
        value.__post_init__()
    session=PipelineSession(source,context=context,ir_level=LEVEL,
        fingerprint=lambda value:sha256(json.dumps(asdict(value) if not options else
            dict(source=asdict(value),weight_options=options),sort_keys=True).encode()).hexdigest(),
        verify_ir=verify)
    if retained_planner is None:
        from .dense_pipeline import _retained_schedule
        retained_planner=_retained_schedule
    if not callable(retained_planner):
        raise ValueError('weight traffic needs the non-dispatching retained planner')
    retained,rr=retained_planner(m,k,n,bias,column_scales,False,geometry,storage_split)
    if storage_split=='half':
        weight,wr=weight_panel_schedule(m,k,n,bias,column_scales,
            geometry=weight_bound,choice_policy='traffic')
    else:
        weight,wr=None,'weight traffic candidate requires original half private assignments'
    artifacts={'retained':(retained,rr),'weight_panels':(weight,wr)}
    identities={key:None if item is None else item.kernel.digest()
        for key,(item,_) in artifacts.items()}
    kt=(k+isa.DIM-1)//isa.DIM
    blocks={}
    def validate():
        verify(source)
        expected=[('lhs',(m,k),'i8','read'),('rhs',(k,n),'i8','read')]
        if bias:expected.append(('bias',(n,),'i32','read'))
        expected.append(('output',(m,n),'i32','write'))
        for key,(item,refusal) in artifacts.items():
            if item is None:
                if not refusal or identities[key]is not None:
                    raise ValueError('weight traffic lost original candidate refusal')
                continue
            item.revalidate()
            if refusal is not None or item.kernel.digest()!=identities[key]:
                raise ValueError('weight traffic current checked artifact changed')
            if [(a.name,a.shape,a.dtype,a.access) for a in item.kernel.args]!=expected:
                raise ValueError('weight traffic candidate implements a different source ABI')
            if key=='weight_panels':
                facts=item.realization['weight_panels']
                if tuple(facts['geometry_bound'])!=weight_bound:
                    raise ValueError('weight traffic changed the selected geometry bounds')
            else:
                facts=item.realization['panel_geometry']
                if facts['selection']!=geometry:
                    raise ValueError('weight traffic retained geometry changed')
            block=tuple(facts['selected_block_tiles'])
            if block[1]!=kt:
                raise ValueError('weight traffic candidates must cover full original K')
            blocks[key]=block
    validate()
    traffic={};traffic_refusals={}
    for key,(item,_) in artifacts.items():
        if item is None:
            traffic[key]=None
            continue
        try:
            traffic[key]=scheduled_input_bytes(item)
        except ValueError as exc:
            # A source-legal artifact remains available to the existing region
            # even if its extent language cannot be accounted analytically.
            traffic[key]=None;traffic_refusals[key]=str(exc)
    rb=None if traffic['retained']is None else traffic['retained']['input_bytes']
    wb=None if traffic['weight_panels']is None else traffic['weight_panels']['input_bytes']
    reduction=None if rb is None or wb is None else rb-wb
    basis_points=0 if rb is None or not rb or reduction is None else max(0,10000*reduction//rb)
    block=blocks.get('weight_panels')
    resources={'m_tiles':(m+isa.DIM-1)//isa.DIM,'uniform_column_readout':int(not column_scales),
        'weight_full_k_legal':int(block is not None),'retained_full_k_legal':int('retained'in blocks),
        'weight_m_tiles':0 if block is None else block[0],
        'input_reduction_basis_points':basis_points,'dim':isa.DIM,
        'spad_rows':isa.SPAD_ROWS,'acc_rows':isa.ACC_ROWS}
    if require_multiple_m_panels:
        resources['weight_counted_m_panels']=0 if weight is None else weight.realization['weight_panels']['counted_m_panels']
    facts=PolicyFacts(session.stamp(),shapes=((m,k),(k,n)),element_types=('i8','i8'),
        layouts=('row_major','row_major'),operations=frozenset({'dense'}),resources=tuple(sorted(resources.items())))
    registry={}
    for key in artifacts:
        def action(current,key=key):
            if current.module!=source:
                raise ValueError('weight traffic registry cannot reuse another source artifact')
            validate()
            if key=='weight_panels' and rb is not None and wb is not None:
                # Bind traffic predicates to the actual artifact consumed now.
                if scheduled_input_bytes(retained)!=traffic['retained'] or scheduled_input_bytes(weight)!=traffic['weight_panels']:
                    raise ValueError('weight traffic current analytical byte accounting changed')
            return {OUTPUT:artifacts[key]}
        name='dense_weight_traffic_'+('panels' if key=='weight_panels' else 'retained')
        registry[name]=RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),mutates_ir=False,ir_level=LEVEL),action)
    selected=execute_policy(session,weight_traffic_policy(
        require_multiple_m_panels=require_multiple_m_panels),registry,facts,output=OUTPUT)
    schedule,refusal=selected.artifact
    receipt={'schema':'authored_dense_weight_traffic.v1','source':asdict(source),'policy':selected.receipt,
        'candidate_kernels':identities,
        'candidate_source_identities':{key:None if item is None else item.source_identity for key,(item,_) in artifacts.items()},
        'selected_kernel_sha256':None if schedule is None else schedule.kernel.digest(),
        'selected_variant':selected.receipt['selected_variant'],
        'candidate_refusals':{key:v[1] for key,v in artifacts.items()},
        'scheduled_input_traffic':traffic,'traffic_accounting_refusals':traffic_refusals,
        'input_byte_reduction':reduction,'input_reduction_basis_points':basis_points,
        'required_reduction_basis_points':MIN_REDUCTION_BPS,'resources':resources,
        'existing_region_preserved':not require_multiple_m_panels,'choice_policy':'traffic',
        'qualification':'provisional authored option; native support on training regions only; holdouts incomplete; not assessed_shape_policy qualification'}
    if options:receipt['weight_options']=options
    return schedule,refusal,receipt


def traffic_weight_schedule(m,k,n,bias=False,column_scales=False,**options):
    """Pair compatibility API; attach selection receipt without mutating caches."""
    schedule,refusal,receipt=execute_dense_weight_traffic(m,k,n,bias,column_scales,**options)
    if schedule is None:return None,refusal
    schedule.revalidate()
    return replace(schedule,realization=dict(schedule.realization,weight_traffic_selection=receipt)),refusal
