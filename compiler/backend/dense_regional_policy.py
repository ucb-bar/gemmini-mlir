"""Reviewed coarse resource classes over existing checked dense schedules.

The library PipelinePolicy owns dispatch. Every artifact is the current-source
checked schedule consumed by the common terminal encoder. Paired cycle support
is an offline snapshot, never a profiling or evidence-file read at compilation.
"""
from dataclasses import replace
from hashlib import sha256
import json

from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PipelineContext, PassSpec

from . import isa
from .dense_geometry_policy import DenseGeometrySource, geometry_policy_session, LEVEL

OUTPUT='checked_regional_dense_schedule'
MIN_FRONTIER_N=2048
SUPPORT={
    'matched_pair_manifest':'artifacts/optimization-pairs.json',
    'confidence':'coarse reviewed regions; boundary profitability UNKNOWN',
    'cycles':'provisional Spike complete-call model; no gSIM claim',
    'resident_axis':{
        'snapshot':'artifacts/optimization-pairs.json',
        'paired_shapes':[[12544,256,64],[512,512,512],[784,256,1024]],
        'complete_call_change_pct':[-14.5535666,-11.6757680,-17.7390807],
        'excluded_equal_operand_traffic_shape':[49,512,2048]},
    'wide_32_frontier':{
        'snapshot':'artifacts/optimization-pairs.json',
        'baseline_request':'2_1793301766930601728_3a395153ee9c8d41136b89e266d4e4c3',
        'candidate_requests':['3_1793301892106324498_4ed7f383aa124069abc13823d4a22cef',
                              '3_1793302100369727903_0248031b073f60da88dd29db6d963c08'],
        'paired_shapes':[[1,5728,2048],[16,5632,2048],[1,2048,32000]],
        'paired_complete_call_cycles':[[7719487,2945626],[7617813,2889072],
                                       [19673813,16602498]],
        'assumed_min_n':MIN_FRONTIER_N,
        'narrow_n':'not promoted; heldout narrow-N includes a 0.3 percent regression',
        'proof_timeout':'quantized heldouts can exhaust continuation/output verification; wide fallback required'},
}

POLICY=PipelinePolicy((
    PipelineVariant('wide_32_frontier',('regional_frontier',),
        (ResourcePredicate('frontier_region',1),)),
    PipelineVariant('resident_axis',('regional_resident_axis',),
        (ResourcePredicate('resident_region',1),
         ResourcePredicate('incumbent_m_panels',2),
         ResourcePredicate('operand_bytes_saved',1))),
    PipelineVariant('wide',('regional_wide',)),
),fallback='wide')


def regional_schedule(m,k,n,bias,column_scales,segmented,storage_split,*,retained_planner):
    source=DenseGeometrySource(m,k,n,bias,column_scales,storage_split)
    capacity={'dim':isa.DIM,'spad_rows':isa.SPAD_ROWS,'acc_rows':isa.ACC_ROWS,
              'storage_split':storage_split}
    context=PipelineContext(resources=sha256(json.dumps(capacity,sort_keys=True).encode()).hexdigest())
    session=geometry_policy_session(source,context=context)
    resident_region=m>=512 and k<=512
    frontier_region=m<=16 and k>=2048 and n>=MIN_FRONTIER_N
    prepared={}
    refusals={}

    def plan(geometry,split_k):
        key=(geometry,split_k)
        if key not in prepared:
            # No context or emitter is mutated by the planner. Library source,
            # storage and completion refusal therefore precedes target emission.
            try:
                result,refusal=retained_planner(m,k,n,bias,column_scales,
                                               split_k,geometry,storage_split)
                if result is not None:
                    result.revalidate()
            except (ValueError,TimeoutError) as error:
                result,refusal=None,type(error).__name__+': '+str(error)
            prepared[key]=(result,refusal)
            if refusal is not None:
                refusals[geometry+(' segmented' if split_k else ' full_k')]=refusal
        return prepared[key]

    saved=0
    panels=0
    traffic={}
    if resident_region:
        incumbent,refusal=plan('wide',False)
        if incumbent is not None:
            p=DenseProblem(m,k,n,isa.DIM,isa.DIM,1 if column_scales else isa.DIM,
                           1,1,4,bias=bias)
            selected=incumbent.realization['panel_geometry']['selected_block_tiles']
            panels=(p.tiles[0]+selected[0]-1)//selected[0]
            if panels>1:
                # Locate the actual incumbent block in the library's hoisted
                # family accounting, rather than choosing a newly ranked block.
                resources=DenseResources(isa.SPAD_ROWS*isa.DIM,isa.ACC_ROWS*4*isa.DIM,
                    max_m_blocks=8,max_n_blocks=8,max_k_blocks=p.tiles[1],
                    partial_output_tiles=True,partial_output_blocks=True)
                storage=DenseLoopStorage(
                    DenseMemoryView('lhs','spad',isa.DIM,0,isa.SPAD_ROWS//2),
                    DenseMemoryView('rhs','spad',isa.DIM,isa.SPAD_ROWS//2,isa.SPAD_ROWS//2),
                    DenseMemoryView('acc','acc',4*isa.DIM,0,isa.ACC_ROWS))
                if storage_split=='adaptive':
                    from .dense_storage_partition import select_full_k_partition
                    partition=select_full_k_partition(p,resources,max_m_blocks=8,max_n_blocks=8)
                    if partition is None:
                        raise ValueError('checked wide incumbent lost its admitted partition')
                    storage=partition[0]
                choices=dense_retained_choices(p,resources,storage,roles=('lhs',))
                actual=next((c for c in choices
                    if [c.block.m,c.block.k,c.block.n]==selected),None)
                if actual is not None:
                    candidate,refusal=plan('resident_axis',False)
                    if candidate is not None:
                        new_bytes=candidate.realization['resident_axis']['requested_operand_bytes']
                        saved=max(0,actual.requested_operand_bytes-new_bytes)
                        traffic={'incumbent_operand_bytes':actual.requested_operand_bytes,
                                 'candidate_operand_bytes':new_bytes,
                                 'incumbent_block_tiles':selected}

    resources={'frontier_region':int(frontier_region),'resident_region':int(resident_region),
               'incumbent_m_panels':panels,'operand_bytes_saved':saved,**{
                   key:value for key,value in capacity.items() if type(value) is int}}
    facts=PolicyFacts(session.stamp(),shapes=((m,k),(k,n)),
        element_types=('i8','i8'),layouts=('row_major','row_major'),
        operations=frozenset({'dense'}),resources=tuple(sorted(resources.items())))

    def entry(name,geometry,split_k):
        def action(current):
            if current.module!=source:
                raise ValueError('regional dense pass lost its original source')
            result,refusal=plan(geometry,split_k)
            actual=geometry
            fallback=None
            if result is None and geometry!='wide':
                fallback={'requested':geometry,'reason':refusal}
                actual='wide'
                result,refusal=plan('wide',segmented)
            return {OUTPUT:(result,refusal,actual,fallback)}
        return name,RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),
            mutates_ir=False,ir_level=LEVEL),action)

    registry=dict((entry('regional_frontier','wide_32_frontier',True),
                   entry('regional_resident_axis','resident_axis',False),
                   entry('regional_wide','wide',segmented)))
    selected=execute_policy(session,POLICY,registry,facts,output=OUTPUT)
    result,refusal,actual,fallback=selected.artifact
    if result is not None:
        # Detach the receipt so explicit geometry calls keep their cache entry.
        realization=dict(result.realization)
        realization['regional_policy']={
            'requested':'regional','selected_rule':actual,
            'selected_variant':selected.receipt['selected_variant'],
            'incumbent_segmented':segmented,'fallback':fallback,
            'refusals':refusals,'traffic':traffic,'support':SUPPORT,
            'policy':selected.receipt}
        result=replace(result,realization=realization)
        result.revalidate()
    return result,refusal
