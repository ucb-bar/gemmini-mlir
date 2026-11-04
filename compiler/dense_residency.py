"""Invocation-local complete RHS cells across explicitly owned resident handles.

The shared library plans the physical stripe and independently checks each
member with its original materialization prefix. The actual cold kernel emits
that prefix; actual warm kernels consume the same live cells as a precondition.
This owner tracks that precondition through a single exclusive RHS window.
No host tensor data, asynchronous execution law, implicit interleaving or
register-resident weights are introduced. Every data effect retains its full
completion. The backend must invalidate on EVICT and foreign scratchpad writes.
"""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
from hashlib import sha256
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.compile.scheduling.block_schedule import Load,Preload
from merlin.transforms.dense_planning import dense_blocks
from merlin.sched.ir import Loop,Var,select

KNOB_DECLARATION={'dense_resident_reuse':{'values':['off','retained','retained_tiny'],
    'mechanism':'complete RHS placement and invocation-local live private-cell retention',
    'note':'Explicit resident handles only; full RHS capacity, ordinary completion and invalidation required. off is incumbent.'}}
INDEX_UNROLL_KNOB_DECLARATION={'dense_index_unroll':{'values':['off','tiny'],
    'mechanism':'ascending literal emission of constant dense index axes of extent at most two',
    'note':'off preserves counted emission; tiny expands singleton/two-trip axes with at most 15 extra iteration callbacks per kernel, bounding each original effect to at most 16 copies. Checked schedule and completion order remain unchanged.'}}


def index_unroll_setting():
    return knob('dense_index_unroll',default='off',values=('off','tiny'),root=Path(__file__).resolve().parent)


def setting():
    return knob('dense_resident_reuse',default='off',values=('off','retained','retained_tiny'),root=Path(__file__).resolve().parent)


# Frozen comparison provenance, never consulted by the geometry predicate.
RETENTION_ADMISSION_EVIDENCE={
    'status':'route-regression diagnosis; isolated paired policy validation pending',
    'baseline_request':'2_1793274739373703589_0296144c7e507d1a77a3a5169dc53583',
    'candidate_request':'2_1793275229567403588_5469863ccbc06038b015eaeaaeeb1757',
    'baseline_submission_sha256':'5c67d0ecb832dac3931c8eca102c6dd4f85b6d78f18af973b7590f8a5a6715f4',
    'candidate_submission_sha256':'c1058272cb98ccf8758ee4e17fa87478a67a2314d552d9d6b4598fca4e59f5d3',
    'model_sha256':'b3ce9aaf7e8e2227e781f5d0bb3bd597f73ab12cd8c67aff64a5cd7f54347b11',
    'scope_sha256':'82dae761bda30488a6760c9ae51764069a6ed9e60544661e90f883477d273525',
    'comparison_domain_sha256':'fa29a9f347bf945f37d95839668cb70a8a190f1669709cf310ee8ca93de0301f',
    'complete_modeled_pairs':[
        {'shape':[16,128,64],'invocations':2,'baseline':5365,'candidate':14982,
         'fences':[5,171],'operand_read_bytes':[20480,24576]},
        {'shape':[16,32,16],'invocations':8,'baseline':4264,'candidate':4521,
         'fences':[19,45],'operand_read_bytes':[8192,4608]}],
    'qualification':'same model, complete-call ROI and numerical pass; snapshots also change backend/ISA so this is not an isolated knob comparison'}


def retention_legality(problem,storage,*,resident_key,rhs_transposed=False,per_column=False):
    """Check route prerequisites and physical fit, without a speed assumption.

    Admission is still conditional on the existing immutable-handle lifetime,
    disjoint caller buffers and invalidation contract. ``prepare`` separately
    verifies the complete actual cold/warm schedules before emitting effects.
    """
    reason=None;placement=None
    if resident_key is None:reason='no explicit reusable resident source handle'
    elif rhs_transposed or per_column:reason='resident route requires ordinary RHS and scalar readout'
    elif not problem.tile_m==problem.tile_k==problem.tile_n:
        reason='resident route requires the selected square primitive'
    else:
        placement=plan_dense_rhs_residency(problem.m,problem.k,problem.n,block=problem.tile_k,
            operand_rows=storage.rhs.base+storage.rhs.capacity,
            accumulator_rows=storage.accumulator.capacity,rhs_base=storage.rhs.base,
            max_static_rhs_tiles=storage.rhs.capacity//problem.tile_k)
        if placement is None:reason='complete padded RHS exceeds selected private stripe'
    return {'eligible':reason is None,'reason':reason,
        'placement':None if placement is None else asdict(placement),
        'effect_check':'prepare independently verifies actual full-completion cold/warm kernels',
        'source_precondition':'immutable explicit resident handle; disjoint buffers; no intervening scratchpad writer'}


def regional_residency_admission(mode,problem,options):
    """Describe competing schedules; do not infer an empirical K cutoff.

    ``observe`` preserves current explicit resident choices while paired trials
    are pending. The optional ``preserve_completion_schedule`` guard retains an
    already selected epoch/readout schedule instead of synchronous retained_tiny.
    It is an experimental route preference, not a source/capacity legality law.
    Plain retained stays the broad explicit experiment. No source dtype, scale,
    activation or case identity establishes a timing exception.
    """
    if mode not in ('off','retained','retained_tiny'):
        raise ValueError('unknown dense resident reuse mode')
    policy=options.get('residency_admission_policy','observe')
    if policy not in ('observe','preserve_completion_schedule'):
        raise ValueError('unknown regional residency admission policy')
    epochs=options.get('completion_epochs','off')
    readout=options.get('readout_schedule','off')
    competing=epochs!='off' or readout!='off'
    admitted=mode!='off' and not (mode=='retained_tiny'
        and policy=='preserve_completion_schedule' and competing)
    reason=('resident reuse disabled' if mode=='off' else
        'preserve the already selected completion/readout schedule pending paired retained-epoch support'
        if not admitted else None)
    return {'schema':'gemmini.regional_residency_admission.v1','requested_mode':mode,
        'admitted':admitted,'selected_path':mode if admitted else 'incumbent',
        'policy':policy,'refusal':reason,'shape':[problem.m,problem.k,problem.n],
        'shape_class':'single_k_slice' if problem.k<=problem.tile_k else 'multiple_k_slices',
        'competes_with_completion_schedule':competing,
        'incumbent_schedule':{'completion_epochs':epochs,'readout_schedule':readout,
            'panel_maxima':{axis:options.get('max_'+axis+'_blocks',1) for axis in ('m','k','n')}},
        'status':'paired policy validation pending; observe does not change admission',
        'legality':'separate original-handle, source views, assigned capacity and actual cold/warm effect verification'}


def retention_admission(problem,storage,*,mode,resident_key,rhs_transposed=False,per_column=False,options=None):
    """Return a reviewable decision; callers keep their incumbent on refusal.

    Use ``decision['admitted']`` before ``prepare``. Publish the decision with
    the actual lowering receipt. This helper neither emits nor changes a state,
    setting, source handle, scheduling alternative or completion boundary.
    """
    legality=retention_legality(problem,storage,resident_key=resident_key,
        rhs_transposed=rhs_transposed,per_column=per_column)
    performance=regional_residency_admission(mode,problem,options or {})
    admitted=legality['eligible'] and performance['admitted']
    return {'schema':'gemmini.retention_admission.v1','requested_mode':mode,
        'shape':[problem.m,problem.k,problem.n],'selected_k_edge':problem.tile_k,
        'admitted':admitted,'selected_path':mode if admitted else 'incumbent',
        'legality':legality,'performance':performance,
        'refusal':None if admitted else legality['reason'] or performance['refusal']}


def invalidate_residency(builder,key=None,reason='explicit ownership invalidation'):
    state=getattr(builder,'dense_residency_state',None)
    if state is not None and (key is None or state['key']==key):
        builder.dense_residency_state=None
    if not hasattr(builder,'dense_residency_invalidations'): builder.dense_residency_invalidations=[]
    builder.dense_residency_invalidations.append({'key':str(key),'reason':reason})


@dataclass
class ResidentDense:
    kernel: object
    proof_kernel: object
    problem: object
    resources: object
    storage: object
    adapter: object
    bias_source: object
    choice: object
    realization: dict
    signature: str

    def revalidate(self):
        if sha256(self.kernel.text().encode()).hexdigest()!=self.signature:
            raise ValueError('resident dense actual emitted kernel changed after checking')
        verify_compact_dense(self.proof_kernel,self.problem,self.resources,self.storage,self.adapter,
            static_budget=256,timeout_ms=5000,bias_source=self.bias_source)


def prepare(builder,problem,resources,storage,adapter,*,key,rhs_ptr,bias_source=None):
    """Return a source-checked cold/warm kernel and its pending context state.

    Pointer identity is part of the proof key, never a workload identity. The
    RHS is immutable under the explicit source handle; disjoint caller buffers
    and the private row assignment establish that member writes cannot change
    it. Parent invalidation is required before every other scratchpad writer.
    """
    edge=problem.tile_k
    if not problem.tile_m==problem.tile_k==problem.tile_n:
        raise ValueError('resident RHS requires square primitive tiles')
    placement=plan_dense_rhs_residency(problem.m,problem.k,problem.n,block=edge,
        operand_rows=storage.rhs.base+storage.rhs.capacity,accumulator_rows=storage.accumulator.capacity,
        rhs_base=storage.rhs.base,max_static_rhs_tiles=storage.rhs.capacity//edge)
    if placement is None:
        raise ValueError('complete padded RHS exceeds selected private stripe')
    # The base compact family stays primitive: each K/N RHS tile gets a unique
    # physical row stripe, independently of output-group repetitions.
    limits=replace(resources,max_m_blocks=1,max_k_blocks=1,max_n_blocks=1)
    block=dense_blocks(problem,limits)[0]
    raw=emit_compact_dense(problem,block,storage,adapter,name='gemmini_resident_dense',
        dtypes=('i8','i8','i32'),bias_source=bias_source)
    nt=problem.tiles[2]
    def rewrite(body):
        result=[]
        for item in body:
            if isinstance(item,Loop):
                nested=rewrite(item.body)
                if nested: result.append(replace(item,body=tuple(nested)))
            else:
                op=adapter.project(item)
                if isinstance(op,Load) and op.role=='weight': continue
                if isinstance(op,Preload):
                    ki,nj=op.weight_block
                    op=replace(op,weight_row=storage.rhs.base+(ki*nt+nj)*edge)
                    item=adapter.encode(op)
                result.append(item)
        return result
    ki,nj=Var('resident_k'),Var('resident_n')
    rows=select('resident_k',problem.tiles[1]-1,problem.k-(problem.tiles[1]-1)*edge,edge)
    cols=select('resident_n',nt-1,problem.n-(nt-1)*edge,edge)
    load=adapter.encode(Load('weight',(ki,nj),ki*edge,nj*edge,rows,cols,
        storage.rhs.base+(ki*nt+nj)*edge,1))
    prefix=(Loop('resident_k',problem.tiles[1],(Loop('resident_n',nt,(load,)),)),)
    body=tuple(rewrite(raw.body))
    proof=replace(raw,body=prefix+body)
    verify_compact_dense(proof,problem,limits,storage,adapter,static_budget=256,
        timeout_ms=5000,bias_source=bias_source)
    identity=(key,rhs_ptr,problem.k,problem.n,edge,storage.rhs.base,placement.rhs_rows)
    previous=getattr(builder,'dense_residency_state',None)
    warm=bool(previous is not None and previous['identity']==identity)
    actual=replace(raw,body=body if warm else prefix+body)
    loads=0 if warm else problem.tiles[1]*nt
    receipt={'mechanism':'invocation_local_complete_rhs','cache_hit':warm,
        'resident_key':str(key),
        'rhs_load_tiles':loads,'rhs_rows':placement.rhs_rows,'placement':asdict(placement),
        'precondition':'same immutable explicit handle and pointer; no intervening scratchpad clobber',
        'proof_kernel':proof.text(),'actual_kernel':actual.text(),
        'completion':'full fence after every load/compute/readout',
        'member_shape':[problem.m,problem.k,problem.n]}
    result=ResidentDense(actual,proof,problem,limits,storage,adapter,bias_source,
        type('ResidentChoice',(),{'block':block})(),receipt,sha256(actual.text().encode()).hexdigest())
    return result,{'identity':identity,'key':key},receipt


def commit(builder,state):
    """Publish the live-cell invariant only after actual emission completes."""
    builder.dense_residency_state=state
