"""Checked immutable-operand residency alternatives on one original block.

Partial output families retain their independently checked reloads. RHS block
DMA changes the command/byte tradeoff: source bytes alone cannot price extra
narrow LHS commands. The optional selector consumes actual final-band load
counts from both checked candidates. It does not profile or simulate a kernel.
"""
from dataclasses import dataclass
from hashlib import sha256
import json


def _policy(source, resources, rhs, lhs):
    from merlin.compiler.transforms.pipeline_policy import (PipelinePolicy, PipelineVariant,
        PolicyFacts, RegisteredPass, ResourcePredicate, execute_policy)
    from merlin.compiler.transforms.pipeline_state import PipelineSession, PipelineContext, PassSpec
    level='dense_operand_residency'
    fingerprint=lambda value:sha256(json.dumps(value).encode()).hexdigest()
    def verify(value):
        if value!=source:raise ValueError('residency source changed')
    session=PipelineSession(source,ir_level=level,fingerprint=fingerprint,verify_ir=verify,
        context=PipelineContext(resources=fingerprint(source)))
    facts=PolicyFacts(session.stamp(),shapes=((source[0],source[1]),(source[1],source[2])),
        element_types=('i8','i8'),layouts=('row_major','row_major'),
        operations=frozenset({'dense'}),resources=tuple(sorted(resources.items())))
    variants=(PipelineVariant('rhs',('retain_rhs',),(ResourcePredicate('rhs_preferred',1),)),
              PipelineVariant('lhs',('retain_lhs',)))
    registry={name:RegisteredPass(PassSpec(name,0,produces=('result',),mutates_ir=False,ir_level=level),
        lambda current,value=value:{'result':value})
        for name,value in [('retain_rhs',rhs),('retain_lhs',lhs)]}
    return execute_policy(session,PipelinePolicy(variants,fallback='lhs'),registry,facts,output='result')


def choose_residency(problem,block,*,selection):
    """Construct a named candidate; legality remains with the shared scheduler."""
    if selection not in ('lhs','rhs_reuse'):raise ValueError('unknown residency construction')
    mt,kt,nt=problem.tiles
    full_m,tail_m=divmod(mt,block.m);full_n,tail_n=divmod(nt,block.n)
    lhs_repeat=int(bool(full_n))+int(bool(tail_n))
    rhs_repeat=int(bool(full_m))+int(bool(tail_m))
    lhs_bytes=problem.m*problem.k*problem.lhs_bytes
    rhs_bytes=problem.k*problem.n*problem.rhs_bytes
    lhs_cost=lhs_bytes*lhs_repeat+rhs_bytes*((mt+block.m-1)//block.m)
    rhs_cost=lhs_bytes*((nt+block.n-1)//block.n)+rhs_bytes*rhs_repeat
    use_rhs=selection=='rhs_reuse' and full_m>1 and rhs_cost<lhs_cost
    resources=dict(multiple_complete_m_groups=int(full_m>1),
        rhs_operand_bytes_saved=int(rhs_cost<lhs_cost),rhs_preferred=int(use_rhs))
    source=(problem.m,problem.k,problem.n,block.m,block.k,block.n,selection)
    selected=_policy(source,resources,'weight','lhs');use_rhs=selected.artifact=='weight'
    return dict(selection=selection,role='weight' if use_rhs else 'lhs',
        outer_order=('n','m') if use_rhs else ('m','n'),
        original_block_tiles=[block.m,block.k,block.n],
        requested_operand_bytes=dict(lhs_retained=lhs_cost,rhs_retained=rhs_cost),
        resources=resources,policy=selected.receipt,
        selected='rhs_reuse' if use_rhs else 'incumbent',
        profitability='requires paired complete-call cycles',
        tail_scope='complete groups share immutable input; separate partial families reload')


def _physical_loads(schedule):
    schedule.revalidate()
    receipt=schedule.realization.get('rhs_block_transfers',{})
    if receipt.get('applied'):
        return receipt['projection']['physical_load_commands']
    source=schedule.realization['source']
    return source['lhs_loads']+source['rhs_loads']


@dataclass(frozen=True)
class DenseResidencySelection:
    checked:object
    realization:dict
    @property
    def kernel(self):return self.checked.kernel
    def revalidate(self):self.checked.revalidate()
    def materialize_scf(self):return self.checked.materialize_scf()
    def emit_llvm(self,builder,lower_call):return self.checked.emit_llvm(builder,lower_call)


def select_checked_residency(lhs,rhs):
    """Choose between final checked schedules using their real transfer receipts.

    This construction filter prices the source's transfer instruction count,
    not elapsed cycles. Paired timing over the affected class is still required
    for promotion. Both candidates retain their original source/storage/epoch
    obligations, terminal encoder and target completion callbacks.
    """
    lhs.revalidate();rhs.revalidate()
    left=lhs.realization['operand_residency'];right=rhs.realization['operand_residency']
    if left['original_block_tiles']!=right['original_block_tiles']:
        raise ValueError('residency candidates changed the admitted block')
    if lhs.kernel.args!=rhs.kernel.args:raise ValueError('residency candidates changed source ABI')
    loads_lhs,loads_rhs=_physical_loads(lhs),_physical_loads(rhs)
    preferred=right['selected']=='rhs_reuse' and loads_rhs<loads_lhs
    resources=dict(lhs_physical_load_commands=loads_lhs,rhs_physical_load_commands=loads_rhs,
        rhs_preferred=int(preferred))
    a=lhs.kernel.args[0].shape;b=lhs.kernel.args[1].shape
    source=(*a,b[1],*left['original_block_tiles'],'rhs_amortized')
    selected=_policy(source,resources,rhs,lhs)
    result=selected.artifact;result.revalidate()
    receipt=dict(selection='rhs_amortized',selected='rhs_reuse' if result is rhs else 'incumbent',
        resources=resources,candidate_kernels=dict(lhs=lhs.kernel.digest(),rhs=rhs.kernel.digest()),
        policy=selected.receipt,cycle_authority='none; current paired complete-call timing required')
    return DenseResidencySelection(result,dict(result.realization,checked_operand_residency=receipt))
