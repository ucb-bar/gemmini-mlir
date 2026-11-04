"""Optional accelerator preparation of head-major signed-i8 attention keys.

Only the final N,D axes are exchanged. Flattening the untouched leading batch
and KV-head axes is a pointer-preserving view, so GQA prepares each KV head once
without materializing query-head copies. The existing checked integer transpose
owns all DMA staging, identity narrowing and real completion boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .transpose import TransposeError, lower_transpose, plan_transpose


@dataclass(frozen=True)
class _Prepared:
    source_shape: tuple
    output_shape: tuple
    destination_shape: tuple
    permutation: tuple
    schedule: str
    proof: dict


class _OutputView:
    """Present an exact unflattened destination type to the common consumer."""
    def __init__(self,ctx,dst,shape):
        self._ctx,self._dst,self._shape=ctx,dst,shape
    def __getattr__(self,name):
        return getattr(self._ctx,name)
    def spec(self,name):
        spec=self._ctx.spec(name)
        return dict(spec,shape=list(self._shape))if name==self._dst else spec


def _prepare(ctx,src,dst,schedule):
    """Read-only admission, including the existing actual DMA projection proof."""
    try:
        if schedule not in ('baseline','batched')or src==dst:return None
        a,b=ctx.spec(src),ctx.spec(dst)
        shape=tuple(a['shape']);destination=tuple(b['shape'])
        if (a['dtype']!='i8'or b['dtype']!='i8'or len(shape)not in (2,3,4)
                or any(type(n)is not int or n<=0 for n in (*shape,*destination))):return None
        rank=len(shape)
        permutation=tuple(range(rank-2))+(rank-1,rank-2)
        output=tuple(shape[i]for i in permutation)
        heads=prod(shape[:-2]);n,d=shape[-2:]
        if destination not in (output,(heads,d,n)):return None
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        sources={x.get('source')for x in aliases.values()if isinstance(x,dict)}
        if {src,dst}&(set(aliases)|sources):return None
        buffers=getattr(ctx,'buffers',{})
        if src in buffers and dst in buffers and buffers[src]is buffers[dst]:return None
        # This is the existing counted DMA transform and address/ownership
        # checker, including field limits and fact-derived ACC tile capacity.
        plan=plan_transpose(shape,permutation,'i8',schedule=schedule)
        return _Prepared(shape,output,destination,permutation,schedule,plan.proof)
    except(KeyError,TypeError,ValueError,OverflowError):
        return None


def can_lower(ctx,src,dst,*,schedule='batched'):
    """Whether K[...,N,D] can prepare the exact existing [...,D,N] buffer."""
    return _prepare(ctx,src,dst,schedule)is not None


def lower(ctx,src,dst,*,schedule='batched'):
    """Emit checked accelerator byte transpose; False precedes any mutation.

    Destination may retain the full leading shape or flatten all batch/KV-head
    axes to one head axis. No head replication, quantization, CPU payload copy,
    or contraction is introduced. The complete drain before return publishes
    every prepared byte before the attention dense consumer may read it.
    """
    prepared=_prepare(ctx,src,dst,schedule)
    if prepared is None:return False
    view=_OutputView(ctx,dst,prepared.output_shape)
    emitted=lower_transpose(view,{'src':src,'dst':dst},
        {'permutation':prepared.permutation,'output_dtype':'i8',
         'transpose_schedule':prepared.schedule})
    ctx.fence()
    receipt=dict(mechanism='attention_device_key_transpose',
        source_shape=list(prepared.source_shape),
        destination_shape=list(prepared.destination_shape),
        logical_permuted_shape=list(prepared.output_shape),
        head_count=prod(prepared.source_shape[:-2]),
        prepared_bytes=prod(prepared.source_shape),
        head_mapping='original batch-major KV head; leading axes unchanged',
        flattened_view='pointer preserving; original complete element cardinality',
        numeric='exact signed-i8 identity movement; scale=1, activation=0',
        completion='checked staging drains and real full completion before consumer',
        transpose=emitted)
    ctx.metadata.setdefault('attention_device_key_transpose',[]).append(receipt)
    return receipt
