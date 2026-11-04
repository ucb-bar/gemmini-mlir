"""Pointer-preserving grouped-query row folding for attention products.

The query heads belonging to one immutable KV head are adjacent in the source
ABI.  Treating their head and row axes as one matrix row axis changes neither
operand bytes nor the order of each integer reduction.  It also lets ordinary
dense tiling use a full array row across several short query heads.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json


@dataclass(frozen=True)
class GroupedRowsPlan:
    batches: int
    query_heads: int
    kv_heads: int
    m: int
    k: int
    n: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('grouped rows need positive static extents')
        if self.query_heads % self.kv_heads:
            raise ValueError('query heads must partition into complete KV groups')
        if max(self.batches*self.query_heads*self.m*self.k,
               self.batches*self.query_heads*self.m*self.n*4,
               self.batches*self.kv_heads*self.k*self.n) >= 1 << 63:
            raise ValueError('grouped attention address range exceeds signed i64')

    @property
    def group(self):
        return self.query_heads // self.kv_heads

    def receipt(self):
        description = asdict(self)
        return dict(mechanism='attention_grouped_query_rows', geometry=description,
            plan_sha256=sha256(json.dumps(description,sort_keys=True).encode()).hexdigest(),
            matrix_shape=[self.group*self.m,self.k,self.n],
            products=self.batches*self.kv_heads,
            source_head='kv_global * group + local_head',
            source_row='local_head * M + query_row',
            lhs_identity='(kv_global*group+local_head)*M*K+query_row*K+depth = kv_global*(group*M*K)+(local_head*M+query_row)*K+depth',
            output_identity='(kv_global*group+local_head)*M*N+query_row*N+column = kv_global*(group*M*N)+(local_head*M+query_row)*N+column',
            rhs_identity='kv_global*K*N+depth*N+column',
            coverage='Euclidean row decomposition local_head=row//M, query_row=row%M; complete disjoint head/row partition',
            numeric='same ordered signed-i8 reduction and unchanged per-column readout attributes',
            payload_copies=0,query_head_rhs_replication=False)


def lower_grouped_rows(ctx,left,right,dst,geometry,attributes):
    """Emit one dense product per original KV head; False precedes mutation."""
    from .dense import lower_dense
    plan=GroupedRowsPlan(*geometry)
    if plan.group < 2:
        return False
    attrs=dict(attributes)
    # A shared M*N initializer repeats once per head and is not an ordinary
    # contiguous group*M*N view. Keep the incumbent for that auxiliary form.
    if attrs.get('accumulator_init') is not None:
        return False
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    names={left,right,dst}|{attrs[key] for key in ('bias',) if attrs.get(key)}
    sources={row.get('source') for row in aliases.values() if isinstance(row,dict)}
    if dst in (left,right) or names & (set(aliases)|sources):
        return False
    if attrs.get('bias'):
        bias=ctx.spec(attrs['bias'])
        if bias['dtype']!='i32' or bias['shape']!=[plan.n]:
            return False
    fb=ctx.fb
    source_index=ctx.current_source_index
    leaves=[]
    def alias(name,offset,shape):
        label=f'_attention_rows_{name}_{len(ctx.tensors)}'
        ctx.register(label,ctx.ptr(name,offset),shape,ctx.spec(name)['dtype'])
        return label
    def product(index):
        ctx.current_source_index=source_index
        a=alias(left,fb.mul_i(index,fb.const(plan.group*plan.m*plan.k)),
                (plan.group*plan.m,plan.k))
        b=alias(right,fb.mul_i(index,fb.const(plan.k*plan.n)),(plan.k,plan.n))
        c=alias(dst,fb.mul_i(index,fb.const(plan.group*plan.m*plan.n)),
                (plan.group*plan.m,plan.n))
        leaves.append(lower_dense(ctx,a,b,c,m=plan.group*plan.m,k=plan.k,n=plan.n,
                                  attributes=attrs))
        ctx.fence()
    fb.for_loop(0,plan.batches*plan.kv_heads,1,product,retain_loop=True)
    ctx.current_source_index=source_index
    receipt=plan.receipt()
    receipt.update(applied=True,dense_leaf_schedules=leaves,
        completion='unchanged dense ownership and complete readout before next KV group')
    ctx.metadata.setdefault('attention_grouped_rows',[]).append(receipt)
    return receipt
