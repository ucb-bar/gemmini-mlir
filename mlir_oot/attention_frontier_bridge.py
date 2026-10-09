"""Target ranked-descriptor/product glue for Merlin's source host executor.

No numeric host algorithm lives here. The caller supplies independently closed
source/consumer proofs and actual compiled xDSL product identities. The bridge
retains the original source body for every host executor refusal.
"""
from __future__ import annotations
from dataclasses import dataclass
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


def _identifier(value: str) -> str:
    if not value or value[0].isdigit() or any(not(c.isascii() and(c.isalnum()or c=='_'))for c in value):
        raise ValueError('explicit C symbol required')
    return value


def _digest(value: str) -> str:
    if len(value)!=64 or any(c not in '0123456789abcdef'for c in value):
        raise ValueError('actual immutable SHA256 required')
    return value


@dataclass(frozen=True)
class GroupedProductImport:
    dimensions: tuple[int,int,int]
    degree: int
    pairs: tuple[tuple[int,int],...]
    accumulator_absolute_bound: int
    symbol: str
    target_ir_sha256: str
    object_sha256: str


def emit_descriptor_bridge(plan,imports,*,writer_symbol:str,executor_symbol:str,
                           fallback_symbol:str,source_binding_sha256:str,
                           consumer_binding_sha256:str,workspace_bytes:int,
                           workspace_alignment:int=64)->str:
    """Emit RV64 ranked memref ABI, with workspace descriptor after output.

    Caller must bind the actual imported objects/IR to these hashes and retain
    fallback through the generic complete-source contract. Return descriptor
    identity and full fresh-destination ownership stay unchanged on either path.
    """
    plan.validate()
    writer=_identifier(writer_symbol);executor=_identifier(executor_symbol);fallback=_identifier(fallback_symbol)
    _digest(source_binding_sha256);_digest(consumer_binding_sha256)
    if type(workspace_bytes)is not int or workspace_bytes<=0 or workspace_bytes>2**63-1:
        raise ValueError('explicit positive compiled workspace size required')
    if type(workspace_alignment) is not int or workspace_alignment<8 or workspace_alignment>2**31 or workspace_alignment&(workspace_alignment-1):
        raise ValueError('target-compatible power-of-two workspace alignment required')
    shapes={(plan.query_rows,plan.chunk,plan.depth),(plan.query_rows,plan.depth,plan.segment),
            (plan.query_rows,plan.depth,plan.chunk-2*plan.segment)}
    expected={(shape,g)for shape in shapes for g in range(5)};selected={};symbols=set()
    for item in imports:
        key=(item.dimensions,item.degree)
        if key not in expected or key in selected:raise ValueError('duplicate or unrelated physical product import')
        _identifier(item.symbol);_digest(item.target_ir_sha256);_digest(item.object_sha256)
        if item.symbol in symbols:raise ValueError("physical product symbols must be unique")
        symbols.add(item.symbol)
        proof=plan_radix_product_groups(radix_bits=7,digits=3,reduction_length=item.dimensions[2])
        group=next(g for g in proof.groups if g.exponent==7*item.degree)
        if item.pairs!=group.pairs or item.accumulator_absolute_bound!=group.accumulator_bound:
            raise ValueError('actual source product group/overflow contract changed')
        selected[key]=item
    if set(selected)!=expected:raise ValueError('incomplete source product coverage')
    inputs=', '.join(f'D4 *i{i}'for i in range(11));arguments=', '.join(f'i{i}'for i in range(11))
    lines=['#include <stdint.h>','#include <stddef.h>','#include "source_attention_frontier_api.h"',
      'typedef struct { void *allocated,*aligned; int64_t offset,size[4],stride[4]; } D4;',
      'typedef struct { void *allocated,*aligned; int64_t offset,size[1],stride[1]; } D1;',
      '_Static_assert(sizeof(void*)==8 && sizeof(D4)==88 && sizeof(D1)==40,"RV64 ranked descriptor ABI required");',
      f'extern int {executor}(const merlin_attention_view*,merlin_attention_view*,void*,size_t,merlin_attention_product,void*);',
      f'extern void _mlir_ciface_{fallback}({inputs}, D4 *out);']
    for item in selected.values():lines.append(f'extern void {item.symbol}(const int8_t*,const int8_t*,int32_t*);')
    lines.append('static int products(void *opaque,const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k,int degree){(void)opaque;')
    for (dims,degree),item in sorted(selected.items()):
        m,n,k=dims;lines.append(f'if(m=={m} && n=={n} && k=={k} && degree=={degree}){{{item.symbol}(a,b,c);return 1;}}')
    lines+=['return 0;}','static merlin_attention_view semantic(D4*d){merlin_attention_view v={d->aligned,d->offset,{0},{0}};for(int i=0;i<4;i++){v.sizes[i]=d->size[i];v.strides[i]=d->stride[i];}return v;}',
       f'void {writer}({inputs},D4*out,D1*workspace){{',
       f'if(workspace && workspace->aligned && workspace->offset==0 && workspace->stride[0]==1 && workspace->size[0]>={workspace_bytes} && (uintptr_t)workspace->aligned%{workspace_alignment}==0){{',
       'merlin_attention_view views[11]={'+','.join(f'semantic(i{i})'for i in range(11))+'};',
       'merlin_attention_view destination=semantic(out);',
       f'if({executor}(views,&destination,workspace->aligned,(size_t)workspace->size[0],products,0))return;',
       '}',f'_mlir_ciface_{fallback}({arguments},out);','}']
    return '\n'.join(lines)+'\n'
