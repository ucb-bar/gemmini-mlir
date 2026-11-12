"""Exact signed attention contractions using dense device schedules.

QK's key transpose is a bounded shared-library layout copy. Products, scaling,
activation and narrowing execute on the accelerator. Softmax has no arithmetic
contract in the named-op ABI and is never inferred here from its mnemonic.
"""
from __future__ import annotations
from math import prod
import os
from pathlib import Path
from merlin.compiler.selection.knobs import knob


class AttentionError(ValueError):
    pass


def _spec(ctx,name):
    s=ctx.spec(name)
    if not s['shape'] or any(type(x) is not int or x <= 0 for x in s['shape']):
        raise AttentionError('attention requires positive static tensor dimensions')
    return s


def _geometry(ctx,left,right,dst,qk):
    a,b,c=(_spec(ctx,x) for x in (left,right,dst))
    sa,sb,sc=a['shape'],b['shape'],c['shape']
    if a['dtype']!='i8' or b['dtype']!='i8' or c['dtype'] not in ('i8','i32'):
        raise AttentionError('selected attention contractions require signed i8 operands and i8/i32 results')
    if len(sa)!=len(sb) or len(sa) not in (2,3,4):
        raise AttentionError('attention requires matching rank-two matrices or head-major rank-three/four tensors')
    if len(sa)==2:
        batches,hq,hkv=1,1,1
    else:
        hq,hkv=sa[-3],sb[-3]
        if sa[:-3]!=sb[:-3]:raise AttentionError('attention batches must agree')
        batches=prod(sa[:-3])
        if hq % hkv:raise AttentionError('query heads must be divisible by key/value heads')
    m,k=sa[-2:]
    if qk:
        n,bk=sb[-2:]
    else:
        bk,n=sb[-2:]
    if bk!=k or sc!=[*sa[:-2],m,n]:
        raise AttentionError('attention result or reduction dimensions disagree')
    return batches,hq,hkv,m,k,n


def _alias(ctx,name,offset,shape):
    label=f'_attention_{name}_{len(ctx.tensors)}'
    ctx.register(label,ctx.ptr(name,offset),shape,ctx.spec(name)['dtype'])
    return label


class _TransposeBuilder:
    def __init__(self,fb):
        from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter
        self.indices=FnBuilderIndexAdapter(fb)
        self.fb=fb
    def op(self,opcode,a,b):return self.indices.op(opcode,a,b)
    def loop(self,start,stop,step,body):return self.indices.loop(start,stop,step,body)
    def gep(self,base,index):
        from xdsl.dialects.builtin import i8
        return self.fb.gep(base,self.indices.value(index),i8)
    def load(self,ptr,dtype):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import i8
        if dtype!='i8':raise AttentionError('key transpose is an i8 byte copy')
        return self.fb.add(llvm.LoadOp(ptr,i8)).results[0]
    def store(self,value,ptr,dtype):
        from xdsl.dialects import llvm
        if dtype!='i8':raise AttentionError('key transpose is an i8 byte copy')
        self.fb.add(llvm.StoreOp(value,ptr))


def _shared_rhs(ctx,left,right,dst,stationary,geometry,attrs):
    """Emit one uninterrupted resident epoch per KV head, or keep dense.

    A global KV index g owns query heads [g*group,(g+1)*group).
    Those intervals partition the batch-major query head axis.  Each member
    reads its own M*K elements and writes its own M*N elements; the common
    K*N RHS is immutable until the complete group's readouts finish.
    """
    from . import isa
    from .dense import lower_dense
    from .resident_rhs import lower_resident_products
    from merlin.compiler.schedule.dense_rhs_residency import plan_dense_rhs_residency
    batches,hq,hkv,m,k,n=geometry
    group=hq//hkv
    report={'mechanism':'attention_grouped_rhs','query_heads':hq,'kv_heads':hkv,
            'batches':batches,'products_per_group':group,'applied':False}
    ctx.metadata.setdefault('attention_rhs_reuse',[]).append(report)
    # Members are authored at compile time, while batches/KV groups remain
    # counted. Bound code growth independently of runtime head/batch extents.
    if group<2 or group>32:
        report['refusal']='no shared heads or group exceeds 32-member static emission budget'
        return False
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    names={left,right,dst}|{attrs[key] for key in ('bias','accumulator_init') if attrs.get(key)}
    aliased_sources={a.get('source') for a in aliases.values() if isinstance(a,dict)}
    if dst in (left,right) or names & (set(aliases)|aliased_sources):
        report['refusal']='source has explicit aliases or output/input overlap'
        return False
    if plan_dense_rhs_residency(m,k,n,block=isa.DIM,operand_rows=isa.SPAD_ROWS,
            accumulator_rows=isa.ACC_ROWS,rhs_base=isa.DIM) is None:
        report['refusal']='complete RHS does not fit disjoint resident operand rows'
        return False
    fb=ctx.fb
    saved_source_index=ctx.current_source_index
    member_attrs=dict(attrs)
    if saved_source_index is not None:
        member_attrs.setdefault('merlin.source_op_index',saved_source_index)
    def kv_head(index):
        common=_alias(ctx,stationary,fb.mul_i(index,fb.const(k*n)),(k,n))
        start=fb.mul_i(index,fb.const(group))
        members=[]
        for local in range(group):
            head=fb.add_i(start,fb.const(local))
            members.append({'lhs':_alias(ctx,left,fb.mul_i(head,fb.const(m*k)),(m,k)),
                'rhs':common,'dst':_alias(ctx,dst,fb.mul_i(head,fb.const(m*n)),(m,n)),
                'm':m,'k':k,'n':n,'attributes':member_attrs})
        receipt=lower_resident_products(ctx,members)
        ctx.current_source_index=saved_source_index
        if receipt is None:
            # The shared adapter refuses before emitting commands. Fall back
            # inside this same partition, so every source head still occurs once.
            report['refusal']='resident adapter rejected epilogue, auxiliary or source schedule'
            for row in members:
                lower_dense(ctx,row['lhs'],row['rhs'],row['dst'],m=m,k=k,n=n,attributes=attrs)
        else:
            report.update(applied=True,rhs_tile_loads_per_group=receipt['rhs_source_loads'],
                resident_schedule_sha256=[x['resident_kernel_sha256'] for x in receipt['members']])
    fb.for_loop(0,batches*hkv,1,kv_head,retain_loop=True)
    report.update(head_partition={'groups':batches*hkv,'group_size':group,
        'total_heads':batches*hq,'query_head':'kv_global * group + local_head',
        'kv_head':'kv_global','all_heads_exactly_once':True},
        lifetime='immutable RHS; contiguous products; complete readouts before next KV group',
        rhs_bytes_per_group=k*n,baseline_rhs_bytes_per_group=group*k*n)
    return True


def _contract(ctx,operands,attributes,qk):
    from .dense import lower_dense
    from merlin.compiler.lower.model.dense_layout_loop import emit_dense_transpose
    left,right,dst=(operands[key] for key in (('q','k','dst') if qk else ('p','v','dst')))
    batches,hq,hkv,m,k,n=_geometry(ctx,left,right,dst,qk)
    attrs=dict(attributes)
    if attrs.get('output_dtype',ctx.spec(dst)['dtype'])!=ctx.spec(dst)['dtype']:
        raise AttentionError('attention output attribute differs from its typed result')
    if any(key in attrs for key in ('transpose_rhs','rhs_transposed','transpose_b')):
        raise AttentionError('QK transpose is fixed by its opcode; PV has no transpose attribute')
    configured=knob('ATTENTION_RHS_REUSE',default='off',values=('off','on'),root=Path(__file__).resolve().parent.parent)
    reuse=attrs.get('attention_rhs_reuse',os.environ.get('ATTENTION_RHS_REUSE',configured))
    if reuse not in ('off','on'):
        raise AttentionError('ATTENTION_RHS_REUSE must be off or on')
    fb=ctx.fb
    stationary=right
    if qk:
        stationary=f'_attention_kt_{len(ctx.tensors)}'
        ctx.alloc(stationary,(batches*hkv,k,n),'i8')
        ctx.fence()
        preparation=knob('TRANSPOSE_TRANSFERS',default='off',values=('off','on','device_keys'),root=Path(__file__).resolve().parent.parent)
        device_prepared=False
        if preparation=='device_keys':
            from .attention_transpose import lower as lower_key_transpose
            device_prepared=bool(lower_key_transpose(ctx,right,stationary))
        if not device_prepared:
            def transpose(head):
                offset=fb.mul_i(head,fb.const(n*k))
                emit_dense_transpose(_TransposeBuilder(fb),ctx.ptr(right,offset),ctx.ptr(stationary,offset),n,k,max_elements=n*k)
            fb.for_loop(0,batches*hkv,1,transpose,retain_loop=True)
            ctx.fence()
        ctx.metadata.setdefault('attention_layout_preparation',[]).append({'operation':'key_transpose','bytes':batches*hkv*n*k,'numeric_arithmetic':False,'route':'device' if device_prepared else 'merlin_host_layout'})
    group=hq//hkv
    if reuse=='on' and _shared_rhs(ctx,left,right,dst,stationary,(batches,hq,hkv,m,k,n),attrs):
        ctx.fence()
        return
    def head(index):
        if hq==hkv:
            kv_index=index
        else:
            batch=fb.udiv_i(index,fb.const(hq))
            qhead=fb.urem_i(index,fb.const(hq))
            kv_index=fb.add_i(fb.mul_i(batch,fb.const(hkv)),fb.udiv_i(qhead,fb.const(group)))
        aname=_alias(ctx,left,fb.mul_i(index,fb.const(m*k)),(m,k))
        bname=_alias(ctx,stationary,fb.mul_i(kv_index,fb.const(k*n)),(k,n))
        cname=_alias(ctx,dst,fb.mul_i(index,fb.const(m*n)),(m,n))
        lower_dense(ctx,aname,bname,cname,m=m,k=k,n=n,attributes=attrs)
    fb.for_loop(0,batches*hq,1,head,retain_loop=True)
    ctx.fence()


def lower_qk(ctx,operands,attributes):
    return _contract(ctx,operands,attributes,True)


def lower_pv(ctx,operands,attributes):
    return _contract(ctx,operands,attributes,False)


lower_attention_qk=lower_qk
lower_attention_pv=lower_pv
