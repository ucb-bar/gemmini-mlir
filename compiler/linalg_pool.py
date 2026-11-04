"""Source-proven signed-i8 NHWC maximum pooling on ordinary target readout.

The complete scalar region, affine maps, geometry and neutral initializer select
this route. Padding can be a deferred immutable view only when every observer is
an independently admitted pool. All other source values retain their buffers.
"""
from functools import lru_cache
import hashlib
import json

from xdsl.dialects.builtin import TensorType, NoneAttr
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr, AffineBinaryOpKind

from merlin.common.mlir_query import op_name
from merlin.compile.logical_abi import pool_params, pool_out_dims, POOL_PAD_VALUE_ATTR
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.llvmlower.integer_readout import derive, evaluate
from elementwise import emit_maxpool
from linalg_device import _constant, _finish


def _shape(value, rank):
    ty=value.type
    if (not isinstance(ty,TensorType) or str(ty.element_type)!='i8'
        or not isinstance(ty.encoding,NoneAttr)):return None
    shape=tuple(tensor_shape(ty))
    return shape if len(shape)==rank and all(0<v<(1<<31) for v in shape) else None


def _constant_tensor(value):
    """Read original literal fills; never infer seed values from runtime inputs."""
    owner=value.owner
    if not isinstance(owner,Operation):return None
    if owner.name=='tensor.splat' and len(owner.operands)==1:
        return _constant(owner.operands[0])
    if owner.name=='linalg.fill' and len(owner.operands)==2:
        if len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1:return None
        block=owner.regions[0].block;body=tuple(block.ops)
        if (len(block.args)!=2 or len(body)!=1 or body[0].name!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],)
            or body[0].attributes or body[0].properties
            or any(not k.startswith('prov.') for k in owner.attributes)
            or set(owner.properties)-{'operandSegmentSizes'}):return None
        return _constant(owner.operands[0])
    if owner.name=='arith.constant':
        raw=owner.properties.get('value',owner.attributes.get('value'))
        get=getattr(raw,'get_values',None)
        values=tuple(get()) if callable(get) else ()
        if values and all(x==values[0] for x in values):return values[0]
    return None


def _signed8(value):
    return ((value+128)%256)-128 if type(value) is int else None


def _linear(expr):
    if isinstance(expr,AffineDimExpr):return ({expr.position:1},0)
    if isinstance(expr,AffineConstantExpr):return ({},expr.value)
    if not isinstance(expr,AffineBinaryOpExpr):return None
    left,right=_linear(expr.lhs),_linear(expr.rhs)
    if left is None or right is None:return None
    a,c=left;b,d=right
    if expr.kind==AffineBinaryOpKind.Add:
        out=dict(a)
        for k,v in b.items():out[k]=out.get(k,0)+v
        return ({k:v for k,v in out.items() if v},c+d)
    if expr.kind==AffineBinaryOpKind.Mul:
        if not a:return ({k:c*v for k,v in b.items() if c*v},c*d)
        if not b:return ({k:d*v for k,v in a.items() if d*v},c*d)
    return None


def _map(mapping,expected):
    return (mapping.num_dims==6 and mapping.num_symbols==0
        and len(mapping.results)==len(expected)
        and all(_linear(x)==v for x,v in zip(mapping.results,expected)))


def _literal_pair(op,name):
    attr=op.properties.get(name,op.attributes.get(name))
    get=getattr(attr,'get_values',None)
    values=tuple(get()) if callable(get) else ()
    return values if len(values)==2 and all(type(x) is int and x>0 for x in values) else None


def _body(op):
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return False
    block=op.regions[0].block;body=tuple(block.ops)
    if len(block.args)!=3 or any(str(v.type)!='i8' for v in block.args):return False
    if len(body)!=2 or body[0].name!='arith.maxsi' or body[1].name!='linalg.yield':return False
    maximum,end=body
    return (not maximum.regions and not maximum.successors and len(maximum.results)==1
        and str(maximum.results[0].type)=='i8' and len(maximum.operands)==2
        and set(maximum.operands)=={block.args[0],block.args[2]}
        and not block.args[1].uses and tuple(end.operands)==tuple(maximum.results)
        and not maximum.attributes and not maximum.properties
        and not end.attributes and not end.properties)


def _plan(op):
    name=op_name(op)
    if name not in ('linalg.generic','linalg.pooling_nhwc_max'):return None
    if len(op.operands)!=3 or len(op.results)!=1 or not _body(op):return None
    source,kernel,seed=op.operands;result=op.results[0]
    shape=_shape(source,4);window=_shape(kernel,2);output=_shape(result,4)
    if shape is None or window is None or output is None or _shape(seed,4)!=output:return None
    if _signed8(_constant_tensor(seed))!=-128:return None
    if shape[0]!=output[0] or shape[3]!=output[3]:return None
    if window[0]!=window[1] or window[0] not in (1,2,3):return None
    if any(not k.startswith('prov.') for k in op.attributes if k not in ('op_name__','strides','dilations')):return None
    if name=='linalg.generic':
        if any(k in op.attributes for k in ('strides','dilations')):return None
        if set(op.properties)-{'indexing_maps','iterator_types','operandSegmentSizes'}:return None
        maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
        if maps is None or len(maps)!=3 or iters is None or len(iters)!=6:return None
        maps=tuple(maps)
        if [str(x) for x in iters]!=['#linalg.iterator_type<parallel>']*4+['#linalg.iterator_type<reduction>']*2:return None
        if not _map(maps[1].data,[({4:1},0),({5:1},0)]):return None
        if not _map(maps[2].data,[({0:1},0),({1:1},0),({2:1},0),({3:1},0)]):return None
        first=maps[0].data
        if first.num_dims!=6 or first.num_symbols or len(first.results)!=4:return None
        y,x=_linear(first.results[1]),_linear(first.results[2])
        if y is None or x is None or y[1]!=0 or x[1]!=0:return None
        sh,sw=y[0].get(1),x[0].get(2)
        if type(sh) is not int or type(sw) is not int or min(sh,sw)<1:return None
        if not _map(first,[({0:1},0),({1:sh,4:1},0),({2:sw,5:1},0),({3:1},0)]):return None
        stride=(sh,sw)
    else:
        if set(op.properties)-{'strides','dilations','operandSegmentSizes'}:return None
        stride=_literal_pair(op,'strides')
        if stride is None or _literal_pair(op,'dilations')!=(1,1):return None
    params=pool_params(dict(pool_in_dims=shape[1:3],pool_size=window,pool_stride=stride,
        pool_padding=(0,0,0,0),**{POOL_PAD_VALUE_ATTR:-128}),op='MAXPOOL2D')
    if pool_out_dims(*shape[1:3],window,stride,(0,0,0,0))!=output[1:3]:return None
    return dict(source=source,seed=seed,result=result,shape=shape,output=output,params=params)


def _pad(value):
    op=value.owner
    if not isinstance(op,Operation) or op.name!='tensor.pad' or len(op.operands)!=1:return None
    if len(op.results)!=1 or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    if set(op.properties)-{'static_low','static_high','operandSegmentSizes','nofold'}:return None
    if any(not k.startswith('prov.') for k in op.attributes):return None
    src=op.operands[0];shape=_shape(src,4);out=_shape(value,4)
    if shape is None or out is None:return None
    low=op.properties.get('static_low');high=op.properties.get('static_high')
    if low is None or high is None:return None
    low,high=tuple(low.get_values()),tuple(high.get_values())
    if (len(low)!=4 or len(high)!=4 or min(*low,*high)<0
        or low[0] or low[3] or high[0] or high[3]
        or tuple(a+b+c for a,b,c in zip(shape,low,high))!=out):return None
    block=op.regions[0].block;body=tuple(block.ops)
    if len(block.args)!=4 or any(str(v.type)!='index' for v in block.args):return None
    if not body or body[-1].name!='tensor.yield' or len(body[-1].operands)!=1:return None
    value=body[-1].operands[0];constant=_signed8(_constant(value))
    if constant not in (0,-128):return None
    used=value.owner
    if set(body[:-1]) not in (set(),{used}):return None
    if isinstance(used,Operation) and used.parent is block and (used.regions or used.successors):return None
    return dict(source=src,shape=shape,padding=(low[1],low[2],high[1],high[2]),pad_value=constant)


@lru_cache(maxsize=1)
def _proof():
    readout=derive([1.0],-128,127,relu=False)
    assert all(evaluate(x,readout)==x for x in range(-128,128))
    return dict(domain=[-128,127],checked_values=256,max_output_lsb_error=0,
        scalar='signed max(input, accumulator), neutral seed -128',
        shared_readout_proof_sha256=hashlib.sha256(json.dumps(readout,sort_keys=True).encode()).hexdigest())


def try_lower(host,op):
    """One interceptor for pool operations and exclusively pool-observed padding."""
    if op.name=='tensor.pad':
        pad=_pad(op.results[0]) if len(op.results)==1 else None
        if pad is None or any(v not in host.ptrs for v in (pad['source'],op.results[0])):return False
        uses=tuple(op.results[0].uses)
        plans=[_plan(use.operation) if use.index==0 else None for use in uses]
        if not plans or any(p is None for p in plans):return False
        for plan in plans:
            kh,kw=plan['params']['pool_size'];sh,sw=plan['params']['pool_stride']
            pt,pl,pb,pr=pad['padding'];h,w=pad['shape'][1:3];oh,ow=plan['output'][1:3]
            if pad['pad_value']==-128 and (kh<=pt or kw<=pl or (oh-1)*sh-pt>=h or (ow-1)*sw-pl>=w):return False
        if not hasattr(host,'pool_padding_views'):host.pool_padding_views={}
        host.pool_padding_views[op.results[0]]=pad
        _finish(host,op)
        host.last_lowering=('storage','immutable_pool_padding_view',None)
        return True
    plan=_plan(op)
    if plan is None:return False
    source=plan['source'];result=plan['result'];params=dict(plan['params']);shape=plan['shape']
    pad=getattr(host,'pool_padding_views',{}).get(source)
    if pad is not None:
        source=pad['source'];shape=pad['shape']
        params.update(pool_in_dims=shape[1:3],pool_padding=pad['padding'],pad_value=pad['pad_value'])
    if any(v not in host.ptrs for v in (source,result,plan['seed'])):return False
    if host.ptrs[source] is host.ptrs[result] or host.ptrs[plan['seed']] is host.ptrs[result]:return False
    actual=emit_maxpool(host.enc,host.ptrs[source],host.ptrs[result],shape,
        **{**params,POOL_PAD_VALUE_ATTR:params['pad_value']},src_dtype='i8',output_dtype='i8',acc_scale=1.0)
    assert tuple(actual)==plan['output']
    _finish(host,op)
    host.last_lowering=('device','exact_signed_i8_nhwc_maxpool',None)
    if hasattr(host,'device_receipts'):
        host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
            emission=dict(kind='standalone_signed_i8_nhwc_maxpool',shape=list(shape),
                output_shape=list(plan['output']),geometry=params,numeric_proof=_proof())))
    return True
