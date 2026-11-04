"""Source-exact, opt-in NHWC convolution/column-readout/maxpool composition.

No intermediate may escape this atomic region. Unsupported source retains its
ordinary placement; operation identifiers never participate in admission.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvColumnPool:
    root: object
    operations: tuple
    ifm: object
    weight: object
    scale_value: object
    attributes: dict

    def record(self):
        return {'form':'source_conv_column_readout_pool',
                'source_order':['signed_i8_convolution_i32','convert_f32',
                                'multiply_f32','round_half_even',
                                'clamp_signed_i8','cast_i8','signed_maxpool'],
                'intermediate_observers':False,
                'proof':'exact convolution maps and scalar recurrence; strict readout matcher; positive monotone spatial readout'}


def enabled():
    from merlin.compile.tuning_knobs import knob
    return knob('POOL_COLUMN_READOUT',default='off',values=('off','on'),
                root=Path(__file__).resolve().parent.parent)=='on'


def match_conv_column_pool(pool):
    from .frontend import _type, _zero
    from .source_groups import _integer_pool
    from .source_dense_readout import match_column_readout
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    import math
    pooled=_integer_pool(pool)
    if pooled is None:return None
    source,pattrs,_=pooled
    if _type(pool.operands[2])!=_type(pool.results[0]):return None
    # Permit only proved contiguous row-major shape views around the original
    # readout. Physical channel order is checked again against the convolution.
    from .source_view_bindings import read_dense_source_view, SourceViewError
    try:
        after=read_dense_source_view(source)
    except SourceViewError:return None
    if any(op.name=='tensor.extract_slice' for op in after.operations):return None
    readout=after.root.owner
    q=match_column_readout(readout,allow_nhwc=True)
    if q is None or any(not math.isfinite(s) or s<=0 for s in q.scales):return None
    try:
        before=read_dense_source_view(q.source)
    except SourceViewError:return None
    if any(op.name=='tensor.extract_slice' for op in before.operations):return None
    conv=before.root.owner
    if (getattr(conv,'name',None) not in ('linalg.generic','linalg.conv_2d_nhwc_hwcf')
            or len(conv.operands)!=3 or len(conv.results)!=1):return None
    x,w,init=conv.operands
    xt,wt,yt=map(_type,(x,w,conv.results[0]))
    if (xt['dtype']!='i8' or wt['dtype']!='i8' or yt['dtype']!='i32'
            or len(xt['shape'])!=4 or len(wt['shape'])!=4 or len(yt['shape'])!=4
            or not _zero(init)):return None
    b,h,width,ci=xt['shape'];kh,kw,wci,co=wt['shape'];nb,oh,ow,nco=yt['shape']
    if (b!=nb or ci!=wci or co!=nco or len(q.scales)!=co
            or _type(source)!={'shape':yt['shape'],'dtype':'i8'}):return None
    allowed=('indexing_maps','iterator_types','operandSegmentSizes') if conv.name=='linalg.generic' else ('strides','dilations','operandSegmentSizes')
    if any(k not in allowed for k in conv.properties) or any(not k.startswith('prov.') for k in conv.attributes):return None
    if conv.name=='linalg.generic':
        maps=tuple(a.data for a in conv.indexing_maps)
        if tuple(a.data.value for a in conv.iterator_types)!=('parallel',)*4+('reduction',)*3:return None
    else:
        maps=tuple(conv.get_default_indexing_maps())
    if len(maps)!=3 or any(m.num_dims!=7 or m.num_symbols for m in maps):return None
    d=tuple(AffineDimExpr(i) for i in range(7))
    if maps[1:]!=(AffineMap(7,0,(d[4],d[5],d[6],d[3])),AffineMap(7,0,d[:4])):return None
    # Extract candidates, then compare the entire original map structurally:
    # neither symbols, offsets nor unproved permutations can survive.
    def coefficient(axis,dimension):
        point=[0]*7;point[dimension]=1
        return maps[0].eval(point,())[axis]
    sh,sw,dh,dw=(coefficient(1,1),coefficient(2,2),coefficient(1,4),coefficient(2,5))
    if min(sh,sw,dh,dw)<=0:return None
    if maps[0]!=AffineMap(7,0,(d[0],d[1]*sh+d[4]*dh,d[2]*sw+d[5]*dw,d[6])):return None
    if oh!=(h-dh*(kh-1)-1)//sh+1 or ow!=(width-dw*(kw-1)-1)//sw+1:return None
    # Exact signed extensions and wrapping i32 multiply/add, without promises.
    body=conv.regions[0].block;ops=list(body.ops)
    if len(body.args)!=3 or [o.name for o in ops]!=['arith.extsi','arith.extsi','arith.muli','arith.addi','linalg.yield']:return None
    a,bext,mul,add,yld=ops
    if (list(a.operands)!=[body.args[0]] or list(bext.operands)!=[body.args[1]]
            or str(a.results[0].type)!='i32' or str(bext.results[0].type)!='i32'
            or set(mul.operands)!={a.results[0],bext.results[0]}
            or set(add.operands)!={mul.results[0],body.args[2]}
            or list(yld.operands)!=[add.results[0]]):return None
    for scalar in ops:
        if any(not k.startswith('prov.') for k in scalar.attributes):return None
        if any(k!='overflowFlags' for k in scalar.properties):return None
        if getattr(getattr(scalar,'overflow_flags',None),'data',()):return None
    if (any(k not in ('indexing_maps','iterator_types','operandSegmentSizes') for k in pool.properties)
            or any(not k.startswith('prov.') for k in pool.attributes)
            or any(s.properties or s.attributes for s in pool.regions[0].block.ops)):return None
    members=(conv,*reversed(before.operations),readout,*reversed(after.operations),pool)
    owned=set(members)
    if any(any(use.operation not in owned for use in value.uses)
           for op in members[:-1] for value in op.results):return None
    # Source pool matcher already proves maps, signed reduction and -128 init.
    attrs={**pattrs,**q.attributes(),'epilogue':['acc_scale','maxpool'],
           'kernel':[kh,kw,ci,co],'stride':[sh,sw],'dilation':[dh,dw],
           'padding':[0,0,0,0],'layout':'nhwc'}
    attrs.pop('acc_scale',None)
    for op in members:op.verify()
    return ConvColumnPool(pool,members,x,w,q.scale_value,attrs)


def source_plans(module):
    if not enabled():return []
    plans=[]
    for op in module.walk():
        if op.name=='linalg.generic' and (match:=match_conv_column_pool(op)) is not None:
            plans.append(match)
    return plans
