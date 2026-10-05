"""Exact removal of a static zero-filled NCHW pad-one shell before direct DMA."""
from xdsl.dialects.builtin import TensorType,NoneAttr,IntegerAttr,i8
from xdsl.ir import Operation


def strip_zero_pad1(value, shape):
    op=value.owner
    if not isinstance(op,Operation) or op.name!='tensor.insert_slice' or len(op.operands)!=2:
        raise ValueError('padding is not a static insert into an explicit zero tensor')
    source,destination=op.operands
    expected=(1,shape.cin,shape.h,shape.w);padded=(1,shape.cin,shape.h+2,shape.w+2)
    for v,extents in ((source,expected),(destination,padded),(value,padded)):
        ty=v.type
        if not isinstance(ty,TensorType) or not isinstance(ty.encoding,NoneAttr) or ty.get_element_type()!=i8 or tuple(ty.get_shape())!=extents or any(x<=0 for x in extents):
            raise ValueError('padding extents or signed-i8 tensor encoding are unproved')
    for key,expected_values in [('static_offsets',(0,0,1,1)),('static_sizes',expected),('static_strides',(1,1,1,1))]:
        a=op.properties.get(key)
        if a is None or tuple(a.get_values())!=expected_values:raise ValueError('padding slice geometry differs from pad1')
    splat=destination.owner
    if not isinstance(splat,Operation) or splat.name!='tensor.splat' or len(splat.operands)!=1:
        raise ValueError('padding initializer is not an explicit scalar splat')
    scalar=splat.operands[0];constant=scalar.owner
    if not isinstance(constant,Operation) or constant.name!='arith.constant' or scalar.type!=i8:
        raise ValueError('padding scalar is not a constant i8 zero')
    attr=constant.properties.get('value')
    if not isinstance(attr,IntegerAttr) or attr.value.data!=0:raise ValueError('nonzero or unknown padding value')
    return source,dict(kind='explicit_static_i8_zero_pad1',source_shape=list(expected),padded_shape=list(padded),offsets=[0,0,1,1])
