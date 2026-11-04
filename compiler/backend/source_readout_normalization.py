"""Exact immutable singleton-prefix column broadcast normalization.

Only the static load expression is normalized; source data, arithmetic order,
scale proofs, liveness, and the existing fused-only callback remain unchanged.
"""
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseIntOrFPElementsAttr
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap,AffineExpr


def singleton_column_literal(value,amap,output_shape,safe_scalar):
    """Return f32 column bits for dense<1x...x1xN> loaded at (0,...,0,dlast)."""
    ty=value.type;owner=value.owner
    if (not isinstance(ty,TensorType) or not isinstance(ty.encoding,NoneAttr)
            or str(ty.element_type)!='f32' or not output_shape
            or not isinstance(owner,Operation) or owner.name!='arith.constant'):
        return None
    shape=tuple(ty.get_shape())
    if not 2<=len(shape)<=8 or shape[:-1]!=(1,)*(len(shape)-1) or shape[-1]!=output_shape[-1] or not 1<=shape[-1]<=65536:
        return None
    expected=AffineMap(len(output_shape),0,tuple(AffineExpr.constant(0)for _ in shape[:-1])+(AffineExpr.dimension(len(output_shape)-1),))
    if amap!=expected:return None
    safe_scalar(owner)
    if owner.operands or len(owner.results)!=1 or owner.results[0] is not value or owner.regions:
        return None
    attr=owner.properties.get('value',owner.attributes.get('value'))
    if not isinstance(attr,DenseIntOrFPElementsAttr) or attr.type!=ty:return None
    values=tuple(float(x)for x in attr.iter_values())
    if len(values)!=shape[-1]:return None
    return values
