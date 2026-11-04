"""Immutable final-axis column scale proof for the existing dense COMMIT ABI.

No global/load argument is treated as immutable. No standalone column readout
ABI is inferred from the fused dense encoder. Arithmetic proof stays complete
signed-i32 and is cached by exact binary32 scale-chain bits.
"""
from functools import lru_cache
import math
import struct
from xdsl.dialects.builtin import TensorType,NoneAttr,DenseIntOrFPElementsAttr
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap,AffineExpr
from merlin.llvmlower.requantization import f32,prove_scale


class ColumnScaleError(ValueError):
    pass


def _bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def column_literal(value,amap,output_shape,safe_scalar):
    """Recognize only immutable f32[N] loaded by exactly the final source axis."""
    from .source_readout_normalization import singleton_column_literal
    normalized=singleton_column_literal(value,amap,output_shape,safe_scalar)
    if normalized is not None:return normalized
    ty=value.type;owner=value.owner
    if (not isinstance(ty,TensorType) or not isinstance(ty.encoding,NoneAttr)
            or str(ty.element_type)!='f32' or not output_shape
            or tuple(ty.get_shape())!=(output_shape[-1],)
            or not isinstance(owner,Operation) or owner.name!='arith.constant'):
        return None
    if amap!=AffineMap(len(output_shape),0,(AffineExpr.dimension(len(output_shape)-1),)):
        return None
    safe_scalar(owner)
    if owner.operands or len(owner.results)!=1 or owner.results[0] is not value or owner.regions:
        raise ColumnScaleError('unproved immutable column literal')
    attr=owner.properties.get('value',owner.attributes.get('value'))
    if not isinstance(attr,DenseIntOrFPElementsAttr) or attr.type!=ty:
        return None
    if output_shape[-1]>65536:
        raise ColumnScaleError('immutable column count exceeds bounded proof budget')
    values=tuple(float(x)for x in attr.iter_values())
    if len(values)!=output_shape[-1]:raise ColumnScaleError('column literal extent disagrees with SSA type')
    # Dense f32 element decoding preserves the actual binary32 values. Numeric
    # positivity/finite constraints are imposed at each actual scale operation.
    return values


def peel_scale(expr):
    a,b=expr.args
    if expr.kind=='arith.mulf' and a.kind in ('constant','column_constant'):
        a,b=b,a
    if b.kind not in ('constant','column_constant') or b.dtype!='f32':
        raise ColumnScaleError('immutable scalar or final-axis binary32 scale required')
    values=b.value if b.kind=='column_constant'else(b.value,)
    result=[]
    for value in values:
        if not isinstance(value,(int,float)):
            raise ColumnScaleError('binary32 scale literal required')
        scale=f32(float(value))
        if not math.isfinite(scale) or scale<=0:raise ColumnScaleError('finite positive binary32 column scale required')
        if expr.kind=='arith.divf':
            if math.frexp(scale)[0]!=0.5:raise ColumnScaleError('non-power-of-two column division cannot become reciprocal multiplication')
            scale=f32(1.0/scale)
            if not math.isfinite(scale) or scale<=0:raise ColumnScaleError('column reciprocal is not finite positive binary32')
        result.append(scale)
    return a,tuple(result)if b.kind=='column_constant'else result[0]


@lru_cache(maxsize=512)
def _prove_bits(bits,relu):
    values=tuple(struct.unpack('<f',struct.pack('<I',word))[0]for word in bits)
    return prove_scale(values,relu=relu)


def prove_columns(scales,count,relu):
    """Keep ordered f32 stages and prove each distinct bit-pattern chain exactly."""
    if any(isinstance(s,tuple)and len(s)!=count for s in scales):
        raise ColumnScaleError('column scale count changed')
    chains=tuple(tuple(s[c]if isinstance(s,tuple)else s for s in scales)for c in range(count))
    identities=tuple(tuple(_bits(s)for s in chain)for chain in chains)
    if len(set(identities))>128:raise ColumnScaleError('distinct column scale chains exceed bounded proof budget')
    proofs=[];indices=[];by_bits={}
    for bits in identities:
        if bits not in by_bits:
            by_bits[bits]=len(proofs)
            proofs.append(dict(_prove_bits(bits,relu),source_scale_bits=[hex(x)for x in bits]))
        indices.append(by_bits[bits])
    attrs={'output_dtype':'i8','acc_scale_by_column':[proofs[i]['scale']for i in indices],
           'epilogue':['acc_scale']+(['relu']if relu else [])}
    proof={'proof':'per-column complete-i32 monotone transition equality',
           'accumulator_min':-(1<<31),'accumulator_max':(1<<31)-1,
           'column_axis':-1,'columns':count,'distinct_scale_chains':proofs,
           'column_proof_indices':indices,'standalone_form':'unsupported; scalar-only ACCUMULATOR_READOUT',
           'selected_form':'resident integer product COMMIT with acc_scale_by_column'}
    return attrs,proof
