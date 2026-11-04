"""Read one exact constant column-scale source readout, without reassociation."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class ColumnReadout:
    source: object
    scale_value: object
    scales: tuple[float, ...]

    def attributes(self):
        return {'epilogue': ['acc_scale'], 'output_dtype': 'i8',
                'acc_scale_by_column': list(self.scales)}

    def record(self):
        return {'form': 'standalone_constant_f32_column_readout',
                'source_order': ['convert_f32', 'multiply_f32', 'round_half_even',
                                 'clamp_signed_i8', 'cast_i8'],
                'column_scales': list(self.scales),
                'proof': 'selected ordered output path; positive scales use shared all-i32 transition proof; zero preserves identical stages'}


@lru_cache(maxsize=1)
def _path():
    from merlin.targetgen.output_path import validate
    path=validate(json.loads(Path(__file__).with_name('source_groups_output_path.json').read_text()))
    # Optional activation is disabled, and pooling is absent from this source.
    stages=[s for s in path['readout'] if s['kind'] not in ('activation','pool')]
    if (not path['exits']['requant'] or path['accumulator'].get('dtype')!='i32'
            or stages != [
                {'kind':'convert','rounding':'half_even','to':'f32'},
                {'kind':'scale','columns_per_scale':16,'dtype':'f32','granularity':'per_command','rounding':'half_even'},
                {'kind':'requant','clamp':[-128,127],'rounding':'half_even','to':'i8'}]):
        raise ValueError('selected output path does not implement the source column readout')
    return path


@lru_cache(maxsize=256)
def _prove_positive(scale):
    from merlin.llvmlower.requantization import prove_scale
    return prove_scale([scale])['scale']


def match_column_readout(op, *, allow_nhwc=False, allow_ranked=False):
    """Return a strict f32 multiply/RNE/clamp match, or None for other source.

    Positive scales use the shared all-i32 transition proof. A zero scale is
    copied unchanged through the identical source/target ordered stages; no
    ``allow_ranked`` selects standalone contiguous tensors of any positive
    static rank, with the last source axis selecting the scale. It does not
    change the separate rank-two/rank-four ``allow_nhwc`` composition route.
    No scale folding or activation movement occurs. Negative scales remain outside
    the available source fusion proof's positive-scale domain.
    """
    from xdsl.dialects.builtin import TensorType, NoneAttr
    from xdsl.ir.affine import AffineMap, AffineDimExpr
    from merlin.llvmlower.requantization import f32
    from merlin.runtime.commandbuffer import declared_acc_scale_by_column
    def typ(v):
        t=v.type
        return (tuple(t.get_shape()),str(t.element_type)) if isinstance(t,TensorType) and isinstance(t.encoding,NoneAttr) else None
    if op.name!='linalg.generic' or len(op.inputs)!=2 or len(op.outputs)!=1 or len(op.results)!=1:
        return None
    if (any(k not in ('indexing_maps','iterator_types','operandSegmentSizes') for k in op.properties)
            or any(not k.startswith('prov.') for k in op.attributes)):
        return None
    source,scale_value=op.inputs
    source_type=typ(source)
    if source_type is None or source_type[1]!='i32':return None
    if (not source_type[0] or any(type(d) is not int or d<=0 for d in source_type[0])
            or not allow_ranked and len(source_type[0]) not in ((2,4) if allow_nhwc else (2,))):return None
    shape=source_type[0];n=shape[-1]
    if (typ(scale_value)!=((n,),'f32') or typ(op.results[0])!=(shape,'i8')
            or op.outputs[0].type!=op.results[0].type):return None
    rank=len(shape)
    identity=AffineMap.identity(rank);columns=AffineMap(rank,0,(AffineDimExpr(rank-1),))
    if (tuple(a.data for a in op.indexing_maps)!=(identity,columns,identity)
            or tuple(i.data.value for i in op.iterator_types)!=('parallel',)*rank):return None
    owner=scale_value.owner
    if getattr(owner,'name',None)!='arith.constant':return None
    attr=owner.properties.get('value',owner.attributes.get('value'))
    if not hasattr(attr,'get_values'):return None
    scales=tuple(f32(s) for s in attr.get_values())
    if len(scales)==1:scales=scales*n
    if len(scales)!=n or any(not math.isfinite(s) or s<0 for s in scales):return None
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
    block=op.regions[0].block
    ops=[s for s in block.ops if s.name!='arith.constant']
    if len(block.args)!=3 or [s.name for s in ops]!=['arith.sitofp','arith.mulf','math.roundeven','arith.maximumf','arith.minimumf','arith.fptosi','linalg.yield']:return None
    for s in block.ops:
        if str(s.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>':return None
        if any(not k.startswith('prov.') for k in s.attributes):return None
        if any(k not in ('value','fastmath') for k in s.properties):return None
    convert,mul,rnd,lo,hi,cast,yieldop=ops
    def constant(v,expected):
        o=v.owner;a=getattr(o,'properties',{}).get('value',getattr(o,'attributes',{}).get('value'))
        return (getattr(o,'name',None)=='arith.constant' and str(v.type)=='f32'
                and getattr(getattr(a,'value',None),'data',None)==expected)
    def binary(step,value,other):return len(step.operands)==2 and value in step.operands and any(other(v) for v in step.operands if v is not value)
    if (list(convert.operands)!=[block.args[0]] or str(convert.results[0].type)!='f32'
            or set(mul.operands)!={convert.results[0],block.args[1]}
            or list(rnd.operands)!=[mul.results[0]]
            or not binary(lo,rnd.results[0],lambda v:constant(v,-128.0))
            or not binary(hi,lo.results[0],lambda v:constant(v,127.0))
            or list(cast.operands)!=[hi.results[0]] or str(cast.results[0].type)!='i8'
            or list(yieldop.operands)!=[cast.results[0]]):return None
    _path()
    declared_acc_scale_by_column({'acc_scale_by_column':list(scales)},n,op='source readout')
    for scale in set(scales):
        if scale>0 and _prove_positive(scale)!=scale:return None
    op.verify()
    return ColumnReadout(source,scale_value,scales)
