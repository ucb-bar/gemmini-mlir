"""Exact lowered per-tensor residual recognition, independent of quant_ext names.

The adapter accepts only dense positive static i8 leaves and a complete scalar
SSA proof of f32 dequantize/add/(ReLU)/reciprocal-multiply/roundeven/clamp/i8.
Every scale is an immutable scalar. All 65,536 source pairs are compared against
the selected existing DMA operand-sum schedule, with zero allowed error.
"""
from dataclasses import dataclass
import math


class ResidualRefusal(ValueError):
    pass


@dataclass(frozen=True)
class LoweredResidual:
    operations: tuple
    lhs: object
    rhs: object
    output: object
    attributes: dict
    proof: dict

    def adapter(self):
        """Return the existing source_groups._quantized_residual adapter tuple."""
        return list(self.operations), {'lhs': self.lhs, 'rhs': self.rhs}, dict(self.attributes), dict(self.proof)


def _spec(value):
    from xdsl.dialects.builtin import TensorType, NoneAttr
    ty=value.type
    if not isinstance(ty,TensorType) or not isinstance(ty.encoding,NoneAttr):
        raise ResidualRefusal('dense unencoded tensor required')
    shape=tuple(ty.get_shape())
    if any(d<=0 for d in shape):raise ResidualRefusal('positive static extents required')
    return shape,str(ty.element_type)


def _literal(value):
    owner=value.owner
    if getattr(owner,'name','')=='tensor.splat':return _literal(owner.operands[0])
    if getattr(owner,'name','')=='arith.constant':
        attr=owner.properties.get('value',owner.attributes.get('value'))
        number=getattr(getattr(attr,'value',None),'data',None)
        if isinstance(number,(int,float)):return number
    return None


def _commutative(expr,kind,constant=None):
    if expr[0]!=kind:return None
    a,b=expr[1:]
    if constant is not None:
        if a==('const',constant):return b
        if b==('const',constant):return a
    elif a[0]=='const':return b,a[1]
    elif b[0]=='const':return a,b[1]
    return None


class _Expression:
    def __init__(self,root,shape):
        self.root=root;self.shape=shape;self.members=set();self.cache={}

    def tensor(self,value,depth=0):
        from xdsl.ir.affine import AffineMap
        if depth>32:raise ResidualRefusal('source expression is too deep')
        shape,dtype=_spec(value)
        literal=_literal(value)
        if literal is not None:
            if dtype not in ('f32','i64','i32'):raise ResidualRefusal('unsupported literal type')
            return ('const',literal)
        if dtype=='i8' and value is not self.root.results[0]:return ('input',value)
        if value in self.cache:return self.cache[value]
        op=value.owner
        if getattr(op,'name','')!='linalg.generic' or len(op.results)!=1 or op.parent is not self.root.parent:
            raise ResidualRefusal('float boundary is not a proved same-block generic expression')
        if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:raise ResidualRefusal('one scalar block required')
        maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
        rank=len(shape)
        if shape not in (self.shape,()):raise ResidualRefusal('tensor shape differs across residual')
        if maps is None or len(maps)!=len(op.operands) or iters is None or [x.data.value for x in iters]!=['parallel']*rank:
            raise ResidualRefusal('parallel pointwise indexing required')
        body=op.regions[0].block
        if len(body.args)!=len(op.operands):raise ResidualRefusal('scalar argument arity differs')
        env={}
        for index,(arg,operand,mapping) in enumerate(zip(body.args,op.operands,maps)):
            operand_shape,_=_spec(operand)
            expected=AffineMap.identity(rank) if operand_shape==shape else AffineMap(rank,0,()) if operand_shape==() else None
            if expected is None or mapping.data!=expected:raise ResidualRefusal('identity or rank-zero scalar map required')
            env[arg]=('init',) if index==len(op.operands)-1 else self.tensor(operand,depth+1)
        yielded=None
        for scalar in body.ops:
            if str(scalar.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>':raise ResidualRefusal('fastmath changes source contract')
            name=scalar.name
            args=[env[v] if v in env else self.scalar_literal(v) for v in scalar.operands]
            if name=='linalg.yield':
                if yielded is not None or len(args)!=1:raise ResidualRefusal('one scalar yield required')
                yielded=args[0];continue
            if len(scalar.results)!=1:raise ResidualRefusal('unsupported scalar arity')
            if name=='arith.constant':
                number=_literal(scalar.results[0])
                if number is None:raise ResidualRefusal('scalar literal required')
                result=('const',number)
            elif name=='arith.sitofp' and str(scalar.results[0].type)=='f32':
                if str(scalar.operands[0].type)not in ('i8','i32','i64'):raise ResidualRefusal('signed integer cast required')
                result=('const',float(args[0][1])) if args[0][0]=='const' else ('sitofp',args[0])
            elif name in ('arith.addf','arith.subf','arith.mulf','arith.divf') and str(scalar.results[0].type)=='f32':
                kind={'arith.addf':'add','arith.subf':'sub','arith.mulf':'mul','arith.divf':'div'}[name]
                result=(kind,*args)
                if all(a[0]=='const' for a in args):
                    import numpy as np
                    fn={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}[kind]
                    with np.errstate(all='ignore'):number=float(fn(np.float32(args[0][1]),np.float32(args[1][1]),dtype=np.float32))
                    if not math.isfinite(number):raise ResidualRefusal('nonfinite constant arithmetic')
                    result=('const',number)
                elif kind=='sub' and args[1]==('const',0):result=args[0]
            elif name=='math.roundeven' and str(scalar.results[0].type)=='f32':result=('round',args[0])
            elif name in ('arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf') and str(scalar.results[0].type)=='f32':
                result=('max'if name in ('arith.maximumf','arith.maxnumf')else'min',*args)
            elif name=='arith.cmpf' and scalar.properties['predicate'].value.data==4:
                result=('olt',*args)
            elif name=='arith.select':
                condition,yes,no=args
                if condition!=('olt',no,('const',0)) or yes!=('const',0):raise ResidualRefusal('only ordered compare/select ReLU admitted')
                result=('max',no,yes)
            elif name=='arith.fptosi' and str(scalar.results[0].type)=='i8':result=('i8',args[0])
            else:raise ResidualRefusal('unsupported scalar operation '+name)
            env[scalar.results[0]]=result
        if yielded is None:raise ResidualRefusal('missing scalar yield')
        self.members.add(op);self.cache[value]=yielded
        return yielded

    @staticmethod
    def scalar_literal(value):
        number=_literal(value)
        if number is None:raise ResidualRefusal('uncaptured runtime scalar')
        return ('const',number)


def _arm(expr):
    scale=1.0
    mul=_commutative(expr,'mul')
    if mul is not None:expr,scale=mul
    if expr[0]!='sitofp' or expr[1][0]!='input':raise ResidualRefusal('dequantized signed i8 leaf required')
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:raise ResidualRefusal('positive finite source scale required')
    return expr[1][1],float(scale)


def recognize(op):
    """Return an exact adapter or None; use prove() to inspect a refusal."""
    try:return prove(op)
    except (ResidualRefusal,ValueError,TypeError,KeyError,OverflowError):return None


def prove(op):
    import numpy as np
    from backend.elementwise import residual_numeric_screen
    if op.name!='linalg.generic' or len(op.results)!=1:raise ResidualRefusal('generic quantizer output required')
    shape,dtype=_spec(op.results[0])
    if dtype!='i8' or not shape:raise ResidualRefusal('ranked i8 output required')
    builder=_Expression(op,shape);expression=builder.tensor(op.results[0])
    expr=expression
    if expr[0]!='i8':raise ResidualRefusal('saturating fptosi required')
    expr=_commutative(expr[1],'min',127)
    if expr is None:raise ResidualRefusal('signed upper clamp 127 required')
    lower=_commutative(expr,'max',-128)
    relu=False
    if lower is None:
        lower=_commutative(expr,'max',0);relu=True
    if lower is None:raise ResidualRefusal('signed lower clamp required')
    expr=lower
    zero=_commutative(expr,'add',0)
    if zero is not None:expr=zero
    if expr[0]!='round':raise ResidualRefusal('one half-even rounding required')
    expr=expr[1];reciprocal=1.0
    mul=_commutative(expr,'mul')
    if mul is not None:expr,reciprocal=mul
    if not isinstance(reciprocal,(int,float)) or not math.isfinite(reciprocal) or reciprocal<=0:raise ResidualRefusal('positive finite reciprocal required')
    activation=_commutative(expr,'max',0)
    if activation is not None:expr=activation;relu=True
    if expr[0]!='add':raise ResidualRefusal('two dequantized operands required')
    lhs,ls=_arm(expr[1]);rhs,rs=_arm(expr[2])
    if _spec(lhs)!=(shape,'i8') or _spec(rhs)!=(shape,'i8'):raise ResidualRefusal('identical dense i8 operands required')
    normalized=(float(np.float32(np.float64(ls)*np.float64(reciprocal))),float(np.float32(np.float64(rs)*np.float64(reciprocal))))
    hardware=residual_numeric_screen(*normalized,0,relu)
    a=np.arange(-128,128,dtype=np.float32)[:,None];b=np.arange(-128,128,dtype=np.float32)[None,:]
    with np.errstate(all='ignore'):
        ap=np.multiply(a,np.float32(ls),dtype=np.float32);bp=np.multiply(b,np.float32(rs),dtype=np.float32)
        source=np.add(ap,bp,dtype=np.float32)
    if not all(np.all(np.isfinite(v)) for v in (ap,bp,source)):raise ResidualRefusal('nonfinite dequantization or addition')
    if activation is not None:source=np.maximum(source,np.float32(0))
    with np.errstate(all='ignore'):source=np.multiply(source,np.float32(reciprocal),dtype=np.float32)
    if not np.all(np.isfinite(source)):raise ResidualRefusal('nonfinite requantization intermediate')
    source=np.rint(source)
    if relu:source=np.maximum(source,np.float32(0))
    source=np.clip(source,-128,127)
    device=np.add(np.clip(np.rint(np.multiply(a,np.float32(hardware['lhs_load_f32']),dtype=np.float32)),-128,127),np.clip(np.rint(np.multiply(b,np.float32(hardware['rhs_load_f32']),dtype=np.float32)),-128,127),dtype=np.float32)
    if relu:device=np.maximum(device,np.float32(0))
    device=np.clip(np.rint(np.multiply(device,np.float32(hardware['readout_scale_f32']),dtype=np.float32)),-128,127)
    if not np.all(np.isfinite(source)) or not np.array_equal(source,device):raise ResidualRefusal('original f32 stages differ from device operand rounding')
    members=builder.members
    for member in members:
        for value in member.results:
            if value is not op.results[0] and any(u.operation not in members for u in value.uses):raise ResidualRefusal('live intermediate fanout requires a separate output')
    ordered=tuple(o for o in op.parent.ops if o in members)
    attrs={'lhs_scale':normalized[0],'rhs_scale':normalized[1],'bound_lsb':0,'output_dtype':'i8','epilogue':['relu']if relu else []}
    proof={'form':'exact_lowered_generic_residual','pairs_checked':65536,'max_error_lsb':0,'source_scales_f32':[ls,rs],'source_reciprocal_f32':float(reciprocal),'source_rounding':'half_even','source_order':['sitofp_f32','dequant_multiply_f32','add_f32']+(['relu']if activation is not None else[])+['reciprocal_multiply_f32','roundeven','signed_clamp','fptosi_i8'],'source_ids':[{k:str(v)for k,v in member.attributes.items()if k.startswith(('prov.','merlin.source'))}for member in ordered]}
    return LoweredResidual(ordered,lhs,rhs,op.results[0],attrs,proof)
