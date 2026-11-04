"""Exact device forms matched directly on original typed graph SSA.

Signed integer movement, widening, additions and saturating narrowing use DMA
into Gemmini's accumulator. Matching is pure; unsupported scalar arithmetic is
never reinterpreted as a device form from provenance or model names.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from xdsl.dialects.builtin import TensorType
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr
from merlin.targetgen.contract import linalg_iface as li
from vector import _expression
from elementwise import lower_movement, _tiles, _address, _geometry
import target


def _dtype(value):
    return str(value.type.element_type) if isinstance(value.type,TensorType) else str(value.type)


def _shape(value):
    return tuple(value.type.get_shape())


@dataclass(frozen=True)
class DeviceForm:
    kind: str
    inputs: tuple
    output: object
    maps: tuple = ()
    extents: tuple = ()
    relu: bool = False
    narrowing: bool = False


def _constant(expr):
    return expr[1] if expr[0]=='const' else None


def _clamp(expr):
    """Return base, lower bound for a complete signed integer [-128/0,127] clamp."""
    if expr[0] != 'arith.trunci' or expr[2] != 'i8': return None
    value = expr[1][0]
    bounds = {}
    for _ in range(2):
        if value[0] not in ('arith.minsi','arith.maxsi') or value[2]!='i32': return None
        left,right = value[1]
        if left[0]=='const': left,right=right,left
        if right[0]!='const': return None
        bounds[value[0]]=right[1]
        value=left
    if bounds.get('arith.minsi')!=127 or bounds.get('arith.maxsi') not in (-128,0): return None
    return value,bounds['arith.maxsi']


def match_device(op):
    """Pure semantic match shared by placement inventory and actual lowering."""
    if op.name in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
        if len(op.operands)!=1 or len(op.results)!=1: return None
        src,out=op.operands[0],op.results[0]
        if (_dtype(src) not in ('i8','i32') or _dtype(src)!=_dtype(out)
            or prod(_shape(src))!=prod(_shape(out)) or min((*_shape(src),*_shape(out)),default=1)<=0): return None
        return DeviceForm('contiguous_movement',(src,),out)
    if op.name not in ('linalg.generic','linalg.transpose'): return None
    ins,outs=tuple(op.inputs),tuple(op.outputs)
    if len(op.results)!=1 or len(outs)!=1: return None
    out=op.results[0]
    if _dtype(out) not in ('i8','i32'): return None
    if op.name=='linalg.transpose':
        if len(ins)!=1 or _dtype(ins[0])!=_dtype(out):return None
        from xdsl.ir.affine import AffineMap, AffineExpr
        permutation=op.properties.get('permutation',op.attributes.get('permutation'))
        perm=tuple(permutation.get_values()) if hasattr(permutation,'get_values') else tuple(x.value.data for x in permutation)
        if sorted(perm)!=list(range(len(_shape(out)))):return None
        # Input coordinate i reads output coordinate inverse_permutation[i].
        inverse=tuple(perm.index(i) for i in range(len(perm)))
        imap=AffineMap(len(perm),0,tuple(AffineExpr.dimension(i) for i in inverse))
        omap=AffineMap.identity(len(perm))
        if any(_shape(ins[0])[i]!=_shape(out)[inverse[i]] for i in range(len(perm))):return None
        return DeviceForm('movement',ins,out,(imap,omap),_shape(out))
    if any('parallel' not in str(x) for x in op.iterator_types):return None
    if li._reduction_dims(op):return None
    maps=tuple(x.data for x in op.indexing_maps)
    if len(maps)!=len(ins)+len(outs):return None
    rank=maps[-1].num_dims
    if maps[-1].num_symbols or tuple(maps[-1].results)!=tuple(AffineDimExpr(i) for i in range(rank)):return None
    extents=_shape(out)
    if len(extents)!=rank or any(x<=0 for x in extents):return None
    for val,amap in zip((*ins,*outs),maps):
        if amap.num_dims!=rank or amap.num_symbols or len(amap.results)!=len(_shape(val)):return None
        for expr,extent in zip(amap.results,_shape(val)):
            if isinstance(expr,AffineDimExpr):
                if extents[expr.position]!=extent:return None
            elif isinstance(expr,AffineConstantExpr):
                if expr.value!=0 or extent!=1:return None
            else:return None
    def operand(value):
        rec={'shape':li._shape(value.type),'dtype':li._dtype(value.type)}
        constant=li._scalar_const(value)
        if constant is not None:rec['const_value']=constant
        return rec
    try: expr=_expression({'scalar_body':li._scalar_body(op,operand)})
    except (ValueError,KeyError,TypeError):return None
    def source(e):
        if e[0]=='arith.extsi' and e[2]=='i32':
            arg=e[1][0]
            if arg[0]!='arg' or arg[2]!='i8':return None
            e=arg
        if e[0]!='arg' or e[1]>=len(ins) or _dtype(ins[e[1]])!=e[2]:return None
        if e[2] not in ('i8','i32'):return None
        return e[1]
    narrow=False;relu=False
    clamp=_clamp(expr)
    if clamp is not None:
        # Scale-one readout equals integer saturation over the whole i32
        # domain: every integer inside [-128,127] is exactly f32, and f32
        # conversion of any value outside that interval remains outside it.
        expr,lower=clamp;narrow=True;relu=lower==0
    elif expr[0]=='arith.maxsi' and expr[2]=='i8' and _dtype(out)=='i8':
        x,y=expr[1]
        if x[0]=='const':x,y=y,x
        if _constant(y)!=0:return None
        expr=x;narrow=True;relu=True
    if expr[0]=='arith.addi' and expr[2]=='i32':
        ids=tuple(source(e) for e in expr[1])
        if any(i is None for i in ids):return None
        if not narrow and _dtype(out)!='i32':return None
        return DeviceForm('add',tuple(ins[i] for i in ids),out,
                          tuple(maps[i] for i in ids)+(maps[-1],),extents,relu,narrow)
    i=source(expr)
    if i is None:return None
    src=ins[i]
    if not narrow:
        # i8->i32 sign extension is the accumulator's exact input conversion.
        if _dtype(src)!=_dtype(out) and not (_dtype(src)=='i8' and _dtype(out)=='i32'):return None
    return DeviceForm('movement',(src,),out,(maps[i],maps[-1]),extents,relu,narrow)


def can_lower_device(op):
    return match_device(op) is not None


def lower_device(graph,op):
    """Realize one matched SSA op and return True; False emits nothing."""
    form=match_device(op)
    if form is None:return False
    b=graph.b;srcs=form.inputs;out=form.output
    shape=_shape(out)
    if form.kind=='contiguous_movement':
        lower_movement(b,graph.ptrs[srcs[0]],graph.ptrs[out],shape or (1,),_dtype(srcs[0]),_dtype(out))
        return True
    # Identity maps can use rectangular DMA tiles, preserving compact code and
    # device issue count for the common row-major copy/widening path.
    if form.kind=='movement' and not form.relu and not form.narrowing and form.maps[0]==form.maps[-1] and _shape(srcs[0])==shape:
        lower_movement(b,graph.ptrs[srcs[0]],graph.ptrs[out],shape or (1,),_dtype(srcs[0]),_dtype(out))
        return True
    full=form.maps[-1]
    rank=len(shape)
    column_map=(AffineDimExpr(rank-1),) if rank else ()
    indexed=[]
    for value,amap in zip(srcs,form.maps):
        if amap==full and _shape(value)==shape:indexed.append('full')
        elif tuple(amap.results)==column_map and _shape(value)==(shape[-1],):indexed.append('column')
        else:indexed.append('other')
    if 'other' not in indexed:
        rows,cols=_geometry(shape or (1,))
        for stream,(value,kind) in enumerate(zip(srcs,indexed)):
            width=1 if _dtype(value)=='i8' else 4
            target.cfg_ld(b,cols*width if kind=='full' else 0,stream=stream,shrunk=width==1)
        width=1 if _dtype(out)=='i8' else 4
        target.cfg_st(b,cols*width,scale=1.,activation=int(form.relu))
        def tile(row,col,nr,nc):
            for stream,(value,kind) in enumerate(zip(srcs,indexed)):
                ib=1 if _dtype(value)=='i8' else 4
                ptr=_address(b,graph.ptrs[value],row,col,cols,ib) if kind=='full' else b.ptr(graph.ptrs[value],b.mul_i(col,b.const(ib)))
                target.mvin(b,ptr,target.ACC|(target.ACCUMULATE if stream else 0),nr,nc,stream=stream)
                b.fence()
            target.mvout(b,_address(b,graph.ptrs[out],row,col,cols,width),target.ACC|(target.FULL if width==4 else 0),nr,nc)
            b.fence()
        _tiles(b,rows,cols,tile)
        return True
    for stream,value in enumerate(srcs):
        target.cfg_ld(b,0,stream=stream,scale=1.,shrunk=_dtype(value)=='i8')
    target.cfg_st(b,0,scale=1.,activation=int(form.relu))
    output_bytes=1 if _dtype(out)=='i8' else 4
    def address(value,amap,indices):
        mapped=[graph.affine(expr,indices) for expr in amap.results]
        flat=graph.flat(_shape(value),mapped)
        return b.ptr(graph.ptrs[value],b.mul_i(flat,b.const(1 if _dtype(value)=='i8' else 4)))
    def emit(indices):
        for stream,(value,amap) in enumerate(zip(srcs,form.maps)):
            target.mvin(b,address(value,amap,indices),target.ACC | (target.ACCUMULATE if stream else 0),1,1,stream=stream)
            b.fence()
        target.mvout(b,address(out,form.maps[-1],indices),target.ACC | (target.FULL if output_bytes==4 else 0),1,1)
        b.fence()
    graph.loops(form.extents,emit)
    return True
