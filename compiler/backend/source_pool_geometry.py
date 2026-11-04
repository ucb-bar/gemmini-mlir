"""Exact unpadded static NHWC signed-i8 max-reduction source geometry.

Height/width and their strides are proved independently from actual affine SSA.
The existing MAXPOOL2D callback retains the rectangular source geometry; this
module grants no arithmetic approximation, padding, placement or ownership.
"""


class PoolGeometryRefusal(ValueError):
    pass


def _tensor(value):
    from xdsl.dialects.builtin import TensorType,NoneAttr
    ty=value.type
    if not isinstance(ty,TensorType) or not isinstance(ty.encoding,NoneAttr):raise PoolGeometryRefusal('dense unencoded ranked tensor required')
    shape=tuple(ty.get_shape())
    if not shape or any(d<=0 for d in shape):raise PoolGeometryRefusal('positive static extents required')
    return shape,str(ty.element_type)


def _safe(op,properties=()):
    if any(not k.startswith(('prov.','merlin.source'))for k in op.attributes):raise PoolGeometryRefusal('unproved source attributes')
    if any(k not in properties for k in op.properties):raise PoolGeometryRefusal('unproved source properties')


def _constant(value):
    owner=value.owner
    name=getattr(owner,'name','')
    if name in ('tensor.splat','linalg.fill'):
        _safe(owner,('operandSegmentSizes',))
        if not owner.operands:raise PoolGeometryRefusal('missing initializer scalar')
        if name=='linalg.fill':
            if len(owner.operands)!=2 or len(owner.results)!=1 or owner.results[0] is not value or owner.operands[1].type!=value.type or owner.operands[0].type!=value.type.element_type:raise PoolGeometryRefusal('exact scalar fill initializer required')
            if len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1:raise PoolGeometryRefusal('one fill scalar block required')
            block=owner.regions[0].block;ops=list(block.ops)
            if len(block.args)!=2 or any(arg.type!=owner.operands[0].type for arg in block.args) or len(ops)!=1 or ops[0].name!='linalg.yield' or tuple(ops[0].operands)!=(block.args[0],):raise PoolGeometryRefusal('initializer fill must yield its scalar input')
            _safe(ops[0])
        elif len(owner.operands)!=1 or len(owner.results)!=1 or owner.results[0] is not value or owner.operands[0].type!=value.type.element_type or owner.regions:raise PoolGeometryRefusal('exact scalar tensor splat required')
        return _constant(owner.operands[0])
    if name=='arith.constant':
        _safe(owner,('value',))
        attr=owner.properties.get('value',owner.attributes.get('value'))
        return getattr(getattr(attr,'value',None),'data',None)
    return None


def prove_integer_pool(op):
    from xdsl.ir.affine import AffineMap,AffineExpr
    if op.name!='linalg.generic' or len(op.operands)!=3 or len(op.results)!=1:raise PoolGeometryRefusal('three-operand one-result generic required')
    _safe(op,('indexing_maps','iterator_types','operandSegmentSizes'))
    src,window,init=op.operands
    a,ad=_tensor(src);c,cd=_tensor(op.results[0]);w,_=_tensor(window);initial,initial_dtype=_tensor(init)
    if ad!='i8' or cd!='i8' or len(a)!=4 or len(c)!=4 or len(w)!=2 or initial!=c or initial_dtype!='i8':raise PoolGeometryRefusal('matching NHWC signed-i8 source/output/initializer required')
    batch,h,width,channels=a;nb,oh,ow,nc=c;ph,pw=w
    if nb!=batch or nc!=channels or not 1<=ph<=3 or not 1<=pw<=3 or h<ph or width<pw:raise PoolGeometryRefusal('independent window extents 1..3 and matching batch/channels required')
    if _constant(init)!=-128:raise PoolGeometryRefusal('signed maximum initializer must be exactly -128')
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if maps is None or len(maps)!=3 or iters is None or [x.data.value for x in iters]!=['parallel']*4+['reduction']*2:raise PoolGeometryRefusal('NHWC parallel plus two reduction axes required')
    maps=tuple(m.data for m in maps);d=tuple(AffineExpr.dimension(i)for i in range(6))
    if maps[1]!=AffineMap(6,0,(d[4],d[5])) or maps[2]!=AffineMap(6,0,d[:4]):raise PoolGeometryRefusal('exact window and output maps required')
    source_map=maps[0]
    if source_map.num_dims!=6 or source_map.num_symbols!=0 or len(source_map.results)!=4:raise PoolGeometryRefusal('static six-axis NHWC source map required')
    zero=(0,)*6
    try:
        sh=source_map.results[1].eval((0,1,0,0,0,0),())-source_map.results[1].eval(zero,())
        sw=source_map.results[2].eval((0,0,1,0,0,0),())-source_map.results[2].eval(zero,())
    except (ValueError,ZeroDivisionError,IndexError) as exc:raise PoolGeometryRefusal('source axis coefficient cannot be proved')from exc
    if not isinstance(sh,int) or not isinstance(sw,int) or min(sh,sw)<=0:raise PoolGeometryRefusal('independently positive source strides required')
    expected=AffineMap(6,0,(d[0],d[1]*sh+d[4],d[2]*sw+d[5],d[3]))
    if source_map!=expected:raise PoolGeometryRefusal('source map must be exactly NHWC independent stride plus window offset')
    if oh!=(h-ph)//sh+1 or ow!=(width-pw)//sw+1:raise PoolGeometryRefusal('output dimensions disagree with unpadded geometry')
    if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:raise PoolGeometryRefusal('one max reduction scalar block required')
    body=op.regions[0].block;ops=list(body.ops)
    if len(body.args)!=3 or any(arg.type!=operand.type.element_type for arg,operand in zip(body.args,op.operands)) or body.args[1].uses:raise PoolGeometryRefusal('scalar types must match; window payload must be unobserved')
    if len(ops)!=2 or ops[0].name!='arith.maxsi' or len(ops[0].results)!=1 or str(ops[0].results[0].type)!='i8' or set(ops[0].operands)!={body.args[0],body.args[2]} or ops[1].name!='linalg.yield' or tuple(ops[1].operands)!=tuple(ops[0].results):raise PoolGeometryRefusal('exact signed max(input,accumulator) yield required')
    for scalar in ops:_safe(scalar)
    attrs={'epilogue':['acc_scale','maxpool'],'acc_scale':1.0,'output_dtype':'i8','pool_in_dims':[h,width],'pool_size':[ph,pw],'pool_stride':[sh,sw],'pool_padding':[0,0,0,0]}
    return src,attrs,{}


def recognize_integer_pool(op):
    try:return prove_integer_pool(op)
    except (PoolGeometryRefusal,ValueError,TypeError,KeyError,IndexError):return None
