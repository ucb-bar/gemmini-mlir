"""Exact float-pool boundary proof and pooled packed-stem rewrite."""
import math
from xdsl.ir import Operation,Region
from xdsl.ir.affine import AffineMap,AffineExpr
from xdsl.dialects import func,tensor
from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,StringAttr,IntegerAttr,UnitAttr,TensorType,i8,i64
from .direct_conv_binding import _transpose


def inspect_pool(insert,value):
    from .captured_requant import scalar
    if len(insert.operands)!=2 or insert.operands[0] is not value:raise ValueError('pool pad must insert the exact epilogue')
    shape=tuple(value.type.get_shape())
    if len(shape)!=4 or shape[0]!=1:raise ValueError('pool requires NCHW one batch')
    n,c,h,w=shape
    if tuple(insert.results[0].type.get_shape())!=(1,c,h+2,w+2):raise ValueError('pool pad extent mismatch')
    for key,want in [('static_offsets',[0,0,1,1]),('static_sizes',list(shape)),('static_strides',[1,1,1,1])]:
        if list(insert.properties[key].get_values())!=want:raise ValueError('pool pad geometry mismatch')
    if scalar(insert.operands[1])!=-math.inf:raise ValueError('pool pad must be negative infinity')
    uses=list(insert.results[0].uses)
    if len(uses)!=1:raise ValueError('pool padding fanout')
    pool=uses[0].operation
    if pool.name!='linalg.generic' or len(pool.operands)!=3 or pool.operands[0] is not insert.results[0]:raise ValueError('expected maxpool reduction')
    if tuple(pool.operands[1].type.get_shape())!=(3,3):raise ValueError('pool window must be3x3')
    if tuple(pool.results[0].type.get_shape())!=(1,c,(h+1)//2,(w+1)//2):raise ValueError('pool output extent mismatch')
    if scalar(pool.operands[2])!=-math.inf:raise ValueError('pool identity must be negative infinity')
    d=[AffineExpr.dimension(i) for i in range(6)]
    expected=[AffineMap(6,0,(d[0],d[1],d[2]*2+d[4],d[3]*2+d[5])),AffineMap(6,0,(d[4],d[5])),AffineMap(6,0,tuple(d[:4]))]
    if [m.data for m in pool.indexing_maps]!=expected:raise ValueError('pool affine maps mismatch')
    if [x.data.value for x in pool.iterator_types]!=['parallel']*4+['reduction']*2:raise ValueError('pool iterator mismatch')
    block=pool.body.block;ops=list(block.ops)
    if len(ops)!=2 or ops[0].name!='arith.maximumf' or set(ops[0].operands)!={block.args[0],block.args[2]} or ops[1].name!='linalg.yield' or list(ops[1].operands)!=list(ops[0].results):raise ValueError('pool reducer is not exact maximum')
    fm=ops[0].properties.get('fastmath')
    if fm is not None and str(fm)!='#arith.fastmath<none>':raise ValueError('pool fastmath is not supported')
    return pool,{'input_shape':list(shape),'output_shape':[1,c,(h+1)//2,(w+1)//2],'kernel':[3,3],'stride':[2,2],'padding':[1,1,1,1],'source_padding':'negative_infinity','device_padding':0,'proof':'ReLU outputs are nonnegative and non-NaN; every window has a valid pixel, so zero and -infinity padding give the same maximum. Positive scalar quantization is monotone and its complete transition proof permits max/quantize exchange.'}


def rewrite(binding,chain,shape,symbol):
    op=binding.contraction;module=op
    while module.parent_op() is not None:module=module.parent_op()
    operations,activation=_transpose(binding.input,[0,2,3,1]);ct=TensorType(i8,[shape.ph*shape.pw,shape.cout]);empty=tensor.EmptyOp([],ct);operations.append(empty)
    inputs=[activation,binding.weight,empty.tensor];call=func.CallOp(symbol,inputs,[ct]);operations.append(call)
    outtype=TensorType(i8,[1,shape.ph,shape.pw,shape.cout]);reassoc=ArrayAttr([ArrayAttr([IntegerAttr(i,i64) for i in [0,1,2]]),ArrayAttr([IntegerAttr(3,i64)])]);view=tensor.ExpandShapeOp(call.results[0],[],reassoc,list(outtype.get_shape()),outtype);operations.append(view)
    trans,result=_transpose(view.result,[0,3,1,2]);operations+=trans
    if result.type!=chain['quantize'].results[0].type:raise ValueError('pooled output layout mismatch')
    op.parent.insert_ops_before(operations,op);chain['quantize'].results[0].replace_all_uses_with(result)
    for old in reversed(chain['operations']):old.parent.erase_op(old)
    op.parent.erase_op(op)
    attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in ['read','read','write']]);declaration=func.FuncOp(symbol,([x.type for x in inputs],[ct]),Region(),visibility='private',arg_attrs=attrs);declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration);return declaration


def validate_layout(binding,chain):
    layouts=chain['layouts'];s=binding.shape
    if [x.name for x in layouts]!=['tensor.collapse_shape','tensor.expand_shape','linalg.transpose']:
        raise ValueError('stem pool requires exact spatial-major to NCHW layout')
    if [tuple(x.results[0].type.get_shape()) for x in layouts]!=[(s.oh*s.ow*s.cout,),(1,s.oh,s.ow,s.cout),(1,s.cout,s.oh,s.ow)] or list(layouts[2].permutation.get_values())!=[0,3,1,2]:
        raise ValueError('stem pool feature/spatial layout proof failed')
    for layout,rank in zip(layouts[:2],[2,4]):
        groups=layout.properties['reassociation']
        if len(groups)!=1 or [x.value.data for x in groups.data[0].data]!=list(range(rank)):
            raise ValueError('stem pool flattening reassociation mismatch')
