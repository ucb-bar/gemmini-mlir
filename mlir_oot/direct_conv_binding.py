"""Prove and replace an integer im2col chain with a direct-convolution call.

Explicit boundary transposes preserve upstream NCHW/KH-KW reduction ordering.
The padded quantized halo is retained verbatim; its zero value is not assumed.
"""
from dataclasses import dataclass
from math import prod
from xdsl.dialects import func, tensor
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.builtin import ArrayAttr, AffineMapAttr, DictionaryAttr, IntegerAttr, StringAttr, UnitAttr, TensorType, i64, i8, i32
from xdsl.ir import Operation, Block, Region, SSAValue
from xdsl.ir.affine import AffineMap, AffineExpr
from .contraction_patterns import match_integer_gemm, _shape
from .golden_conv import ConvShape


@dataclass
class DirectConvBinding:
    contraction: Operation
    gather: Operation
    input: SSAValue
    shape: ConvShape
    orientation: str = "channels_first"

    @property
    def weight(self):
        return self.contraction.operands[0 if self.orientation == "channels_first" else 1]


def _flat_reshape(value, rank):
    expand = value.owner
    if not isinstance(expand,Operation) or expand.name != 'tensor.expand_shape':
        raise ValueError('gather must reach matmul through a flat reshape')
    collapse = expand.operands[0].owner
    if not isinstance(collapse,Operation) or collapse.name != 'tensor.collapse_shape':
        raise ValueError('gather must reach matmul through a flat reshape')
    for op,n in [(collapse,rank),(expand,2)]:
        reassoc = op.properties.get('reassociation')
        if reassoc is None or len(reassoc.data)!=1 or [x.value.data for x in reassoc.data[0].data] != list(range(n)):
            raise ValueError('reshape reassociation is not row-major flattening')
    if _shape(collapse.results[0],'i8') != (prod(collapse.operands[0].type.get_shape()),):
        raise ValueError('reshape is not a complete flattening')
    return collapse.operands[0]


def _match_gather(op):
    dims = match_integer_gemm(op)
    if dims is None or dims.batch != 1:
        raise ValueError('expected exact from-zero signed i8/i32 matmul')
    value = _flat_reshape(op.operands[1],6)
    gather = value.owner
    if not isinstance(gather,Operation) or gather.name != 'linalg.generic' or len(gather.operands)!=2 or len(gather.results)!=1:
        raise ValueError('expected a one-input gather')
    shape = _shape(value,'i8')
    if shape is None or len(shape)!=6 or shape[1:4]!=(3,3,1):
        raise ValueError('expected Cin x 3 x 3 x 1 x OH x OW gather')
    cin,_,_,_,oh,ow=shape
    inp=_shape(gather.operands[0],'i8')
    if inp is None or len(inp)!=4 or inp[:2]!=(1,cin):
        raise ValueError('expected padded NCHW input')
    if (dims.k,dims.n)!=(cin*9,oh*ow):
        raise ValueError('matmul shape disagrees with gather linearization')
    body=gather.regions[0].blocks[0]
    if len(body.args)!=2 or len(list(body.ops))!=1 or body.last_op.name!='linalg.yield' or list(body.last_op.operands)!=[body.args[0]]:
        raise ValueError('gather must copy its input exactly')
    iters=gather.properties['iterator_types'].data
    if len(iters)!=6 or any(x.data.value!='parallel' for x in iters):
        raise ValueError('gather must have six parallel iterators')
    maps=gather.properties['indexing_maps'].data
    d=[AffineExpr.dimension(i) for i in range(6)]
    stride=None
    for candidate in (1,2):
        expected=AffineMap(6,0,(d[3],d[0],d[4]*candidate+d[1],d[5]*candidate+d[2]))
        if len(maps)==2 and maps[0].data==expected and maps[1].data==AffineMap.identity(6):
            stride=candidate
    if stride is None:
        raise ValueError('gather affine map is not supported 3x3 convolution')
    s=ConvShape(inp[2]-2,inp[3]-2,cin,dims.m,stride,wide_b=True,explicit_halo=True)
    s.validate()
    if (s.oh,s.ow)!=(oh,ow):
        raise ValueError('input halo/output geometry disagrees')
    return DirectConvBinding(op,gather,gather.operands[0],s)


def _transpose(value, permutation):
    shape=value.type.get_shape();dtype=value.type.get_element_type();rank=len(shape)
    outtype=TensorType(dtype,[shape[i] for i in permutation])
    empty=tensor.EmptyOp([],outtype)
    block=Block(arg_types=[dtype,dtype]);block.add_op(YieldOp(block.args[0]))
    # Loop order is destination order; each source axis reads its inverse permutation.
    inv=[permutation.index(i) for i in range(rank)]
    op=GenericOp([value],[empty.tensor],Region(block),
        [AffineMapAttr(AffineMap(rank,0,tuple(AffineExpr.dimension(i) for i in inv))),AffineMapAttr(AffineMap.identity(rank))],
        [IteratorTypeAttr.parallel()]*rank,[outtype])
    return [empty,op],op.results[0]


def rewrite(binding, symbol='gemmini_direct_conv_boundary'):
    """Replace the selected matmul, retaining downstream result type and halo.

    Returns declaration and source operation names. Dead gather/reshape producers
    are left for canonical DCE so shared producers remain correct.
    """
    op=binding.contraction;s=binding.shape;module=op
    while module.parent_op() is not None: module=module.parent_op()
    if any(x.name=='func.func' and x.sym_name.data==symbol for x in module.body.block.ops):
        raise ValueError('callee symbol already exists')
    before=[]
    ops,activation=_transpose(binding.input,[0,2,3,1]);before+=ops
    if binding.orientation == 'channels_first':
        reassoc=ArrayAttr([ArrayAttr([IntegerAttr(0,i64)]),ArrayAttr([IntegerAttr(i,i64) for i in (1,2,3)])])
        wt=TensorType(i8,[s.cout,s.cin,3,3])
        expand=tensor.ExpandShapeOp(binding.weight,[],reassoc,[s.cout,s.cin,3,3],wt)
        before.append(expand)
        ops,weight=_transpose(expand.result,[2,3,1,0]);before+=ops
    else:
        reassoc=ArrayAttr([ArrayAttr([IntegerAttr(i,i64) for i in (0,1,2)]),ArrayAttr([IntegerAttr(3,i64)])])
        wt=TensorType(i8,[3,3,s.cin,s.cout])
        expand=tensor.ExpandShapeOp(binding.weight,[],reassoc,[3,3,s.cin,s.cout],wt)
        before.append(expand);weight=expand.result
    ct=TensorType(i32,[s.oh*s.ow,s.cout]);empty=tensor.EmptyOp([],ct);before.append(empty)
    call=func.CallOp(symbol,[activation,weight,empty.tensor],[ct]);before.append(call)
    result=call.results[0]
    if binding.orientation == "channels_first":
        ops,result=_transpose(result,[1,0]);before+=ops
    op.parent.insert_ops_before(before,op)
    op.results[0].replace_all_uses_with(result);op.parent.erase_op(op)
    attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in ['read','read','write']])
    declaration=func.FuncOp(symbol,([activation.type,weight.type,ct],[ct]),Region(),visibility='private',arg_attrs=attrs)
    declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
    module.body.block.add_op(declaration)
    module.verify()
    return declaration


def capture(binding):
    """Outline the proven gather+matmul with only weight and activation inputs."""
    from xdsl.dialects.builtin import ModuleOp
    original=binding.contraction
    block=Block(arg_types=[binding.weight.type,binding.input.type])
    mapping={binding.weight:block.args[0],binding.input:block.args[1]}
    def clone(value):
        if value in mapping: return mapping[value]
        producer=value.owner
        if not isinstance(producer,Operation): raise ValueError('uncaptured block argument')
        for operand in producer.operands: clone(operand)
        copied=producer.clone(value_mapper=mapping);block.add_op(copied)
        return mapping[value]
    out=clone(original.results[0]);block.add_op(func.ReturnOp(out))
    module=ModuleOp([func.FuncOp('captured_conv',([x.type for x in block.args],[out.type]),Region(block))])
    module.body.block.first_op.attributes["llvm.emit_c_interface"] = UnitAttr()
    module.verify()
    return module


def emit_c_adapter(s, symbol='gemmini_direct_conv_boundary', kernel_symbol='gemmini_golden_conv'):
    """C-interface adapter consumes bufferized descriptors; no layout inference."""
    def checks(name,shape):
        conditions=[f'{name}->offset < 0',f'!{name}->aligned']
        for i,extent in enumerate(shape):
            conditions += [f'{name}->sizes[{i}] != {extent}',f'{name}->strides[{i}] != {prod(shape[i+1:])}']
        return ' || '.join(conditions)
    text = '''#include <stdint.h>
#include <stddef.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
extern void gemmini_golden_conv(int8_t*,int8_t*,int32_t*);
''' + f'''void _mlir_ciface_{symbol}(memref2 *r,memref4 *a,memref4 *b,memref2 *c) {{
 if ({checks('a',[1,s.h+2,s.w+2,s.cin])} || {checks('b',[3,3,s.cin,s.cout])} || {checks('c',[s.oh*s.ow,s.cout])}) __builtin_trap();
 gemmini_golden_conv((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int32_t*)c->aligned+c->offset);
 *r=*c;
}}
'''
    text=text.replace("gemmini_golden_conv",kernel_symbol)
    if s.output_dtype == "i8":
        text=text.replace("int32_t*","int8_t*")
    return text


def serialize(module, declarations):
    """Keep external argument properties through the bodyless function parser."""
    import io
    from xdsl.printer import Printer
    if isinstance(declarations,Operation): declarations=[declarations]
    declarations=list(declarations)
    # Composition may already contain external calls from an earlier binder.
    # Preserve their properties as well as the newly introduced declarations.
    for op in module.walk():
        if op.name=='func.func' and not op.body.blocks and op.properties.get('arg_attrs') is not None and op not in declarations:
            declarations.append(op)
    text=str(module)
    for declaration in declarations:
        old=str(declaration)
        if text.count(old)!=1: raise ValueError('expected one exact generated declaration')
        stream=io.StringIO()
        Printer(stream=stream,print_generic_format=True).print_op(declaration)
        text=text.replace(old,stream.getvalue(),1)
    return text


def _slice_coordinates(value):
    """Compose static rank-preserving slices into base coordinates."""
    offsets=[0]*4;strides=[1]*4
    while isinstance(value.owner,Operation) and value.owner.name=='tensor.extract_slice':
        op=value.owner
        if len(op.operands)!=1 or len(value.type.get_shape())!=4:
            raise ValueError('dynamic or rank-reducing slice is unsupported')
        arrays=[list(op.properties[x].get_values()) for x in ('static_offsets','static_sizes','static_strides')]
        off,size,step=arrays;source=op.operands[0]
        if any(len(x)!=4 for x in arrays) or tuple(size)!=value.type.get_shape():
            raise ValueError('slice geometry is not rank preserving')
        for i in range(4):
            if min(off[i],size[i]-1,step[i]-1)<0 or off[i]+(size[i]-1)*step[i]>=source.type.get_shape()[i]:
                raise ValueError('slice bounds are not statically proven')
            offsets[i]=off[i]+offsets[i]*step[i];strides[i]*=step[i]
        value=source
    return value,offsets,strides


def _match_concat(op):
    dims=match_integer_gemm(op)
    if dims is None or dims.batch!=1:raise ValueError('expected signed integer matmul')
    concat=op.operands[0].owner
    if not isinstance(concat,Operation) or concat.name!='tensor.concat' or len(concat.operands)!=9 or concat.dim.value.data!=1:
        raise ValueError('expected nine ordered 3x3 tap panels concatenated along K')
    base=None;geometry=None
    for tap,panel in enumerate(concat.operands):
        value=_flat_reshape(panel,4);trans=value.owner
        if not isinstance(trans,Operation) or trans.name!='linalg.transpose' or list(trans.permutation.get_values())!=[0,2,3,1]:
            raise ValueError('tap panel must transpose NCHW to NHWC')
        ty=_shape(trans.operands[0],'i8')
        if ty is None or len(ty)!=4 or ty[0]!=1:raise ValueError('tap must have one NCHW batch')
        source,off,step=_slice_coordinates(trans.operands[0])
        if off!=[0,0,tap//3,tap%3] or step[:2]!=[1,1] or step[2]!=step[3] or step[2] not in (1,2):
            raise ValueError('tap order or stride does not describe 3x3 convolution')
        if base is None:base=source;geometry=(ty,step[2])
        elif source is not base or geometry!=(ty,step[2]):raise ValueError('tap paths disagree on input or geometry')
        if _shape(panel,'i8')!=(ty[2]*ty[3],ty[1]):raise ValueError('tap flattening dimensions disagree')
    ty,stride=geometry;inp=_shape(base,'i8')
    if inp is None or inp[:2]!=ty[:2]:raise ValueError('base input batch/channel geometry differs')
    s=ConvShape(inp[2]-2,inp[3]-2,ty[1],dims.n,stride,wide_b=True,explicit_halo=True)
    s.validate()
    if (s.oh,s.ow)!=(ty[2],ty[3]) or (dims.m,dims.k)!=(s.oh*s.ow,s.cin*9):raise ValueError('concat contraction geometry disagrees')
    return DirectConvBinding(op,concat,base,s,'spatial_first')


def match(op):
    """Accept either exact upstream im2col family; provenance is not evidence."""
    try:return _match_gather(op)
    except ValueError as first:
        try:return _match_concat(op)
        except ValueError as second:raise ValueError(f'{first}; {second}') from second


def emit_scalar_oracle(s, kernel_symbol):
    """Native functional stand-in for whole-graph checks; never device performance."""
    if not s.explicit_halo or s.output_dtype!='i32':
        raise ValueError('native boundary oracle requires explicit halo and i32 output')
    return f'''
void {kernel_symbol}(int8_t *a,int8_t *b,int32_t *c) {{
 for(int y=0;y<{s.oh};y++) for(int x=0;x<{s.ow};x++) for(int n=0;n<{s.cout};n++) {{
   uint32_t acc=0;
   for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) for(int ci=0;ci<{s.cin};ci++)
     acc+=(uint32_t)((int32_t)a[(((y*{s.stride}+ky)*{s.w+2}+x*{s.stride}+kx)*{s.cin})+ci]*(int32_t)b[((ky*3+kx)*{s.cin}+ci)*{s.cout}+n]);
   ((uint32_t*)c)[(y*{s.ow}+x)*{s.cout}+n]=acc;
 }}
}}
'''
