"""Structural binding for captured 49-tap RGB stems, retaining host epilogues."""
from dataclasses import dataclass
from xdsl.ir import Operation,Region
from xdsl.dialects import func,tensor
from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,StringAttr,UnitAttr,TensorType,i32
from .contraction_patterns import match_integer_gemm,_shape
from .direct_conv_binding import _flat_reshape,_slice_coordinates,_transpose
from .golden_stem import StemShape

@dataclass
class StemBinding:
    contraction:Operation
    input:object
    shape:StemShape
    @property
    def weight(self):return self.contraction.operands[1]


def match(op):
    dims=match_integer_gemm(op)
    if dims is None or dims.batch!=1:raise ValueError('stem requires signed from-zero integer GEMM')
    concat=op.operands[0].owner
    if not isinstance(concat,Operation) or concat.name!='tensor.concat' or len(concat.operands)!=49 or concat.dim.value.data!=1:raise ValueError('stem requires49 ordered tap panels concatenated on K')
    base=None;geometry=None
    for tap,panel in enumerate(concat.operands):
        value=_flat_reshape(panel,4);trans=value.owner
        if not isinstance(trans,Operation) or trans.name!='linalg.transpose' or list(trans.permutation.get_values())!=[0,2,3,1]:raise ValueError('stem tap must transpose NCHW to NHWC')
        ty=_shape(trans.operands[0],'i8')
        if ty is None or len(ty)!=4 or ty[:2]!=(1,3):raise ValueError('stem requires one RGB NCHW batch')
        source,off,step=_slice_coordinates(trans.operands[0])
        if off!=[0,0,tap//7,tap%7] or step!=[1,1,2,2]:raise ValueError('stem tap order or stride mismatch')
        if base is None:base=source;geometry=ty
        elif base is not source or geometry!=ty:raise ValueError('stem tap paths disagree')
        if _shape(panel,'i8')!=(ty[2]*ty[3],3):raise ValueError('stem tap flattening mismatch')
    inp=_shape(base,'i8')
    if inp is None or inp[:2]!=(1,3):raise ValueError('stem base geometry mismatch')
    s=StemShape(inp[2]-6,inp[3]-6,dims.n);s.validate()
    if (s.oh,s.ow)!=geometry[2:] or (dims.m,dims.k)!=(s.oh*s.ow,147):raise ValueError('stem contraction geometry mismatch')
    return StemBinding(op,base,s)


def rewrite(binding,symbol='gemmini_packed_stem'):
    op=binding.contraction;s=binding.shape;module=op
    while module.parent_op() is not None:module=module.parent_op()
    operations,activation=_transpose(binding.input,[0,2,3,1]);ct=TensorType(i32,[s.oh*s.ow,s.cout]);empty=tensor.EmptyOp([],ct);operations.append(empty)
    inputs=[activation,binding.weight,empty.tensor];call=func.CallOp(symbol,inputs,[ct]);operations.append(call)
    op.parent.insert_ops_before(operations,op);op.results[0].replace_all_uses_with(call.results[0]);op.parent.erase_op(op)
    attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in ['read','read','write']]);declaration=func.FuncOp(symbol,([x.type for x in inputs],[ct]),Region(),visibility='private',arg_attrs=attrs);declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration);return declaration


def emit_adapter(s,symbol,kernel):
    def check(name,shape):
        strides=[];stride=1
        for n in reversed(shape):strides.insert(0,stride);stride*=n
        return ' || '.join([f'!{name}->aligned',f'{name}->offset<0']+[f'{name}->sizes[{i}]!={n} || {name}->strides[{i}]!={strides[i]}' for i,n in enumerate(shape)])
    return '''#include <stdint.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
'''+f'''extern void {kernel}(int8_t*,int8_t*,int32_t*);
void _mlir_ciface_{symbol}(memref2*r,memref4*a,memref2*b,memref2*c) {{
 if ({check('a',[1,s.h+6,s.w+6,3])} || {check('b',[147,s.cout])} || {check('c',[s.oh*s.ow,s.cout])}) __builtin_trap();
 {kernel}((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int32_t*)c->aligned+c->offset);*r=*c;
}}
'''


def emit_oracle(s,kernel):
    return f'''void {kernel}(int8_t*a,int8_t*b,int32_t*c) {{
 for(int y=0;y<{s.oh};y++)for(int x=0;x<{s.ow};x++)for(int n=0;n<{s.cout};n++){{
 uint32_t acc=0;for(int ky=0;ky<7;ky++)for(int kx=0;kx<7;kx++)for(int ci=0;ci<3;ci++)
 acc+=(uint32_t)((int32_t)a[((y*2+ky)*{s.w+6}+x*2+kx)*3+ci]*(int32_t)b[((ky*7+kx)*3+ci)*{s.cout}+n]);
 ((uint32_t*)c)[(y*{s.ow}+x)*{s.cout}+n]=acc;}}
}}
'''
