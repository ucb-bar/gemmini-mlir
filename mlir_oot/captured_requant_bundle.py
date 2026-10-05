"""Source/weight-bound exact unary requant fusion, preserving direct convolution.

Only immutable zero-bias captured channels are currently specialized here.
The general scalar proof supports nonzero bias, but this bundle refuses it until
its descriptor ABI explicitly binds the synthesized integer preload table.
"""
import argparse
from dataclasses import asdict,replace
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
from xdsl.dialects import func,tensor
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.builtin import ArrayAttr,DictionaryAttr,IntegerAttr,StringAttr,TensorType,UnitAttr,i8,i64
from xdsl.ir import Region
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .captured_requant import inspect_chain
from .golden_requant import synthesize_bias
from .golden_contraction_upstream import choose_shape
from .golden_gemm import GoldenGemm
from .conv_schedule import select_kernel
from .golden_device_compile import compile_module
from .direct_conv_binding import match as match_conv,_transpose,serialize,emit_c_adapter
from .no_fsm_audit import audit_elf


def dense_adapter(s,symbol,kernel):
    def checks(name,shape):
        return ' || '.join([f'!{name}->aligned',f'{name}->offset<0']+[f'{name}->sizes[{i}]!={size} || {name}->strides[{i}]!={shape[1] if i==0 else 1}' for i,size in enumerate(shape)])
    return '''#include <stdint.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
'''+f'''extern void {kernel}(int8_t*,int8_t*,int8_t*);
void _mlir_ciface_{symbol}(memref2 *r,memref2 *a,memref2 *b,memref2 *c) {{
 if ({checks('a',[s.m,s.k])} || {checks('b',[s.k,s.n])} || {checks('c',[s.m,s.n])}) __builtin_trap();
 {kernel}((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int8_t*)c->aligned+c->offset);*r=*c;
}}
'''


def scalar_oracle(s,kernel,direct):
    scale=s.scale.hex()+'f';low=0 if s.relu else -128
    if direct:
        loops=f'''for(int y=0;y<{s.oh};y++) for(int x=0;x<{s.ow};x++) for(int n=0;n<{s.cout};n++) {{
 uint32_t acc=0;
 for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) for(int ci=0;ci<{s.cin};ci++)
 acc+=(uint32_t)((int32_t)a[((y*{s.stride}+ky)*{s.w+2}+x*{s.stride}+kx)*{s.cin}+ci]*(int32_t)b[((ky*3+kx)*{s.cin}+ci)*{s.cout}+n]);
 int index=(y*{s.ow}+x)*{s.cout}+n;'''
    else:
        loops=f'''for(int m=0;m<{s.m};m++) for(int n=0;n<{s.n};n++) {{
 uint32_t acc=0;for(int k=0;k<{s.k};k++)acc+=(uint32_t)((int32_t)a[m*{s.k}+k]*(int32_t)b[k*{s.n}+n]);
 int index=m*{s.n}+n;'''
    return f'''#include <math.h>
void {kernel}(int8_t*a,int8_t*b,int8_t*c) {{
{loops}
 float value=nearbyintf((float)(int32_t)acc*{scale});
 if(value<{low})value={low};if(value>127)value=127;c[index]=(int8_t)value;
 }}
}}
'''


def _replay_layouts(value,layouts):
    operations=[]
    for original in layouts:
        ty=TensorType(i8,original.results[0].type.get_shape())
        if original.name=='tensor.collapse_shape':
            op=tensor.CollapseShapeOp(operands=[value],result_types=[ty],properties={'reassociation':original.properties['reassociation']})
        elif original.name=='tensor.expand_shape':
            op=tensor.ExpandShapeOp(value,[],original.properties['reassociation'],list(ty.get_shape()),ty)
        else:
            empty=tensor.EmptyOp([],ty);operations.append(empty)
            op=TransposeOp(value,empty.tensor,original.permutation,ty)
        operations.append(op);value=op.results[0]
    return operations,value


def rewrite_path(op,chain,symbol,direct):
    dims=chain['dimensions'];operations=[]
    if direct:
        if direct.orientation!='spatial_first':raise ValueError('fused captured direct path needs spatial-first contraction')
        ops,activation=_transpose(direct.input,[0,2,3,1]);operations+=ops
        s=direct.shape;wt=TensorType(i8,[3,3,s.cin,s.cout])
        reassoc=ArrayAttr([ArrayAttr([IntegerAttr(i,i64) for i in (0,1,2)]),ArrayAttr([IntegerAttr(3,i64)])])
        weight=tensor.ExpandShapeOp(direct.weight,[],reassoc,list(wt.get_shape()),wt);operations.append(weight)
        inputs=[activation,weight.result]
    else:inputs=list(op.operands[:2])
    ct=TensorType(i8,[dims.m,dims.n]);empty=tensor.EmptyOp([],ct);operations.append(empty)
    call=func.CallOp(symbol,[*inputs,empty.tensor],[ct]);operations.append(call)
    views,result=_replay_layouts(call.results[0],chain['layouts']);operations+=views
    if result.type!=chain['quantize'].results[0].type:raise ValueError('replayed output layout differs')
    module=op
    while module.parent_op() is not None:module=module.parent_op()
    op.parent.insert_ops_before(operations,op)
    chain['quantize'].results[0].replace_all_uses_with(result)
    for old in reversed(chain['operations']):old.parent.erase_op(old)
    op.parent.erase_op(op)
    attrs=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in ['read','read','write']])
    declaration=func.FuncOp(symbol,([x.type for x in inputs]+[ct],[ct]),Region(),visibility='private',arg_attrs=attrs)
    declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration)
    return declaration


def build(capture:Path,llvm_bin:Path,output:Path,*,flat_spatial=False):
    from merlin.runtime.captured_constants import verify_capture_constant
    output.mkdir(parents=True,exist_ok=False)
    source=(capture/'model.mlir').read_text();module=parse_module(source)
    source_sha=hashlib.sha256(source.encode()).hexdigest();pins=json.loads((capture/'capture_receipt.json').read_text())['artifacts']
    if source_sha!=pins['model.mlir']['sha256']:raise ValueError('capture source hash changed')
    objects=[];declarations=[];native=[];routes=[];refused=[]
    for op in list(module.walk()):
        dims=match_integer_gemm(op)
        if dims is None:continue
        rid=getattr(op.attributes.get('prov.region_id'),'data','')
        try:chain=inspect_chain(op)
        except ValueError as e:refused.append(dict(region=rid,reason=str(e)));continue
        constant=verify_capture_constant(manifest_path=capture/'weights.safetensors.manifest.json',manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],safetensors_path=capture/'weights.safetensors',safetensors_sha256=pins['weights.safetensors']['sha256'],entry_argument_index=chain['bias'].index,source_shape=[dims.n],source_dtype='f32',max_payload_bytes=dims.n*4)
        biases=np.frombuffer(constant.logical_payload,dtype='<f4');bound=dims.k*16384
        proof=synthesize_bias(chain['scales'],biases,chain['reciprocal'],max(-(1<<31),-bound),min((1<<31)-1,bound),chain['relu'])
        if proof['accepted_channels']!=dims.n:
            refused.append(dict(region=rid,reason='float transition proof refused',accepted_channels=proof['accepted_channels']));continue
        if np.any(biases!=0) or any(x!=0 for x in proof['integer_bias']):
            refused.append(dict(region=rid,reason='nonzero bias requires explicit integer table ABI'));continue
        try:direct=match_conv(op)
        except ValueError:direct=None
        base=direct.shape if direct else choose_shape(dims)
        schedule=replace(base,output_dtype='i8',scale=proof['scale'],relu=chain['relu'])
        if not direct:schedule=replace(schedule,wide_store=True)
        symbol=f'gemmini_exact_requant_{len(routes)}';kernel=symbol+'_kernel';work=output/symbol
        bias_index=chain['bias'].index
        declaration=rewrite_path(op,chain,symbol,direct);declarations.append(declaration)
        if direct:
            generator,schedule_kind=select_kernel(schedule,flat_spatial=flat_spatial);schedule=generator.conv
        else:
            generator,schedule_kind=GoldenGemm(schedule),'dense_gemm'
        device=generator.build();device.body.block.first_op.properties['sym_name']=StringAttr(kernel)
        compilation=compile_module(device,llvm_bin,work)
        adapter=emit_c_adapter(schedule,symbol,kernel) if direct else dense_adapter(schedule,symbol,kernel)
        (work/'adapter.c').write_text(adapter)
        subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-O2','-ffreestanding','-fno-builtin','-c',str(work/'adapter.c'),'-o',str(work/'adapter.o')],check=True,capture_output=True)
        objects.extend([work/'kernel.o',work/'adapter.o']);native.append(adapter+scalar_oracle(schedule,kernel,bool(direct)))
        routes.append(dict(region=rid,symbol=symbol,kernel=kernel,direct_conv=bool(direct),schedule_kind=schedule_kind,schedule=asdict(schedule),bias_argument=bias_index,bias_payload_sha256=constant.payload_sha256,proof=proof,compilation=compilation))
    if not routes:raise ValueError('no exactly provable captured epilogues')
    module.verify();printed=serialize(module,declarations);parse_module(printed).verify();(output/'rewritten.mlir').write_text(printed)
    (output/'native_oracle.c').write_text('\n'.join(native))
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():linker=Path(shutil.which('ld.lld'))
    linked=output/'requant.o';subprocess.run([str(linker),'-r',*[str(p) for p in objects],'-o',str(linked)],check=True,capture_output=True)
    audit=audit_elf(linked.read_bytes())
    if audit['status']!='pass':raise ValueError('forbidden device instruction')
    result=dict(schema='gemmini_exact_captured_requant_bundle_v1',source_sha256=source_sha,weights_sha256=pins['weights.safetensors']['sha256'],manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],rewritten_sha256=hashlib.sha256(printed.encode()).hexdigest(),object_sha256=hashlib.sha256(linked.read_bytes()).hexdigest(),routes=routes,refused=refused,nofsm_audit=audit,scope='Specializes only hash-bound captured zero-bias parameters. Runtime input images remain variable. Other weights/bias blobs require recompilation. Whole-model accuracy still must be checked.')
    (output/'requant.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--flat-spatial',action='store_true');a=p.parse_args()
    result=build(a.capture,a.llvm_bin,a.output,flat_spatial=a.flat_spatial);print(json.dumps(dict(routes=len(result['routes']),direct=sum(x['direct_conv'] for x in result['routes']),refused=len(result['refused']),object_sha256=result['object_sha256']),indent=2))

if __name__=='__main__':main()
