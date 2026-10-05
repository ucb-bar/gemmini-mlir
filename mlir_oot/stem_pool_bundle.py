"""Hash-bound exact captured stem+float-epilogue+maxpool specialization."""
import json,subprocess,shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from xdsl.dialects.builtin import StringAttr
from .fused_mixed_catalog import sha
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .stem_binding import match,emit_adapter
from .stem_pool_binding import rewrite
from .captured_requant import inspect_chain
from .golden_requant import synthesize_bias
from .golden_stem_pool import StemPoolShape,GoldenStemPool,command_counts
from .golden_device_compile import compile_module
from .direct_conv_binding import serialize
from .no_fsm_audit import audit_elf


def emit_oracle(s,kernel):
    return f'''#include <math.h>
void {kernel}(int8_t*a,int8_t*b,int8_t*c){{
 for(int py=0;py<{s.ph};py++)for(int px=0;px<{s.pw};px++)for(int n=0;n<{s.cout};n++){{
 int32_t best=0;for(int dy=0;dy<3;dy++)for(int dx=0;dx<3;dx++){{
 int y=py*2+dy-1,x=px*2+dx-1;if(y<0||x<0||y>={s.oh}||x>={s.ow})continue;
 int32_t acc=0;for(int ky=0;ky<7;ky++)for(int kx=0;kx<7;kx++)for(int ci=0;ci<3;ci++)
 acc+=(int32_t)a[((y*2+ky)*{s.w+6}+x*2+kx)*3+ci]*(int32_t)b[((ky*7+kx)*3+ci)*{s.cout}+n];
 if(acc>best)best=acc;}}
 float v=nearbyintf((float)best*{s.scale.hex()}f);if(v>127)v=127;c[(py*{s.pw}+px)*{s.cout}+n]=(int8_t)v;
 }}}}
'''


def build(capture,llvm_bin,output):
    from merlin.runtime.captured_constants import verify_capture_constant
    capture,llvm_bin,output=map(Path,(capture,llvm_bin,output));output.mkdir(parents=True,exist_ok=False)
    source=capture/'model.mlir';pins=json.loads((capture/'capture_receipt.json').read_text())['artifacts']
    if sha(source)!=pins['model.mlir']['sha256']:raise ValueError('captured stem source identity changed')
    module=parse_module(source.read_text());selected=[]
    for op in module.walk():
        if match_integer_gemm(op):
            try:selected.append(match(op))
            except ValueError:pass
    if len(selected)!=1:raise ValueError('exactly one structural stem is required')
    binding=selected[0];chain=inspect_chain(binding.contraction,allow_maxpool=True)
    if not chain['pool']:raise ValueError('stem has no exactly matched pool')
    from .stem_pool_binding import validate_layout
    validate_layout(binding,chain)
    constant=verify_capture_constant(manifest_path=capture/'weights.safetensors.manifest.json',manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],safetensors_path=capture/'weights.safetensors',safetensors_sha256=pins['weights.safetensors']['sha256'],entry_argument_index=chain['bias'].index,source_shape=[binding.shape.cout],source_dtype='f32',max_payload_bytes=binding.shape.cout*4)
    biases=np.frombuffer(constant.logical_payload,dtype='<f4');proof=synthesize_bias(chain['scales'],biases,chain['reciprocal'],-147*16384,147*16384,True)
    if proof['accepted_channels']!=binding.shape.cout or np.any(biases!=0) or any(x!=0 for x in proof['integer_bias']):raise ValueError('stem requires completely proved zero bias thresholds')
    shape=StemPoolShape(binding.shape.h,binding.shape.w,binding.shape.cout,proof['scale']);symbol='gemmini_exact_stem_pool';kernel=symbol+'_kernel';declaration=rewrite(binding,chain,shape,symbol)
    device=GoldenStemPool(shape).build();device.body.block.first_op.properties['sym_name']=StringAttr(kernel);compilation=compile_module(device,llvm_bin,output/'device')
    adapter=emit_adapter(SimpleNamespace(h=shape.h,w=shape.w,cout=shape.cout,oh=shape.ph,ow=shape.pw),symbol,kernel).replace('int32_t*','int8_t*');(output/'adapter.c').write_text(adapter);(output/'native_oracle.c').write_text(adapter+emit_oracle(shape,kernel))
    subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-c',str(output/'adapter.c'),'-o',str(output/'adapter.o')],check=True,capture_output=True)
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():linker=Path(shutil.which('ld.lld'))
    obj=output/'stem_pool.o';subprocess.run([str(linker),'-r',str(output/'device/kernel.o'),str(output/'adapter.o'),'-o',str(obj)],check=True,capture_output=True);audit=audit_elf(obj.read_bytes())
    if audit['status']!='pass':raise ValueError('pool object no-FSM audit failed')
    module.verify();printed=serialize(module,[declaration]);parse_module(printed).verify();(output/'rewritten.mlir').write_text(printed)
    result={'schema':'gemmini_exact_stem_pool_bundle_v1','source_sha256':sha(source),'weights_sha256':pins['weights.safetensors']['sha256'],'manifest_sha256':pins['weights.safetensors.manifest.json']['sha256'],'bias_payload_sha256':constant.payload_sha256,'rewritten_sha256':sha(output/'rewritten.mlir'),'object_sha256':sha(obj),'symbol':symbol,'shape':asdict(shape),'scalar_transition_proof':proof,'pool_proof':chain['pool'],'commands':command_counts(shape),'compilation':compilation,'nofsm_audit':audit}
    (output/'stem_pool.json').write_text(json.dumps(result,indent=2)+'\n');return result


def apply_capture(capture,bundle):
    capture,bundle=map(Path,(capture,bundle));manifest=json.loads((bundle/'stem_pool.json').read_text())
    if sha(capture/'model.mlir')!=manifest['source_sha256'] or sha(bundle/'rewritten.mlir')!=manifest['rewritten_sha256']:raise ValueError('pooled capture identity mismatch')
    receipt=json.loads((capture/'capture_receipt.json').read_text());receipt.setdefault('transforms',[]).append({'source_sha256':manifest['source_sha256'],'stem_pool_manifest_sha256':sha(bundle/'stem_pool.json')})
    shutil.copyfile(bundle/'rewritten.mlir',capture/'model.mlir');receipt['artifacts']['model.mlir']={'bytes':(capture/'model.mlir').stat().st_size,'sha256':manifest['rewritten_sha256']};(capture/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
