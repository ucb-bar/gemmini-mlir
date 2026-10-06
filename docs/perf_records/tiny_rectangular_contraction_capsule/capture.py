from pathlib import Path
from datetime import datetime,timezone
import re,json,hashlib,subprocess
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower import quant_hoist
work=Path(__file__).resolve().parent/'capture';work.mkdir(exist_ok=False)
original=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-two-products-outline-composed-whole-20261006/whole/host_llvm/expanded.native.ll')
source=original.read_text();old=source
proof={}
for name,sizes in [('qk',(32*8*8,32*8*64,32*64*8)),('pv',(32*8*64,32*8*8,32*8*64))]:
 symbol={'qk':'forward.extracted.118','pv':'forward.extracted.112'}[name]
 pattern=r'^define internal void @'+re.escape(symbol)+r'\(ptr %0, ptr %1, ptr %2\)[^\n]*\{.*?^}'
 match=re.search(pattern,source,re.M|re.S);assert match
 body=match.group();proof[name]={'source_symbol':symbol,'original_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'sizes_f32':sizes,'calls':len(re.findall(r'call void @'+re.escape(symbol)+r'\(',source))}
 changed=body.replace('define internal void @'+symbol,'define void @capture_source_'+name)
 source=source[:match.start()]+changed+source[match.end():]
 source=source.replace('call void @'+symbol+'(', 'call void @tap_'+name+'(')
 source+='\ndeclare void @tap_'+name+'(ptr,ptr,ptr)\n'
(work/'tap.native.ll').write_text(source)
c='#include <stdio.h>\n#include <stdint.h>\n#include <stdlib.h>\nstatic void save(const char*n,const void*p,size_t s){FILE*f=fopen(n,"wb");if(!f||fwrite(p,1,s,f)!=s||fclose(f))abort();}\n'
for name in ['qk','pv']:
 z,a,b=proof[name]['sizes_f32'];prefix=str(work/name)
 c+=f'extern void capture_source_{name}(float*,const float*,const float*);static unsigned {name}_count;void tap_{name}(float*c,const float*a,const float*b){{unsigned take={name}_count++==0;if(take){{save("{prefix}_a.bin",a,{a*4});save("{prefix}_b.bin",b,{b*4});save("{prefix}_initial.bin",c,{z*4});}}capture_source_{name}(c,a,b);if(take)save("{prefix}_expected.bin",c,{z*4});}}\n'
(work/'tap.c').write_text(c)
base=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
build=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
bundle=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
clang=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
runtime=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/abi/mlir_runtime.c')
compile=[str(clang),'-O2','-fPIC','-c',str(work/'tap.native.ll'),'-o',str(work/'model.o')]
link=['cc','-O3','-march=native','-fPIC','-shared',str(work/'model.o'),str(work/'tap.c'),str(base/'reference.c'),str(base/'device/device_catalog_shim.c'),str(runtime),'-lm','-o',str(work/'model.so')]
subprocess.run(compile,check=True,capture_output=True);subprocess.run(link,check=True,capture_output=True)
args=resolve_forward_args(bundle);values=quant_hoist.read_values(build);args.extend(np.ascontiguousarray(values[e.key]) for e in quant_hoist.read_plan(build))
gold=np.load(bundle/'golden.npy');out=np.zeros(gold.shape,dtype=np.float32)
HostModel.load(str(work/'model.so'))([(a.ctypes.data,a.shape) for a in args]+[(out.ctypes.data,out.shape)])
rawsha=hashlib.sha256(out.astype('<f4').tobytes()).hexdigest();assert rawsha=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3' and np.allclose(out,gold,atol=.03125,rtol=.02)
np.save(work/'output.npy',out)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pins={str(path):sha(path) for path in [original,work/'tap.native.ll',work/'tap.c',work/'model.o',work/'model.so',runtime,base/'reference.c',base/'device/device_catalog_shim.c',build/'quant_hoist_args.json',bundle/'golden.npy',*work.glob('*.bin')]}
record={'schema':'original1989_source_exact_host_contraction_readonly_tap_v1','status':'pass','recorded_utc':datetime.now(timezone.utc).isoformat(),'scope':'Fresh native compiler build from immutable qualified1989 nativeLLVM; only two source-exact arithmetic helper exports and readonly before/after taps, all22calls per helper preserved. Original inputs/weights/source ABI/runtime/device standins/output gate unchanged. No performance claim. Source helper names select experimental taps only.','compile_argv':compile,'link_argv':link,'helpers':proof,'pins':pins,'original256000_words':True,'raw_output_sha256':rawsha,'torch_gate':True,'torch_atol':.03125,'torch_rtol':.02,'token_usage_available':False}
(work/'validation.json').write_text(json.dumps(record,indent=2)+'\n');print('SOURCE_EXACT_QK_PV_CAPTURE_PASS',rawsha,flush=True)
