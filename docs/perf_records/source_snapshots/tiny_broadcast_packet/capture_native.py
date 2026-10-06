"""Read-only tap of immutable original1880 first gate input/output tensors."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,ctypes
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower import quant_hoist

W=Path(__file__).resolve().parent
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
BASE=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
BUNDLE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
FROZEN=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005')
CLANG=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
source=B/'host_llvm/model.native.ll'
old_gate=json.loads((BASE/'validation.json').read_text())
assert old_gate['original_compiled_bits_exact'] and old_gate['llvm_sha256']==sha(source)
case=W/'capture';case.mkdir(exist_ok=False)
needle='  %6039 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5895, 1\n'
text=source.read_text();assert text.count(needle)==1
text=text.replace(needle,needle+'  call void @merlin_root_gate_tap(ptr %5876, ptr %5877, ptr %273, ptr %283, ptr %6039)\n')
text+='\ndeclare void @merlin_root_gate_tap(ptr, ptr, ptr, ptr, ptr)\n'
selected=case/'model.native.ll';selected.write_text(text)
tap=case/'tap.c'
tap.write_text('''#include <stdint.h>
#include <string.h>
static int32_t a[45056],b[45056];
static float sa[5632],sb[5632];
static int8_t expected[45056];
static int calls;
void merlin_root_gate_tap(const void *pa,const void *pb,const void *psa,const void *psb,const void *pe){
 memcpy(a,pa,sizeof(a));memcpy(b,pb,sizeof(b));memcpy(sa,psa,sizeof(sa));memcpy(sb,psb,sizeof(sb));memcpy(expected,pe,sizeof(expected));++calls;
}
int merlin_root_gate_calls(void){return calls;}
void *merlin_root_gate_tensor(unsigned i){void *values[]={a,sa,b,sb,expected};return i<5?values[i]:0;}
''')
argv=[str(CLANG),'-O2','-fPIC','-c',str(selected),'-o',str(case/'model.o')]
bound=[BASE/'reference.c',BASE/'device/device_catalog_shim.c',FROZEN/'merlin/runtime/abi/mlir_runtime.c',tap]
link=['cc','-O3','-march=native','-fPIC','-shared',str(case/'model.o'),*map(str,bound),'-lm','-o',str(case/'model.so')]
record={'schema':'original_tiny1880_readonly_gate_tap_v1','started_utc':datetime.now(timezone.utc).isoformat(),
 'source_llvm_sha256':sha(source),'selected_llvm_sha256':sha(selected),'compile_argv':argv,'link_argv':link,
 'boundary_pins':{str(p):sha(p) for p in bound},'source_capsule_sha256':sha(W/'source_capsule.mlir'),
 'modification':'One read-only five-pointer tap after original first gate result before next dispatch. Original source arithmetic, inputs, weights, runtime and native integer device standins unchanged.',
 'accuracy_atol':.03125,'accuracy_rtol':.02}
receipt=case/'receipt.json';receipt.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(argv,check=True);subprocess.run(link,check=True)
print('CAPTURE_NATIVE_COMPILED',flush=True)
args=resolve_forward_args(BUNDLE);plan=quant_hoist.read_plan(B)
if plan:
 values=quant_hoist.read_values(B);args.extend(np.ascontiguousarray(values[e.key]) for e in plan)
golden=np.load(BUNDLE/'golden.npy');out=np.zeros_like(golden)
HostModel.load(str(case/'model.so'))([(a.ctypes.data,a.shape) for a in args]+[(out.ctypes.data,out.shape)])
raw=hashlib.sha256(out.astype('<f4').tobytes()).hexdigest()
assert np.array_equal(out.view('u4'),np.load(BASE/'output.npy').view('u4'))
assert np.allclose(out,golden,atol=.03125,rtol=.02)
assert raw=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
lib=ctypes.CDLL(str(case/'model.so'));lib.merlin_root_gate_calls.restype=ctypes.c_int
assert lib.merlin_root_gate_calls()==1
lib.merlin_root_gate_tensor.argtypes=[ctypes.c_uint];lib.merlin_root_gate_tensor.restype=ctypes.c_void_p
types=[np.dtype('i4'),np.dtype('f4'),np.dtype('i4'),np.dtype('f4'),np.dtype('i1')]
shapes=[(1,8,5632),(5632,),(1,8,5632),(5632,),(1,8,5632)]
names=['a','scale_a','b','scale_b','expected'];pins={}
for i,(name,dtype,shape) in enumerate(zip(names,types,shapes)):
 size=int(np.prod(shape))*dtype.itemsize
 value=np.frombuffer(ctypes.string_at(lib.merlin_root_gate_tensor(i),size),dtype=dtype).copy().reshape(shape)
 path=case/(name+'.npy');np.save(path,value)
 pins[name]={'path':str(path),'sha256':sha(path),'shape':list(shape),'dtype':str(dtype)}
record.update(status='pass',finished_utc=datetime.now(timezone.utc).isoformat(),outputs=out.size,
 original_compiled_bits_exact=True,torch_original_gate_pass=True,raw_output_sha256=raw,tap_calls=1,
 captured_tensors=pins,model_object_sha256=sha(case/'model.o'),shared_object_sha256=sha(case/'model.so'))
receipt.write_text(json.dumps(record,indent=2)+'\n');print('FULL_ORIGINAL_CAPTURE_PASS',raw,flush=True)
