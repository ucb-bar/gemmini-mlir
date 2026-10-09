"""Read-only first dequant/residual tap on the frozen stock1967 native image."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import subprocess
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower import quant_hoist

W = Path(__file__).resolve().parent
B = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
BASE = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
CONTROL = W.parent/'tiny-multiplication-packet4-whole-20261006/whole'
BUNDLE = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
FROZEN = Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005')
CLANG = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

source = CONTROL/'host_llvm/expanded.native.ll'
gate = json.loads((CONTROL/'host/validation.json').read_text())
assert gate['original_compiled_bits_exact'] and gate['allclose']
assert gate['selected_llvm_sha256'] == sha(source)
text = source.read_text()
call = '  call void @merlin_dev_gemmini_0__fresh_tensor_result(ptr %5784, ptr %5789, i64 0, i64 8, i64 2048, i64 2048, i64 1, ptr %2900, ptr %2901, i64 %2902, i64 %2903, i64 %2904, i64 %2905, i64 %2906, ptr %4702, ptr %4707, i64 0, i64 8, i64 2048, i64 2048, i64 1, ptr %5846, ptr %5846, i64 0, i64 8, i64 2048, i64 2048, i64 1)\n'
after = '5878:                                             ; preds = %5847\n'
# Bind the actual scalar loop's pointers, numeric literal and complete source order.
for statement in ['%5863 = getelementptr inbounds nuw float, ptr %3979, i64 %5862',
                  '%5865 = getelementptr inbounds nuw i32, ptr %5846, i64 %5862',
                  '%5867 = getelementptr inbounds nuw float, ptr %263, i64 %5856',
                  '%5869 = sitofp i32 %5866 to float',
                  '%5870 = fmul float %5869, f0x3A33A2DF',
                  '%5871 = fmul float %5870, %5868',
                  '%5872 = fadd float %5864, %5871',
                  'store float %5872, ptr %5863, align 4']:
    assert text.count(statement) == 1, statement
assert text.count(call) == text.count(after) == 1
text = text.replace(call,call+'  call void @two_product_tap_before(ptr %3979, ptr %5846, ptr %263)\n')
text = text.replace(after,after+'  call void @two_product_tap_after(ptr %3979)\n')
text += '\ndeclare void @two_product_tap_before(ptr,ptr,ptr)\ndeclare void @two_product_tap_after(ptr)\n'
case = W/'capture'
case.mkdir(exist_ok=False)
selected = case/'model.native.ll'
selected.write_text(text)
tap = case/'tap.c'
tap.write_text('''#include <stdint.h>
#include <string.h>
static float residual[16384],scale[2048],expected[16384];
static int32_t accumulation[16384];
static unsigned before_calls,after_calls;
void two_product_tap_before(const void*r,const void*a,const void*s){
 memcpy(residual,r,sizeof(residual));memcpy(accumulation,a,sizeof(accumulation));memcpy(scale,s,sizeof(scale));++before_calls;
}
void two_product_tap_after(const void*p){memcpy(expected,p,sizeof(expected));++after_calls;}
unsigned two_product_tap_calls(unsigned i){return i?after_calls:before_calls;}
void*two_product_tap_tensor(unsigned i){void*v[]={residual,accumulation,scale,expected};return i<4?v[i]:0;}
''')
argv = [str(CLANG),'-O2','-fPIC','-c',str(selected),'-o',str(case/'model.o')]
bound = [BASE/'reference.c',BASE/'device/device_catalog_shim.c',FROZEN/'merlin/runtime/abi/mlir_runtime.c',tap]
link = ['cc','-O3','-march=native','-fPIC','-shared',str(case/'model.o'),*map(str,bound),'-lm','-o',str(case/'model.so')]
record = {'schema':'original_tiny1967_readonly_dequant_residual_tap_v1',
          'started_utc':datetime.now(timezone.utc).isoformat(),
          'source_llvm_sha256':sha(source),'selected_llvm_sha256':sha(selected),
          'source_gate_path':str(CONTROL/'host/validation.json'),'source_gate_sha256':sha(CONTROL/'host/validation.json'),
          'compile_argv':argv,'link_argv':link,'boundary_pins':{str(p):sha(p)for p in bound},
          'source_capsule_sha256':sha(W/'source_capsule.mlir'),
          'modification':'Two read-only experimental taps around the unchanged original first dequant/residual loop. No source arithmetic, input, weight, scale, runtime or integer-device stand-in change.',
          'atol':.03125,'rtol':.02,'token_usage_available':False}
receipt = case/'receipt.json'
receipt.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(argv,check=True)
subprocess.run(link,check=True)
print('READONLY_NATIVE_COMPILED',flush=True)
args = resolve_forward_args(BUNDLE)
plan = quant_hoist.read_plan(B)
if plan:
    values = quant_hoist.read_values(B)
    args.extend(np.ascontiguousarray(values[e.key])for e in plan)
golden = np.load(BUNDLE/'golden.npy')
out = np.zeros_like(golden)
HostModel.load(str(case/'model.so'))([(a.ctypes.data,a.shape)for a in args]+[(out.ctypes.data,out.shape)])
assert np.array_equal(out.view('u4'),np.load(BASE/'output.npy').view('u4'))
assert np.allclose(out,golden,atol=.03125,rtol=.02)
raw = hashlib.sha256(out.astype('<f4').tobytes()).hexdigest()
assert raw == 'ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
lib = ctypes.CDLL(str(case/'model.so'))
lib.two_product_tap_calls.argtypes = [ctypes.c_uint]
lib.two_product_tap_calls.restype = ctypes.c_uint
assert lib.two_product_tap_calls(0) == lib.two_product_tap_calls(1) == 1
lib.two_product_tap_tensor.argtypes = [ctypes.c_uint]
lib.two_product_tap_tensor.restype = ctypes.c_void_p
pins = {}
for i,(name,dtype,shape)in enumerate([('residual','f4',(1,8,2048)),('accumulation','i4',(1,8,2048)),('scale','f4',(2048,)),('expected','f4',(1,8,2048))]):
    dtype = np.dtype(dtype)
    value = np.frombuffer(ctypes.string_at(lib.two_product_tap_tensor(i),int(np.prod(shape))*dtype.itemsize),dtype=dtype).copy().reshape(shape)
    path = case/(name+'.npy')
    np.save(path,value)
    pins[name] = {'path':str(path),'sha256':sha(path),'shape':list(shape),'dtype':str(dtype)}
record.update(status='pass',finished_utc=datetime.now(timezone.utc).isoformat(),outputs=256000,
              original_compiled_bits_exact=True,torch_original_gate_pass=True,
              raw_output_sha256=raw,tap_calls=[1,1],captured_tensors=pins,
              model_object_sha256=sha(case/'model.o'),shared_object_sha256=sha(case/'model.so'))
receipt.write_text(json.dumps(record,indent=2)+'\n')
print('FULL_ORIGINAL_READONLY_CAPTURE_PASS',raw,flush=True)
