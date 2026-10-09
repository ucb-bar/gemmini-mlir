"""Fresh immutable original whole-model native gate, independent integer devices."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess
import numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower import quant_hoist

W=Path(__file__).resolve().parent/'normal_whole_v3'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
BASE_NATIVE=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
BUNDLE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
FROZEN=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005')
CLANG=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
host=W/'host';host.mkdir(exist_ok=False)
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
source=W/'host_llvm/expanded.native.ll'
reference=BASE_NATIVE/'reference.c'
shim=BASE_NATIVE/'device/device_catalog_shim.c'
runtime=FROZEN/'merlin/runtime/abi/mlir_runtime.c'
old_gate=json.loads((BASE_NATIVE/'validation.json').read_text())
assert old_gate['original_compiled_bits_exact'] and old_gate['allclose']
assert old_gate['llvm_sha256']==sha(B/'host_llvm/model.native.ll')
argv=[str(CLANG),'-O2','-fPIC','-c',str(source),'-o',str(host/'model.o')]
link=['cc','-O3','-march=native','-fPIC','-shared',str(host/'model.o'),str(reference),str(shim),str(runtime),'-lm','-o',str(host/'model.so')]
record={'schema':'tiny_source_interval_table_full_native_gate_v1','status':'compiling','started_utc':now(),'selected_llvm_path':str(source),'selected_llvm_sha256':sha(source),'original_llvm_sha256':sha(B/'host_llvm/model.native.ll'),'original_native_gate_path':str(BASE_NATIVE/'validation.json'),'original_native_gate_sha256':sha(BASE_NATIVE/'validation.json'),'compile_argv':argv,'link_argv':link,'native_boundary_sources':{str(path):sha(path) for path in [reference,shim,runtime]},'original_quant_plan_sha256':sha(B/'quant_hoist_args.json'),'torch_golden_path':str(BUNDLE/'golden.npy'),'torch_golden_sha256':sha(BUNDLE/'golden.npy'),'torch_atol':.03125,'torch_rtol':.02,'scope':'Fresh normal hostLLVMhook output and selected modelobject containing one8MiBreadonlytable,44typed source-bound scalarlookup endpoints/22complete-helper rounding-mode sourcefallback guards; original scalar int8/i32 device stand-ins/source-bound shim and frozen runtimeC. Full original22layers/eighttokens/all256000output/originalTorchgate, no newcapture/weights/scales/golden. Functionalnative timing is not targetcycles.','token_usage_available':False}
path=host/'validation.json';path.write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(argv,check=True)
subprocess.run(link,check=True)
print('NATIVE_COMPILED',flush=True)
args=resolve_forward_args(BUNDLE)
plan=quant_hoist.read_plan(B)
if plan:
    values=quant_hoist.read_values(B)
    args.extend(np.ascontiguousarray(values[entry.key]) for entry in plan)
golden=np.load(BUNDLE/'golden.npy');output=np.zeros(golden.shape,dtype=golden.dtype,order='C')
HostModel.load(str(host/'model.so'))([(arg.ctypes.data,arg.shape) for arg in args]+[(output.ctypes.data,output.shape)])
np.save(host/'output.npy',output)
baseline=np.load(BASE_NATIVE/'output.npy')
raw_sha=hashlib.sha256(output.astype('<f4').tobytes()).hexdigest()
exact=bool(np.array_equal(output.view('u4'),baseline.view('u4')))
allclose=bool(np.allclose(output,golden,atol=.03125,rtol=.02))
record.update(finished_utc=now(),status='pass' if exact and allclose else 'fail',outputs=output.size,shape=list(output.shape),dtype=str(output.dtype),allclose=allclose,original_compiled_bits_exact=exact,raw_output_sha256=raw_sha,reference_path=str(host/'output.npy'),reference_sha256=sha(host/'output.npy'),max_abs_error=float(np.max(np.abs(output-golden))),relative_l2=float(np.linalg.norm(output-golden)/(np.linalg.norm(golden)+1e-12)),finite=bool(np.isfinite(output).all()),model_object_sha256=sha(host/'model.o'),shared_object_sha256=sha(host/'model.so'))
path.write_text(json.dumps(record,indent=2)+'\n')
assert output.shape==(1,8,32000) and output.dtype==np.float32 and output.size==256000
assert record['torch_golden_sha256']=='3a4d0d1105d4fad150b9ff1b80d4b7838d9fcdb3933a72b71000d6ecc0f5caee'
assert exact and allclose and raw_sha=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
print('FULL_ORIGINAL_NATIVE_PASS',output.size,raw_sha,flush=True)
