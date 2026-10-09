from pathlib import Path
import subprocess,json,hashlib,numpy as np
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005');host=root/'host_outlined';w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
from merlin.runtime.host_math import host_math_recipe
(w/'host_math.c').write_text(host_math_recipe('expf_via_double').source)
old=json.loads((root/'build_outlined/device_signatures.json').read_text());new=json.loads((w/'build/device_signatures.json').read_text());assert {r['symbol']:r['dtypes'] for r in old['routed']}=={r['symbol']:r['dtypes'] for r in new['routed']}
subprocess.run(['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang','-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(w/'build/lower/model.ll'),'-o',str(w/'native_model.o')],check=True)
subprocess.run(['cc','-O2','-fPIC','-fno-builtin','-ffp-contract=off','-c',str(w/'host_math.c'),'-o',str(w/'native_wrapper.o')],check=True)
subprocess.run(['cc','-O2','-fPIC','-c',str(w/'tap.c'),'-o',str(w/'tap.o')],check=True)
# Reuse accepted native SO as the provider of unchanged runtime and exact integer stand-ins.
subprocess.run(['cc','-shared',str(w/'native_model.o'),str(w/'native_wrapper.o'),str(w/'tap.o'),'-Wl,--wrap=expf',str(host/'model.so'),'-lm','-o',str(w/'native.so')],check=True)
args=resolve_forward_args(root/'bundle');g=np.load(root/'bundle/golden.npy');x=np.zeros_like(g);HostModel.load(str(w/'native.so'))([(a.ctypes.data,a.shape) for a in args]+[(x.ctypes.data,x.shape)]);np.save(w/'native_output.npy',x);r={'scope':'Actual original closed source group partition and read-only live-leaf/endpoint native taps. All384source FMAs retain tile8/both-widen original order; unchanged other source/runtime/integer stand-ins and expf-via-double.','model_o_sha256':sha(w/'native_model.o'),'original_runtime_provider_so_sha256':sha(host/'model.so'),'native_so_sha256':sha(w/'native.so'),'original_golden_sha256':sha(root/'bundle/golden.npy'),'elements':x.size,'failed_count':int(np.count_nonzero(~np.isclose(x,g,atol=.03125,rtol=.02))),'finite':bool(np.isfinite(x).all()),'relative_l2':float(np.linalg.norm((x-g).astype(np.float64))/np.linalg.norm(g.astype(np.float64))),'bitwise_mismatches':int(np.count_nonzero(x.view('u4')!=g.view('u4'))),'allclose':bool(np.allclose(x,g,atol=.03125,rtol=.02)),'maxabs':float(np.max(np.abs(x-g))),'original_atol':.03125,'original_rtol':.02};(w/'native_validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

assert r["bitwise_mismatches"] == 0 and r["allclose"]
print("ACCEPTED_GROUP_TAP",flush=True)
