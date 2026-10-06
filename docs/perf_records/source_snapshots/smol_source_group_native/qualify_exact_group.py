from pathlib import Path
import json,hashlib,subprocess,numpy as np
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.abi import HostModel
from merlin.xdsl_dialects._common import text
w=Path(__file__).resolve().parent;out=w/'independent_group';out.mkdir(exist_ok=True)
module=parse_mlir_text((w/'source_group_exact.mlir').read_text())
rewrite=rewrite_ordered_fma_contractions(module,output_tile=8,pre_widen_operands='both',assume_rne=True,assume_finite_intermediates=True)
source=text(module);(out/'ordered_source.mlir').write_text(source)
llvm=lower_to_llvm_ir(source,workdir=out/'lower',features={'lower_fma_to_intrinsic','outline_llvm_loops'})
(out/'group.ll').write_text(llvm)
clang=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
subprocess.run([str(clang),'-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(out/'group.ll'),'-o',str(out/'group.o')],check=True)
provider=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/host_outlined/model.so')
subprocess.run(['cc','-shared',str(out/'group.o'),str(w/'native_wrapper.o'),'-Wl,--wrap=expf',str(provider),'-lm','-o',str(out/'group.so')],check=True)
receipt=json.loads((w/'accepted_taps.json').read_text());arguments=[np.load(row['npy_path'])for row in receipt['records']if row['role']=='input'];expected=np.load(w/'endpoint.npy');result=np.full_like(expected,0xa55a)
HostModel.load(str(out/'group.so'),name='source_group_exact')([(a.ctypes.data,a.shape)for a in arguments]+[(result.ctypes.data,result.shape)])
np.save(out/'output.npy',result)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
r={'schema':'actual_source_group_exact_native_endpoint_v1','source_group_reference_sha256':sha(w/'source_group_exact.mlir'),'accepted_taps_sha256':sha(w/'accepted_taps.json'),'llvm_sha256':sha(out/'group.ll'),'object_sha256':sha(out/'group.o'),'library_sha256':sha(out/'group.so'),'runtime_integer_provider_sha256':sha(provider),'endpoint_words':result.size,'physical_dtype':'bf16','bitwise_mismatches':int(np.count_nonzero(result!=expected)),'output_sha256':sha(out/'output.npy'),'original_endpoint_sha256':sha(w/'endpoint.npy'),'source_fma_rewrite':rewrite,'numeric_changes':'None; normal source lowering, exact cast/order/polyexp/max/denominator/PV source DAG.','scope':'Independent actual original source group endpoint replay on accepted full-source live tensors; no target or timing result.','token_usage_available':False}
(out/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);assert not r['bitwise_mismatches']
