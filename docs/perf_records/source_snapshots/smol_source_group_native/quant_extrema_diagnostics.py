"""Compiled original source extrema plus existing first4 observation outputs."""
from pathlib import Path
import hashlib,json,subprocess,numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.linalg.ops import ReduceOp
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.abi import HostModel
w=Path(__file__).resolve().parent;out=w/'quant_frontier'
module=parse_mlir_text((out/'source_quant_frontier.mlir').read_text());f=list(module.body.block.ops)[0];body=f.body.block;ret=list(body.ops)[-1]
extra=[op.results[0] for op in body.ops if isinstance(op,ReduceOp)]
assert len(extra)==2
ret.operands=(*ret.operands,*extra)
f.properties['function_type']=FunctionType.from_lists([value.type for value in body.args],[value.type for value in ret.operands])
module.verify();ir=text(module);(out/'source_quant_extrema.mlir').write_text(ir)
llvm=lower_to_llvm_ir(ir,workdir=out/'extrema_lower');(out/'extrema.ll').write_text(llvm)
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
subprocess.run([clang,'-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(out/'extrema.ll'),'-o',str(out/'extrema.o')],check=True)
provider='/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/host_outlined/model.so'
subprocess.run(['cc','-shared',str(out/'extrema.o'),provider,'-lm','-o',str(out/'extrema.so')],check=True)
run=HostModel.load(str(out/'extrema.so'),name='source_quant_frontier');outputs=[]
for directory in ['exact_native_control','center_native_screen']:
    data=np.concatenate([np.load(w/directory/f'group_{i}_output.npy') for i in range(4)],axis=2)
    result=[np.zeros((1,1024,768),np.int8),*[np.zeros((1,1024),np.uint16) for i in range(3)]]
    run([(data.ctypes.data,data.shape)]+[(value.ctypes.data,value.shape) for value in result]);outputs.append(result)
    for index in (0,1):assert np.array_equal(result[index],np.load(out/(directory+'_output_'+str(index)+'.npy')))
    for index in (2,3):np.save(out/(directory+'_output_'+str(index)+'.npy'),result[index])
comparison=[dict(role=role,changed_words=int(np.count_nonzero(a!=b)),elements=int(a.size)) for role,a,b in zip(['i8','BF16scale','BF16minimum','BF16maximum'],*outputs)]
r=dict(schema='original_source_quant_extrema_first4_v1',source_frontier_sha256=hashlib.sha256((out/'source_quant_frontier.mlir').read_bytes()).hexdigest(),diagnostic_source_sha256=hashlib.sha256(ir.encode()).hexdigest(),llvm_sha256=hashlib.sha256(llvm.encode()).hexdigest(),library_sha256=hashlib.sha256((out/'extrema.so').read_bytes()).hexdigest(),comparison=comparison,retained_original_i8_scale_outputs_exact=True,scope='Compiled actual source min/max/quant DAG on same original first4 source inputs. No candidate changes; source arithmetic retained.',token_usage_available=False)
(out/'extrema_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
