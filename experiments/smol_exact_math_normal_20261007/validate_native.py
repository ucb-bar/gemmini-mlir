"""Normal prepared physical workspace/retained-source ABI, native product stand-ins."""
import ctypes as C,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args
w=Path('/scratch/agustin/tmp/merlin-latest-20261004/out/artifacts/probes/smol-exact-math-normal-20261007');h=w/'native';h.mkdir(exist_ok=True)
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005')
frozen=Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007/out/artifacts/probes/prepared-endpoint-normal/provider/native_numeric')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007/out/artifacts/probes/prepared-endpoint-normal/provider/build.json')).read_text())
for p,digest in manifest['pins'].items():
 assert sha(p)==digest
runtime=root/'host_outlined/model.so';assert sha(runtime)=='a3dda3fccafc08aff62e749223eda9a839bec3e2cd9a8fd91d926f4c65d9198e'
a=json.loads((root/'build_outlined/device_signatures.json').read_text());b=json.loads((w/'build/device_signatures.json').read_text());assert {r['symbol']:r['dtypes']for r in a['routed']}=={r['symbol']:r['dtypes']for r in b['routed']}
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang';commands=[]
def run(cmd):commands.append(cmd);subprocess.run(cmd,check=True)
import shutil
prior=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_i64/native')
reuse=json.loads((prior/'compiled_model_reuse.json').read_text())
prior_ir=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_i64/build/lower/model.ll')
selected_ir=w/'build/lower/model.ll'
old_lines=prior_ir.read_bytes().splitlines(keepends=True);new_lines=selected_ir.read_bytes().splitlines(keepends=True)
module_comment_only=(old_lines[0].startswith(b'; ModuleID = ') and new_lines[0].startswith(b'; ModuleID = ') and old_lines[1:]==new_lines[1:])
if (sha(selected_ir)==reuse['llvm'] or module_comment_only) and sha(prior/'model.o')==reuse['object']:
 shutil.copyfile(prior/'model.o',h/'model.o');reuse['reuse_scope']='Only nonsemantic first ModuleID comment differs; all remaining LLVM bytes identical';reuse['selected_llvm']=sha(selected_ir);reuse['prior_llvm']=sha(prior_ir)
else:
 cmd=[clang,'-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(w/'build/lower/model.ll'),'-o',str(h/'model.o')];run(cmd);reuse={'llvm':sha(w/'build/lower/model.ll'),'object':sha(h/'model.o'),'compile_command':cmd}
(h/'compiled_model_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
# Actual bridge body unchanged. Native primitive symbols replace only device execution.
bridge=Path('/scratch/agustin/tmp/gemmini-prepared-rhs-owner-20261006/out/prepared_rhs_normal_materialized/bridge.c')
run([clang,'-O2','-fPIC','-fno-builtin','-ffp-contract=off','-I'+str(frozen),'-D_mlir_ciface_attention_frontier_prepared_borrowed=native_attention_frontier_bridge','-D_mlir_ciface_prepare_attention_rhs_borrowed=native_prepare_attention_rhs_bridge','-c',str(bridge),'-o',str(h/'bridge.o')])
from merlin.runtime.host_math import host_math_recipe
(h/'host_math.c').write_text(host_math_recipe('expf_via_double').source)
run([clang,'-O2','-fPIC','-fno-builtin','-ffp-contract=off','-c',str(h/'host_math.c'),'-o',str(h/'math.o')])
args=','.join(f'void*a{i}'for i in range(14))+',int64_t epoch,int64_t ordinal';vals=','.join(f'a{i}'for i in range(14))+',epoch,ordinal'
p_args=','.join(f'void*a{i}'for i in range(12))+',int64_t epoch,int64_t uses';p_vals=','.join(f'a{i}'for i in range(12))+',epoch,uses'
fargs=','.join(f'void*a{i}'for i in range(12));fvals=','.join(f'a{i}'for i in range(12))
s=['#include <stdint.h>','#include <stddef.h>','typedef void(*product_fn)(const int8_t*,const int8_t*,int32_t*,int,int,int,int);','static product_fn product;','void register_product(product_fn p){product=p;}','static uint64_t calls,fallbacks,preparations,owner_changes,lease_errors;static uintptr_t first_workspace,first_owner;static uint64_t workspace_changes;',f'extern void native_attention_frontier_bridge({args});',f'extern void native_prepare_attention_rhs_bridge({p_args});',f'void _mlir_ciface_attention_frontier_prepared_borrowed({args}){{uintptr_t p=((uintptr_t*)a12)[1];if(!calls)first_workspace=p;else if(p!=first_workspace)workspace_changes++;if(((uintptr_t*)a13)[1]!=first_owner)owner_changes++;if(epoch!=(int64_t)(calls/4)||ordinal!=(int64_t)(calls%4))lease_errors++;calls++;native_attention_frontier_bridge({vals});}}',f'void _mlir_ciface_prepare_attention_rhs_borrowed({p_args}){{uintptr_t p=((uintptr_t*)a11)[1];if(!preparations)first_owner=p;else if(p!=first_owner)owner_changes++;if(epoch!=(int64_t)preparations||uses!=4)lease_errors++;preparations++;native_prepare_attention_rhs_bridge({p_vals});}}',f'extern void __real__mlir_ciface_source_attention_frontier_fallback({fargs});',f'void __wrap__mlir_ciface_source_attention_frontier_fallback({fargs}){{fallbacks++;__real__mlir_ciface_source_attention_frontier_fallback({fvals});}}','void get_counts(uint64_t*out){out[0]=calls;out[1]=fallbacks;out[2]=workspace_changes;out[3]=first_workspace;out[4]=preparations;out[5]=owner_changes;out[6]=lease_errors;out[7]=first_owner;}']
for name,n,k in [('qk',512,64),('pv192',64,192),('pv128',64,128)]:
 for degree in range(5):s.append(f'void {name}_products_{degree}(const int8_t*a,const int8_t*b,int32_t*c){{product(a,b,c,256,{n},{k},{degree});}}')
(h/'products.c').write_text('\n'.join(s))
run([clang,'-O2','-fPIC','-c',str(h/'products.c'),'-o',str(h/'products.o')])
run([clang,'-shared',str(h/'model.o'),str(h/'bridge.o'),str(h/'products.o'),str(h/'math.o'),str(frozen/'provider.so'),str(runtime),'-Wl,--wrap=expf','-Wl,--wrap=_mlir_ciface_source_attention_frontier_fallback','-lm','-o',str(h/'model.so')])
print('NORMAL_NATIVE_LINK_CLOSED',flush=True)
lib=C.CDLL(str(h/'model.so'));CALL=C.CFUNCTYPE(None,C.POINTER(C.c_int8),C.POINTER(C.c_int8),C.POINTER(C.c_int32),C.c_int,C.c_int,C.c_int,C.c_int);errors=[];product_calls=0
@CALL
def products(a,b,c,m,n,k,degree):
 global product_calls
 try:
  av=np.ctypeslib.as_array(a,shape=(3*m*k,)).reshape(3,m,k);bv=np.ctypeslib.as_array(b,shape=(3*k*n,)).reshape(3,k,n);dest=np.ctypeslib.as_array(c,shape=(m*n,)).reshape(m,n);total=np.zeros((m,n),np.float64)
  for ad in range(3):
   bd=degree-ad
   if 0<=bd<3:total+=av[ad].astype(np.float64)@bv[bd].astype(np.float64)
  assert np.isfinite(total).all() and (total==np.rint(total)).all() and (abs(total)<=2**31-1).all();dest[:]=total.astype(np.int32);product_calls+=1
  if product_calls%480==0:print('PRODUCT_GROUP',product_calls//480,flush=True)
 except BaseException as e:errors.append(repr(e));raise
lib.register_product.argtypes=[CALL];lib.register_product(products)
args=resolve_forward_args(root/'bundle');g=np.load(root/'bundle/golden.npy');out=np.zeros_like(g);start=time.time();HostModel.load(str(h/'model.so'))([(x.ctypes.data,x.shape)for x in args]+[(out.ctypes.data,out.shape)])
counts=(C.c_uint64*8)();lib.get_counts(counts);np.save(h/'output.npy',out)
r={'policy':'explicit approximate_source_roundoff_rms4; no exact observer theorem', 'scope':'New normal prepared48 group physical descriptor/workspace/fallback implementation; native exact integer product stand-ins, not hardware','commands':commands,'source_llvm_sha256':sha(w/'build/lower/model.ll'),'model_object_sha256':sha(h/'model.o'),'library_sha256':sha(h/'model.so'),'numeric_library_sha256':sha(frozen/'provider.so'),'calls':list(counts),'product_calls':product_calls,'callback_errors':errors,'original_golden_sha256':sha(root/'bundle/golden.npy'),'elements':out.size,'bitwise_mismatches':int(np.count_nonzero(out.view('u4')!=g.view('u4'))),'allclose':bool(np.allclose(out,g,atol=.03125,rtol=.02)),'maxabs':float(np.max(abs(out-g))),'native_seconds':time.time()-start,'token_usage_available':False};(h/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert not errors and counts[0]==48 and counts[1]==0 and counts[2]==0 and counts[4]==12 and counts[5]==0 and counts[6]==0 and product_calls==48*480 and r['bitwise_mismatches']==0 and r['allclose']
