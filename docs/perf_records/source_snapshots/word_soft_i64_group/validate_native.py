"""Normal prepared physical workspace/retained-source ABI, native product stand-ins."""
import ctypes as C,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args
w=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_i64');h=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group/native');h.mkdir(exist_ok=True)
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005')
base=w.parent/'closed_group_endpoint'; frozen=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group/native_numeric_frozen')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((frozen/'manifest.json').read_text())
for p,digest in manifest['local_pins'].items():assert sha(frozen/p)==digest
runtime=root/'host_outlined/model.so';assert sha(runtime)=='a3dda3fccafc08aff62e749223eda9a839bec3e2cd9a8fd91d926f4c65d9198e'
a=json.loads((root/'build_outlined/device_signatures.json').read_text());b=json.loads((w/'build/device_signatures.json').read_text());assert {r['symbol']:r['dtypes']for r in a['routed']}=={r['symbol']:r['dtypes']for r in b['routed']}
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang';commands=[]
def run(cmd):commands.append(cmd);subprocess.run(cmd,check=True)
import shutil
prior=w/'native'
reuse=json.loads((prior/'compiled_model_reuse.json').read_text())
assert sha(w/'build/lower/model.ll')==reuse['llvm'] and sha(prior/'model.o')==reuse['object']
shutil.copyfile(prior/'model.o',h/'model.o')
reuse['scope']='Exact accepted current i64 ordinary host object reused; only native numeric library changes'
(h/'compiled_model_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
# Actual bridge body unchanged. Native primitive symbols replace only device execution.
bridge=w/'bridge/bridge.c'
run([clang,'-O2','-fPIC','-fno-builtin','-ffp-contract=off','-I'+str(frozen),'-Dattention_frontier_writer=native_attention_frontier_bridge','-c',str(bridge),'-o',str(h/'bridge.o')])
from merlin.runtime.host_math import host_math_recipe
(h/'host_math.c').write_text(host_math_recipe('expf_via_double').source)
run([clang,'-O2','-fPIC','-fno-builtin','-ffp-contract=off','-c',str(h/'host_math.c'),'-o',str(h/'math.o')])
args=','.join(f'void*a{i}'for i in range(13));vals=','.join(f'a{i}'for i in range(13));fargs=','.join(f'void*a{i}'for i in range(12));fvals=','.join(f'a{i}'for i in range(12))
s=['#include <stdint.h>','#include <stddef.h>','typedef void(*product_fn)(const int8_t*,const int8_t*,int32_t*,int,int,int,int);','static product_fn product;','void register_product(product_fn p){product=p;}','static uint64_t calls,fallbacks;static uintptr_t first_workspace;static uint64_t workspace_changes;',f'extern void native_attention_frontier_bridge({args});',f'void _mlir_ciface_attention_frontier_writer_borrowed({args}){{uintptr_t p=((uintptr_t*)a12)[1];if(!calls)first_workspace=p;else if(p!=first_workspace)workspace_changes++;calls++;native_attention_frontier_bridge({vals});}}',f'extern void __real__mlir_ciface_source_attention_frontier_fallback({fargs});',f'void __wrap__mlir_ciface_source_attention_frontier_fallback({fargs}){{fallbacks++;__real__mlir_ciface_source_attention_frontier_fallback({fvals});}}','void get_counts(uint64_t*out){out[0]=calls;out[1]=fallbacks;out[2]=workspace_changes;out[3]=first_workspace;}']
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
counts=(C.c_uint64*4)();lib.get_counts(counts);np.save(h/'output.npy',out)
r={'scope':'Frozen accepted current i64 normal host plus explicit word/private-soft numeric composition; native exact integer product stand-ins, not new target or normal binding qualification','commands':commands,'source_llvm_sha256':sha(w/'build/lower/model.ll'),'model_object_sha256':sha(h/'model.o'),'library_sha256':sha(h/'model.so'),'numeric_library_sha256':sha(frozen/'provider.so'),'calls':list(counts),'product_calls':product_calls,'callback_errors':errors,'original_golden_sha256':sha(root/'bundle/golden.npy'),'elements':out.size,'bitwise_mismatches':int(np.count_nonzero(out.view('u4')!=g.view('u4'))),'allclose':bool(np.allclose(out,g,atol=.03125,rtol=.02)),'maxabs':float(np.max(abs(out-g))),'native_seconds':time.time()-start,'token_usage_available':False};(h/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert not errors and counts[0]==48 and counts[1]==0 and counts[2]==0 and product_calls==48*480 and r['bitwise_mismatches']==0 and r['allclose']
