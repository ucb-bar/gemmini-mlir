"""Diagnostic initial endpoint snapshots; never a source-containment admission."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import shutil
import subprocess
import time
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args

base=Path(__file__).resolve().parents[2]
w=base/'out/normal_composition'
h=base/'out/observation_frontier/rounded_monotone_native'
if h.exists(): raise ValueError('fresh evidence directory required')
h.mkdir(parents=True)
src=w/'provider/native_numeric'; dest=h/'numeric'; dest.mkdir()
for p in src.iterdir():
 if p.suffix in ('.h','.c'): shutil.copyfile(p,dest/p.name)
s=(src/'provider.c').read_text()
needle='static int certify_frontier(struct attention_workspace *w){'
assert s.count(needle)==1
s=s.replace(needle,'''typedef void (*diagnostic_endpoint_fn)(int,const float*,const float*,const float*,size_t);
static diagnostic_endpoint_fn diagnostic_endpoint;
void diagnostic_set_endpoint(diagnostic_endpoint_fn fn){diagnostic_endpoint=fn;}
'''+needle)
needle='  int changed=0,unfinished=0;'
assert s.count(needle)==1
s=s.replace(needle,'''  if(pass==0 && diagnostic_endpoint)for(int head=0;head<HEADS;head++){
   struct attention_head *h=&w->heads[head];
   diagnostic_endpoint(head,h->endpoint_lo,h->endpoint_hi,h->out,ROWS*DEPTH);
  }
'''+needle)
(dest/'provider.c').write_text(s)
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity, consume_rounded_polynomial_monotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
mon=base/'out/observation_frontier/rounded_monotonicity'
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
header=dest/'monotone_bit_polynomial.h'
proof=RoundedPolynomialMonotonicity((0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000),1118743633,hash_file(mon/'independent_qualification.json'),hash_file(mon/'independent.c'),hash_file(header),True,True,True)
contract=SourceNumericContract(*([True]*7),standard_floor_values=True,floor_interposition_unobserved=True)
header.write_text(consume_rounded_polynomial_monotonicity(header.read_text(),proof,contract))
(h/'monotonicity_binding.json').write_text(json.dumps(dict(proof=vars(proof),contract=vars(contract)),indent=2)+'\n')
manifest=json.loads((w/'provider/build.json').read_text())
commands=[]
for cmd in manifest['roles']['native_numeric']['commands']:
 cmd=[v.replace('out/normal_composition/provider/native_numeric',str(dest)) for v in cmd]
 subprocess.run(cmd,check=True);commands.append(cmd)
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005')
runtime=root/'host_outlined/model.so'
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
cmd=[clang,'-shared',*[str(w/'native_v3'/x) for x in ('model.o','bridge.o','products.o','math.o')],str(dest/'provider.so'),str(runtime),'-Wl,--wrap=expf','-Wl,--wrap=_mlir_ciface_source_attention_frontier_fallback','-lm','-o',str(h/'model.so')]
subprocess.run(cmd,check=True);commands.append(cmd)
lib=C.CDLL(str(h/'model.so'))
END=C.CFUNCTYPE(None,C.c_int,C.POINTER(C.c_float),C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_size_t)
heads=[]; snapshots=0; errors=[]
@END
def endpoint(head,lo,hi,candidate,count):
 global snapshots
 try:
  assert head==len(heads) and count==256*64
  heads.append(np.stack([np.ctypeslib.as_array(p,shape=(count,)).copy().reshape(256,64) for p in (lo,hi,candidate)]))
  if head==11:
   np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))
   snapshots+=1;heads.clear()
 except BaseException as e:errors.append(repr(e))
lib.diagnostic_set_endpoint.argtypes=[END];lib.diagnostic_set_endpoint(endpoint)
CALL=C.CFUNCTYPE(None,C.POINTER(C.c_int8),C.POINTER(C.c_int8),C.POINTER(C.c_int32),C.c_int,C.c_int,C.c_int,C.c_int)
product_calls=0
@CALL
def products(a,b,c,m,n,k,degree):
 global product_calls
 try:
  av=np.ctypeslib.as_array(a,shape=(3*m*k,)).reshape(3,m,k);bv=np.ctypeslib.as_array(b,shape=(3*k*n,)).reshape(3,k,n)
  total=np.zeros((m,n),np.float64)
  for ad in range(3):
   bd=degree-ad
   if 0<=bd<3:total+=av[ad].astype(np.float64)@bv[bd].astype(np.float64)
  assert np.isfinite(total).all() and (total==np.rint(total)).all() and (abs(total)<=2**31-1).all()
  np.ctypeslib.as_array(c,shape=(m*n,))[:]=total.astype(np.int32).ravel();product_calls+=1
 except BaseException as e:errors.append(repr(e))
lib.register_product.argtypes=[CALL];lib.register_product(products)
args=resolve_forward_args(root/'bundle');gold=np.load(root/'bundle/golden.npy');out=np.zeros_like(gold)
start=time.time();HostModel.load(str(h/'model.so'))([(x.ctypes.data,x.shape) for x in args]+[(out.ctypes.data,out.shape)])
counts=(C.c_uint64*8)();lib.get_counts(counts);np.save(h/'output.npy',out)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=dict(scope='Complete rounded-DAG monotonicity removes only encoded endpoint budget; source/RMS4/consumer/fallback unchanged. Diagnostic initial intervals retained.',snapshots=snapshots,product_calls=product_calls,counts=list(counts),errors=errors,bit_mismatches=int(np.count_nonzero(out.view('u4')!=gold.view('u4'))),allclose=bool(np.allclose(out,gold,atol=.03125,rtol=.02)),seconds=time.time()-start,commands=commands,pins={str(p):sha(p) for p in h.rglob('*') if p.is_file()})
(h/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('pins','commands')}),flush=True)
assert snapshots==48 and not errors and product_calls==23040 and not record['bit_mismatches'] and counts[1]==0
