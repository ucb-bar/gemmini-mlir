"""Full native selected-value graph with conditional observation permission.

Original continuation executes eagerly in this semantic screen. No runtime gain
is claimed; admission cannot read reference output values or final golden data.
"""
from pathlib import Path
import ctypes as C
import hashlib,importlib.util,json,subprocess,time
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.integer_projection_enclosure import prepare_projection_weight_bounds
from merlin.llvmlower.conditional_observation_policy import ConditionalBF16ObservationPolicy,admit_conditional_bf16_row,admit_conditional_bf16_rows
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/conditional_source_off';normal=base/'out/normal_composition'
import shutil
w.mkdir(parents=True,exist_ok=False)
for name in ('model.o','source_binding.json'):shutil.copyfile(base/'out/observation_frontier/conditional_native_v2'/name,w/name)
root=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005');args=resolve_forward_args(root/'bundle')
bindings=json.loads((base/'out/observation_frontier/initial_bounds/projection_screen.json').read_text())['rows']
provider=base/'out/observation_frontier/initial_bounds/numeric/provider.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
permission=dict(policy='conditional_bf16_row_max_spacing_v1',budget='one BF16 spacing at largest absolute candidate word in each row',estimate_policy='approximate_source_roundoff_rms4',selection='candidate and conditional endpoints only; no reference or golden values',continuation='unchanged source-compatible normal projection result for every nonadmitted row',final_gate=dict(atol=.03125,rtol=.02),scope='Matched source-off identity control: conditional selection disabled, same compiled hook/model/provider')
(w/'policy_predeclaration.json').write_text(json.dumps(permission,indent=2)+'\n')
policy=ConditionalBF16ObservationPolicy(sha(w/'source_binding.json'),sha(w/'policy_predeclaration.json'),sha(provider),'approximate_source_roundoff_rms4',True,True,True,True)
proto=Path('/scratch/agustin/tmp/merlin-bf16-quantizer-enclosure-20261007/out/artifacts/quantizer-enclosure/prototype.py');spec=importlib.util.spec_from_file_location('numeric',proto);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
prepared=[]
for b in bindings:
 weight=np.ascontiguousarray(args[b['weight_argument']].T);plan=prepare_projection_weight_bounds(weight.tobytes(),reduction=768,columns=768,source_min=-127,source_max=127)
 assert plan.weight_sha256==b['weight_sha256']
 prepared.append((weight,plan,p.widen(args[b['channel_scale_argument']].view(np.uint16))[None,:],p.widen(args[b['bias_argument']].view(np.uint16))[None,:]))
(w/'observer.c').write_text('''#include <stdint.h>
#include <stddef.h>
typedef struct{uint16_t*allocated,*aligned;int64_t offset,sizes[3],strides[3];}D;
typedef void(*FN)(D*,D*);static FN selected;
void register_observer(FN f){selected=f;}
void _mlir_ciface_conditional_projection_observer_borrowed(D*a,D*b){selected(a,b);}
''')
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
cmd=[clang,'-O2','-fPIC','-c',str(w/'observer.c'),'-o',str(w/'observer.o')];subprocess.run(cmd,check=True)
link=[clang,'-shared',str(w/'model.o'),str(w/'observer.o'),*[str(normal/'native_v3'/x) for x in ('bridge.o','products.o','math.o')],str(provider),str(root/'host_outlined/model.so'),'-Wl,--wrap=expf','-Wl,--wrap=_mlir_ciface_source_attention_frontier_fallback','-lm','-o',str(w/'model.so')];subprocess.run(link,check=True)
lib=C.CDLL(str(w/'model.so'));errors=[];heads=[];quarters=[];groups=0;rows=[]
END=C.CFUNCTYPE(None,C.c_int,C.POINTER(C.c_float),C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_size_t)
@END
def endpoint(head,lo,hi,center,count):
 global groups
 try:
  assert head==len(heads) and count==16384
  heads.append(np.stack([np.ctypeslib.as_array(v,shape=(count,)).copy().reshape(256,64) for v in (lo,hi,center)]))
  if head==11:quarters.append(np.stack(heads,axis=1));heads.clear();groups+=1
 except BaseException as e:errors.append(repr(e))
lib.diagnostic_set_endpoint.argtypes=[END];lib.diagnostic_set_endpoint(endpoint)
class D(C.Structure):_fields_=[('allocated',C.POINTER(C.c_uint16)),('aligned',C.POINTER(C.c_uint16)),('offset',C.c_int64),('sizes',C.c_int64*3),('strides',C.c_int64*3)]
def array(d):
 assert tuple(d.sizes)==(1,1024,768) and tuple(d.strides)==(786432,768,1)
 return np.ctypeslib.as_array(d.aligned,shape=(d.offset+786432,))[d.offset:].reshape(1024,768)
OBS=C.CFUNCTYPE(None,C.POINTER(D),C.POINTER(D))
@OBS
def observe(original,destination):
 try:
  original=array(original.contents);output=array(destination.contents);np.copyto(output,original)
  assert C.CDLL(None).fegetround()==0 and len(quarters)==4
  block=len(rows);weight,plan,cs,bias=prepared[block]
  a=np.concatenate(quarters,axis=2).transpose(0,2,1,3).reshape(3,1024,768);quarters.clear()
  assert np.isfinite(a).all() and np.all(a[0]<=a[1])
  words=lambda x:(x.view(np.uint32)>>16).astype(np.uint16)
  lo,hi,center=[words(p.bf(v)) for v in a]
  kw=dict(divisor=127.,epsilon_word=0x3728,qmin=-127,qmax=127)
  ql,qh,sl,sh=p.enclose(lo,hi,**kw);qc,_,sc,_=p.enclose(center,center,**kw)
  delta=np.maximum(abs(qc.astype(np.int16)-ql),abs(qc.astype(np.int16)-qh))
  product=qc.astype(np.float64)@weight.astype(np.float64);assert np.all(product==np.rint(product))
  radius=np.minimum(delta.sum(axis=1,dtype=np.int64)[:,None]*np.asarray(plan.column_linf),delta.max(axis=1).astype(np.int64)[:,None]*np.asarray(plan.column_l1))
  low=np.maximum(product-radius,plan.source_lower);high=np.minimum(product+radius,plan.source_upper)
  candidates=[p.bf(x*s[:,None]) for x in (p.bf(low),p.bf(high)) for s in (sl,sh)]
  candidates=[p.bf(x*cs) for x in (np.minimum.reduce(candidates),np.maximum.reduce(candidates))]
  lower=p.bf(np.minimum.reduce(candidates)+bias);upper=p.bf(np.maximum.reduce(candidates)+bias)
  selected=p.bf(p.bf(p.bf(p.bf(product)*sc[:,None])*cs)+bias);cw=words(selected)
  admitted=admit_conditional_bf16_rows(policy,lower=words(lower),upper=words(upper),candidate=cw)
  for row in (0,511,1023):
   reference=admit_conditional_bf16_row(policy,lower=tuple(map(int,words(lower[row]))),upper=tuple(map(int,words(upper[row]))),candidate=tuple(map(int,cw[row])))
   assert not admitted[row] or reference
  admitted[:]=False  # Matched identity control; original continuation copied unchanged.
  output[admitted]=cw[admitted]
  eager=np.all(ql==qh,axis=1)&(sl==sh)
  record=dict(block=block,admitted_rows=int(admitted.sum()),additional_rows=int(np.count_nonzero(admitted&~eager)),continuation_rows=int(np.count_nonzero(~admitted)),changed_output_words=int(np.count_nonzero(output!=original)),guard_failures=0)
  rows.append(record);print(json.dumps(record),flush=True)
 except BaseException as e:errors.append(repr(e));print('OBSERVER_ERROR',repr(e),flush=True)
lib.register_observer.argtypes=[OBS];lib.register_observer(observe)
CALL=C.CFUNCTYPE(None,C.POINTER(C.c_int8),C.POINTER(C.c_int8),C.POINTER(C.c_int32),C.c_int,C.c_int,C.c_int,C.c_int);products_count=0
@CALL
def products(a,b,c,m,n,k,degree):
 global products_count
 try:
  aa=np.ctypeslib.as_array(a,shape=(3*m*k,)).reshape(3,m,k);bb=np.ctypeslib.as_array(b,shape=(3*k*n,)).reshape(3,k,n);total=np.zeros((m,n),np.float64)
  for ad in range(3):
   bd=degree-ad
   if 0<=bd<3:total+=aa[ad].astype(np.float64)@bb[bd].astype(np.float64)
  assert np.all(total==np.rint(total)) and np.max(abs(total))<2**31
  np.ctypeslib.as_array(c,shape=(m*n,))[:]=total.astype(np.int32).ravel();products_count+=1
 except BaseException as e:errors.append(repr(e))
lib.register_product.argtypes=[CALL];lib.register_product(products)
gold=np.load(root/'bundle/golden.npy');output=np.zeros_like(gold);started=time.time();HostModel.load(str(w/'model.so'))([(x.ctypes.data,x.shape) for x in args]+[(output.ctypes.data,output.shape)])
counts=(C.c_uint64*8)();lib.get_counts(counts);np.save(w/'output.npy',output)
diff=abs(output.astype(np.float64)-gold.astype(np.float64));failed=diff>(.03125+.02*abs(gold.astype(np.float64)))
record=dict(policy=permission,source_groups=groups,calls=list(counts),products=products_count,blocks=rows,errors=errors,original_gate_failures=int(failed.sum()),bit_changes=int(np.count_nonzero(output.view('u4')!=gold.view('u4'))),maxabs=float(diff.max()),rms=float(np.sqrt(np.mean(diff**2))),seconds=time.time()-started,golden_sha256=sha(root/'bundle/golden.npy'),commands=[cmd,link],pins={str(p):sha(p) for p in w.rglob('*') if p.is_file()})
(w/'validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('pins','commands')}),flush=True)
assert not errors and groups==48 and len(rows)==12 and counts[1]==0 and products_count==23040
