from pathlib import Path
import ctypes as C,json,hashlib,time
import numpy as np
w=Path(__file__).resolve().parent;t=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class View(C.Structure):_fields_=[('data',C.c_void_p),('offset',C.c_int64),('sizes',C.c_int64*4),('strides',C.c_int64*4)]
def view(a):return View(a.ctypes.data,0,(C.c_int64*4)(*a.shape),(C.c_int64*4)(*[s//a.itemsize for s in a.strides]))
def desc_type(n):
 class D(C.Structure):_fields_=[('allocated',C.c_void_p),('aligned',C.c_void_p),('offset',C.c_int64),('sizes',C.c_int64*n),('strides',C.c_int64*n)]
 return D
D4,D3,D2=[desc_type(i)for i in(4,3,2)]
def desc(a,D):return D(a.ctypes.data,a.ctypes.data,0,(C.c_int64*a.ndim)(*a.shape),(C.c_int64*a.ndim)(*[s//a.itemsize for s in a.strides]))
quant_path=t/'quant_frontier/frontier.so';qlib=C.CDLL(str(quant_path));qfn=qlib._mlir_ciface_source_quant_frontier;qfn.argtypes=[C.POINTER(D4),C.POINTER(D3),C.POINTER(D2)]
def quant(a):
 x=np.concatenate([a]*4,axis=2);saved=x.copy();qraw=np.full(1024*768+64,0xA5,np.uint8);sraw=np.full(1024+32,0x5A5A,np.uint16);q=qraw[:1024*768].view(np.int8).reshape(1,1024,768);s=sraw[:1024].reshape(1,1024);ds=[desc(x,D4),desc(q,D3),desc(s,D2)];prior=[bytes(d)for d in ds];qfn(*[C.byref(d)for d in ds]);assert prior==[bytes(d)for d in ds] and np.array_equal(x,saved) and (qraw[-64:]==0xA5).all()and(sraw[-32:]==0x5A5A).all();return q.copy(),s.copy()
inputs=[np.load(t/f'input_{i}.npy')for i in range(11)];before=[hashlib.sha256(x.tobytes()).hexdigest()for x in inputs];vs=(View*11)(*[view(x)for x in inputs]);gold=np.fromfile(t/'endpoint.bin',np.uint16).reshape(1,12,256,64);goldq,golds=quant(gold)
CALL=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(C.c_int8),C.POINTER(C.c_int8),C.POINTER(C.c_int32),C.c_int,C.c_int,C.c_int,C.c_int)
results={}
for arm in ['control','candidate']:
 lib=C.CDLL(str(w/arm/'provider.so'));lib.group_provider_workspace_bytes.restype=C.c_size_t;capacity=lib.group_provider_workspace_bytes();lib.group_provider.argtypes=[C.POINTER(View),C.POINTER(View),C.c_void_p,C.c_size_t,CALL,C.c_void_p];lib.group_provider_statistics.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_ulonglong),C.c_size_t];calls=[];errors=[]
 @CALL
 def product(opaque,a,b,c,m,n,k,degree):
  try:
   aa=np.ctypeslib.as_array(a,shape=(3*m*k,)).reshape(3,m,k);bb=np.ctypeslib.as_array(b,shape=(3*k*n,)).reshape(3,k,n);cc=np.ctypeslib.as_array(c,shape=(m*n,)).reshape(m,n);total=np.zeros((m,n),np.float64)
   for ad in range(3):
    bd=degree-ad
    if 0<=bd<3:total+=aa[ad].astype(np.float64)@bb[bd].astype(np.float64)
   assert (total==np.rint(total)).all()and(abs(total)<=2**31-1).all();cc[:]=total.astype(np.int32);calls.append((m,n,k,degree));return 1
  except BaseException as e:errors.append(repr(e));return 0
 storage=np.full(capacity+128,0xA5,np.uint8);ptr=(storage.ctypes.data+63)&~63;start_offset=ptr-storage.ctypes.data;raw=np.full(gold.size+32,0xDEAD,np.uint16);out=raw[:gold.size].reshape(gold.shape);vout=view(out);start=time.time();status=lib.group_provider(vs,C.byref(vout),ptr,capacity,product,None);seconds=time.time()-start
 assert not errors and before==[hashlib.sha256(x.tobytes()).hexdigest()for x in inputs];assert(storage[:start_offset]==0xA5).all()and(storage[start_offset+capacity:]==0xA5).all()and(raw[-32:]==0xDEAD).all()
 counts=(C.c_ulonglong*8)();qdiff=sdiff=None
 if status:
  assert lib.group_provider_statistics(ptr,capacity,counts,8)==1
  q,s=quant(out);qdiff=int(np.count_nonzero(q!=goldq));sdiff=int(np.count_nonzero(s!=golds));np.save(w/arm/'output.npy',out)
 record={'status':int(status),'original_i8_mismatches':qdiff,'original_scale_mismatches':sdiff,'unobserved_carrier_differences':int(np.count_nonzero(out!=gold)),'workspace_bytes':capacity,'callback_calls':len(calls),'calls':calls,'stats':list(counts),'native_seconds_functional_only':seconds,'guards_and_input_bytes':True,'source_qk_fmas':64*(counts[0]+counts[2]+counts[3]),'source_pv_fmas':int(counts[7]),'provider_sha256':sha(w/arm/'provider.so')};results[arm]=record;print(arm,{k:v for k,v in record.items()if k!='calls'},flush=True)
 assert status and qdiff==0 and sdiff==0
results['scope']='Complete actual original12-head source group, native exact integer product stand-ins plus compiled original54-op consumer; functional/replay census, no target cycles or performance claim.';results['pins']={str(t/f'input_{i}.npy'):sha(t/f'input_{i}.npy')for i in range(11)};results['pins'].update({str(t/'endpoint.bin'):sha(t/'endpoint.bin'),str(quant_path):sha(quant_path)});(w/'native_group.json').write_text(json.dumps(results,indent=2)+'\n')
