from pathlib import Path
import sys,subprocess,ctypes as C,json,hashlib
import numpy as np
sys.path.insert(0,'/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006/merlin/tests/runtime')
import test_source_attention_frontier as t
from dataclasses import replace
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from merlin.common.paths import merlin_dir
w=Path(__file__).resolve().parent;p=replace(t.PLAN,chunk=6,segment=2);(w/'test.c').write_text(emit_source_attention_frontier(p,symbol='test_provider',word_interval_enclosure=True,prepare_softmax_domain=True,polynomial_batch_four=True)+t.EXTRA)
cc='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang';cmd=[cc,'-O2','-shared','-fPIC','-ffp-contract=off','-I',str(merlin_dir()/'runtime/c'),str(w/'test.c'),'-lm','-o',str(w/'test.so')];subprocess.run(cmd,check=True)
l=C.CDLL(str(w/'test.so'));l.test_provider_workspace_bytes.restype=C.c_size_t;l.run.argtypes=[C.POINTER(t.View),C.POINTER(t.View),C.c_void_p,C.c_size_t,C.c_int];l.oracle.argtypes=[C.POINTER(t.View),C.c_void_p]
for seed in range(8):
 rng=np.random.default_rng(seed);a=[]
 for i in range(11):
  if i in(2,7):x=rng.integers(0,2,(1,1,3,6),dtype=np.uint8);x[:]=0 if seed==0 else x
  else:
   rows=3 if i==0 else 6 if i in(1,6)else 2
   f=rng.integers(-8,9,(1,2,rows,4)).astype(np.float32)/8;x=(f.view(np.uint32)>>16).astype(np.uint16)
  backing=np.zeros((*x.shape[:-1],x.shape[-1]*2),dtype=x.dtype);backing[...,::2]=x;a.append(backing[...,::2])
 views=(t.View*11)(*[t.view(x)for x in a]);size=l.test_provider_workspace_bytes();storage=np.full(size+128,0xa5,np.uint8);ptr=(storage.ctypes.data+63)&~63
 for repeat in range(2):
  out=np.full((1,2,3,4),0xdead,np.uint16);gold=np.empty_like(out);assert l.run(views,C.byref(t.view(out)),ptr,size,0)==1;assert l.oracle(views,gold.ctypes.data)==1;assert all(np.array_equal(x,y)for x,y in zip(t.quant(out),t.quant(gold)));assert np.all(storage[:ptr-storage.ctypes.data]==0xa5)and np.all(storage[ptr-storage.ctypes.data+size:]==0xa5)
r={'status':'pass','scope':'Independent six-column polynomial batch tail with three two-column PV parts; eight masks/inputs including all-masked, strided reads, two dirty workspace invocations each','invocations':16,'compile':cmd,'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(cc),w/'test.c',w/'test.so',Path(__file__)]}};(w/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS16')
