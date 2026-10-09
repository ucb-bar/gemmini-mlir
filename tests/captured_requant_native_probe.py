"""Compare selected bundle native kernels against original ordered f32 epilogues."""
import argparse,ctypes,json,numpy as np
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle',type=Path);args=parser.parse_args();p=args.bundle;r=json.loads((p/'requant.json').read_text());lib=ctypes.CDLL(str((p/'native_oracle.so').resolve()));rng=np.random.default_rng(73);results=[]
for route in r['routes'][:2]:
 s=route['schedule'];proof=route['proof']
 if route['direct_conv']:
  h,w,ci,co=[s[k] for k in ['h','w','cin','cout']];a=rng.integers(-128,128,(h+2,w+2,ci),dtype=np.int8);b=rng.integers(-128,128,(3,3,ci,co),dtype=np.int8);acc=np.zeros((h,w,co),np.int32)
  for ky in range(3):
   for kx in range(3):acc+=a[ky:ky+h,kx:kx+w].astype(np.int32)@b[ky,kx].astype(np.int32)
 else:
  m,n,k=[s[k] for k in ['m','n','k']];a=rng.integers(-128,128,(m,k),dtype=np.int8);b=rng.integers(-128,128,(k,n),dtype=np.int8);acc=a.astype(np.int32)@b.astype(np.int32)
 source=acc.astype(np.float32)
 for scale in proof['source_scales']:source=source*np.float32(scale)
 source=source+np.float32(0)
 if proof['relu']:source=np.maximum(source,np.float32(0))
 expected=np.clip(np.rint(source*np.float32(proof['output_reciprocal'])),-128,127).astype(np.int8)
 actual=np.empty(acc.shape,np.int8);fn=getattr(lib,route['kernel']);fn.argtypes=[ctypes.c_void_p]*3;fn(a.ctypes.data,b.ctypes.data,actual.ctypes.data)
 assert np.array_equal(actual,expected),(route['symbol'],np.count_nonzero(actual!=expected))
 results.append({'symbol':route['symbol'],'direct_conv':route['direct_conv'],'values':actual.size,'source_float_order_exact':True,'nonzero_halo':route['direct_conv']})
print(json.dumps(results,indent=2));(p/'native_selected_result.json').write_text(json.dumps(results,indent=2)+'\n')
