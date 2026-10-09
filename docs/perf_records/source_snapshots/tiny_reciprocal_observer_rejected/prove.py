from fractions import Fraction as F
from pathlib import Path
import json,ctypes as C,hashlib
import numpy as np
w=Path(__file__).resolve().parent
S=2**23;M=0x7311c3
extrema=[]
for lo,hi,offset,den in [(0,M,1,2),(M+1,S-1,2,4)]:
 f=lambda t:(1+F(t,S))*(offset+F(M-t,S))/den
 vertex=F(M+(offset-1)*S,2)
 points=[F(lo),F(hi)]+([vertex]if lo<=vertex<=hi else[])
 extrema.extend([(str(x),str(f(x))) for x in points])
 assert max(abs(1-f(x)) for x in points)<F(13,256)
u=F(1,2**24);E=F(13,256);errs=[]
for _ in range(2):
 E=E*E+(1+E)**2*u+(1+E*E+(1+E)**2*u)*u;errs.append(str(E))
lo=(1-u)/(1+E)*((1-u)/(1+u))**3
hi=(1+u)/(1-E)*((1+u)/(1-u))**3
beta=max(1-lo,hi-1)
assert beta<F(1,2**15)
assert beta*256<F(1,128)
# Certified candidates keep each product in [2^-125,2^126); this bounds
# the source perturbed product away from subnormals/overflow at each step.
lib=C.CDLL(str(w/'observer.so'));lib.vector_test.argtypes=[C.c_void_p]*7+[C.c_ulong]
rng=np.random.default_rng(738);reports=[]
def check(name,vs):
 d,a,b,c=[np.ascontiguousarray(x,np.float32) for x in vs];n=d.size
 out=[np.empty(n,np.int32) for _ in range(3)]
 lib.vector_test(*[x.ctypes.data for x in [d,a,b,c,*out]],n)
 assert np.array_equal(out[0],out[1]),(name,np.where(out[0]!=out[1])[0][:5])
 reports.append(dict(name=name,elements=n,fast=int(out[2].sum()),mismatches=0))
n=1000000
d=np.exp2(rng.uniform(-120,120,n)).astype(np.float32);a=np.exp2(rng.uniform(-30,30,n))*rng.choice([-1,1],n);b=np.exp2(rng.uniform(-30,30,n))*rng.choice([-1,1],n);c=np.exp2(rng.uniform(-20,20,n))*rng.choice([-1,1],n)
check('unrelated_signed_dynamic_chain',[d,a,b,c])
# Round boundaries and their representable neighbors, varied scales/order.
n=262144;d=np.exp2(rng.uniform(-100,100,n)).astype(np.float32);c=np.exp2(rng.uniform(-12,12,n)).astype(np.float32);b=np.exp2(rng.uniform(-12,12,n)).astype(np.float32);target=(rng.integers(-128,128,n)+.5).astype(np.float32)
a=np.asarray(target.astype(np.float64)*d.astype(np.float64)/(b.astype(np.float64)*c.astype(np.float64)),np.float32)
for tag,aa in [('ties',a),('below',np.nextafter(a,np.float32(-np.inf))),('above',np.nextafter(a,np.float32(np.inf)))]:check(tag,[d,aa,b,c])
# Raw bits include zero, subnormal, nonfinite, huge products; unsupported paths
# execute original C operations and identical clamps/conversion.
raw=[rng.integers(0,2**32,262144,dtype=np.uint32).view(np.float32) for _ in range(4)]
check('raw_special_overflow_underflow',raw)
r=dict(schema='experimental_closed_reciprocal_observer_rational_screen_v1',initial_piecewise_extrema=extrema,initial_error_bound='13/256',newton_relative_error_bounds=errs,source_to_candidate_ratio_lower=str(lo),source_to_candidate_ratio_upper=str(hi),beta=str(beta),beta_float=float(beta),absolute_error_below_256='1/128',normal_product_guard='each candidate p,q,z exponent[2,252] with positive finite seed domain; source remains normal via ratio',rounding='RNE only; gradual underflow/nontrapping/flags+errno unobserved required, no policy assumed by this screen',observations='Bounded [-128,127] integerRNE only, carrierfloat may differ',cases=reports,source_sha256=hashlib.sha256((w/'observer.c').read_bytes()).hexdigest(),library_sha256=hashlib.sha256((w/'observer.so').read_bytes()).hexdigest(),compiler_promotion=False,token_usage_available=False)
(w/'rational_native_screen.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in ['newton_relative_error_bounds','source_to_candidate_ratio_lower','source_to_candidate_ratio_upper','beta']}))
