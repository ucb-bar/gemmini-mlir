"""Private finite positive schedule admission and exact source arithmetic."""
import ctypes
import math
import random
import shutil
import subprocess

import pytest
from merlin.common.paths import merlin_dir

SOURCE = r'''
#include "prepared_softmax_interval.h"
#include "positive_scalar_interval.h"
static merlin_bit_polynomial_plan source(const float *a) {
 return (merlin_bit_polynomial_plan){a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
}
int domain(const float *a,float scale,size_t chunk,size_t lanes,size_t chunks,float *out){
 merlin_fma_bound e=merlin_fma_bound_begin();merlin_bit_polynomial_plan s=source(a);
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,0);
 merlin_prepared_softmax_domain d=merlin_softmax_domain_prepare(&e,&p,scale,chunk,lanes,chunks);
 out[0]=d.probability_upper;out[1]=d.denominator_upper;return d.valid;
}
int check(const float *a,float lo,float hi,float *out){
 merlin_fma_bound e=merlin_fma_bound_begin();merlin_bit_polynomial_plan s=source(a);
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,0);
 merlin_f32_interval y=merlin_monotone_bit_polynomial_apply(merlin_interval(lo,hi),&p);
 out[0]=y.lo;out[1]=y.hi;return y.valid;
}
int word_domain(const float *a,float scale,size_t chunk,size_t lanes,size_t chunks,float *out){
 merlin_fma_bound e=merlin_fma_bound_begin();merlin_bit_polynomial_plan s=source(a);
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,0);
 merlin_prepared_softmax_domain d=merlin_softmax_word_domain_prepare(&e,&p,scale,chunk,lanes,chunks);
 out[0]=d.probability_upper;out[1]=d.denominator_upper;return d.valid;
}
int word_check(const float *a,float lo,float hi,float *out){
 merlin_fma_bound e=merlin_fma_bound_begin();merlin_bit_polynomial_plan s=source(a);
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,0);
 merlin_f32_interval y=merlin_monotone_bit_polynomial_apply_words(merlin_interval(lo,hi),&p);
 out[0]=y.lo;out[1]=y.hi;return y.valid;
}
int word_overflow(const float *a){
 merlin_fma_bound e=merlin_fma_bound_begin();merlin_bit_polynomial_plan s=source(a);
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,0);
 p.word_budget=UINT32_MAX;
 return merlin_softmax_word_domain_prepare(&e,&p,.125f,8,8,2).valid;
}
int admission(const float *lo,const float *hi,const unsigned char *mask,size_t n){
 merlin_prepared_softmax_domain d={0};d.valid=1;
 return merlin_softmax_admit_active_spans(&d,lo,hi,mask,n);
}
int alpha(float x){return merlin_softmax_admit_alpha(x);}
int mode(int x){return fesetround(x?FE_DOWNWARD:FE_TONEAREST);}
int compare(float lo,float hi,float mx,float scale){
 merlin_prepared_softmax_domain d={0};d.scale=scale;
 merlin_soft_interval a=merlin_softmax_score(&d,lo,hi,mx);
 merlin_f32_interval b=merlin_interval_sub(merlin_interval_positive_scale(merlin_interval(lo,hi),scale),merlin_interval_point(mx));
 return b.valid&&merlin_interval_bits(a.lo)==merlin_interval_bits(b.lo)&&merlin_interval_bits(a.hi)==merlin_interval_bits(b.hi);
}
'''
PLANS = [
 [-87.3365478515625,1.4426950216293335,-.079204238951206207,-.22433836758136749,.3035426139831543,.00010703434963943437,8388608.,1065353216.],
 [-40.,.5,.02,-.1,.2,.004,8388608.,1065353216.],
 [-32.,.75,-.01,-.02,.1,.001,4194304.,1065353216.],
]

@pytest.fixture(scope='module')
def native(tmp_path_factory):
 cc=shutil.which('clang') or shutil.which('cc')
 if not cc: pytest.skip('native compiler required')
 w=tmp_path_factory.mktemp('private-soft');(w/'test.c').write_text(SOURCE)
 subprocess.run([cc,'-O2','-fno-fast-math','-ffp-contract=off','-frounding-math','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(w/'test.c'),'-lm','-o',str(w/'test.so')],check=True)
 lib=ctypes.CDLL(str(w/'test.so'));f=ctypes.c_float;p=ctypes.POINTER(f);z=ctypes.c_size_t
 lib.domain.argtypes=[p,f,z,z,z,p];lib.check.argtypes=[p,f,f,p]
 lib.word_domain.argtypes=[p,f,z,z,z,p];lib.word_check.argtypes=[p,f,f,p];lib.word_overflow.argtypes=[p]
 lib.admission.argtypes=[p,p,ctypes.POINTER(ctypes.c_ubyte),z]
 lib.alpha.argtypes=[f];lib.compare.argtypes=[f,f,f,f]
 return lib

@pytest.mark.parametrize('plan',PLANS)
def test_uniform_bound_contains_checked_polynomial_intervals(native,plan):
 a=(ctypes.c_float*8)(*plan);out=(ctypes.c_float*2)()
 assert native.domain(a,.125,513,8,2,out)
 upper,denominator=out[:];assert 0<upper<=denominator<math.inf
 rng=random.Random(511)
 points=[plan[0]-1,plan[0],-2**-149,-2**-126,-0.,0.]
 for _ in range(1000): points.append(rng.uniform(plan[0]-1,0))
 for lo in points:
  for hi in [lo,min(0,lo+.00001),0.]:
   assert native.check(a,lo,hi,out)
   assert 0<=out[0]<=out[1]<=upper

@pytest.mark.parametrize('scale,chunk,lanes,chunks',[(0,8,2,2),(-1,8,2,2),(1,8,2,2),(math.inf,8,2,2),(.125,0,2,2),(.125,8,3,2),(.125,8,2,0)])
def test_refuses_unknown_or_overflowing_domain(native,scale,chunk,lanes,chunks):
 assert not native.domain((ctypes.c_float*8)(*PLANS[0]),scale,chunk,lanes,chunks,(ctypes.c_float*2)())

def test_rounding_mode_requires_fresh_rne_admission(native):
 native.mode(1)
 try: assert not native.domain((ctypes.c_float*8)(*PLANS[0]),.125,8,2,2,(ctypes.c_float*2)())
 finally: native.mode(0)

def test_active_spans_masks_and_alpha(native):
 f=ctypes.c_float;u=ctypes.c_ubyte
 assert native.admission((f*3)(-0.,math.nan,-1),(f*3)(0,math.inf,2),(u*3)(1,0,1),3)
 for lo,hi,mask in [(math.nan,0,1),(0,math.inf,1),(1,0,1),(0,0,2)]:
  assert not native.admission((f*1)(lo),(f*1)(hi),(u*1)(mask),1)
 for a in [-0.,0,2**-149,.5,1]:assert native.alpha(a)
 for a in [-2**-149,1.01,math.nan,math.inf]:assert not native.alpha(a)

def test_source_score_bits_signed_zero_and_underflow(native):
 for lo,hi,mx in [(-0.,0.,0.),(-2**-149,2**-149,0.),(-1,1,.5),(-1e30,1e30,1e29)]:
  assert native.compare(lo,hi,mx,.125)

def test_option_is_explicit_and_word_policy_separately_admitted():
 from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
 from test_source_attention_frontier import PLAN
 assert emit_source_attention_frontier(PLAN,symbol='f')==emit_source_attention_frontier(PLAN,symbol='f',prepare_softmax_domain=False)
 for invalid in [0,1,None,'yes']:
  with pytest.raises(ValueError):emit_source_attention_frontier(PLAN,symbol='f',prepare_softmax_domain=invalid)
 generated=emit_source_attention_frontier(PLAN,symbol='f',prepare_softmax_domain=True,word_interval_enclosure=True)
 assert 'merlin_softmax_word_domain_prepare(' in generated
 assert 'soft_details_checked(' in generated

@pytest.mark.parametrize('plan',PLANS)
def test_word_enclosure_complete_domain_and_overflow(native,plan):
 a=(ctypes.c_float*8)(*plan);out=(ctypes.c_float*2)()
 assert native.word_domain(a,.125,513,8,2,out)
 upper,denominator=out[:]
 assert 0<upper<=denominator<math.inf
 rng=random.Random(431)
 points=[plan[0]-1,plan[0],-2**-149,-2**-126,-0.,0.]
 points += [rng.uniform(plan[0]-1,0) for _ in range(1000)]
 for lo in points:
  for hi in [lo,min(0,lo+.00001),0.]:
   assert native.word_check(a,lo,hi,out)
   assert 0<=out[0]<=out[1]<=upper
 assert not native.word_overflow(a)
 native.mode(1)
 try:assert not native.word_domain(a,.125,8,2,2,out)
 finally:native.mode(0)
 for scale in [0,-1,math.nan,math.inf,1]:
  assert not native.word_domain(a,scale,8,2,2,out)
