"""One source evaluation must enclose independent interior source words."""
import ctypes as C
from fractions import Fraction
import math
import random
import shutil
import subprocess

import pytest
from merlin.common.paths import merlin_dir

PLANS = [
    [-87.3365478515625,1.4426950216293335,-.079204238951206207,-.22433836758136749,.3035426139831543,.00010703434963943437,8388608.,1065353216.],
    [-40.,.5,.02,-.1,.2,.004,8388608.,1065353216.],
    [-32.,.75,-.01,-.02,.1,.001,4194304.,1065353216.],
    [-32.,1.,0.,0.,.0625,0.,1048576.,1065353216.],
]
SOURCE = r'''
#include "one_endpoint_word_polynomial.h"
static merlin_bit_polynomial_plan plan(const float *a){
 return (merlin_bit_polynomial_plan){a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
}
void bound(const float*a,float upper,float lo,float hi,uint32_t*out){
 merlin_bit_polynomial_plan s=plan(a);merlin_fma_bound e=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&s,upper);
 merlin_one_endpoint_word_polynomial n=merlin_one_endpoint_word_polynomial_prepare(&p);
 merlin_f32_interval x=merlin_interval(lo,hi);
 merlin_f32_interval y=merlin_one_endpoint_word_polynomial_apply(x,&n);
 merlin_f32_interval old=merlin_monotone_bit_polynomial_apply_words(x,&p);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);
 out[2]=y.valid;out[3]=p.fast_valid;out[4]=n.valid;out[5]=n.source_upper_word;
 out[6]=merlin_interval_bits(old.lo);out[7]=merlin_interval_bits(old.hi);out[8]=old.valid;
}
uint32_t original(const float*a,float x){
 merlin_bit_polynomial_plan p=plan(a);if(x<p.cutoff)return 0;
 float s=x*p.scale,f=s-floorf(s),v=p.coefficients[0];
 for(int i=1;i<4;i++)v=fmaf(f,v,p.coefficients[i]);
 return (uint32_t)(int32_t)fmaf(p.bit_multiplier,s-v,p.bit_bias);
}
uint32_t distance(uint32_t d,int s){return merlin_polynomial_word_distance_bound(d,s);}
int down(void){return fesetround(FE_DOWNWARD);}
int nearest(void){return fesetround(FE_TONEAREST);}
'''


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc:pytest.skip('native C compiler required')
    root=tmp_path_factory.mktemp('one-endpoint-word')
    (root/'test.c').write_text(SOURCE)
    subprocess.run([cc,'-O2','-fno-fast-math','-ffp-contract=off','-frounding-math',
        '-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(root/'test.c'),
        '-lm','-o',str(root/'test.so')],check=True)
    lib=C.CDLL(str(root/'test.so'));f=C.c_float;u=C.c_uint32;p=C.POINTER(f)
    lib.bound.argtypes=[p,f,f,f,C.POINTER(u)]
    lib.original.argtypes=[p,f];lib.original.restype=u
    lib.distance.argtypes=[u,C.c_int];lib.distance.restype=u
    return lib


def bounds(native,plan,lo,hi,upper=0):
    a=(C.c_float*8)(*plan);out=(C.c_uint32*9)()
    native.bound(a,upper,lo,hi,out)
    return a,list(out)


@pytest.mark.parametrize('plan',PLANS)
def test_encloses_independent_interior_source_arithmetic(native,plan):
    rng=random.Random(4031)
    for _ in range(1000):
        lo=rng.uniform(plan[0]-1,0);hi=min(0,lo+rng.choice([0,1e-7,1e-5,.002,.3,2]))
        a,b=bounds(native,plan,lo,hi)
        assert b[2:5]==[1,1,1]
        lo,hi=C.c_float(lo).value,C.c_float(hi).value
        for x in [lo,hi]+[rng.uniform(lo,hi) for _ in range(24)]:
            word=native.original(a,x)
            assert b[0]<=word<=b[1],(lo,hi,x,b,word)
            assert word<=b[5]


@pytest.mark.parametrize('plan',PLANS)
def test_floor_cutoff_sign_zero_and_subnormal_domains(native,plan):
    points=[plan[0],-0.,0.,-2.**-149,2.**-149,-2.**-126,2.**-126]
    points += [i/plan[1] for i in range(math.ceil(plan[0]*plan[1]),5)]
    for x in points:
        for width in [0,2.**-149,1e-7,1e-5]:
            lo=max(plan[0]-1,x-width);hi=min(8,x+width)
            a,b=bounds(native,plan,lo,hi,upper=8)
            assert b[2:5]==[1,1,1]
            for t in [lo,hi,x]:
                if lo<=t<=hi:assert b[0]<=native.original(a,t)<=b[1]


@pytest.mark.parametrize('change',[(4,1.25),(2,-.5),(6,-1.),(1,0.)])
def test_unproved_plan_preserves_checked_refusal(native,change):
    p=PLANS[0].copy();p[change[0]]=change[1]
    _,b=bounds(native,p,-1,-.5)
    assert not b[3] and not b[4] and b[:3]==b[6:9]


def test_invalid_domain_and_rounding(native):
    for lo,hi in [(math.nan,0),(0,math.nan),(1,0),(-1,.1)]:
        _,b=bounds(native,PLANS[0],lo,hi)
        assert b[:3]==b[6:9]
    try:
        assert native.down()==0
        _,b=bounds(native,PLANS[0],-1,-.5)
        assert b[2:5]==[0,0,0]
    finally:assert native.nearest()==0


def test_distance_bound_with_independent_rationals(native):
    for d in [0,1,3,127,2**16-1,2**31,2**32-1]:
        for shift in [-1200,-32,-31,-1,0,1,30,31,1200]:
            exact=Fraction(d)*Fraction(2)**shift
            expected=min(2139095039,math.ceil(exact))
            assert native.distance(d,shift)==expected
