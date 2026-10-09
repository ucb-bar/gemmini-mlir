"""Directed binary32 radius versus exact original source FMA execution."""
import ctypes as C
import shutil
import subprocess
import numpy as np
import pytest
from merlin.common.paths import merlin_dir

PREFIX=r'''
#include <fenv.h>
#include <math.h>
#pragma STDC FENV_ACCESS ON
static float cu(double x){int m=fegetround();fesetround(FE_UPWARD);volatile double a=x;volatile float r=(float)a;fesetround(m);return r;}
static float cd(double x){int m=fegetround();fesetround(FE_DOWNWARD);volatile double a=x;volatile float r=(float)a;fesetround(m);return r;}
static float au(float a,float b){int m=fegetround();fesetround(FE_UPWARD);volatile float r=a+b;fesetround(m);return r;}
static float ad(float a,float b){int m=fegetround();fesetround(FE_DOWNWARD);volatile float r=a+b;fesetround(m);return r;}
static float fu(float a,float b,float c){int m=fegetround();fesetround(FE_UPWARD);volatile float r=fmaf(a,b,c);fesetround(m);return r;}
#pragma STDC FENV_ACCESS OFF
#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) cu(x)
#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) cd(x)
#define MERLIN_F32_RADIUS_FMA_UP(a,b,c) fu(a,b,c)
#define MERLIN_F32_RADIUS_ADD_UP(a,b) au(a,b)
#define MERLIN_F32_RADIUS_ADD_DOWN(a,b) ad(a,b)
#include "binary32_separable_radius.h"
int compare(const float*a,const float*b,int k){
 double center=0,l1=0,maximum=0;float ordered=0;
 for(int i=0;i<k;i++){center+=(double)a[i]*b[i];l1=merlin_fma_up_add(l1,fabs((double)a[i]));maximum=fmax(maximum,fabs((double)b[i]));ordered=fmaf(a[i],b[i],ordered);}
 merlin_fma_bound e=merlin_fma_bound_begin();
 merlin_fma_zero_gamma_plan g=merlin_fma_zero_gamma_prepare(&e,k);
 if(!g.valid)return 0;
 merlin_fma_zero_gamma_batch gb=merlin_fma_zero_gamma_batch_prepare(&g);
 merlin_admitted_dot_norms norm={0};norm.maximum=maximum;
 merlin_fma_exact_columns cols={&norm,1,k,maximum,1};
 double factor=merlin_fma_up_mul(gb.gamma_upper,l1);
 double envelope=merlin_fma_up_add(merlin_fma_up_mul(l1,maximum),merlin_fma_up_add(merlin_fma_up_mul(factor,maximum),gb.subnormal_error_upper));
 if(!isfinite(envelope)||envelope>=FLT_MAX)return 0;
 merlin_fma_separable_radius s={&cols,&center,factor,gb.subnormal_error_upper,1};
 merlin_binary32_separable_radius p=merlin_binary32_radius_prepare(&s);
 float lo,hi;if(!p.valid)return 0;
 if(!merlin_binary32_radius_apply(&p,0,&lo,&hi))return -1;
 return lo<=ordered&&ordered<=hi&&fegetround()==FE_TONEAREST?1:-2;
}
'''
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc:pytest.skip('compiler required')
    w=tmp_path_factory.mktemp('f32radius');(w/'p.c').write_text(PREFIX)
    subprocess.run([cc,'-O2','-ffp-contract=off','-frounding-math','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(w/'p.c'),'-lm','-o',str(w/'p.so')],check=True)
    lib=C.CDLL(str(w/'p.so'));lib.compare.argtypes=[C.c_void_p,C.c_void_p,C.c_int];return lib

@pytest.mark.parametrize('k',[1,3,17,64,192])
@pytest.mark.parametrize('exponent',[-140,-80,-1,0,40,120])
def test_source_order_and_finite_refusal(native,k,exponent):
    rng=np.random.default_rng(591+k)
    # Integer coefficients and common powers ensure exact real binary64 sums.
    a=np.ldexp(rng.integers(-16,17,k).astype(np.float32),exponent)
    b=np.ldexp(rng.integers(-16,17,k).astype(np.float32),-2)
    expected=0 if exponent==120 else 1
    result=native.compare(a.ctypes.data,b.ctypes.data,k)
    assert result in (0,1) if exponent==120 else result==expected
