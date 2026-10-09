"""Independent original ordered FMA and exact absolute-product admission."""
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from test_source_attention_frontier import PLAN


SOURCE = r'''
#include "exact_absolute_dot_bounds.h"
#include <fenv.h>
int compare(const float*a,const float*b,int m,int n,int k){
 double centers[1024],absolute[1024];if(m*n>1024)return -1;
 for(int r=0;r<m;r++)for(int c=0;c<n;c++){
  double s=0,t=0;for(int z=0;z<k;z++){
   double v=(double)a[r*k+z]*(double)b[c*k+z];s+=v;t+=fabs(v);
  }centers[r*n+c]=s;absolute[r*n+c]=t;
 }
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_fma_zero_gamma_plan g=merlin_fma_zero_gamma_prepare(&env,k);
 merlin_fma_zero_gamma_batch gamma=merlin_fma_zero_gamma_batch_prepare(&g);
 merlin_exact_absolute_dot_bounds p=merlin_exact_absolute_dot_prepare(&gamma,centers,absolute,m*n,k);
 if(!p.valid)return 0;
 for(int r=0;r<m;r++)for(int c=0;c<n;c++){
  float lo,hi,s=0;for(int z=0;z<k;z++)s=fmaf(a[r*k+z],b[c*k+z],s);
  if(!merlin_exact_absolute_dot_apply(&p,r*n+c,&lo,&hi)||!(lo<=s&&s<=hi))return -2;
 }
 return 1;
}
int refused(void){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_fma_zero_gamma_plan g=merlin_fma_zero_gamma_prepare(&env,1);
 merlin_fma_zero_gamma_batch gamma=merlin_fma_zero_gamma_batch_prepare(&g);
 double s=1,t=1;merlin_exact_absolute_dot_bounds p;
 p=merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1);if(!p.valid)return 0;
 float lo=42,hi=43;
 if(merlin_exact_absolute_dot_apply(&p,1,&lo,&hi)||lo!=42||hi!=43)return 0;
 if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,2).valid)return 0;
 t=-1;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 t=.5;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 t=NAN;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 t=INFINITY;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 t=1;s=NAN;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 s=0;t=FLT_MAX;if(merlin_exact_absolute_dot_prepare(&gamma,&s,&t,1,1).valid)return 0;
 int saved=fegetround(),modes[]={FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};
 for(int i=0;i<3;i++){fesetround(modes[i]);env=merlin_fma_bound_begin();
  g=merlin_fma_zero_gamma_prepare(&env,1);if(g.valid){fesetround(saved);return 0;}}
 fesetround(saved);return 1;
}
'''


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc = shutil.which('clang') or shutil.which('cc')
    if not cc:
        pytest.skip('native compiler required')
    root = tmp_path_factory.mktemp('absolute-dot')
    (root / 'test.c').write_text(SOURCE)
    subprocess.run([cc, '-O2', '-fno-fast-math', '-ffp-contract=off', '-shared',
                    '-fPIC', '-I', str(merlin_dir() / 'runtime/c'),
                    str(root / 'test.c'), '-lm', '-o', str(root / 'test.so')], check=True)
    lib = ctypes.CDLL(str(root / 'test.so'))
    lib.compare.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return lib


@pytest.mark.parametrize('shape', [(1, 1, 1), (3, 5, 7), (2, 17, 31)])
@pytest.mark.parametrize('case', ['zero', 'signed', 'cancellation', 'tiny', 'rounding', 'overflow'])
def test_exact_sums_enclose_original_fma(native, shape, case):
    m, n, k = shape
    rng = np.random.default_rng(4113)
    a = rng.integers(-128, 128, (m, k)).astype(np.float32) / 16
    b = rng.integers(-128, 128, (n, k)).astype(np.float32) / 16
    if case == 'zero':
        a[:] = -0.
    if case == 'cancellation':
        a[:, ::2] = 1
        a[:, 1::2] = -1
        b[:] = 1
    if case == 'tiny':
        a[:] = np.float32(2**-149)
        b[:] = .5
    if case == 'rounding':
        a = rng.integers(-(2**20), 2**20, (m, k)).astype(np.float32) / 65536
        b = rng.integers(-(2**20), 2**20, (n, k)).astype(np.float32) / 65536
    if case == 'overflow':
        a[:] = np.finfo(np.float32).max
        b[:] = 2
    assert native.compare(a.ctypes.data, b.ctypes.data, m, n, k) == (0 if case == 'overflow' else 1)


def test_invalid_arrays_environment_and_domain_refuse(native):
    assert native.refused() == 1


def test_default_source_and_explicit_proof_requirements():
    assert emit_source_attention_frontier(PLAN, symbol='p') == emit_source_attention_frontier(
        PLAN, symbol='p', prepare_absolute_products=False)
    with pytest.raises(ValueError, match='canonical integer and encoded-row'):
        emit_source_attention_frontier(PLAN, symbol='p', prepare_absolute_products=True)
    with pytest.raises(ValueError, match='policy must be bool'):
        emit_source_attention_frontier(PLAN, symbol='p', prepare_absolute_products=1)
    with pytest.raises(ValueError, match='not yet admitted'):
        emit_source_attention_frontier(PLAN, symbol='p', prepare_absolute_products=True,
                                      prepare_encoded_rows=True, integer_reconstruction=True,
                                      prepare_softmax_spans=True)
