"""Four independent cells preserve scalar endpoint bits and checked refusal."""
import ctypes
import random
import shutil
import subprocess
import pytest
from merlin.common.paths import merlin_dir

C=r'''
#include <fenv.h>
#include "prepared_polynomial_batch.h"
int probe(const float *a,const float *v,int mode,int invalid){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 int old=fegetround();fesetround(modes[mode]);
 merlin_bit_polynomial_plan source={a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&source,0);
 if(mode==0&&!p.fast_valid){fesetround(old);return -1;}
 merlin_f32_interval x[4],out[4];
 for(int i=0;i<4;i++)x[i]=merlin_interval(v[2*i],v[2*i+1]);
 if(invalid)x[2].valid=0;
 merlin_polynomial_words_four(x,&p,out);
 int pass=1;
 for(int i=0;i<4;i++){
  merlin_f32_interval scalar=merlin_monotone_bit_polynomial_apply_words(x[i],&p);
  if(scalar.valid!=out[i].valid||merlin_interval_bits(scalar.lo)!=merlin_interval_bits(out[i].lo)||merlin_interval_bits(scalar.hi)!=merlin_interval_bits(out[i].hi))pass=0;
 }
 if(fegetround()!=modes[mode])pass=0;
 fesetround(old);return pass;
}
'''
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    w=tmp_path_factory.mktemp('polynomial-four');(w/'test.c').write_text(C)
    cc=shutil.which('clang')or shutil.which('cc')
    subprocess.run([cc,'-O2','-frounding-math','-ffp-contract=off','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(w/'test.c'),'-lm','-o',str(w/'test.so')],check=True)
    lib=ctypes.CDLL(str(w/'test.so'));lib.probe.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int];return lib
@pytest.mark.parametrize('mode',range(4))
@pytest.mark.parametrize('invalid',[0,1])
def test_scalar_identity(native,mode,invalid):
    plan=(ctypes.c_float*8)(-87.3365478515625,1.4426950216293335,-.079204238951206207,-.22433836758136749,.30354261398315430,.00010703434963943437,8388608,1065353216)
    cases=[[-81,-80,-80,-79.99,-0.,0.,0.,0.],[-1,-.999999,-2,-1.999999,-10,-9.999,-70,-69.9],[float('nan'),0,float('-inf'),0,0,float('inf'),1,-1]]
    rng=random.Random(921)
    for _ in range(2000):
        v=[]
        for _ in range(4):
            lo=rng.uniform(-82,0);v.extend((lo,min(0,lo+rng.choice([0,1e-6,.01,.5]))))
        cases.append(v)
    for v in cases:assert native.probe(plan,(ctypes.c_float*8)(*v),mode,invalid)==1,v

@pytest.mark.parametrize('bad',[None,1,'yes'])
def test_explicit_bool_refusal(bad):
    from test_source_attention_frontier import PLAN
    from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
    with pytest.raises(ValueError,match='must be bool'):
        emit_source_attention_frontier(PLAN,symbol='test',polynomial_batch_four=bad)

def test_source_schedule_refusal():
    from merlin.llvmlower.prepared_polynomial_batch import prepare_polynomial_batch_four
    with pytest.raises(ValueError):prepare_polynomial_batch_four('unknown source')
