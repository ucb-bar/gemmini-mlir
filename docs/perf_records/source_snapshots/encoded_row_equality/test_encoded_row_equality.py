"""Immutable encoder row identity and exact norm reuse refusals."""
import ctypes
import shutil
import subprocess
import pytest
from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from test_source_attention_frontier import PLAN

C=r'''
#include "encoded_row_equality.h"
#include "representation_error_norms.h"
int probe(int test){
 float source[6]={-0.f,2,-3,4,5,6},encoded[6]={0,2,-3,4,5,6};
 float lower[6],upper[6];double output[6];unsigned char flags[2]={9,9};
 memcpy(lower,source,sizeof(source));memcpy(upper,source,sizeof(source));
 merlin_fma_bound env=merlin_fma_bound_begin();
 if(test==1)encoded[4]=7;
 if(test==2)upper[4]=7;
 if(test==3)source[4]=NAN;
 if(test==4)encoded[4]=INFINITY;
 if(test==5)env.valid=0;
 merlin_encoded_row_equality p=merlin_encoded_rows_widen(&env,source,encoded,lower,upper,output,flags,2,3);
 if(test==5)return !p.valid;
 if(!p.valid||flags[0]!=1||flags[1]!=(test==0))return 0;
 if(!merlin_encoded_row_matches(&p,0,source,output,lower,upper,3))return 0;
 if(merlin_encoded_row_matches(&p,0,source+1,output,lower,upper,3)||
    merlin_encoded_row_matches(&p,0,source,output+1,lower,upper,3)||
    merlin_encoded_row_matches(&p,0,source,output,upper,lower,3)||
    merlin_encoded_row_matches(&p,0,source,output,lower,upper,2)||
    merlin_encoded_row_matches(&p,2,source,output,lower,upper,3))return 0;
 for(int i=0;i<6;i++)if(!isnan(encoded[i])&&output[i]!=(double)encoded[i])return 0;
 if(test==0){
  merlin_dot_norms a=merlin_dot_norms_begin(&env),r=a,e=a;
  for(int i=0;i<3;i++){merlin_dot_norms_add(&a,fabs(source[i]));merlin_dot_norms_add(&r,fabs(output[i]));merlin_representation_error_add(&e,source[i],output[i]);}
  merlin_dot_norms_finish(&a);merlin_dot_norms_finish(&r);merlin_representation_error_finish(&e);
  if(a.l1!=r.l1||a.maximum!=r.maximum||a.l2!=r.l2||e.l1!=0||e.maximum!=0||e.l2!=0)return 0;
 }
 return !merlin_encoded_rows_widen(&env,source,encoded,lower,0,output,flags,2,3).valid&&
 !merlin_encoded_rows_widen(&env,source,encoded,0,0,output,flags,SIZE_MAX,2).valid;
}
'''
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang')or shutil.which('cc')
    if not cc:pytest.skip('C compiler required')
    w=tmp_path_factory.mktemp('encoded-equality');(w/'proof.c').write_text(C)
    subprocess.run([cc,'-O2','-shared','-fPIC','-ffp-contract=off','-fno-fast-math','-I',str(merlin_dir()/'runtime/c'),str(w/'proof.c'),'-lm','-o',str(w/'proof.so')],check=True)
    return ctypes.CDLL(str(w/'proof.so'))
@pytest.mark.parametrize('case',range(6))
def test_equality_uncertainty_and_identity(native,case):
    assert native.probe(case)==1
@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_boolean_policy(bad):
    with pytest.raises(ValueError,match='encoded row'):
        emit_source_attention_frontier(PLAN,symbol='test',prepare_encoded_rows=bad)
def test_requires_admitted_norms():
    with pytest.raises(ValueError,match='norm requirements'):
        emit_source_attention_frontier(PLAN,symbol='test',prepare_encoded_rows=True)
