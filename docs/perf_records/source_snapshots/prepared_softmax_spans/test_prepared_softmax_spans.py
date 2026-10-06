"""Private producer coverage, identity, epochs and one-use consumption."""
import ctypes
import shutil
import subprocess
import pytest
from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from merlin.llvmlower.prepared_softmax_spans import prepare_softmax_produced_spans
from test_source_attention_frontier import PLAN

C=r'''
#include "prepared_softmax_spans.h"
int probe(int test){
 float lo[12]={0},hi[12]={0};unsigned char mask[12]={0},epoch,other;
 merlin_softmax_produced_spans p=merlin_softmax_spans_begin(lo,hi,mask,2,3,&epoch);
 if(test==0){
  if(!merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,0,&epoch))return 0;
  if(!merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,1,&epoch))return 0;
  if(!merlin_softmax_spans_consume(&p,lo,hi,mask,12,&epoch))return 0;
  return !merlin_softmax_spans_consume(&p,lo,hi,mask,12,&epoch);
 }
 if(test==1)return !merlin_softmax_spans_consume(&p,lo,hi,mask,12,&epoch);
 if(test==2)return !merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,1,&epoch)&&!p.valid;
 if(!merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,0,&epoch))return 0;
 if(test==3)return !merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,0,&epoch)&&!p.valid;
 if(test==4)return !merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,1,&other)&&!p.valid;
 if(!merlin_softmax_spans_record_tile(&p,lo,hi,mask,2,3,1,&epoch))return 0;
 if(test==5)return !merlin_softmax_spans_consume(&p,lo+1,hi,mask,12,&epoch);
 if(test==6)return !merlin_softmax_spans_consume(&p,lo,hi,mask+1,12,&epoch);
 if(test==7)return !merlin_softmax_spans_consume(&p,lo,hi,mask,11,&epoch);
 if(test==8)return !merlin_softmax_spans_consume(&p,lo,hi,mask,12,&other);
 if(test==9)return !merlin_softmax_spans_begin(lo,hi,mask,SIZE_MAX,3,&epoch).valid;
 return !merlin_softmax_spans_begin(lo,hi,mask,2,0,&epoch).valid;
}
'''
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang')or shutil.which('cc')
    if not cc:pytest.skip('C compiler required')
    w=tmp_path_factory.mktemp('producer-spans');(w/'proof.c').write_text(C)
    subprocess.run([cc,'-O2','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(w/'proof.c'),'-o',str(w/'proof.so')],check=True)
    return ctypes.CDLL(str(w/'proof.so'))
@pytest.mark.parametrize('case',range(11))
def test_private_coverage_identity_epoch(native,case):
    assert native.probe(case)==1
@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_explicit_policy(bad):
    with pytest.raises(ValueError,match='spans policy'):
        emit_source_attention_frontier(PLAN,symbol='test',prepare_softmax_spans=bad)
def test_requires_complete_producer_domain():
    with pytest.raises(ValueError,match='domain and source-radius'):
        emit_source_attention_frontier(PLAN,symbol='test',prepare_softmax_spans=True)
@pytest.mark.parametrize('old,new',[
 ('if(value>1)return 0;h->mask','h->mask'),
 ('if(!evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque))return 0;','evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque);'),
 ('h->qhi[r*KEYS+tile*CHUNK+j]=w->upper[r*CHUNK+j];','h->qhi[r*KEYS+tile*CHUNK+j]=0;'),
 ('for(int tile=0;tile<2;tile++){\n   memcpy','for(int tile=0;tile<1;tile++){\n   memcpy'),
])
def test_changed_producer_refused(old,new):
    source=emit_source_attention_frontier(PLAN,symbol='test',prepare_product_domain=True,separable_source_radius=True,prepare_softmax_domain=True)
    assert old in source
    with pytest.raises(ValueError,match='grammar changed'):
        prepare_softmax_produced_spans(source.replace(old,new))
