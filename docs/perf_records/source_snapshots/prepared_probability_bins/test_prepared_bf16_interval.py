"""Exact private BF16 endpoint snapshots and source midpoint reuse."""
import ctypes
import shutil
import subprocess

import pytest
from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from test_source_attention_frontier import PLAN

C = r'''
#include <fenv.h>
#include "prepared_bf16_interval.h"
int check(float lo,float hi){
 merlin_f32_interval x={lo,hi,1};
 merlin_bf16_interval_bins b=merlin_bf16_interval_prepare(x);
 float wanted=merlin_interval_bf16((float)(((double)lo+hi)*.5));
 float got=merlin_bf16_interval_midpoint(x,b);
 return merlin_interval_bits(wanted)==merlin_interval_bits(got)&&
 b.low_bits==merlin_interval_bits(merlin_interval_bf16(lo))&&
 b.high_bits==merlin_interval_bits(merlin_interval_bf16(hi));
}
int probe(void){
 int old=fegetround(),modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint32_t seed=1729;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(unsigned word=0;word<65536;word++){
   uint32_t base=word<<16;
   for(int delta=-2;delta<=2;delta++){
    float a=merlin_interval_float(base+(uint32_t)delta),b=merlin_interval_float(base+32768);
    if(!check(a,b)||!check(b,a))return 2;
   }
  }
  for(int i=0;i<20000;i++){
   seed=1664525*seed+1013904223;float a=merlin_interval_float(seed);
   seed=1664525*seed+1013904223;float b=merlin_interval_float(seed);
   if(!check(a,b))return 3;
  }
  if(!check(-0.f,0.f)||!check(-0.f,-0.f)||!check(0.f,-0.f))return 4;
 }
 return fesetround(old)?5:0;
}
'''

def test_original_midpoint_raw_corpus(tmp_path):
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc:pytest.skip('C compiler required')
    source=tmp_path/'probe.c';source.write_text(C)
    out=tmp_path/'probe.so'
    subprocess.run([cc,'-O2','-frounding-math','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(source),'-lm','-o',str(out)],check=True)
    assert ctypes.CDLL(str(out)).probe()==0


def test_default_identity_and_refinement_lifecycle():
    default=emit_source_attention_frontier(PLAN,symbol='test')
    assert default==emit_source_attention_frontier(PLAN,symbol='test',prepare_probability_bins=False)
    selected=emit_source_attention_frontier(PLAN,symbol='test',prepare_probability_bins=True,prepare_softmax_domain=True)
    assert selected.count('bins=merlin_bf16_interval_prepare(y); counts[2]++;')==2
    assert selected.count('merlin_bf16_interval_midpoint(y,bins)')==2

@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_explicit_boolean_refusal(bad):
    with pytest.raises(ValueError,match='probability bin'):
        emit_source_attention_frontier(PLAN,symbol='test',prepare_probability_bins=bad)
