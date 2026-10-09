"""Exact benchmark checks preserve arbitrary alignment, guards and byte tails."""
import os
import shutil
import subprocess

import pytest

from merlin.common.paths import merlin_dir

SOURCE = r'''
#include "benchmark_buffer.h"
#include <assert.h>
static size_t oracle(const unsigned char *a,const unsigned char *b,size_t n){
 for(size_t i=0;i<n;i++)if(a[i]!=b[i])return i;
 return n;
}
int main(void){
 unsigned char a[320],b[320],saved[320];
 assert(merlin_benchmark_first_difference(0,0,0)==0);
 merlin_benchmark_fill(0,123,0);
 for(size_t ax=0;ax<16;ax++)for(size_t bx=0;bx<16;bx++){
  for(size_t i=0;i<320;i++)a[i]=(unsigned char)(i*73+19);
  for(size_t n=0;n<=257;n++){
   for(size_t i=0;i<320;i++)b[i]=0x5a;
   for(size_t i=0;i<n;i++)b[bx+i]=a[ax+i];
   memcpy(saved,b,sizeof(b));
   assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==n);
   assert(merlin_benchmark_first_difference(a+ax,a+ax,n)==n);
   assert(!memcmp(b,saved,sizeof(b)));
   for(size_t position=0;position<n;position++){
    b[bx+position]^=0x80;
    assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==position);
    assert(merlin_benchmark_first_difference(b+bx,a+ax,n)==position);
    assert(oracle(a+ax,b+bx,n)==position);
    if(position+1<n){
     b[bx+position+1]^=1;
     assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==position);
     b[bx+position+1]^=1;
    }
    b[bx+position]^=0x80;
   }
  }
 }
 for(unsigned value=0;value<256;value++)for(size_t offset=0;offset<16;offset++){
  for(size_t n=0;n<=257;n++){
   for(size_t i=0;i<320;i++)b[i]=0xa5;
   merlin_benchmark_fill(b+offset,(unsigned char)value,n);
   for(size_t i=0;i<320;i++)
    assert(b[i]==(i>=offset&&i<offset+n?(unsigned char)value:0xa5));
  }
 }
 return 0;
}
'''


def test_actual_exact_comparison_and_dirty_fill(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    source = tmp_path / "check.c"
    source.write_text(SOURCE)
    executable = tmp_path / "check"
    subprocess.run(
        [cc, "-O2", "-fno-builtin", "-I", str(merlin_dir() / "runtime/c"),
         str(source), "-o", str(executable)], check=True,
    )
    subprocess.run([str(executable)], check=True)
