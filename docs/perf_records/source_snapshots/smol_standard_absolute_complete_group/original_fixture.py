from dataclasses import replace
import os,subprocess,shutil
import pytest
from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_numeric_capability import SourceNumericContract,emit_source_numeric_capability
GOOD=SourceNumericContract(True,True,True,True,True,True,True)

def test_default_prefix_is_empty():
 assert emit_source_numeric_capability(GOOD)==''

@pytest.mark.parametrize('field',['round_to_nearest_even','fused_single_rounding','errno_unobserved','nontrapping','exception_flags_unobserved'])
def test_fma_contract_refuses(field):
 with pytest.raises(ValueError):emit_source_numeric_capability(replace(GOOD,**{field:False}),inline_fma=True)

@pytest.mark.parametrize('field',['standard_bitcast_copy','copy_interposition_unobserved'])
def test_copy_contract_refuses(field):
 with pytest.raises(ValueError):emit_source_numeric_capability(replace(GOOD,**{field:False}),inline_bitcasts=True)

CLASSIFICATION=replace(GOOD,standard_fp_classification=True,
                       classification_interposition_unobserved=True)

@pytest.mark.parametrize('field',['standard_fp_classification',
 'classification_interposition_unobserved','nontrapping','exception_flags_unobserved'])
def test_classification_contract_refuses(field):
 with pytest.raises(ValueError):
  emit_source_numeric_capability(replace(CLASSIFICATION,**{field:False}),inline_classification=True)

def test_existing_contract_does_not_admit_classification():
 with pytest.raises(ValueError):
  emit_source_numeric_capability(GOOD,inline_classification=True)

@pytest.mark.parametrize('choice',['inline_fma','inline_bitcasts','inline_classification','inline_absolute_values'])
def test_nonboolean_choice_refuses(choice):
 with pytest.raises(ValueError):
  emit_source_numeric_capability(CLASSIFICATION,**{choice:1})

SOURCE=r'''
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
static uint32_t bits(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static float value(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
int main(void){
 float (*volatile original)(float,float,float)=fmaf;
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint32_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(int i=0;i<10000;i++){
   rng=rng*1664525u+1013904223u;float a=value(rng&0xfeffffffu);
   rng=rng*1664525u+1013904223u;float b=value(rng&0xfeffffffu);
   rng=rng*1664525u+1013904223u;float c=value(rng&0xfeffffffu);
   float expected=original(a,b,c),actual=MERLIN_SOURCE_F32_FMA(a,b,c);
   if(bits(expected)!=bits(actual))return 2;
  }
 }
 unsigned char a[19],b[19];for(int i=0;i<19;i++)a[i]=(unsigned char)(13*i);
 for(int n=2;n<=8;n*=2)for(int off=0;off<8;off++){
  for(int i=0;i<19;i++)b[i]=0xa5;
  MERLIN_SOURCE_BITCAST_COPY(b+off,a+off,n);
  for(int i=0;i<19;i++)if(b[i]!=(i>=off&&i<off+n?a[i]:0xa5))return 3;
 }
 return 0;
}
'''
@pytest.mark.parametrize('fma,copy',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_native_rounding_and_unaligned_copy(tmp_path,fma,copy):
 cc=os.environ.get('MERLIN_CLANG') or shutil.which('clang')
 if not cc:pytest.skip('Clang builtin capability test requires Clang')
 c=tmp_path/'test.c';c.write_text(emit_source_numeric_capability(GOOD,inline_fma=fma,inline_bitcasts=copy)+SOURCE)
 exe=tmp_path/'test'
 subprocess.run([cc,'-O2','-fno-builtin','-fno-fast-math','-frounding-math','-ffp-contract=off','-I',str(merlin_dir()/'runtime/c'),str(c),'-lm','-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)

def test_capability_cannot_arrive_after_numeric_headers(tmp_path):
 cc=os.environ.get('MERLIN_CLANG') or shutil.which('clang')
 if not cc:pytest.skip('Clang capability test requires Clang')
 c=tmp_path/'late.c';c.write_text('#include "source_f32_math.h"\n'+emit_source_numeric_capability(GOOD,inline_fma=True)+'int main(void){return 0;}\n')
 result=subprocess.run([cc,'-I',str(merlin_dir()/'runtime/c'),'-c',str(c),'-o',str(tmp_path/'late.o')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 assert result.returncode!=0
 assert 'must precede numeric headers' in result.stderr

CLASSIFICATION_SOURCE=r'''
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
#include <errno.h>
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static int seen;
static double once(void){seen++;return 1;}
static int check32(uint32_t n){
 return !!MERLIN_SOURCE_ISFINITE(f32(n))==((n&0x7f800000u)!=0x7f800000u);
}
static int check64(uint64_t n){
 return !!MERLIN_SOURCE_ISFINITE(f64(n))==((n&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000));
}
int main(void){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint64_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  errno=123;
  /* Entire BF16 representation, including both signed zeros, infinities,
     quiet/signaling NaNs; no floating conversion supplies the oracle. */
  for(uint32_t i=0;i<65536;i++)if(!check32(i<<16))return 2;
  uint32_t mf[]={0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff};
  uint64_t md[]={0,1,UINT64_C(0x7ffffffffffff),UINT64_C(0x8000000000000),UINT64_C(0xffffffffffffe),UINT64_C(0xfffffffffffff)};
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<256;e++)for(unsigned m=0;m<6;m++)
   if(!check32((s<<31)|(e<<23)|mf[m]))return 3;
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<2048;e++)for(unsigned m=0;m<6;m++)
   if(!check64(((uint64_t)s<<63)|((uint64_t)e<<52)|md[m]))return 4;
  for(unsigned i=0;i<10000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!check32((uint32_t)rng)||!check64(rng))return 5;
  }
  seen=0;if(!MERLIN_SOURCE_ISFINITE(once())||seen!=1)return 6;
  if(fegetround()!=modes[mode]||errno!=123)return 7;
 }
 return 0;
}
'''

@pytest.mark.parametrize('inline',[False,True])
def test_classification_all_bf16_and_f32_f64_boundaries(tmp_path,inline):
 cc=os.environ.get('MERLIN_CLANG') or shutil.which('clang')
 if not cc:pytest.skip('Clang classification capability test requires Clang')
 c=tmp_path/'classification.c'
 c.write_text(emit_source_numeric_capability(CLASSIFICATION,inline_classification=inline)+CLASSIFICATION_SOURCE)
 exe=tmp_path/'classification'
 subprocess.run([cc,'-O2','-fno-builtin','-fno-fast-math','-frounding-math','-ffp-contract=off','-I',str(merlin_dir()/'runtime/c'),str(c),'-lm','-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)

ABSOLUTE=replace(GOOD,standard_absolute_value=True,
                absolute_value_interposition_unobserved=True,
                absolute_value_nan_payload_unobserved=True)

@pytest.mark.parametrize('field',['standard_absolute_value',
 'absolute_value_interposition_unobserved','absolute_value_nan_payload_unobserved',
 'errno_unobserved','nontrapping','exception_flags_unobserved'])
def test_absolute_value_contract_refuses(field):
 with pytest.raises(ValueError):
  emit_source_numeric_capability(replace(ABSOLUTE,**{field:False}),inline_absolute_values=True)

ABSOLUTE_SOURCE=r'''
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static uint32_t b32(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static uint64_t b64(double x){uint64_t n;__builtin_memcpy(&n,&x,8);return n;}
static float (*volatile source32)(float)=fabsf;
static double (*volatile source64)(double)=fabs;
static int check32(uint32_t n){
 uint32_t expected=n&UINT32_C(0x7fffffff);
 return b32(source32(f32(n)))==expected&&b32(MERLIN_SOURCE_F32_ABS(f32(n)))==expected;
}
static int check64(uint64_t n){
 uint64_t expected=n&UINT64_C(0x7fffffffffffffff);
 return b64(source64(f64(n)))==expected&&b64(MERLIN_SOURCE_F64_ABS(f64(n)))==expected;
}
static int seen;
static double once(void){seen++;return -1;}
int main(void){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint64_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(uint32_t i=0;i<65536;i++)if(!check32(i<<16))return 2;
  uint32_t mf[]={0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff};
  uint64_t md[]={0,1,UINT64_C(0x7ffffffffffff),UINT64_C(0x8000000000000),UINT64_C(0xffffffffffffe),UINT64_C(0xfffffffffffff)};
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<256;e++)for(unsigned m=0;m<6;m++)
   if(!check32((s<<31)|(e<<23)|mf[m]))return 3;
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<2048;e++)for(unsigned m=0;m<6;m++)
   if(!check64(((uint64_t)s<<63)|((uint64_t)e<<52)|md[m]))return 4;
  for(unsigned i=0;i<10000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!check32((uint32_t)rng)||!check64(rng))return 5;
  }
  seen=0;if(MERLIN_SOURCE_F64_ABS(once())!=1||seen!=1)return 6;
  if(fegetround()!=modes[mode])return 7;
 }
 return 0;
}
'''

@pytest.mark.parametrize('inline',[False,True])
def test_absolute_values_all_bf16_and_f32_f64_boundaries(tmp_path,inline):
 cc=os.environ.get('MERLIN_CLANG') or shutil.which('clang')
 if not cc:pytest.skip('Clang absolute-value capability test requires Clang')
 c=tmp_path/'absolute.c'
 c.write_text(emit_source_numeric_capability(ABSOLUTE,inline_absolute_values=inline)+ABSOLUTE_SOURCE)
 exe=tmp_path/'absolute'
 subprocess.run([cc,'-O2','-fno-builtin','-fno-fast-math','-frounding-math','-ffp-contract=off','-I',str(merlin_dir()/'runtime/c'),str(c),'-lm','-o',str(exe)],check=True)
 subprocess.run([str(exe)],check=True)
