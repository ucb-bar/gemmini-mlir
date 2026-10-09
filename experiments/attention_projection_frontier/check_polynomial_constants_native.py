"""Compare specialized endpoints to the original exact-word batch consumer."""
from pathlib import Path
import json,subprocess,hashlib
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants';D=W/'candidate/native_numeric';T=W/'independent';T.mkdir(exist_ok=False)
c=T/'check.c'
c.write_text(r'''
#include "provider.c"
#include <stdio.h>
#include <stdlib.h>
static uint32_t rng=0x91a73c29u;
static uint32_t next(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void compare(merlin_f32_interval x[4],merlin_monotone_bit_polynomial *p,int admitted){
 merlin_f32_interval a[4],b[4];
 merlin_polynomial_words_four(x,p,a);
 merlin_polynomial_constants_four(x,p,admitted,b);
 for(int i=0;i<4;i++)if(a[i].valid!=b[i].valid||(a[i].valid&&
 (merlin_interval_bits(a[i].lo)!=merlin_interval_bits(b[i].lo)||merlin_interval_bits(a[i].hi)!=merlin_interval_bits(b[i].hi))))abort();
}
int main(void){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&plan,0);
 if(!merlin_polynomial_constants_admit(&p))return 2;
 unsigned count=0;
 for(int j=0;j<100000;j++){
  merlin_f32_interval x[4];
  for(int i=0;i<4;i++){
   float a=merlin_interval_float(0x80000000u+(next()%0x7f800000u));
   float b=merlin_interval_float(0x80000000u+(next()%0x7f800000u));
   if(a>b){float tmp=a;a=b;b=tmp;}
   x[i]=(merlin_f32_interval){a,b,1};
  }
  compare(x,&p,1);count+=4;
 }
 const float corners[]={0.0f,-0.0f,plan.cutoff,nextafterf(plan.cutoff,-INFINITY),nextafterf(plan.cutoff,INFINITY),-FLT_MAX,-FLT_MIN,-0x1p-149f};
 for(unsigned a=0;a<8;a++)for(unsigned b=0;b<8;b++){
  float lo=fminf(corners[a],corners[b]),hi=fmaxf(corners[a],corners[b]);
  merlin_f32_interval x[4];for(int i=0;i<4;i++)x[i]=(merlin_f32_interval){lo,hi,1};compare(x,&p,1);count+=4;
 }
 for(int k=0;k<8;k++){
  merlin_monotone_bit_polynomial q=p;float *fields[]={&q.checked.source.cutoff,&q.checked.source.scale,&q.checked.source.coefficients[0],&q.checked.source.coefficients[1],&q.checked.source.coefficients[2],&q.checked.source.coefficients[3],&q.checked.source.bit_multiplier,&q.checked.source.bit_bias};
  *fields[k]=nextafterf(*fields[k],INFINITY);if(merlin_polynomial_constants_admit(&q))return 3;
 }
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(int mode=0;mode<4;mode++){
  fesetround(modes[mode]);feraiseexcept(FE_INVALID);
  int admitted=merlin_polynomial_constants_admit(&p);if(admitted!=(mode==0)||!(fetestexcept(FE_INVALID)))return 4;
  merlin_f32_interval x[4];for(int i=0;i<4;i++)x[i]=(merlin_f32_interval){-1,-0.5f,1};compare(x,&p,admitted);count+=4;
 }
 fesetround(FE_TONEAREST);
 merlin_f32_interval bad[4]={{0,0,0},{0,0,1},{0,0,1},{0,0,1}};compare(bad,&p,1);
 p.word_budget=1;if(merlin_polynomial_constants_admit(&p))return 5;
 p.word_budget=0;p.fast_valid=0;if(merlin_polynomial_constants_admit(&p))return 6;
 if(merlin_polynomial_constants_admit(NULL))return 7;
 printf("POLYNOMIAL_CONSTANTS_PASS intervals=%u modes=4 plan_refusals=8\n",count);
 return 0;
}
''')
cmd=['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang','-O2','-fno-builtin','-fno-fast-math','-ffp-contract=off','-include',str(D/'native_exact_conversion.h'),'-include',str(D/'numeric_capability.h'),'-I',str(D),str(c),'-lm','-o',str(T/'check')]
r=subprocess.run(cmd,capture_output=True,text=True);(T/'compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
r=subprocess.run([str(T/'check')],capture_output=True,text=True);(T/'stdout').write_text(r.stdout);(T/'stderr').write_text(r.stderr);assert r.returncode==0,(r.returncode,r.stderr)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(T/'qualification.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,pins={str(p):sha(p) for p in [Path(__file__),c,T/'check',D/'provider.c',D/'prepared_polynomial_constants.h',D/'monotone_bit_polynomial.h']}),indent=2)+'\n');print(r.stdout)
