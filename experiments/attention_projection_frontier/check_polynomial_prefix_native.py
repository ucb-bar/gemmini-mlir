"""Exhaust every table cell boundary/tail against the retained source evaluator."""
from pathlib import Path
import json,subprocess,hashlib
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/source-polynomial-prefix-table';D=W/'candidate/native_numeric';T=W/'independent_v3';T.mkdir(exist_ok=False)
meta=json.load(open(W/'generation/metadata.json'));c=T/'check.c'
c.write_text(r'''
#include "provider.c"
#include <stdio.h>
#include <stdlib.h>
static unsigned long checks=0;
static void point(uint32_t word){
 float x=merlin_interval_float(word);
 merlin_f32_interval r=merlin_polynomial_prefix_point(x);
 float exact=x<plan.cutoff?0.0f:merlin_interval_float(merlin_monotone_polynomial_source_word(x,&plan));
 if(!(r.valid && r.lo<=exact && exact<=r.hi)){
  fprintf(stderr,"FAIL word=%08x lo=%a exact=%a hi=%a\n",word,r.lo,exact,r.hi);abort();
 }
 checks++;
}
int main(void){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&plan,0);
 if(!merlin_polynomial_prefix_admit(&p))return 2;
 const uint32_t base=BASEu,cutoff=CUTOFFu,step=256;
 for(uint32_t w=base;w<cutoff;w+=step){
  point(w);point(w+(cutoff-w<step/2?cutoff-w:step/2));
  point(w+(cutoff-w<step-1?cutoff-w:step-1));
 }
 point(cutoff);point(cutoff+1);point(0);point(0x80000000u);
 point(base-1);point(0xff7fffffu);point(0x80000001u);
 uint32_t rng=0x96534a13u;
 for(int j=0;j<100000;j++){
  merlin_f32_interval x[4],a[4],b[4];
  for(int i=0;i<4;i++){
   rng=1664525*rng+1013904223;float lo=merlin_interval_float(0x80000000u+rng%0x7f800000u);
   rng=1664525*rng+1013904223;float hi=merlin_interval_float(0x80000000u+rng%0x7f800000u);
   if(lo>hi){float t=lo;lo=hi;hi=t;}x[i]=(merlin_f32_interval){lo,hi,1};
  }
  merlin_polynomial_constants_four(x,&p,1,a);merlin_polynomial_prefix_four(x,&p,1,b);
  for(int i=0;i<4;i++){if(a[i].valid!=b[i].valid||b[i].lo>a[i].lo||b[i].hi<a[i].hi)abort();checks++;}
 }
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(int mode=0;mode<4;mode++){
  fesetround(modes[mode]);feraiseexcept(FE_INVALID);
  int admitted=merlin_polynomial_prefix_admit(&p);
  if(admitted!=(mode==0)||!(fetestexcept(FE_INVALID)))return 3;
  merlin_f32_interval x[4],a[4],b[4];for(int i=0;i<4;i++)x[i]=(merlin_f32_interval){-1,-.5f,1};
  merlin_polynomial_constants_four(x,&p,admitted,a);merlin_polynomial_prefix_four(x,&p,admitted,b);
  for(int i=0;i<4;i++)if(mode && (memcmp(a+i,b+i,sizeof(a[i]))))return 4;
 }
 fesetround(FE_TONEAREST);
 printf("PREFIX_TABLE_PASS checks=%lu modes=4\n",checks);return 0;
}
'''.replace('BASE',str(meta['base_word'])).replace('CUTOFF',str(meta['cutoff_word'])))
cmd=['/usr/bin/clang','-O2','-fno-builtin','-fno-fast-math','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all','-include',str(D/'native_exact_conversion.h'),'-include',str(D/'numeric_capability.h'),'-I',str(D),str(c),'-lm','-o',str(T/'check')]
r=subprocess.run(cmd,capture_output=True,text=True);(T/'compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
r=subprocess.run([str(T/'check')],capture_output=True,text=True);(T/'stdout').write_text(r.stdout);(T/'stderr').write_text(r.stderr);assert r.returncode==0,(r.returncode,r.stderr)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(T/'qualification.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,pins={str(p):sha(p) for p in [Path(__file__),c,T/'check',D/'provider.c',D/'polynomial_prefix_table.h',D/'prepared_polynomial_constants.h',D/'monotone_bit_polynomial.h',W/'generation/metadata.json']}),indent=2)+'\n');print(r.stdout)
