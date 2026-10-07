"""Bounded exact-source graph counterexample search; incomplete is not proof."""
from pathlib import Path
import hashlib,json,subprocess,time
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/rounded_monotonicity';provider=base/'out/normal_composition/provider/native_numeric/provider.c'
lines=provider.read_text().splitlines();plan=next(x for x in lines if x.startswith('static const merlin_bit_polynomial_plan plan='));source=next(x for x in lines if x.startswith('static float source_poly('))
text='''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
typedef struct {float cutoff,scale,coefficients[4],bit_multiplier,bit_bias;} merlin_bit_polynomial_plan;
static float merlin_interval_float(uint32_t bits){float value;memcpy(&value,&bits,4);return value;}
#define MERLIN_SOURCE_F32_FMA fmaf
'''+plan+'\n'+source+'''
int main(void){if(fegetround()!=FE_TONEAREST)return 2;
 uint32_t stop;memcpy(&stop,&plan.cutoff,4);stop&=0x7fffffff;
 uint32_t prior=0,previous_input=0;
 for(uint32_t bits=0;bits<=stop;bits++){
  uint32_t input=bits|0x80000000;float x;memcpy(&x,&input,4);float y=source_poly(x);uint32_t word;memcpy(&word,&y,4);
  if(!isfinite(y)||y<0)return 3;
  if(bits && word>prior){printf("COUNTEREXAMPLE previous_x=%08x x=%08x previous_y=%08x y=%08x checked=%u total=%u\\n",previous_input,input,prior,word,bits,stop+1);return 0;}
  prior=word;previous_input=input;
 }
 printf("COMPLETE_MONOTONE checked=%u\\n",stop+1);return 0;}
'''
(w/'check.c').write_text(text);compiler='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang';cmd=[compiler,'-O3','-march=native','-fno-fast-math','-ffp-contract=off',str(w/'check.c'),'-lm','-o',str(w/'check')];subprocess.run(cmd,check=True);started=time.time()
try:
 p=subprocess.run([str(w/'check')],capture_output=True,text=True,timeout=60);status='COUNTEREXAMPLE' if 'COUNTEREXAMPLE' in p.stdout else 'COMPLETE' if 'COMPLETE_MONOTONE' in p.stdout else 'REFUSAL';stdout=p.stdout;stderr=p.stderr;code=p.returncode
except subprocess.TimeoutExpired as e:status='INCOMPLETE_TIMEOUT_NO_PROOF';stdout=e.stdout or '';stderr=e.stderr or '';code=None
if isinstance(stdout,bytes):stdout=stdout.decode()
if isinstance(stderr,bytes):stderr=stderr.decode()
(w/'stdout.txt').write_text(stdout);(w/'stderr.txt').write_text(stderr)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=dict(status=status,scope='Exact source-plan graph native RNE counterexample screen; no target/compile-time certificate or new interval policy. Timeout is not proof.',command=cmd,seconds=time.time()-started,returncode=code,stdout=stdout,pins={str(p):sha(p) for p in [provider,w/'check.c',w/'check',w/'stdout.txt',Path(__file__)]});(w/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
