"""Independent source-word oracle and five target rounding-mode refusal checks."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,struct,subprocess,time
import numpy as np
from merlin.llvmlower.bf16_projection_enclosure import _round,_decode
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];w=base/'out/rounded_monotone_target_v3';w.mkdir(exist_ok=False)
d=base/'out/rounded_monotone_group/candidate/target_numeric'
words=[0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000]
cutoff,scale,*tail=[_decode(x,23) for x in words];coeff=tail[:4];mult,bias=tail[4:]
rounded=lambda x:_decode(_round(x,23),23)
def oracle(bits):
 x=_decode(bits,23)
 if x<cutoff:return 0
 s=rounded(x*scale);f=rounded(s-Fraction(s.numerator//s.denominator));v=coeff[0]
 for c in coeff[1:]:v=rounded(f*v+c)
 return int(rounded(mult*rounded(s-v)+bias))
rng=np.random.default_rng(471)
samples=[0,0x80000000,0x80000001,0x807fffff,0x80800000,0xc2aeac50,0xc2aeac51]+list(map(int,rng.integers(0x80000000,0xc2aeac51,size=1024,dtype=np.uint32)))
expected=[oracle(x) for x in samples]
source='''#include "monotone_bit_polynomial.h"
#include <stdio.h>
static const uint32_t inputs[]={INPUTS},expected[]={EXPECTED};
int main(void){
 uint32_t words[8]={WORDS};float a[8];memcpy(a,words,sizeof(a));
 merlin_bit_polynomial_plan s={a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
 unsigned checks=0;
 for(unsigned mode=0;mode<5;mode++){
  asm volatile("csrw frm,%0"::"r"(mode):"memory");
  merlin_fma_bound env=merlin_fma_bound_begin();
  merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&s,0.0f);
  if(p.fast_valid!=(mode==0))return 2;
  if(mode)continue;
  if(p.word_budget)return 3;
  for(unsigned i=0;i<sizeof(inputs)/sizeof(inputs[0]);i++){
   float x;memcpy(&x,&inputs[i],4);
   uint32_t got=x<s.cutoff?0:merlin_monotone_polynomial_source_word(x,&s);
   if(got!=expected[i])return 4;
   merlin_f32_interval interval=merlin_monotone_bit_polynomial_apply_words(merlin_interval(x,x),&p);
   if(!interval.valid||merlin_interval_bits(interval.lo)!=got||merlin_interval_bits(interval.hi)!=got)return 5;
   checks++;
  }
  s.bit_bias=nextafterf(s.bit_bias,INFINITY);
  p=merlin_monotone_bit_polynomial_prepare(&env,&s,0.0f);
  if(!p.fast_valid||!p.word_budget)return 6;s.bit_bias=a[7];
  p=merlin_monotone_bit_polynomial_prepare(&env,&s,-1.0f);
  if(!p.fast_valid||!p.word_budget)return 7;
 }
 asm volatile("csrw frm,zero":::"memory");
 printf("MONOTONE_TARGET_PASS words=%u modes=5 plan_domain_refusals=2\\n",checks);
 return 0;
}
'''
for key,v in [('INPUTS',samples),('EXPECTED',expected),('WORDS',words)]:source=source.replace(key,','.join(hex(x)+'u' for x in v))
(w/'probe.c').write_text(source)
cmd=json.loads((d/'compile.json').read_text())['commands'][0]
cmd=[str(w/'probe.c') if x==str(d/'provider.c') else str(w/'probe.o') if x==str(d/'provider.o') else str(w/'probe.d') if x==str(d/'provider.d') else x for x in cmd]
subprocess.run(cmd,check=True)
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')
link=json.loads((b/'build.json').read_text())['link'];link=[x for x in link if not x.endswith('.o') or Path(x).name in ('crt.o','syscalls.o')]
link=[str(w/'model.elf') if x==str(b/'model.elf') else x for x in link];link.insert(link.index('-lm'),str(w/'probe.o'));subprocess.run(link,check=True)
audit=audit_elf((w/'model.elf').read_bytes());assert audit['status']=='pass'
run=['timeout','60s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc',str(w/'model.elf')]
p=subprocess.run(run,capture_output=True,text=True);(w/'stdout').write_text(p.stdout);(w/'stderr').write_text(p.stderr)
assert p.returncode==0 and f'MONOTONE_TARGET_PASS words={len(samples)} modes=5 plan_domain_refusals=2' in p.stdout,(p.returncode,p.stdout,p.stderr)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(w/'qualification.json').write_text(json.dumps(dict(words=len(samples),oracle='independent exact rational per-operation binary32 RNE',commands=[cmd,link,run],stdout=p.stdout,audit=audit,pins={str(p):sha(p) for p in w.iterdir() if p.is_file()}),indent=2)+'\n');print(p.stdout)
