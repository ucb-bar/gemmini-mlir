"""Independent compiler trace digest and integer-rational source probes."""
from pathlib import Path
import ctypes as C
from fractions import Fraction
import hashlib,json,struct,subprocess,time
import numpy as np
from merlin.llvmlower.bf16_projection_enclosure import _round,_decode
base=Path(__file__).resolve().parents[2];w=base/'out/observation_frontier/rounded_monotonicity';source=(w/'check.c').read_text()
source=source.replace('int main(void)', 'uint32_t evaluate_word(uint32_t bits){float x;memcpy(&x,&bits,4);float y=source_poly(x);uint32_t out;memcpy(&out,&y,4);return out;}\nint main(void)')
source=source.replace('uint32_t prior=0,previous_input=0;', 'uint32_t prior=0,previous_input=0;uint64_t digest=UINT64_C(1469598103934665603);')
source=source.replace('prior=word;previous_input=input;', 'digest=(digest^word)*UINT64_C(1099511628211);prior=word;previous_input=input;')
source=source.replace('printf("COMPLETE_MONOTONE checked=%u\\n",stop+1);','printf("COMPLETE_MONOTONE checked=%u digest=%016llx\\n",stop+1,(unsigned long long)digest);')
(w/'independent.c').write_text(source);commands=[];results=[]
for label,compiler in [('clang','/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'),('gcc','/usr/bin/gcc')]:
 common=[compiler,'-O2','-march=native','-fno-fast-math','-ffp-contract=off','-frounding-math',str(w/'independent.c')]
 cmd=common+['-lm','-o',str(w/label)];subprocess.run(cmd,check=True);commands.append(cmd)
 start=time.time()
 try:p=subprocess.run([str(w/label)],capture_output=True,text=True,timeout=60);record=dict(engine=label,seconds=time.time()-start,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired:record=dict(engine=label,status='INCOMPLETE_NO_PROOF')
 results.append(record);(w/f'{label}_trace.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
 if label=='gcc':
  cmd=common+['-shared','-fPIC','-lm','-o',str(w/'oracle.so')];subprocess.run(cmd,check=True);commands.append(cmd)
lib=C.CDLL(str(w/'oracle.so'));lib.evaluate_word.argtypes=[C.c_uint32];lib.evaluate_word.restype=C.c_uint32
# Independently rounded exact-rational implementation of each source operation.
words=[0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000]
cutoff,scale,*tail=[_decode(x,23) for x in words];coeff=tail[:4];mult,bias=tail[4:]
def rounded(x):return _decode(_round(x,23),23)
def oracle(bits):
 x=_decode(bits,23)
 if x<cutoff:return 0
 s=rounded(x*scale);floor=s.numerator//s.denominator;f=rounded(s-Fraction(floor));v=coeff[0]
 for c in coeff[1:]:v=rounded(f*v+c)
 q=rounded(s-v);q=rounded(mult*q+bias);return int(q)
samples=[bits<<16 for bits in range(0x8000,0xc2af)] + [0,0xc2aeac50,0xc2aeac51]
rng=np.random.default_rng(472);samples.extend(map(int,rng.integers(0x80000000,0xc2aeac51,size=1024,dtype=np.uint32)))
for bits in samples:
 expected=oracle(bits);actual=lib.evaluate_word(bits)
 if expected!=actual:raise AssertionError((hex(bits),hex(expected),hex(actual)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=dict(scope='Two independently compiled complete finite-domain monotonicity traces and independent rational exact per-operation checks; RNE source graph only, no numerical permission for arbitrary plans.',results=results,rational_checks=len(samples),commands=commands,pins={str(p):sha(p) for p in [w/'independent.c',w/'clang',w/'gcc',w/'oracle.so',Path(__file__)]})
(w/'independent_qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('RATIONAL_CHECKS',len(samples),flush=True)
assert all(x.get('returncode')==0 and x.get('stdout','').startswith('COMPLETE_MONOTONE') for x in results)
assert results[0]['stdout']==results[1]['stdout']
