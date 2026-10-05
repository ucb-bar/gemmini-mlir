"""Build a bitwise RISC-V roundeven probe against independently computed f32 results."""
from pathlib import Path
import json
import numpy as np
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
root = Path(__file__).resolve().parent.parent
out = root/'out/roundeven_probe';out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(739)
bits=rng.integers(0,2**32,size=16000,dtype=np.uint32)
values=np.concatenate([bits.view(np.float32),np.arange(-4096,4097,dtype=np.float32)*np.float32(.5), np.array([0.,-0.,2**23,-2**23,np.inf,-np.inf],np.float32)])
with np.errstate(invalid='ignore'):
 expected=np.rint(values)
# NaN payload/canonicalization is implementation-dependent; classify NaNs.
source=out/'probe.c'
source.write_text('#include <stdint.h>\n#include <stdio.h>\nextern float gemmini_golden_roundevenf(float);\n'+
 'static const uint32_t input[]={'+','.join(str(int(x)) for x in values.view(np.uint32))+'};\n'+
 'static const uint32_t expected[]={'+','.join(str(int(x)) for x in expected.view(np.uint32))+'};\n'+r'''
int main(void) {
 for(int mode=0;mode<5;++mode) {
  __asm__ volatile("fsrm %0" :: "r"(mode));
  for(unsigned i=0;i<sizeof(input)/sizeof(input[0]);++i) {
   union {float f;uint32_t u;} in={.u=input[i]},out;
   __asm__ volatile("fsflags %0" :: "r"(8));
   out.f=gemmini_golden_roundevenf(in.f);
   unsigned flags;__asm__ volatile("frflags %0":"=r"(flags));
   int nan=(input[i]&0x7fffffff)>0x7f800000;
   if ((nan ? ((out.u&0x7fffffff)<=0x7f800000) : out.u!=expected[i]) || (!nan && flags!=8)) {
    printf("ROUND_FAIL mode=%d i=%d got=%u expected=%u flags=%u\n",mode,i,out.u,expected[i],flags);return 1;
   }
  }
 }
 __asm__ volatile("fsrm zero");
 printf("GOLDEN_ROUND PASS\n");return 0;
}
''')
b=build_program([source,root/'runtime/roundevenf_rv64gc.S'],out,target='gemmini',extra_cflags=['-march=rv64gc','-fno-builtin'],max_loaded_bytes=None)
audit=audit_elf(b.elf.read_bytes());(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
assert audit['status']=='pass'
print(b.elf)
