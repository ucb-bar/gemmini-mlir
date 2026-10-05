"""Independent full bounded-RNE probe over all frm modes and boundary neighbors."""
from pathlib import Path
import json,subprocess
import numpy as np
from mlir_oot.late_quant_rne import rewrite
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program
root=Path(__file__).resolve().parent.parent
out=root/'out/bounded_quant_rne_probe';out.mkdir(parents=True,exist_ok=True)
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
source=root/'tests/fixtures/bounded_quant_rne.ll'
text,proof=rewrite(source.read_text());(out/'quant.ll').write_text(text)
subprocess.run([str(llvm/'llvm-as'),str(out/'quant.ll'),'-o',str(out/'quant.bc')],check=True)
subprocess.run([str(llvm/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-O2','-c',str(out/'quant.ll'),'-o',str(out/'quant.o')],check=True)
rng=np.random.default_rng(715)
random=rng.integers(0,2**32,20000,dtype=np.uint32).view(np.float32)
random=random[~np.isnan(random)]
ties=np.arange(-260,260,dtype=np.float32)*np.float32(.5)
values=np.concatenate([random,ties,np.nextafter(ties,np.float32(np.inf)),np.nextafter(ties,np.float32(-np.inf)),np.array([np.inf,-np.inf,0.,-0.,1e-40,-1e-40],np.float32)])
expected=np.rint(np.clip(values,-128,127)).astype(np.int8)
p=out/'probe.c'
p.write_text('#include <stdint.h>\n#include <stdio.h>\nextern int8_t quant(float);\nstatic const uint32_t input[]={'+','.join(str(int(x)) for x in values.view(np.uint32))+'};\nstatic const int8_t expected[]={'+','.join(str(int(x)) for x in expected)+'};\n'+r'''
int main(void){
 for(int mode=0;mode<5;++mode){
  __asm__ volatile("fsrm %0" :: "r"(mode));
  for(unsigned i=0;i<sizeof(input)/sizeof(input[0]);++i){
   union {uint32_t u;float f;} value={.u=input[i]};
   int8_t got=quant(value.f);
   unsigned actual_mode;__asm__ volatile("frrm %0":"=r"(actual_mode));
   if(got!=expected[i]||actual_mode!=(unsigned)mode){printf("RNE_FAIL %d %d %d %d\n",mode,i,got,expected[i]);return 1;}
  }
 }
 __asm__ volatile("fsrm zero");printf("BOUNDED_RNE PASS\n");return 0;
}
''')
b=build_program([p,out/'quant.o'],out,target='gemmini',extra_cflags=['-march=rv64gc','-fno-builtin'],max_loaded_bytes=None)
audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass'
r=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,check=True)
(out/'spike.log').write_text(r.stdout+r.stderr);assert 'BOUNDED_RNE PASS' in r.stdout+r.stderr
proof.update(values=len(values),frm_modes=5,checks=len(values)*5,nofsm_audit=audit)
(out/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
