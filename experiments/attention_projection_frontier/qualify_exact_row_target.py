"""Independent fixed-format word domain and five-mode source equivalence."""
from pathlib import Path
import hashlib,json,subprocess
from merlin.llvmlower.exact_row_radix_pack import c_header
from merlin.llvmlower.fused_encoded_witness import fused_encoded_row_header
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];w=base/'out/exact_row_target';w.mkdir(exist_ok=False)
(w/'old.h').write_text(fused_encoded_row_header().replace('MERLIN_FUSED_ENCODED_ROW_H','OLD_ROW_H').replace('merlin_bf16_radix_row_widen','old_row'))
(w/'new.h').write_text(c_header())
(w/'probe.c').write_text(r'''
#include "old.h"
#include "new.h"
#include <stdio.h>
#include <string.h>
static int check(uint32_t a,uint32_t b,unsigned digits){
 float input[2];memcpy(input,&a,4);memcpy(input+1,&b,4);
 double old[4]={17,19,23,29},fresh[4]={17,19,23,29};
 int8_t op[8]={1,2,3,4,5,6,7,8},np[8]={1,2,3,4,5,6,7,8};
 float os=13,ns=13;unsigned char of=3,nf=3;
 merlin_fma_bound e=merlin_fma_bound_begin();
 int x=old_row(&e,input,2,1,old,1,op,2,1,digits,&os,0,0,&of);
 int y=merlin_bf16_radix_row_widen(&e,input,2,1,fresh,1,np,2,1,digits,&ns,0,0,&nf);
 if(x!=y)return 0;
 if(!x)return 1;
 return memcmp(old,fresh,sizeof old)==0&&memcmp(op,np,sizeof op)==0&&memcmp(&os,&ns,4)==0&&of==nf;
}
int main(void){
 const uint32_t maxima[3]={0x3f800000,0x71800000,0x0d800000};unsigned checks=0;
 for(unsigned mode=0;mode<5;mode++){
  asm volatile("csrw frm,%0"::"r"(mode):"memory");
  if(mode){if(merlin_fma_bound_begin().valid)return 2;if(!check(0x3f800000,0x3f000000,3))return 3;continue;}
  for(unsigned m=0;m<3;m++)for(unsigned v=0;v<65536;v++)for(unsigned digits=1;digits<=3;digits++){
   if(!check(maxima[m],v<<16,digits)){printf("FAIL %u %u %u\n",m,v,digits);return 4;}checks++;
  }
 }
 asm volatile("csrw frm,zero":::"memory");
 printf("EXACT_ROW_TARGET_PASS comparisons=%u modes=5 guards=PASS\n",checks);return 0;
}
''')
d=base/'out/exact_row_group/candidate/target_numeric';manifest=json.load(open(base/'out/exact_row_group/candidate/build.json'))
cmd=next(c for c in manifest['commands']if str(d/'provider.c')in c);cmd=[str(w/'probe.c')if x==str(d/'provider.c')else str(w/'probe.o')if x==str(d/'provider.o')else str(w/'probe.d')if x==str(d/'provider.d')else x for x in cmd]
subprocess.run(cmd,check=True)
link=[x for x in manifest['commands'][-1]if not x.endswith('.o')or Path(x).name in ('crt.o','syscalls.o')]
link=[str(w/'model.elf')if x==str(d.parent/'model.elf')else x for x in link];link.insert(link.index('-lm'),str(w/'probe.o'));subprocess.run(link,check=True)
audit=audit_elf((w/'model.elf').read_bytes());assert audit['status']=='pass'
run=['timeout','120s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc',str(w/'model.elf')]
p=subprocess.run(run,capture_output=True,text=True);(w/'stdout').write_text(p.stdout);(w/'stderr').write_text(p.stderr)
assert p.returncode==0 and 'comparisons=589824 modes=5 guards=PASS'in p.stdout,(p.returncode,p.stdout,p.stderr)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(w/'qualification.json').write_text(json.dumps(dict(comparisons=589824,modes=5,scope='Independent finite BF16 domain/reference canonical helper equivalence; original signed-zero/rounding/plane/flag and guard bytes',commands=[cmd,link,run],audit=audit,pins={str(p):sha(p)for p in w.iterdir()if p.is_file()}),indent=2)+'\n');print(p.stdout)
