"""Validate nonunit residual load rounds across every signed-i8 operand pair."""
import argparse,json,re
from pathlib import Path
import numpy as np
from mlir_oot.golden_resadd import build
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program,run_on_gsim

p=argparse.ArgumentParser();p.add_argument('--lhs',type=float,required=True);p.add_argument('--rhs',type=float,required=True);p.add_argument('--readout',type=float,default=1.);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--workdir',type=Path,required=True);args=p.parse_args()
w=args.workdir.resolve();w.mkdir(parents=True,exist_ok=False)
a=np.repeat(np.arange(-128,128,dtype=np.float32),256);b=np.tile(np.arange(-128,128,dtype=np.float32),256)
x=np.clip(np.rint(a*np.float32(args.lhs)),-128,127).astype(np.int32)
y=np.clip(np.rint(b*np.float32(args.rhs)),-128,127).astype(np.int32)
expected=np.clip(np.rint((x+y).astype(np.float32)*np.float32(args.readout)),0,127).astype(np.int8)
(w/'expected.h').write_text('static const int8_t expected[65536]={'+','.join(map(str,expected.tolist()))+'};\n')
source=w/'probe.c';source.write_text('''#include <stdint.h>
#include <stdio.h>
#include "expected.h"
static int8_t a[65536] __attribute__((aligned(64))),b[65536] __attribute__((aligned(64)));
static int8_t identity[256] __attribute__((aligned(64))),scratch[1024] __attribute__((aligned(64)));
static struct {int8_t output[65536];uint8_t guard[2048];} c __attribute__((aligned(64)));
extern void gemmini_golden_resadd(int8_t*,int8_t*,int8_t*,int8_t*,int8_t*);
int main(void) {
 for(int i=0;i<65536;i++){a[i]=(int8_t)(i/256-128);b[i]=(int8_t)(i%256-128);c.output[i]=-37;}
 for(int i=0;i<16;i++)identity[i*16+i]=1;
 for(int i=0;i<2048;i++)c.guard[i]=0x5a;
 uint64_t begin,end;__asm__ volatile("rdcycle %0":"=r"(begin)::"memory");
 gemmini_golden_resadd(a,b,c.output,identity,scratch);
 __asm__ volatile("rdcycle %0":"=r"(end)::"memory");
 printf("GOLDEN_RESADD_CYCLES %d\\n",(int)(end-begin));
 for(int i=0;i<65536;i++)if(c.output[i]!=expected[i]){printf("FAIL %d got%d want%d\\n",i,c.output[i],expected[i]);return 1;}
 for(int i=0;i<2048;i++)if(c.guard[i]!=0x5a){puts("FAIL guard");return 2;}
 puts("GOLDEN_RESADD PASS full65536pairs");return 0;
}
''')
compile_module(build(1024,64,lhs_scale=args.lhs,rhs_scale=args.rhs,output_scale=args.readout,relu=True),args.llvm_bin,w)
built=build_program([source,w/'kernel.o'],w,target='gemmini',extra_cflags=[f'-I{w}'],max_loaded_bytes=None)
audit=audit_elf(built.elf.read_bytes());(w/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
assert audit['status']=='pass'
run=run_on_gsim(built.elf,target='gemmini',max_cycles=10000000,timeout_s=600,backdoor=True,stdout_path=w/'gsim.stdout')
passed=bool(run.completed and run.returncode==0 and 'GOLDEN_RESADD PASS full65536pairs' in run.stdout_tail)
cycles=re.findall(r'GOLDEN_RESADD_CYCLES (\d+)',run.stdout_tail)
r=dict(status='pass' if passed else 'fail',full_pair_count=65536,kernel_cycles=int(cycles[0]) if cycles else None,lhs=args.lhs,rhs=args.rhs,readout=args.readout,elf_sha256=built.elf_sha256,gsim_engine_sha256=run.engine.get('binary_sha256'),stdout=run.stdout_tail,stderr=run.stderr_tail)
(w/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if passed else 1)
