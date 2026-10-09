"""Packed stem hardware pool with an independent ordered-f32 source oracle."""
import argparse
import numpy as np
from dataclasses import asdict
import json
from pathlib import Path
import re
from mlir_oot.golden_stem_pool import StemPoolShape, GoldenStemPool
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program, run_on_gsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--llvm-bin', type=Path, required=True)
    for key, value in [('h',5),('w',35),('cout',19)]:
        ap.add_argument('--'+key,type=int,default=value)
    ap.add_argument('--scale',type=float,default=0.03125)
    ap.add_argument('--build-only',action='store_true')
    ap.add_argument('--max-cycles',type=int,default=3000000)
    ap.add_argument('--timeout-s',type=int,default=600)
    ap.add_argument('--source-scales',nargs='+',type=float,default=[1.0])
    ap.add_argument('--output-reciprocal',type=float,default=0.03125)
    a = ap.parse_args()
    s = StemPoolShape(a.h,a.w,a.cout,scale=a.scale)
    out = a.workdir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    receipt = compile_module(GoldenStemPool(s).build(),a.llvm_bin,out)
    av=((np.arange((s.h+6)*(s.w+6)*3,dtype=np.int32)*73+7)%255-128).reshape(s.h+6,s.w+6,3)
    bv=((np.arange(147*s.cout,dtype=np.int32)*37+11)%255-128).reshape(7,7,3,s.cout)
    acc=np.zeros((s.oh,s.ow,s.cout),np.int32)
    for ky in range(7):
        for kx in range(7):acc+=av[ky:ky+2*s.oh:2,kx:kx+2*s.ow:2]@bv[ky,kx]
    # Independent original f32 dequant/bias/ReLU -> float pool -> requant.
    source=acc.astype(np.float32)
    for scale in a.source_scales:source=source*np.float32(scale)
    source=np.maximum(source+np.float32(0),np.float32(0))
    padded=np.pad(source,((1,1),(1,1),(0,0)),constant_values=-np.inf)
    pooled=np.full((s.ph,s.pw,s.cout),-np.inf,np.float32)
    for ky in range(3):
        for kx in range(3):pooled=np.maximum(pooled,padded[ky:ky+2*s.ph:2,kx:kx+2*s.pw:2])
    expected=np.clip(np.rint(pooled*np.float32(a.output_reciprocal)),-128,127).astype(np.int8)
    values=[str(int(x)) for x in expected.reshape(-1)]
    source = out/'probe.c'
    ctype = 'int8_t'
    source.write_text('''#include <stdint.h>
#include <stdio.h>
''' + f'''extern void gemmini_golden_stem_pool(int8_t*,int8_t*,{ctype}*);
static int8_t a[{(s.h+6)*(s.w+6)*3}] __attribute__((aligned(64)));
static int8_t b[{49*3*s.cout}] __attribute__((aligned(64)));
static struct {{{ctype} c[{len(values)}]; uint8_t guard[2048];}} box __attribute__((aligned(64)));
static const int32_t expected[] = {{{','.join(values)}}};
''' + '''static uint64_t cycles(void) {uint64_t v; __asm__ volatile("rdcycle %0":"=r"(v)::"memory"); return v;}
int main(void) {
 for (int i=0;i<sizeof(a);i++) a[i]=(i*73+7)%255-128;
 for (int i=0;i<sizeof(b);i++) b[i]=(i*37+11)%255-128;
 for (int i=0;i<2048;i++) box.guard[i]=0x5a;
 uint64_t t=cycles(); gemmini_golden_stem_pool(a,b,box.c); t=cycles()-t;
 printf("GOLDEN_STEM_POOL_CYCLES %d\\n",(int)t);
 for(int i=0;i<sizeof(expected)/sizeof(expected[0]);i++) if(box.c[i]!=expected[i]) {
 printf("GOLDEN_STEM_POOL FAIL i=%d got=%d expected=%d\\n",i,box.c[i],expected[i]); return 1;}
 for(int i=0;i<2048;i++) if(box.guard[i]!=0x5a) {printf("GUARD_FAIL\\n"); return 2;}
 printf("GOLDEN_STEM_POOL PASS\\n"); return 0;
}
''')
    built = build_program([source,out/'kernel.o'],out,target='gemmini',extra_cflags=['-march=rv64gc'],max_loaded_bytes=None)
    audit = audit_elf(built.elf.read_bytes())
    (out/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['status'] != 'pass':
        raise RuntimeError('forbidden instruction in linked ELF')
    if a.build_only:
        print(built.elf)
        return 0
    run = run_on_gsim(built.elf,target='gemmini',max_cycles=a.max_cycles,timeout_s=a.timeout_s,backdoor=True,stdout_path=out/'gsim.stdout')
    match = re.search(r'GOLDEN_STEM_POOL_CYCLES (\d+)',run.stdout_tail)
    passed = run.completed and run.returncode == 0 and 'GOLDEN_STEM_POOL PASS' in run.stdout_tail
    result = dict(shape=asdict(s),status='pass' if passed else 'fail',completed=run.completed,returncode=run.returncode,stderr=run.stderr_tail,kernel_cycles=int(match[1]) if match else None,
                  elf_sha256=built.elf_sha256,compilation=receipt,nofsm_audit=audit,gsim_engine=run.engine,stdout=run.stdout_tail)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('compilation','nofsm_audit','gsim_engine')},indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
