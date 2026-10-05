"""Numerical direct-convolution probe; no host im2col, linked ELF audit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import struct
from mlir_oot.golden_conv import ConvShape, GoldenConv
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program, run_on_gsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--llvm-bin', type=Path, required=True)
    for key, value in [('h',3),('w',19),('cin',20),('cout',19),('stride',1)]:
        ap.add_argument('--'+key,type=int,default=value)
    ap.add_argument('--wide-b',action='store_true')
    ap.add_argument('--output-dtype',choices=['i8','i32'],default='i32')
    ap.add_argument('--scale',type=float,default=1.0)
    ap.add_argument('--relu',action='store_true')
    ap.add_argument('--build-only',action='store_true')
    a = ap.parse_args()
    s = ConvShape(a.h,a.w,a.cin,a.cout,a.stride,wide_b=a.wide_b,output_dtype=a.output_dtype,scale=a.scale,relu=a.relu)
    out = a.workdir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    receipt = compile_module(GoldenConv(s).build(),a.llvm_bin,out)
    values = []
    for y in range(s.oh):
        for x in range(s.ow):
            for n in range(s.cout):
                acc = 0
                for ky in range(3):
                    for kx in range(3):
                        iy,ix = y*s.stride+ky-1,x*s.stride+kx-1
                        if 0 <= iy < s.h and 0 <= ix < s.w:
                            for c in range(s.cin):
                                ai = (iy*s.w+ix)*s.cin+c
                                bi = ((ky*3+kx)*s.cin+c)*s.cout+n
                                acc += (ai%11-5)*(bi%13-6)
                if s.output_dtype == 'i8':
                    scale = struct.unpack('<f',struct.pack('<f',s.scale))[0]
                    acc = round(struct.unpack('<f',struct.pack('<f',float(acc)*scale))[0])
                    acc = min(127,max(0 if s.relu else -128,acc))
                values.append(str(acc))
    source = out/'probe.c'
    ctype = 'int32_t' if s.output_dtype == 'i32' else 'int8_t'
    source.write_text('''#include <stdint.h>
#include <stdio.h>
''' + f'''extern void gemmini_golden_conv(int8_t*,int8_t*,{ctype}*);
static int8_t a[{s.h*s.w*s.cin}] __attribute__((aligned(64)));
static int8_t b[{9*s.cin*s.cout}] __attribute__((aligned(64)));
static struct {{{ctype} c[{len(values)}]; uint8_t guard[2048];}} box __attribute__((aligned(64)));
static const int32_t expected[] = {{{','.join(values)}}};
''' + '''static uint64_t cycles(void) {uint64_t v; __asm__ volatile("rdcycle %0":"=r"(v)::"memory"); return v;}
int main(void) {
 for (int i=0;i<sizeof(a);i++) a[i]=i%11-5;
 for (int i=0;i<sizeof(b);i++) b[i]=i%13-6;
 for (int i=0;i<2048;i++) box.guard[i]=0x5a;
 uint64_t t=cycles(); gemmini_golden_conv(a,b,box.c); t=cycles()-t;
 printf("GOLDEN_CONV_CYCLES %d\\n",(int)t);
 for(int i=0;i<sizeof(expected)/sizeof(expected[0]);i++) if(box.c[i]!=expected[i]) {
 printf("GOLDEN_CONV FAIL i=%d got=%d expected=%d\\n",i,box.c[i],expected[i]); return 1;}
 for(int i=0;i<2048;i++) if(box.guard[i]!=0x5a) {printf("GUARD_FAIL\\n"); return 2;}
 printf("GOLDEN_CONV PASS\\n"); return 0;
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
    run = run_on_gsim(built.elf,target='gemmini',max_cycles=3000000,timeout_s=600,backdoor=True,stdout_path=out/'gsim.stdout')
    match = re.search(r'GOLDEN_CONV_CYCLES (\d+)',run.stdout_tail)
    passed = run.completed and run.returncode == 0 and 'GOLDEN_CONV PASS' in run.stdout_tail
    result = dict(shape=asdict(s),status='pass' if passed else 'fail',kernel_cycles=int(match[1]) if match else None,
                  elf_sha256=built.elf_sha256,compilation=receipt,nofsm_audit=audit,gsim_engine=run.engine,stdout=run.stdout_tail)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('compilation','nofsm_audit','gsim_engine')},indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
