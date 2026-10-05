"""Numerical direct-convolution probe; no host im2col, linked ELF audit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
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
    ap.add_argument('--bn',type=int,default=4)
    ap.add_argument('--wide-a',action='store_true')
    ap.add_argument('--separate-b-bank',action='store_true')
    ap.add_argument('--band-rows',type=int)
    ap.add_argument('--full-range',action='store_true',help='exercise signed i8 dynamic range with nontrivial requantized outputs')
    ap.add_argument('--static-inputs',action='store_true',help='embed deterministic input bytes to exclude large scalar setup loops')
    ap.add_argument('--flat-spatial',action='store_true',help='flatten spatial tiles; input includes explicit nonzero halo')
    ap.add_argument('--output-dtype',choices=['i8','i32'],default='i32')
    ap.add_argument('--scale',type=float,default=1.0)
    ap.add_argument('--relu',action='store_true')
    ap.add_argument('--build-only',action='store_true')
    ap.add_argument('--max-cycles',type=int,default=3000000)
    ap.add_argument('--timeout-s',type=int,default=600)
    a = ap.parse_args()
    if (a.wide_a or a.separate_b_bank or a.band_rows is not None) and not a.flat_spatial:
        ap.error("wide A or separate B bank requires --flat-spatial")
    s = ConvShape(a.h,a.w,a.cin,a.cout,a.stride,bn=a.bn,wide_b=a.wide_b,output_dtype=a.output_dtype,scale=a.scale,relu=a.relu,explicit_halo=a.flat_spatial)
    out = a.workdir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if a.flat_spatial:
        from mlir_oot.golden_flat_conv import GoldenFlatConv
        kernel = GoldenFlatConv(s,wide_a=a.wide_a,separate_b_bank=a.separate_b_bank,band_rows=a.band_rows)
    else:
        kernel = GoldenConv(s)
    receipt = compile_module(kernel.build(),a.llvm_bin,out)
    symbol = 'gemmini_golden_flat_conv' if a.flat_spatial else 'gemmini_golden_conv'
    ih, iw = (s.h+2,s.w+2) if s.explicit_halo else (s.h,s.w)
    import numpy as np
    am,ao,bm,bo = (251,125,241,120) if a.full_range else (11,5,13,6)
    inp = (np.arange(ih*iw*s.cin,dtype=np.int64)%am-ao).reshape(ih,iw,s.cin)
    if not s.explicit_halo:
        inp = np.pad(inp,((1,1),(1,1),(0,0)))
    weight = (np.arange(9*s.cin*s.cout,dtype=np.int64)%bm-bo).reshape(3,3,s.cin,s.cout)
    acc = np.zeros((s.oh,s.ow,s.cout),dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            acc += inp[ky:ky+s.oh*s.stride:s.stride,kx:kx+s.ow*s.stride:s.stride,:] @ weight[ky,kx]
    if s.output_dtype == 'i8':
        acc = np.clip(np.rint(acc.astype(np.float32)*np.float32(s.scale)),0 if s.relu else -128,127).astype(np.int64)
    values = [str(v) for v in acc.reshape(-1)]
    source = out/'probe.c'
    ctype = 'int32_t' if s.output_dtype == 'i32' else 'int8_t'
    source.write_text(('''#include <stdint.h>
#include <stdio.h>
''' + f'''extern void {symbol}(int8_t*,int8_t*,{ctype}*);
static int8_t a[{ih*iw*s.cin}] __attribute__((aligned(64)));
static int8_t b[{9*s.cin*s.cout}] __attribute__((aligned(64)));
static struct {{{ctype} c[{len(values)}]; uint8_t guard[2048];}} box __attribute__((aligned(64)));
static const int32_t expected[] = {{{','.join(values)}}};
''' + '''static uint64_t cycles(void) {uint64_t v; __asm__ volatile("rdcycle %0":"=r"(v)::"memory"); return v;}
int main(void) {
 for (int i=0;i<sizeof(a);i++) a[i]=i%11-5;
 for (int i=0;i<sizeof(b);i++) b[i]=i%13-6;
 for (int i=0;i<2048;i++) box.guard[i]=0x5a;
 uint64_t t=cycles(); KERNEL_SYMBOL(a,b,box.c); t=cycles()-t;
 printf("GOLDEN_CONV_CYCLES %d\\n",(int)t);
 for(int i=0;i<sizeof(expected)/sizeof(expected[0]);i++) if(box.c[i]!=expected[i]) {
 printf("GOLDEN_CONV FAIL i=%d got=%d expected=%d\\n",i,box.c[i],expected[i]); return 1;}
 for(int i=0;i<2048;i++) if(box.guard[i]!=0x5a) {printf("GUARD_FAIL\\n"); return 2;}
 printf("GOLDEN_CONV PASS\\n"); return 0;
}
''').replace('KERNEL_SYMBOL',symbol).replace('i%11-5',f'i%{am}-{ao}').replace('i%13-6',f'i%{bm}-{bo}'))
    inputs = []
    if a.static_inputs:
        assembly = []
        for name, count, modulus, offset in [('a',ih*iw*s.cin,am,ao),('b',9*s.cin*s.cout,bm,bo)]:
            data = out/(name+'.bin')
            (np.arange(count,dtype=np.int64)%modulus-offset).astype(np.int8).tofile(data)
            assembly.append(f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{data}"\n')
        asm = out/'input_data.S';asm.write_text(''.join(assembly))
        obj = out/'input_data.o'
        subprocess.run([str(a.llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(asm),'-o',str(obj)],check=True,capture_output=True)
        inputs.append(obj)
        text = source.read_text()
        text = re.sub(r'static int8_t ([ab])\[(\d+)\] __attribute__\(\(aligned\(64\)\)\);',r'extern int8_t \1[\2];',text)
        text = re.sub(r' for \(int i=0;i<sizeof\([ab]\);i\+\+\) [ab]\[i\]=i%\d+-\d+;\n','',text)
        source.write_text(text)
    built = build_program([source,out/'kernel.o',*inputs],out,target='gemmini',extra_cflags=['-march=rv64gc'],max_loaded_bytes=None)
    audit = audit_elf(built.elf.read_bytes())
    (out/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['status'] != 'pass':
        raise RuntimeError('forbidden instruction in linked ELF')
    if a.build_only:
        print(built.elf)
        return 0
    run = run_on_gsim(built.elf,target='gemmini',max_cycles=a.max_cycles,timeout_s=a.timeout_s,backdoor=True,stdout_path=out/'gsim.stdout')
    match = re.search(r'GOLDEN_CONV_CYCLES (\d+)',run.stdout_tail)
    passed = run.completed and run.returncode == 0 and 'GOLDEN_CONV PASS' in run.stdout_tail
    result = dict(shape=asdict(s),wide_a=a.wide_a,separate_b_bank=a.separate_b_bank,band_rows=a.band_rows,flat_spatial=a.flat_spatial,static_inputs=a.static_inputs,input_value_rules=[am,ao,bm,bo],status='pass' if passed else 'fail',completed=run.completed,returncode=run.returncode,stderr=run.stderr_tail,kernel_cycles=int(match[1]) if match else None,
                  elf_sha256=built.elf_sha256,compilation=receipt,nofsm_audit=audit,gsim_engine=run.engine,stdout=run.stdout_tail)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('compilation','nofsm_audit','gsim_engine')},indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
