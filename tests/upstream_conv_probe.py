"""Compile one actual upstream gather/matmul through host layouts and device IR."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from mlir_oot.frontend.parse import parse_module
from mlir_oot.direct_conv_binding import match, capture, rewrite, emit_c_adapter, serialize
from mlir_oot.golden_conv import GoldenConv
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from mlir_oot.contraction_patterns import match_integer_gemm
from merlin.perf.layer_bench import build_program


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--region',required=True);p.add_argument('--workdir',type=Path,required=True);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--spike',type=Path,required=True)
    a=p.parse_args();out=a.workdir.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.source.read_text();m=parse_module(source)
    candidates=[op for op in m.walk() if op.name in ('linalg.matmul','linalg.generic') and getattr(op.attributes.get('prov.region_id'),'data','')==a.region]
    matched=[]
    for op in candidates:
        try: matched.append(match(op))
        except ValueError: pass
    if len(matched)!=1: raise ValueError(f'need exactly one proven direct conv, got {len(matched)}')
    b=matched[0];s=b.shape;c=capture(b);(out/'original.mlir').write_text(str(c))
    declaration=rewrite(match(next(op for op in c.walk() if match_integer_gemm(op))))
    (out/'rewritten.mlir').write_text(serialize(c,declaration))
    compilation=compile_module(GoldenConv(s).build(),a.llvm_bin,out/'device')
    commands=[
      [str(a.llvm_bin/'mlir-opt'),str(out/'rewritten.mlir'),'--canonicalize','--one-shot-bufferize=bufferize-function-boundaries','--convert-linalg-to-loops','--expand-strided-metadata','--lower-affine','--convert-scf-to-cf','--convert-to-llvm','--reconcile-unrealized-casts','-o',str(out/'host.llvm.mlir')],
      [str(a.llvm_bin/'mlir-translate'),'--mlir-to-llvmir',str(out/'host.llvm.mlir'),'-o',str(out/'host.ll')],
      [str(a.llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-O2','-c',str(out/'host.ll'),'-o',str(out/'host.o')]]
    for command in commands: subprocess.run(command,check=True)
    inp=(np.arange(s.cin*(s.h+2)*(s.w+2),dtype=np.int32)%11-5).reshape(s.cin,s.h+2,s.w+2)
    weights=(np.arange(s.cout*s.cin*9,dtype=np.int32)%13-6).reshape(s.cout,s.cin,3,3)
    if b.orientation=='spatial_first':
        weights=weights.reshape(3,3,s.cin,s.cout).transpose(3,2,0,1)
    expected=np.zeros((s.cout,s.oh,s.ow),dtype=np.int32)
    for ky in range(3):
        for kx in range(3):
            pixels=inp[:,ky:ky+s.oh*s.stride:s.stride,kx:kx+s.ow*s.stride:s.stride].reshape(s.cin,-1)
            expected+=(weights[:,:,ky,kx]@pixels).reshape(expected.shape)
    if b.orientation=='spatial_first':
        expected=expected.reshape(s.cout,-1).T.copy()
    weight_shape=b.weight.type.get_shape()
    result_shape=b.contraction.results[0].type.get_shape()
    adapter=emit_c_adapter(s)
    main='''
#include <stdio.h>
extern void _mlir_ciface_captured_conv(memref2*,memref2*,memref4*);
static unsigned char heap[16*1024*1024] __attribute__((aligned(64)));
static uintptr_t used;
void *malloc(size_t n) {uintptr_t p=(used+63)&~63UL; if(p+n>sizeof(heap)) exit(4);used=p+n;return heap+p;}
void free(void *p) {(void)p;}
'''+f'''
static int8_t weights[{weights.size}] __attribute__((aligned(64)));
static int8_t activation[{inp.size}] __attribute__((aligned(64)));
static const int32_t expected[{expected.size}]={{{','.join(map(str,expected.flat))}}};
int main(void) {{
 for(int i=0;i<{weights.size};i++) weights[i]=i%13-6;
 for(int i=0;i<{inp.size};i++) activation[i]=i%11-5;
 memref2 w={{weights,weights,0,{{{weight_shape[0]},{weight_shape[1]}}},{{{weight_shape[1]},1}}}};
 memref4 a={{activation,activation,0,{{1,{s.cin},{s.h+2},{s.w+2}}},{{{inp.size},{(s.h+2)*(s.w+2)},{s.w+2},1}}}};
 memref2 r;
 _mlir_ciface_captured_conv(&r,&w,&a);
 for(int n=0;n<{result_shape[0]};n++) for(int p=0;p<{result_shape[1]};p++) {{
 int got=((int32_t*)r.aligned)[r.offset+n*r.strides[0]+p*r.strides[1]];
 if(got!=expected[n*{result_shape[1]}+p]) {{printf("UPSTREAM_CONV FAIL n=%d p=%d got=%d expected=%d\\n",n,p,got,expected[n*{result_shape[1]}+p]);return 1;}}
 }}
 printf("UPSTREAM_CONV PASS values={expected.size} heap=%d\\n",(int)used);return 0;
}}
'''
    (out/'probe.c').write_text(adapter+main)
    built=build_program([out/'probe.c',out/'host.o',out/'device/kernel.o'],out,target='gemmini',extra_cflags=['-march=rv64gc','-fno-builtin'],max_loaded_bytes=None)
    audit=audit_elf(built.elf.read_bytes())
    if audit['status']!='pass':raise ValueError('final ELF contains forbidden instructions')
    run=subprocess.run([str(a.spike),'--extension=gemmini',str(built.elf)],capture_output=True,text=True,timeout=120)
    passed=run.returncode==0 and 'UPSTREAM_CONV PASS' in run.stdout
    result=dict(status='pass' if passed else 'fail',source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_region=a.region,orientation=b.orientation,shape=asdict(s),elf_sha256=built.elf_sha256,nofsm_audit=audit,compilation=compilation,host_commands=commands,host_object_sha256=hashlib.sha256((out/'host.o').read_bytes()).hexdigest(),rewritten_sha256=hashlib.sha256((out/'rewritten.mlir').read_bytes()).hexdigest(),stdout=run.stdout,stderr=run.stderr,scope='Selected captured gather/matmul plus generated host boundary conversions; functional Spike only; nonzero quantized halo retained; no whole-model or FireSim claim')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(run.stdout);return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
