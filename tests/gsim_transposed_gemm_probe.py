"""Screen exact AB versus (B.T A.T).T with explicit online layout costs."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
from mlir_oot.golden_gemm import GoldenGemm,Shape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program,run_on_gsim


def run(work,llvm,n,k,transpose,separate):
    work.mkdir(parents=True,exist_ok=False)
    a=(((np.arange(8)[:,None]*7+np.arange(k)*3)%11)-5).astype(np.int8)
    b=(((np.arange(k)[:,None]*5+np.arange(n)*2)%13)-6).astype(np.int8)
    expected=a.astype(np.int32)@b.astype(np.int32)
    arrays={'activation':a,'weight':b.T.copy() if transpose else b,'expected':expected}
    asm=[]
    for name,v in arrays.items():
        path=work/(name+'.bin');v.tofile(path)
        asm.append(f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n')
    (work/'data.S').write_text(''.join(asm))
    subprocess.run([str(llvm/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(work/'data.S'),'-o',str(work/'data.o')],check=True)
    shape=Shape(n,8,k,output_dtype='i32',bm=16,bn=1,reuse_b=True,cache_b=True,separate_b_bank=separate) if transpose else Shape(8,n,k,output_dtype='i32',bm=1,bn=min(64,(n+15)//16),wide_b=True,cache_a=True)
    compile_module(GoldenGemm(shape).build(),llvm,work)
    source=r'''
#include <stdint.h>
#include <stdio.h>
extern int8_t activation[8][K];
extern int8_t weight[];
extern int32_t expected[8][N];
extern void gemmini_golden_gemm(int8_t*,int8_t*,int32_t*);
static int8_t at[K][8] __attribute__((aligned(64)));
static struct {int32_t values[8*N];uint8_t guard[2048];} out __attribute__((aligned(64)));
static int32_t restored[8][N] __attribute__((aligned(64)));
static inline uint64_t ticks(void){uint64_t t;asm volatile("rdcycle %0":"=r"(t)::"memory");return t;}
int main(void){
 for(int x=0;x<2048;x++)out.guard[x]=0x5a;
 uint64_t t=ticks();
#ifdef TRANSPOSE
 for(int q=0;q<K;q++)for(int i=0;i<8;i++)at[q][i]=activation[i][q];
#endif
 uint64_t activation_cycles=ticks()-t;t=ticks();
#ifdef TRANSPOSE
 gemmini_golden_gemm(weight,&at[0][0],out.values);
#else
 gemmini_golden_gemm(&activation[0][0],weight,out.values);
#endif
 uint64_t kernel_cycles=ticks()-t;t=ticks();
#ifdef TRANSPOSE
 for(int j=0;j<N;j++)for(int i=0;i<8;i++)restored[i][j]=out.values[j*8+i];
#endif
 uint64_t output_cycles=ticks()-t;
 for(int i=0;i<8;i++)for(int j=0;j<N;j++){
#ifdef TRANSPOSE
 int32_t actual=restored[i][j];
#else
 int32_t actual=out.values[i*N+j];
#endif
 if(actual!=expected[i][j]){printf("ORIENTATION FAIL %d %d %d %d\n",i,j,actual,expected[i][j]);return 1;}}
 for(int x=0;x<2048;x++)if(out.guard[x]!=0x5a){printf("ORIENTATION GUARD_FAIL\n");return 2;}
 printf("ORIENTATION_CYCLES %d %d %d\n",(int)activation_cycles,(int)kernel_cycles,(int)output_cycles);
 printf("ORIENTATION PASS\n");return 0;
}
'''
    (work/'probe.c').write_text(source)
    built=build_program([work/'probe.c',work/'kernel.o',work/'data.o'],work,target='gemmini',extra_cflags=[f'-DN={n}',f'-DK={k}','-march=rv64gc']+(['-DTRANSPOSE'] if transpose else []),max_loaded_bytes=None)
    audit=audit_elf(built.elf.read_bytes());assert audit['status']=='pass';(work/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    result=run_on_gsim(built.elf,target='gemmini',max_cycles=5000000,timeout_s=300,backdoor=True,stdout_path=work/'gsim.stdout')
    match=re.search(r'^ORIENTATION_CYCLES (\d+) (\d+) (\d+)$',result.stdout_tail,re.M)
    passed=result.completed and result.returncode==0 and 'ORIENTATION PASS' in result.stdout_tail and match is not None
    receipt=dict(status='pass' if passed else 'fail',shape=asdict(shape),original_shape=[8,n,k],transposed=transpose,separate_b_bank=separate,cycles=list(map(int,match.groups())) if match else None,cycles_order=['activation_transpose','kernel','output_transpose'],elf_sha256=built.elf_sha256,nofsm_audit=audit,fixture_hashes={name:hashlib.sha256(v.tobytes()).hexdigest() for name,v in arrays.items()},engine=result.engine,stdout_tail=result.stdout_tail,weight_transpose_policy='offline constant materialization, not included in online cycles')
    (work/'result.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n');print(json.dumps({key:receipt[key] for key in ['status','shape','cycles','elf_sha256']}),flush=True)
    if not passed:raise RuntimeError('GSIM orientation numerical gate failed')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--n',type=int,default=512);p.add_argument('--k',type=int,default=2048);p.add_argument('--variant',choices=['baseline','transpose','transpose-banked'],required=True);a=p.parse_args();run(a.work.resolve(),a.llvm_bin,a.n,a.k,a.variant!='baseline',a.variant=='transpose-banked')
