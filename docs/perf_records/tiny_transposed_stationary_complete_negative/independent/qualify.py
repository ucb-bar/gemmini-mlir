"""Unrelated full-i8 domain/tail matrices for existing transposed schedules."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from merlin.perf.layer_bench import build_program
from mlir_oot.golden_gemm import Shape,GoldenGemm
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
rng=np.random.default_rng(122013);objects=[];commands=[];cases=[];protos=[];inits=[];asm='.section .rodata\n'
for i,(m,n,k) in enumerate([(3,35,32),(7,73,64),(17,65,32),(16,64,32)]):
 a=rng.integers(-128,128,size=(m,k),dtype='i1');b=rng.integers(-128,128,size=(k,n),dtype='i1')
 a[0,:]=np.resize(np.array([-128,127,127,-128],dtype='i1'),k);b[:,0]=np.resize(np.array([127,127,-128,-128],dtype='i1'),k)
 bt=np.ascontiguousarray(b.T);gold=a.astype('i4')@b.astype('i4');assert k*128*128<(1<<31)
 for kind,value in [('a',a),('b',b),('bt',bt),('gold',gold.astype('<i4'))]:
  path=W/f'case{i}_{kind}.bin';value.tofile(path);asm+=f'.balign 64\n.global case{i}_{kind}\ncase{i}_{kind}:\n.incbin "{path}"\n'
  protos.append(f'extern const {"int32_t" if kind=="gold" else "int8_t"} case{i}_{kind}[];')
 shapes={'dense':Shape(m,n,k,output_dtype='i32',bm=(m+15)//16,bn=min(64,(n+15)//16),cache_a=True,wide_b=True,prefetch_b=True),'transposed':Shape(n,m,k,output_dtype='i32',bm=2,bn=(m+15)//16,cache_b=True,reuse_b=True,wide_a=True,pipeline_m=True,prefetch_m=True,banked_m=True)}
 receipts={}
 for arm,s in shapes.items():
  mod=GoldenGemm(s).build();from xdsl.dialects.builtin import StringAttr
  mod.body.block.first_op.sym_name=StringAttr(f'case{i}_{arm}')
  receipts[arm]=compile_module(mod,L,W/f'case{i}_{arm}');objects.append(W/f'case{i}_{arm}/kernel.o');protos.append(f'extern void case{i}_{arm}(const int8_t*,const int8_t*,int32_t*);')
 cases.append({'dimensions':[m,n,k],'prefix_bound':k*128*128,'schedules':{a:s.__dict__ for a,s in shapes.items()},'device_receipts':receipts})
 inits.append('{'+f'{m},{n},{k},case{i}_a,case{i}_b,case{i}_bt,case{i}_gold,case{i}_dense,case{i}_transposed'+'}')
(W/'data.S').write_text(asm)
main=r'''#include <stdint.h>
extern int printf(const char*,...);
PROTOTYPES
struct entry{unsigned m,n,k;const int8_t*a,*b,*bt;const int32_t*gold;void(*dense)(const int8_t*,const int8_t*,int32_t*),(*transposed)(const int8_t*,const int8_t*,int32_t*);};
static struct entry cases[]={ENTRIES};
static int8_t at[4096+128] __attribute__((aligned(64)));
static int32_t ct[4096+128] __attribute__((aligned(64))),out[4096+128] __attribute__((aligned(64)));
static uint32_t sum8(const int8_t*p,unsigned n){uint32_t h=0;for(unsigned i=0;i<n;i++)h=h*31+(uint8_t)p[i];return h;}
int main(void){for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
struct entry*e=&cases[ci];unsigned cn=e->m*e->n,an=e->m*e->k,bn=e->k*e->n;uint32_t ah=sum8(e->a,an),bh=sum8(e->b,bn),bth=sum8(e->bt,bn);
for(unsigned arm=0;arm<2;arm++){
for(unsigned i=0;i<cn+128;i++)out[i]=ct[i]=0x4d4d4d4d;for(unsigned i=0;i<an+128;i++)at[i]=0x4d;
if(arm){for(unsigned r=0;r<e->m;r++)for(unsigned c=0;c<e->k;c++)at[c*e->m+r]=e->a[r*e->k+c];e->transposed(e->bt,at,ct);for(unsigned r=0;r<e->m;r++)for(unsigned c=0;c<e->n;c++)out[r*e->n+c]=ct[c*e->m+r];}else e->dense(e->a,e->b,out);
for(unsigned i=0;i<cn;i++)if(out[i]!=e->gold[i]){printf("VALUE_FAIL %u %u %u %d %d\n",ci,arm,i,out[i],e->gold[i]);return 1;}
for(unsigned i=cn;i<cn+128;i++)if(out[i]!=0x4d4d4d4d||ct[i]!=0x4d4d4d4d){printf("OUTPUT_GUARD_FAIL %u\n",ci);return 2;}
for(unsigned i=an;i<an+128;i++)if(at[i]!=0x4d){printf("A_GUARD_FAIL %u\n",ci);return 3;}
if(ah!=sum8(e->a,an)||bh!=sum8(e->b,bn)||bth!=sum8(e->bt,bn)){printf("INPUT_FAIL %u\n",ci);return 4;}
}printf("INDEPENDENT_TRANSPOSED_I8_CASE PASS %u %u\n",ci,cn);
}printf("INDEPENDENT_TRANSPOSED_STATIONARY PASS\n");return 0;}'''.replace('PROTOTYPES','\n'.join(protos)).replace('ENTRIES',','.join(inits))
(W/'main.c').write_text(main)
for kind,source in [('data',W/'data.S'),('main',W/'main.c')]:
 argv=[str(L/'clang'),*flags,'-c',str(source),'-o',str(W/(kind+'.o'))];commands.append(argv);subprocess.run(argv,check=True,capture_output=True);objects.append(W/(kind+'.o'))
build=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass'
argv=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(build.elf)]
p=subprocess.run(argv,capture_output=True,text=True,timeout=300);log=p.stdout+p.stderr;(W/'spike.log').write_text(log)
assert p.returncode==0 and 'INDEPENDENT_TRANSPOSED_STATIONARY PASS' in log,log[-3000:]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r={'schema':'transposed_stationary_signed_i8_independent_actual_isa_v1','status':'pass','cases':cases,'oracle':'Independent NumPy mathematical signed-i32 integer dot, full-range signed-i8 operands/mixed cancellation. Each prefix bound below2^31. M/N tails and multiple output tiles, both original and transposed xDSL primitives, poisoned scratch/output and immutable inputs. Experimental C exact copies for independent cases; actual current timed copies separately upstream MLIR.','commands':commands,'spike_argv':argv,'elf_sha256':sha(build.elf),'spike_log_sha256':sha(W/'spike.log'),'audit':audit,'token_usage_available':False}
(W/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('TRANSPOSED_INDEPENDENT_SIGNED_I8_PASS',len(cases),flush=True)
