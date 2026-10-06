"""Independent original unary source arithmetic, all target rounding modes/flags."""
from pathlib import Path
import hashlib,json,subprocess,importlib.util
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_squared_sum import FEATURE
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
P=Path(__file__).resolve().parent
C=Path('/scratch/agustin/tmp/merlin-scalar-squared-sum-20261006')
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
FROZEN=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
text='''module {func.func @forward(%a:tensor<12x7xf32>,%seed:tensor<12xf32>) -> tensor<12xf32> attributes {llvm.emit_c_interface} {
 %r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0)>],iterator_types=["parallel","reduction"]}
 ins(%a:tensor<12x7xf32>) outs(%seed:tensor<12xf32>) {^bb0(%x:f32,%z:f32):
 %p=arith.mulf %x,%x:f32 %v=arith.addf %p,%z:f32 linalg.yield %v:f32} -> tensor<12xf32>
 return %r:tensor<12xf32>
}}'''
(P/'source.mlir').write_text(text)
objects=[]
for name,features in [('control',set()),('selected',{FEATURE})]:
 out=P/name
 raw=lower_to_llvm_ir(text,workdir=out,features=features)
 renamed=raw.replace('forward',name)
 (out/'actual.ll').write_text(renamed)
 obj=out/'model.o';argv=[str(L/'clang'),*flags,'-c',str(out/'actual.ll'),'-o',str(obj)]
 subprocess.run(argv,check=True,capture_output=True);objects.append(obj)
main=r'''#include <stdint.h>
extern int printf(const char*,...);extern void *memcpy(void*,const void*,unsigned long);
struct d1{void*a,*p;long o,s,t;};struct d2{void*a,*p;long o,s[2],t[2];};
extern void _mlir_ciface_control(struct d2*,struct d1*,struct d1*),_mlir_ciface_selected(struct d2*,struct d1*,struct d1*);
static const uint32_t words[]={0,0x80000000,1,0x80000001,0x3f800001,0xbf800001,0x7f800000,0xff800000,0x7fc00123,0x7f800123,0x7f7fffff,0xff7fffff,0x45800000,0xcb800000};
static uint32_t a[84],save[84],seeds[12],reference[12],initial[12],result[12+32];
static unsigned char arena[8192];static unsigned cursor;
static int errno_value;int *__errno(void){return &errno_value;}
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
int main(void){struct d2 in={a,a,0,{12,7},{7,1}};struct d1 s={seeds,seeds,0,12,1},out={result+16,result+16,0,12,1};
 for(unsigned rot=0;rot<14;rot++){
  for(unsigned i=0;i<84;i++)a[i]=save[i]=words[(i+rot)%14];
  for(unsigned i=0;i<12;i++)initial[i]=words[(i+rot+4)%14];
  for(unsigned frm=0;frm<5;frm++){
   unsigned long oldflags,newflags;
   for(unsigned i=0;i<12;i++)seeds[i]=initial[i];for(unsigned i=0;i<44;i++)result[i]=0xdeadbeef;cursor=0;
   asm volatile("csrw frm,%0;csrw fflags,%1"::"r"((unsigned long)frm),"r"(8UL):"memory");
   _mlir_ciface_control(&in,&s,&out);asm volatile("csrr %0,fflags":"=r"(oldflags)::"memory");
   for(unsigned i=0;i<12;i++)reference[i]=result[i+16];
   for(unsigned i=0;i<12;i++)seeds[i]=initial[i];for(unsigned i=0;i<44;i++)result[i]=0xdeadbeef;cursor=0;
   asm volatile("csrw fflags,%0"::"r"(8UL):"memory");
   _mlir_ciface_selected(&in,&s,&out);asm volatile("csrr %0,fflags":"=r"(newflags)::"memory");
   if(oldflags!=newflags){printf("SQUARED_FLAGS_FAIL %u %u %lu %lu\n",rot,frm,oldflags,newflags);return 1;}
   for(unsigned i=0;i<12;i++)if(result[i+16]!=reference[i]){printf("SQUARED_WORD_FAIL %u %u %u %u %u\n",rot,frm,i,result[i+16],reference[i]);return 2;}
   for(unsigned i=0;i<16;i++)if(result[i]!=0xdeadbeef||result[i+28]!=0xdeadbeef){printf("SQUARED_GUARD_FAIL\n");return 3;}
   for(unsigned i=0;i<84;i++)if(a[i]!=save[i]){printf("SQUARED_INPUT_FAIL\n");return 4;}
  }
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 printf("INDEPENDENT_SQUARED_SUM_FIVE_FRM_STICKY PASS 840\nDONE\n");return 0;
}'''
(P/'main.c').write_text(main)
subprocess.run([str(L/'clang'),*flags,'-c',str(P/'main.c'),'-o',str(P/'main.o')],check=True,capture_output=True)
objects.extend([P/'main.o',FROZEN/'merlin/runtime/baremetal/spike/mlir_rt.o']if (FROZEN/'merlin/runtime/baremetal/spike/mlir_rt.o').exists()else[P/'main.o',Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/mlir_rt.o')])
b=build_program(objects,P/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass'
argv=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)]
r=subprocess.run(argv,capture_output=True,text=True,timeout=180);console=r.stdout+r.stderr;(P/'spike.log').write_text(console)
assert r.returncode==0 and 'INDEPENDENT_SQUARED_SUM_FIVE_FRM_STICKY PASS 840'in console and 'DONE'in console,console[-3000:]
receipt=dict(schema='independent_source_squared_sum_target_v1',status='pass',shape=[12,7],rotations=14,rounding_modes=5,raw_final_f32_word_comparisons=840,initial_sticky_flags=8,signed_zero_subnormal_infinity_qnan_snan_overflow=True,input_unchanged=True,dirty_prefix_suffix_guard_words=32,zero_FSM=True,argv=argv,elf_sha256=b.elf_sha256,spike_sha256=sha(P/'spike.log'),core_module_sha256=sha(C/'src/merlin/llvmlower/scalar_squared_sum.py'),token_usage_available=False)
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('INDEPENDENT_SQUARED_SUM_TARGET_PASS',flush=True)
