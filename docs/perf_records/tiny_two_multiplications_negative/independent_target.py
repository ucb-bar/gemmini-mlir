"""Independent typed unequal dimensions/tails and raw floating corner cases."""
from pathlib import Path
import hashlib
import json
import subprocess
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import TWO_MULTIPLY_FOUR_FEATURE
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf

W = Path(__file__).resolve().parent/'independent_target_v1'
W.mkdir(exist_ok=False)
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
flags = ['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
def sha(p):
    with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
source = (W.parent/'capsule_abi.mlir').read_text().replace('1x8x2048','1x3x7').replace('tensor<2048xf32>','tensor<7xf32>')
(W/'source.mlir').write_text(source)
objects = []
for name,features in [('edge_control',set()),('edge_packet',{TWO_MULTIPLY_FOUR_FEATURE})]:
    raw = lower_to_llvm_ir(source,workdir=W/(name+'_lower'),features=features)
    raw = raw.replace('forward',name).replace('dealloc_helper',name+'_dealloc_helper')
    ll,obj = W/(name+'.ll'),W/(name+'.o')
    ll.write_text(raw)
    subprocess.run([str(L/'clang'),*flags,'-c',str(ll),'-o',str(obj)],check=True,capture_output=True)
    objects.append(obj)
main = r'''
#include <stdint.h>
extern int printf(const char*,...);
struct d3{void*alloc,*ptr;long off,size[3],stride[3];};struct d1{void*alloc,*ptr;long off,size,stride;};
extern void _mlir_ciface_edge_control(struct d3*,struct d3*,struct d1*,struct d3*);
extern void _mlir_ciface_edge_packet(struct d3*,struct d3*,struct d1*,struct d3*);
static float r[21],s[7],oldout[53],newout[53];static int32_t a[21];
static uint8_t arena[65536];static unsigned cursor;
void*malloc(unsigned long n){cursor=(cursor+63)&~63u;if(n>sizeof(arena)-cursor)return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static void word(float*p,uint32_t v){__builtin_memcpy(p,&v,4);}static uint32_t bits(const float*p){uint32_t v;__builtin_memcpy(&v,p,4);return v;}
int main(void){
 const uint32_t words[]={0,0x80000000,1,0x80000001,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc12345,0x7f812345,0x3f800001,0xbf800001};
 const int32_t integers[]={0,-1,2147483647,(-2147483647-1),16777217,-16777217,123456789};
 struct d3 dr={r,r,0,{1,3,7},{21,7,1}},da={a,a,0,{1,3,7},{21,7,1}};
 struct d1 ds={s,s,0,7,1};struct d3 old={oldout,oldout+16,0,{1,3,7},{21,7,1}},new={newout,newout+16,0,{1,3,7},{21,7,1}};
 unsigned checks=0;
 for(unsigned data=0;data<12;data++){
  for(unsigned i=0;i<21;i++){word(r+i,words[(i+data)%12]);a[i]=integers[(i+data)%7];}for(unsigned i=0;i<7;i++)word(s+i,words[(i*3+data)%12]);
  for(unsigned frm=0;frm<5;frm++){
   for(unsigned i=0;i<53;i++){word(oldout+i,0x4f123456);word(newout+i,0x4f123456);}cursor=0;
   asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");_mlir_ciface_edge_control(&dr,&da,&ds,&old);unsigned fa;asm volatile("csrr %0,fflags":"=r"(fa)::"memory");cursor=0;
   asm volatile("csrw fflags,%0"::"r"(8):"memory");_mlir_ciface_edge_packet(&dr,&da,&ds,&new);unsigned fb;asm volatile("csrr %0,fflags":"=r"(fb)::"memory");
   if(fa!=fb){printf("EDGE_FLAGS_FAIL %u %u %u %u\n",data,frm,fa,fb);return 1;}
   for(unsigned i=0;i<21;i++){if(bits(oldout+16+i)!=bits(newout+16+i)){printf("EDGE_BITS_FAIL %u %u %u\n",data,frm,i);return 2;}checks++;}
   for(unsigned i=0;i<16;i++)if(bits(oldout+i)!=0x4f123456||bits(newout+i)!=0x4f123456||bits(oldout+37+i)!=0x4f123456||bits(newout+37+i)!=0x4f123456)return 3;
   for(unsigned i=0;i<21;i++)if(bits(r+i)!=words[(i+data)%12]||a[i]!=integers[(i+data)%7])return 4;for(unsigned i=0;i<7;i++)if(bits(s+i)!=words[(i*3+data)%12])return 5;
  }
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("TWO_MUL_INDEPENDENT PASS %u 5\n",checks);printf("DONE\n");return 0;
}
'''
(W/'main.c').write_text(main)
obj=W/'main.o'
subprocess.run([str(L/'clang'),*flags,'-c',str(W/'main.c'),'-o',str(obj)],check=True,capture_output=True)
build=build_program([*objects,obj],W/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(build.elf.read_bytes())
assert audit['status']=='pass'
spike=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(build.elf)],capture_output=True,text=True,timeout=300)
console=spike.stdout+spike.stderr
(W/'spike.log').write_text(console)
assert spike.returncode==0 and 'TWO_MUL_INDEPENDENT PASS 1260 5' in console and 'DONE' in console
receipt={'schema':'two_multiplications_independent_actual_target_v1','status':'pass','compiled_source_shape':[1,3,7],'complete_scalar_operations':'signedi32cast,twoorderedf32multiplications,residualf32add','compared_raw_words':1260,'rounding_modes':5,'sticky_flags_match':True,'source_inputs_unchanged':True,'guard_bytes_each_output':128,'cases':'12 unrelated rawword rotations including signedzero,subnormal,overflow,infinities,quiet/signalingNaN,int32limits andinexactcasts; unequal3rows/7columns andthree-lane tail','zero_FSM':True,'pins':{str(p):sha(p)for p in [Path(__file__),W/'source.mlir',W/'main.c',*objects,obj,build.elf,W/'spike.log',L/'clang']},'token_usage_available':False}
(W/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('TWO_MUL_INDEPENDENT_ACTUAL_TARGET_PASS',1260,flush=True)
