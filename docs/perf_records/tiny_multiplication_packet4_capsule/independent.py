from pathlib import Path
import hashlib,json,subprocess,runpy
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import MULTIPLY_FOUR_FEATURE
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent/'independent';W.mkdir(exist_ok=True)
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
source=runpy.run_path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/merlin/tests/ir/test_scalar_pointwise_packet.py')['multiplication_source'](3,11,alias=True)
objects=[]
for name,features in [('control',set()),('candidate',{MULTIPLY_FOUR_FEATURE})]:
 d=W/name;d.mkdir(exist_ok=True);(d/'source.mlir').write_text(source)
 llvm=lower_to_llvm_ir(source,workdir=d/'lower',features=features).replace('forward',name).replace('dealloc_helper',name+'_dealloc_helper')
 (d/'model.ll').write_text(llvm);subprocess.run([str(LLVM/'clang'),*flags,'-c',str(d/'model.ll'),'-o',str(d/'model.o')],check=True,capture_output=True);objects.append(d/'model.o')
c=r'''#include <stdint.h>
extern int printf(const char*,...);extern void*memcpy(void*,const void*,unsigned long);
struct d1{void*a,*p;long off,size,stride;};struct d2{void*a,*p;long off,size[2],stride[2];};
extern void _mlir_ciface_control(struct d2*,struct d1*,struct d2*,struct d2*),_mlir_ciface_candidate(struct d2*,struct d1*,struct d2*,struct d2*);
static float input[33],weight[11],row[3],out[33+32];static uint32_t reference[33];static uint8_t arena[65536];static unsigned cursor;static int errno_value;int*__errno(void){return &errno_value;}
void*malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint32_t bits(float f){uint32_t u;memcpy(&u,&f,4);return u;}static void word(float*p,uint32_t u){memcpy(p,&u,4);}
static void fillout(void){for(unsigned i=0;i<65;i++)word(out+i,0xdeadbeef);}
static unsigned fflags(void){uint64_t r;asm volatile("csrr %0,fflags":"=r"(r));return r;}
static void setenv(unsigned frm,unsigned flags){asm volatile("csrw frm,%0\ncsrw fflags,%1"::"r"((uint64_t)frm),"r"((uint64_t)flags):"memory");}
int main(void){static const uint32_t values[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x3f7fffff,0x3f800001,0x4b800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0x7fa54321,0xffa12345};unsigned checks=0;uint32_t seed=1357911;
struct d2 a={input,input,0,{3,11},{11,1}},r={row,row,0,{3,1},{1,1}},o={out,out,0,{3,11},{11,1}};struct d1 w={weight,weight,0,11,1};
for(unsigned mode=0;mode<5;mode++)for(unsigned trial=0;trial<128;trial++){
for(unsigned i=0;i<33;i++){seed=seed*1664525u+1013904223u;word(input+i,trial<36?values[(trial+i)%18]:seed);}for(unsigned i=0;i<11;i++){seed=seed*1664525u+1013904223u;word(weight+i,trial<36?values[(trial+3*i)%18]:seed);}for(unsigned i=0;i<3;i++){seed=seed*1664525u+1013904223u;word(row+i,trial<36?values[(trial+5*i)%18]:seed);}
uint32_t original[47];for(unsigned i=0;i<33;i++)original[i]=bits(input[i]);for(unsigned i=0;i<11;i++)original[33+i]=bits(weight[i]);for(unsigned i=0;i<3;i++)original[44+i]=bits(row[i]);
fillout();cursor=0;setenv(mode,trial&31);_mlir_ciface_control(&a,&w,&r,&o);unsigned control_flags=fflags();for(unsigned i=0;i<33;i++)reference[i]=bits(out[i]);for(unsigned i=33;i<65;i++)if(bits(out[i])!=0xdeadbeef){printf("CONTROL_GUARD\n");return 1;}
fillout();cursor=0;setenv(mode,trial&31);_mlir_ciface_candidate(&a,&w,&r,&o);unsigned candidate_flags=fflags();if(control_flags!=candidate_flags){printf("FLAGS_FAIL %u %u %u %u\n",mode,trial,control_flags,candidate_flags);return 2;}for(unsigned i=0;i<33;i++){if(reference[i]!=bits(out[i])){printf("RAW_FAIL %u %u %u %u %u\n",mode,trial,i,reference[i],bits(out[i]));return 3;}checks++;}for(unsigned i=33;i<65;i++)if(bits(out[i])!=0xdeadbeef){printf("CANDIDATE_GUARD\n");return 4;}
for(unsigned i=0;i<33;i++)if(original[i]!=bits(input[i]))return 5;for(unsigned i=0;i<11;i++)if(original[33+i]!=bits(weight[i]))return 6;for(unsigned i=0;i<3;i++)if(original[44+i]!=bits(row[i]))return 7;
}
setenv(0,0);printf("INDEPENDENT_MULTIPLY_PASS %u fivefrm stickyflags dirtytails immutable_inputs\n",checks);return 0;}
'''
(W/'main.c').write_text(c);subprocess.run([str(LLVM/'clang'),*flags,'-c',str(W/'main.c'),'-o',str(W/'main.o')],check=True,capture_output=True);objects.append(W/'main.o');objects.append(Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/mlir_rt.o'))
b=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass'
r=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(b.elf)],capture_output=True,text=True,timeout=300);log=r.stdout+r.stderr;(W/'spike.log').write_text(log);assert r.returncode==0 and 'INDEPENDENT_MULTIPLY_PASS 21120' in log,log[-3000:]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(schema='independent_typed_multiplication_packet_raw_f32_v1',original_model_operands_used=False,shape=[3,11],packet_width=4,raw_output_bit_checks=21120,all5_rounding_modes_and_seeded_sticky_flags_match=True,subnormals_overflow_infinity_signedzeros_nan_patterns=True,dirty_guard_bytes=128,source_alias_input_unchanged=True,ELF_sha256=b.elf_sha256,source_sha256=hashlib.sha256(source.encode()).hexdigest(),object_pins={str(p):sha(p)for p in objects},log_sha256=sha(W/'spike.log'),audit=audit,token_usage_available=False);(W/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
