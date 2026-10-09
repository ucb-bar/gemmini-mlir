"""Complete current primitive with exact transposition and stationary reuse."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.perf.layer_bench import build_program
from mlir_oot.golden_gemm import Shape,GoldenGemm
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
CAP=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture')
BASE=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_catalog/kernel.o')
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
M,N,K=8,5632,2048
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(argv):
 commands.append(list(map(str,argv)));subprocess.run(commands[-1],check=True,capture_output=True)
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
b=np.fromfile(CAP/'b.bin',dtype='i1').reshape(K,N);bt=np.ascontiguousarray(b.T)
assert np.array_equal(bt.T,b);bt.tofile(W/'parameter_transposed.bin')
assert K*128*128<(1<<31)
s=Shape(N,M,K,output_dtype='i32',bm=2,bn=1,cache_b=True,reuse_b=True,wide_a=True,pipeline_m=True,prefetch_m=True,banked_m=True)
s.validate();module=GoldenGemm(s).build();compiled=compile_module(module,L,W/'transposed_device')
# Pure generic memref permutations use upstream lowering and caller-owned
# scratch, with no alias assertion, public output substitution or target code.
def transpose_function(name,rows,cols,dtype):
 return f'''func.func @{name}(%a: memref<{rows}x{cols}x{dtype}>,%out: memref<{cols}x{rows}x{dtype}>) attributes {{llvm.emit_c_interface}} {{
 %z=arith.constant 0:index
 %one=arith.constant 1:index
 %r=arith.constant {rows}:index
 %c=arith.constant {cols}:index
 scf.for %i=%z to %r step %one {{
  scf.for %j=%z to %c step %one {{
   %v=memref.load %a[%i,%j]:memref<{rows}x{cols}x{dtype}>
   memref.store %v,%out[%j,%i]:memref<{cols}x{rows}x{dtype}>
  }}
 }}
 return
}}'''
source='module {'+transpose_function('prepare_activation',M,K,'i8')+transpose_function('restore_output',N,M,'i32')+'}'
(W/'permutations.mlir').write_text(source)
llvm=lower_to_llvm_ir(source,workdir=W/'permutations')
(W/'permutations/model.ll').write_text(llvm)
run([L/'clang',*flags,'-c',W/'permutations/model.ll','-o',W/'permutations/model.o'])
main=r'''#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],input_b[];extern const int32_t expected[];
extern const uint64_t chosen_arm,expected_b_xor,expected_a_xor;
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*),gemmini_golden_gemm(const int8_t*,const int8_t*,int32_t*);
struct d2{void*a,*p;long off,size[2],stride[2];};
extern void _mlir_ciface_prepare_activation(struct d2*,struct d2*),_mlir_ciface_restore_output(struct d2*,struct d2*);
static int8_t activation_t[2048*8+128] __attribute__((aligned(64)));
static int32_t output_t[5632*8+128] __attribute__((aligned(64))),destination[8*5632+128] __attribute__((aligned(64)));
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t words_xor(const void*ptr,unsigned long n){const uint64_t*p=ptr;uint64_t x=0;for(unsigned long i=0;i<n/8;i++)x^=p[i];return x;}
static int check(void){for(unsigned i=0;i<8*5632;i++)if(destination[i]!=expected[i]){printf("VALUE_FAIL %u %d %d\n",i,destination[i],expected[i]);return 0;}for(unsigned i=8*5632;i<8*5632+128;i++)if(destination[i]!=0x4d4d4d4d||output_t[i]!=0x4d4d4d4d){printf("OUTPUT_GUARD_FAIL %u\n",i);return 0;}for(unsigned i=2048*8;i<2048*8+128;i++)if(activation_t[i]!=0x4d){printf("A_GUARD_FAIL %u\n",i);return 0;}return 1;}
int main(void){
 struct d2 a={(void*)input_a,(void*)input_a,0,{8,2048},{2048,1}},at={activation_t,activation_t,0,{2048,8},{8,1}},ct={output_t,output_t,0,{5632,8},{8,1}},c={destination,destination,0,{8,5632},{5632,1}};
 if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_FAIL\n");return 1;}
 for(unsigned repeat=0;repeat<3;repeat++){
 for(unsigned i=0;i<8*5632+128;i++)destination[i]=output_t[i]=0x4d4d4d4d;for(unsigned i=0;i<2048*8+128;i++)activation_t[i]=0x4d;
 uint64_t t=tick();
 if(chosen_arm){_mlir_ciface_prepare_activation(&a,&at);gemmini_golden_gemm(input_b,activation_t,output_t);_mlir_ciface_restore_output(&ct,&c);}else gemmini_golden_a5705ab56e324ba1(input_a,input_b,destination);
 t=tick()-t;
 if(!check())return 2;
 if(chosen_arm)for(unsigned i=0;i<2048;i++)for(unsigned j=0;j<8;j++)if(activation_t[i*8+j]!=input_a[j*2048+i]){printf("A_PERM_FAIL\n");return 3;}
 printf("TRANSPOSED_STATIONARY_COMPLETE_CYCLES %lu %u %lu\n",chosen_arm,repeat,t);
 }
 if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_MUTATED\n");return 4;}
 printf("TRANSPOSED_STATIONARY_COMPLETE PASS %lu 45056\n",chosen_arm);return 0;
}'''
(W/'main.c').write_text(main);run([L/'clang',*flags,'-c',W/'main.c','-o',W/'main.o'])
xor=lambda x:int(np.bitwise_xor.reduce(np.frombuffer(x,dtype='<u8')))
elfs={};audits={}
for arm,data in [('dense',CAP/'b.bin'),('transposed',W/'parameter_transposed.bin')]:
 D=W/arm;D.mkdir()
 assembly='.section .rodata\n'+''.join(f'.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n'for name,path in [('input_a',CAP/'a.bin'),('input_b',data),('expected',CAP/'expected.bin')])
 assembly+=f'.balign 8\n.global chosen_arm\nchosen_arm: .quad {int(arm=="transposed")}\n.global expected_b_xor\nexpected_b_xor: .quad {xor(data.read_bytes())}\n.global expected_a_xor\nexpected_a_xor: .quad {xor((CAP/"a.bin").read_bytes())}\n'
 (D/'data.S').write_text(assembly);run([L/'clang',*flags,'-c',D/'data.S','-o',D/'data.o'])
 build=build_program([BASE,W/'transposed_device/kernel.o',W/'permutations/model.o',W/'main.o',D/'data.o'],D/'build',target='gemmini',max_loaded_bytes=None)
 audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass';audits[arm]=audit;elfs[arm]=build.elf
 run([GCC.parent/'riscv64-unknown-elf-objcopy','--only-section=.text','-O','binary',build.elf,D/'text.bin'])
 (D/'symbols.txt').write_bytes(subprocess.check_output([GCC.parent/'riscv64-unknown-elf-nm','-n',str(build.elf)]))
 sp=subprocess.run([str(GCC.parent/'spike'),'--isa=rv64gc','--extension=gemmini',str(build.elf)],capture_output=True,text=True,timeout=180)
 (D/'spike.log').write_text(sp.stdout+sp.stderr);assert sp.returncode==0 and 'TRANSPOSED_STATIONARY_COMPLETE PASS' in sp.stdout+sp.stderr,(sp.stdout+sp.stderr)[-3000:]
 print('CURRENT_TRANSPOSED_NUMERIC_PASS',arm,sha(build.elf),flush=True)
assert (W/'dense/text.bin').read_bytes()==(W/'transposed/text.bin').read_bytes()
assert (W/'dense/symbols.txt').read_bytes()==(W/'transposed/symbols.txt').read_bytes()
q={'schema':'current_exact_transposed_stationary_complete_capsule_v1','status':'pass','dimensions_original':[M,N,K],'dimensions_primitive':[N,M,K],'original_full_signed_i8_range_int32_prefix_bound':K*128*128,'all45056_original_i32_outputs_exact':True,'poisoned_scratch_output_tails':128,'immutable_inputs_preserved':True,'same_executable_bytes_and_all_symbol_addresses':True,'source_permutation_ownership':'Generic exact memref copy permutations lowered upstream; output/activation private scratch explicit. Offline readonly parameter transpose exact reversible permutation, no public output layout change. No production transform/route yet.','source_arithmetic':'Every output sums same original signed-i8 products in increasingK, no bias/requantization. Int32 prefix range proved for full i8domain; no overflow/reassociation needed.','resources':{'cached_B_rows':2048,'A_rows_per_slot':4096,'SPAD_banks_A':[0,1],'SPAD_bank_B':2,'accumulator_rows_two_slots':64},'source_schedule':s.__dict__,'target_compilation':compiled,'runtime_copy_mlir_sha256':sha(W/'permutations.mlir'),'actual_build_commands':commands,'baseline_object_sha256':sha(BASE),'captured_original_inputs':{str(p):sha(p)for p in CAP.glob('*.bin')},'audits':audits,'elfs':{a:{'path':str(p),'sha256':sha(p)}for a,p in elfs.items()},'ROI':'Complete original contraction including runtime A transpose, primitive config/DMA/CPUissue/compute/readout/fence, restoring dense C and ABI calls. Readonly parameter permutation offline. All checker/poison/printing outsideROI, scratch guard and original45056i32 check afterward.','performance':None,'token_usage_available':False}
(W/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print('CURRENT_TRANSPOSED_COMPLETE_PAIR_READY',flush=True)
