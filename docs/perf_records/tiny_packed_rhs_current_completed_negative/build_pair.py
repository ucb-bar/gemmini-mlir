from pathlib import Path
from dataclasses import asdict
import hashlib, json, os, subprocess, sys
import numpy as np
from merlin.llvmlower.weight_panel import PackedArg, pack_bytes
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.golden_gemm import Shape,GoldenGemm
from mlir_oot.golden_packed_rhs import PackedRhs, GoldenPackedRhsGemm
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf

work=Path(__file__).resolve().parent
capture=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture')
baseline=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_catalog/kernel.o')
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
gcc=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
spike=gcc.parent/'spike'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
def command(argv):
 subprocess.run([str(x)for x in argv],check=True,capture_output=True,text=True)
 commands.append([str(x)for x in argv])
commands=[]
M,N,K,NR=8,5632,2048,64
a=np.fromfile(capture/'a.bin',dtype=np.int8).reshape(M,K)
b=np.fromfile(capture/'b.bin',dtype=np.int8).reshape(K,N)
gold=np.fromfile(capture/'expected.bin',dtype='<i4').reshape(M,N)
packed=pack_bytes(b,PackedArg(0,(K,N),'i8',(),M,K,N,1,NR))
assert np.array_equal(packed.transpose(1,0,2).reshape(K,N),b)
assert sha(baseline)=='f8f49fbe81db69b7a0d8765eb618cbcfd31859b55c15efe9e78c3abe3e1e55f5'
(work/'packed_b.bin').write_bytes(packed.tobytes())
s=Shape(M,N,K,output_dtype='i32',bm=1,bn=64,cache_a=True,wide_b=True,prefetch_b=True)
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.llvm import LLVM
from mlir_oot.ir.gemmini_dialect import GEMMINI
ctx=Context();ctx.load_dialect(Builtin);ctx.load_dialect(LLVM);ctx.load_dialect(GEMMINI)
original_ir=baseline.parent/'kernel.gemmini.mlir'
original_module=Parser(ctx,original_ir.read_text()).parse_module()
original_function=next(fn for fn in original_module.body.block.ops if getattr(fn,'sym_name',None) is not None and fn.sym_name.data=='gemmini_golden_a5705ab56e324ba1')
regenerated=GoldenGemm(s).build().body.block.first_op
assert original_function.body.is_structurally_equivalent(regenerated.body), 'Current generated control differs from frozen original device semantics'
module=GoldenPackedRhsGemm(s,PackedRhs(K,N,NR)).build()
candidate_dir=work/'packed_device';device_receipt=compile_module(module,llvm,candidate_dir)
candidate=candidate_dir/'kernel.o'
# A/B/output virtual addresses and executable bytes are identical in the two
# linked images. Only readonly B's exact permutation and the same-size choice
# word differ; both kernels are retained in BOTH images.
main=r'''#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],input_b[];extern const int32_t expected[];
extern const uint64_t chosen_arm,expected_b_xor,expected_a_xor;
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*);
extern void gemmini_golden_gemm(const int8_t*,const int8_t*,int32_t*);
static int32_t destination[8*5632+128] __attribute__((aligned(64)));
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t words_xor(const void *ptr,unsigned long n){const uint64_t*p=ptr;uint64_t x=0;for(unsigned long i=0;i<n/8;i++)x^=p[i];return x;}
static int check(void){for(unsigned i=0;i<8*5632;i++)if(destination[i]!=expected[i]){printf("VALUE_FAIL %u %d %d\n",i,destination[i],expected[i]);return 0;}for(unsigned i=8*5632;i<8*5632+128;i++)if(destination[i]!=0x4d4d4d4d){printf("GUARD_FAIL %u\n",i);return 0;}return 1;}
int main(void){void(*fn)(const int8_t*,const int8_t*,int32_t*)=chosen_arm?gemmini_golden_gemm:gemmini_golden_a5705ab56e324ba1;
if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_FAIL\n");return 1;}
for(unsigned repeat=0;repeat<3;repeat++){for(unsigned i=0;i<8*5632+128;i++)destination[i]=0x4d4d4d4d;uint64_t t=tick();fn(input_a,input_b,destination);t=tick()-t;if(!check())return 2;printf("PACKED_RHS_COMPLETE_CYCLES %lu %u %lu\n",chosen_arm,repeat,t);}
if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_MUTATED\n");return 3;}
printf("PACKED_RHS_COMPLETE PASS %lu 45056\n",chosen_arm);return 0;}
'''
(work/'main.c').write_text(main)
command([llvm/'clang',*flags,'-c',work/'main.c','-o',work/'main.o'])
elfs={};audits={}
xor=lambda x:int(np.bitwise_xor.reduce(np.frombuffer(x,dtype='<u8')))
for arm,data in [('dense',capture/'b.bin'),('packed',work/'packed_b.bin')]:
 d=work/arm;d.mkdir(exist_ok=True)
 asm='.section .rodata\n'+''.join(f'.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n'for name,path in [('input_a',capture/'a.bin'),('input_b',data),('expected',capture/'expected.bin')])
 asm+=f'.balign 8\n.global chosen_arm\nchosen_arm: .quad {int(arm=="packed")}\n.global expected_b_xor\nexpected_b_xor: .quad {xor(data.read_bytes())}\n.global expected_a_xor\nexpected_a_xor: .quad {xor(a.tobytes())}\n'
 (d/'data.S').write_text(asm);command([llvm/'clang',*flags,'-c',d/'data.S','-o',d/'data.o'])
 built=build_program([baseline,candidate,work/'main.o',d/'data.o'],d/'build',target='gemmini',max_loaded_bytes=None)
 audits[arm]=audit_elf(built.elf.read_bytes());assert audits[arm]['status']=='pass'
 elfs[arm]=built.elf
 command([gcc.parent/'riscv64-unknown-elf-objcopy','--only-section=.text','-O','binary',built.elf,d/'text.bin'])
 nm=subprocess.check_output([gcc.parent/'riscv64-unknown-elf-nm','-n',str(built.elf)],text=True)
 (d/'symbols.txt').write_text(nm)
 sp=subprocess.run([str(spike),'--isa=rv64gc','--extension=gemmini',str(built.elf)],capture_output=True,text=True,timeout=120)
 (d/'spike.log').write_text(sp.stdout+sp.stderr)
 assert sp.returncode==0 and 'PACKED_RHS_COMPLETE PASS' in sp.stdout+sp.stderr
 print('PACKED_RHS_STRICT_FULL_PASS',arm,sha(built.elf),flush=True)
assert (work/'dense/text.bin').read_bytes()==(work/'packed/text.bin').read_bytes()
assert (work/'dense/symbols.txt').read_bytes()==(work/'packed/symbols.txt').read_bytes()
recipe={'schema':'current_source_bound_packed_rhs_matched_capsule_v1','m':M,'n':N,'k':K,'nr':NR,'output_block_bn_tiles':s.bn,'original_control_typed_body_structurally_identical':True,'same_executable_bytes':True,'same_all_symbol_addresses':True,'baseline_device_object_path':str(baseline),'baseline_device_object_sha256':sha(baseline),'candidate':device_receipt,'offline_packing':'Merlin weight_panel.pack_bytes; exact reversible permutation; excluded from runtime ROI as immutable parameter preparation. No byte savings claim.','input_scope':'First actual current1983 source-bound gate projection, original full eight-token model capture passes256000originalbits/Torch; complete45056i32 output, no shape narrowing.','roi':'Whole actual primitive kernel, including config, A/B DMA, CPU command arithmetic, compute, all output writes and final fence. Repeat0 cold, repeats1/2 warm separately. Input/checker/printing outside ROI, same code/addresses.','captured_inputs':{str(p):sha(p)for p in [capture/'a.bin',capture/'b.bin',capture/'expected.bin',capture/'validation.json']},'actual_build_commands':commands,'audits':audits,'elfs':{a:{'path':str(p),'sha256':sha(p)}for a,p in elfs.items()},'token_usage_available':False}
(work/'qualification.json').write_text(json.dumps(recipe,indent=2)+'\n')
print('PACKED_RHS_MATCHED_IDENTITIES_CLOSED',flush=True)
if '--gsim' in sys.argv:
 for arm in ['dense','packed']:
  result=run_on_gsim(elfs[arm],target='gemmini',max_cycles=50000000,timeout_s=4500,stdout_path=work/arm/'gsim.stdout')
  (work/arm/'gsim_receipt.json').write_text(json.dumps(asdict(result),indent=2,default=str)+'\n')
  print('PACKED_RHS_GSIM',arm,result.completed,flush=True)
  assert result.completed
