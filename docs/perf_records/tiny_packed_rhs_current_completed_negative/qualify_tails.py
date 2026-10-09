from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from xdsl.dialects.builtin import StringAttr
from mlir_oot.golden_gemm import GoldenGemm,Shape
from mlir_oot.golden_packed_rhs import GoldenPackedRhsGemm,PackedRhs
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from merlin.llvmlower.weight_panel import PackedArg,pack_bytes
from merlin.perf.layer_bench import build_program
work=Path(__file__).resolve().parent/'independent';work.mkdir(exist_ok=True)
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
spike='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
objects=[];declarations=[];calls=[];proofs=[]
rng=np.random.default_rng(31927)
for index,(m,n,k,nr,bn)in enumerate([(3,32,9,16,1),(7,128,65,64,4),(17,96,47,32,2),(16,64,32,64,4),(7,192,65,64,8)]):
 s=Shape(m,n,k,output_dtype='i32',bm=(m+15)//16,bn=bn,cache_a=True,wide_b=True,prefetch_b=k>16)
 a=rng.integers(-128,128,(m,k),dtype=np.int8);b=rng.integers(-128,128,(k,n),dtype=np.int8)
 gold=a.astype(np.int32)@b.astype(np.int32);packed=pack_bytes(b,PackedArg(0,(k,n),'i8',(),m,k,n,1,nr))
 for tag,provider in [('dense',GoldenGemm(s)),('packed',GoldenPackedRhsGemm(s,PackedRhs(k,n,nr)))]:
  module=provider.build();module.body.block.first_op.properties['sym_name']=StringAttr(f'{tag}_{index}')
  directory=work/f'{tag}_{index}';receipt=compile_module(module,llvm,directory);objects.append(directory/'kernel.o')
  proofs.append({'m':m,'n':n,'k':k,'nr':nr,'tag':tag,'receipt':receipt})
 def array(name,ctype,value):return f'static const {ctype} {name}[] __attribute__((aligned(64)))={{'+','.join(map(str,value.ravel()))+'};\n'
 declarations.extend([array(f'a{index}','int8_t',a),array(f'b{index}','int8_t',b),array(f'p{index}','int8_t',packed),array(f'g{index}','int32_t',gold),f'extern void dense_{index}(const int8_t*,const int8_t*,int32_t*),packed_{index}(const int8_t*,const int8_t*,int32_t*);\n'])
 calls.append(f'for(unsigned i=0;i<{m*n+128};i++)out[i]=0x73737373;dense_{index}(a{index},b{index},out);if(!check(out,g{index},{m*n}))return {index*2+1};for(unsigned i=0;i<{m*n+128};i++)out[i]=0x73737373;packed_{index}(a{index},p{index},out);if(!check(out,g{index},{m*n}))return {index*2+2};')
source='#include <stdint.h>\nextern int printf(const char*,...);\n'+''.join(declarations)+'''static int32_t out[17*128+128] __attribute__((aligned(64)));
static int check(const int32_t*out,const int32_t*expected,unsigned n){for(unsigned i=0;i<n;i++)if(out[i]!=expected[i]){printf("FAIL %u %d %d\\n",i,out[i],expected[i]);return 0;}for(unsigned i=n;i<n+128;i++)if(out[i]!=0x73737373){printf("GUARD_FAIL %u\\n",i);return 0;}return 1;}
int main(void){'''+''.join(calls)+'printf("PACKED_RHS_INDEPENDENT PASS 5\\n");return 0;}\n'
(work/'main.c').write_text(source)
subprocess.run([str(llvm/'clang'),*flags,'-c',str(work/'main.c'),'-o',str(work/'main.o')],check=True,capture_output=True)
built=build_program([*objects,work/'main.o'],work/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(built.elf.read_bytes());assert audit['status']=='pass'
sp=subprocess.run([spike,'--isa=rv64gc','--extension=gemmini',str(built.elf)],capture_output=True,text=True,timeout=120)
(work/'spike.log').write_text(sp.stdout+sp.stderr)
assert sp.returncode==0 and 'PACKED_RHS_INDEPENDENT PASS 5' in sp.stdout+sp.stderr,sp.stdout+sp.stderr
receipt={'schema':'packed_rhs_independent_actual_isa_v1','status':'pass','cases':proofs,'oracle':'Independent NumPy exact signed-i32 dot of unrelated full-range signed-i8 operands; mixed negative/cancellation, M/K tails and three legal NR widths. All output words and128 poisoned-tail words each arm. Static address proofs separately cover every B byte.','elf_sha256':built.elf_sha256,'spike_console_sha256':hashlib.sha256((work/'spike.log').read_bytes()).hexdigest(),'audit':audit}
(work/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PACKED_RHS_INDEPENDENT_ACTUAL_ISA PASS 5',flush=True)
