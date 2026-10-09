from pathlib import Path
import json,hashlib,subprocess,os
from dataclasses import asdict
import numpy as np
from safetensors import safe_open
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.broadcast_math_hoist import FEATURE as HOIST
from merlin.llvmlower.scalar_pointwise_packet import FEATURE as FMA_PACKET, MULTIPLY_FOUR_FEATURE,TWO_MULTIPLY_FOUR_FEATURE
from merlin.llvmlower.bufferized_result_identity import FEATURE as IDENTITY
from merlin.llvmlower.scalar_squared_sum import FEATURE
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=Path(__file__).resolve().parent;source=(work/'source.mlir').read_text()
frozen=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2')
runtime=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/abi/mlir_runtime.c')
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
a=np.fromfile(work/'activation.bin',dtype='<f4').reshape(1,8,2048)
w=np.fromfile(work/'weight.bin',dtype='<f4')
assert a.dtype==w.dtype==np.float32
objects=[];references=[];proofs=[]
base_features={HOIST,MULTIPLY_FOUR_FEATURE,TWO_MULTIPLY_FOUR_FEATURE,IDENTITY,FMA_PACKET,'fuse_quantize_round_convert','fuse_activation_polynomial_fma'}
for arm,feature in [('control',None),('accumulated',FEATURE)]:
 directory=work/arm;features=base_features if feature is None else base_features|{feature}
 ll=lower_to_llvm_ir(source,workdir=directory,features=features).replace('forward',arm).replace('dealloc_helper',arm+'_dealloc_helper')
 native,npf=rewrite(ll,host_isa='portable');target,tpf=rewrite(ll,host_isa='rv64gc',combine_clamp=True)
 (directory/'native.ll').write_text(native);(directory/'target.ll').write_text(target)
 so=directory/f'{arm}.so';subprocess.run([str(llvm/'clang'),'-O3','-shared','-fPIC','-ffp-contract=off',str(directory/'native.ll'),str(runtime),'-lm','-o',str(so)],check=True,capture_output=True)
 output=np.zeros(a.shape,dtype='i1');HostModel.load(str(so),name=arm)([(w.ctypes.data,w.shape),(a.ctypes.data,a.shape),(output.ctypes.data,output.shape)])
 references.append(output)
 if len(references)==2:assert np.array_equal(*references)
 target_obj=directory/'model.o';subprocess.run([str(llvm/'clang'),*flags,'-c',str(directory/'target.ll'),'-o',str(target_obj)],check=True,capture_output=True);objects.append(target_obj)
 proofs.append({'arm':arm,'features':sorted(features),'target_rne_sites':len(tpf['routes']),'llvm_sha256':hashlib.sha256(ll.encode()).hexdigest(),'target_llvm_sha256':hashlib.sha256(target.encode()).hexdigest(),'native_output_sha256':hashlib.sha256(output.tobytes()).hexdigest()})
 print('SOURCE_SQUARED_SUM_NORM_NATIVE_EXACT',arm,flush=True)
assert references[0].tobytes()==(work/'expected.bin').read_bytes()
assembly='.section .data\n'+''.join(f'.balign 64\n.global {name}\n{name}:\n.incbin "{work}/{file}"\n' for name,file in [('weight','weight.bin'),('activation','activation.bin'),('expected','expected.bin')])
(work/'data.S').write_text(assembly);subprocess.run([str(llvm/'clang'),*flags,'-c',str(work/'data.S'),'-o',str(work/'data.o')],check=True)
c=r'''#include <stdint.h>
extern int printf(const char *,...);
extern float weight[],activation[];extern int8_t expected[];
struct d1 {void*a,*p;long off,size,stride;};struct d3{void*a,*p;long off,size[3],stride[3];};
extern void _mlir_ciface_control(struct d1*,struct d3*,struct d3*),_mlir_ciface_accumulated(struct d1*,struct d3*,struct d3*);
static int8_t out[16384+128];static uint8_t arena[256*1024];static unsigned cursor;static int errno_value;
int *__errno(void){return &errno_value;}
extern void *memcpy(void*d,const void*s,unsigned long n);
extern void *memset(void*d,int v,unsigned long n);
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static void mode(unsigned m){asm volatile("csrw frm,%0"::"r"((uint64_t)m):"memory");}
static unsigned flags(void){unsigned long f;asm volatile("csrr %0,fflags":"=r"(f));return (unsigned)f;}
static void clear(void){asm volatile("csrw fflags,zero":::"memory");}
int main(void){struct d1 w={weight,weight,0,2048,1};struct d3 a={activation,activation,0,{1,8,2048},{16384,2048,1}},o={out,out,0,{1,8,2048},{16384,2048,1}};unsigned control_flags[5];static int8_t mode_reference[16384];
for(unsigned m=0;m<5;m++){mode(m);clear();cursor=0;_mlir_ciface_control(&w,&a,&o);control_flags[m]=flags();memcpy(mode_reference,out,16384);if(m==0)for(unsigned i=0;i<16384;i++)if(out[i]!=expected[i]){printf("FAIL_NATIVE %u\n",i);return 1;}
clear();cursor=0;_mlir_ciface_accumulated(&w,&a,&o);unsigned candidate_flags=flags();if(candidate_flags!=control_flags[m]){printf("FLAG_FAIL %u %u %u\n",m,control_flags[m],candidate_flags);return 2;}for(unsigned i=0;i<16384;i++)if(out[i]!=mode_reference[i]){printf("MODE_FAIL %u %u\n",m,i);return 3;}}
mode(0);for(unsigned r=0;r<2;r++)for(unsigned step=0;step<2;step++){unsigned arm=r==0?step:1-step;for(unsigned i=0;i<sizeof(out);i++)out[i]=73;cursor=0;uint64_t t=tick();if(arm==0)_mlir_ciface_control(&w,&a,&o);else _mlir_ciface_accumulated(&w,&a,&o);t=tick()-t;
for(unsigned i=0;i<16384;i++)if(out[i]!=expected[i]){printf("VALUE_FAIL %u %u\n",arm,i);return 4;}for(unsigned i=16384;i<sizeof(out);i++)if(out[i]!=73){printf("GUARD_FAIL\n");return 5;}printf("SQUARED_SUM_NORM_COMPLETE_CYCLES %u %u %lu\n",r,arm,t);}
printf("SOURCE_SQUARED_SUM_NORM_COMPLETE PASS\n");return 0;}
'''
(work/'main.c').write_text(c);subprocess.run([str(llvm/'clang'),*flags,'-ffp-contract=off','-c',str(work/'main.c'),'-o',str(work/'main.o')],check=True,capture_output=True)
objects.extend([work/'main.o',work/'data.o',frozen/'build/mlir_rt.o'])
b=build_program(objects,work/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass'
sp=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,timeout=240);console=sp.stdout+sp.stderr;(work/'spike.log').write_text(console);assert sp.returncode==0 and 'SOURCE_SQUARED_SUM_NORM_COMPLETE PASS' in console,console[-2000:]
qualification={'schema':'source_bound_scalar_squared_sum_complete_norm_capsule_v1','scope':'Exact first normalization extracted through typed dependencies; pinned original8x2048embedding/weights; actualcurrent1983-compatible hoist/puremul4/identity/RNE control, add only typed squared-sum scalaraccumulator. Complete reduction/divide/epsilon/rsqrt/weight multiplication/quant RNE/output incl allocations/traffic and original1880runtime object. Five strict rounding modes and matching sticky flags, all16384i8 results plus128 dirtyguards.','native_all_exact':True,'strict_all_exact':True,'strict_all5_rounding_modes_and_flags':True,'original_runtime_object':str(frozen/'build/mlir_rt.o'),'original_runtime_object_sha256':hashlib.sha256((frozen/'build/mlir_rt.o').read_bytes()).hexdigest(),'elf_sha256':b.elf_sha256,'proofs':proofs,'audit':audit,'token_usage_available':False}
(work/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n');print('SOURCE_SQUARED_SUM_NORM_STRICT5_PASS',flush=True)
(work/'ready.json').write_text(json.dumps({'elf_path':str(b.elf),'elf_sha256':b.elf_sha256,'qualification_path':str(work/'qualification.json')},indent=2)+'\n');
print('SOURCE_SQUARED_SUM_NORM_TIMING_READY',flush=True)
r=run_on_gsim(b.elf,target='gemmini',timeout_s=1800,max_cycles=35000000,stdout_path=work/'gsim.stdout');(work/'gsim_receipt.json').write_text(json.dumps(asdict(r),indent=2,default=str)+'\n');print('SOURCE_SQUARED_SUM_NORM_GSIM',r,flush=True)
