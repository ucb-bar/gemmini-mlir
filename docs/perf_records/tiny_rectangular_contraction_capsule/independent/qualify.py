"""Actual RV64GC source/candidate independent shapes and all five FRMs."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,subprocess,importlib.util
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_contraction import EIGHT_OUTPUTS_FEATURE,RECTANGULAR_FEATURE,REDUCTION_UNROLL_FEATURE,apply_for_test
from merlin.llvmlower.bufferized_result_identity import FEATURE as IDENTITY
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
C=Path('/scratch/agustin/tmp/merlin-scalar-contraction-rectangular-20261006')
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
spec=importlib.util.spec_from_file_location('source_fixture',C/'merlin/tests/ir/test_scalar_contraction.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(92334)
cases=[]
for m,n,k,transpose,alias,empty,special in [(4,12,5,False,False,False,False),(2,8,7,True,False,False,False),(4,4,4,False,True,False,False),(2,8,0,False,False,True,False),(2,8,3,False,False,False,True),(3,4,5,False,False,False,False),(0,8,3,False,False,True,False),(2,0,3,False,False,True,False)]:
 a=rng.normal(size=(2,m,k)).astype('f4');b=rng.normal(size=(2,k,n)).astype('f4');c=rng.normal(size=(2,m,n)).astype('f4')
 if alias:c=a
 if k and m and n and not special:
  a[:,0,:]=0;a[:,0,:min(k,3)]=np.array([16777216,1,-16777216],dtype='f4')[:min(k,3)];b[:,:,0]=1
 if k==0:
  c.view('u4').flat[:4]=[0x7fc12345,0x80000000,0x00000001,0xff800000]
 if special:
  a.view('u4')[:]=np.array([0x7f800000,0x00000001,0x80000000,0x7fc12345,0x00800000,0xff800000],dtype='u4').reshape(1,2,3)
  b.view('u4')[0,0,:]=[0,0x3f800001,0x80000000,0x7f7fffff,0xff800000,0x7f800001,1,0x80000001]
 storage=np.ascontiguousarray(b.swapaxes(-1,-2)) if transpose else b
 src=module.source(a.shape,storage.shape,c.shape,transpose=transpose,alias=alias,reversed_add=transpose)
 transformed,count=apply_for_test(src,outputs=4,rows=2)
 assert count==(0 if m%2 or n%4 else 1)
 idx=len(cases);(W/f'case{idx}.mlir').write_text(src)
 for kind,x in [('a',a),('b',storage),('c',c)]:x.astype('<f4').tofile(W/f'case{idx}_{kind}.bin')
 cases.append(dict(m=m,n=n,k=k,transpose=transpose,alias=alias,matched=count,arrays=[a,storage,c],special=special))
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
objects=[];commands=[]
def run(argv):commands.append(list(map(str,argv)));subprocess.run(commands[-1],check=True,capture_output=True)
assembly='.section .rodata\n'
protos=[];caseinit=[]
for i,case in enumerate(cases):
 for kind in ['a','b','c']:assembly+=f'.balign 64\n.global case{i}_{kind}\ncase{i}_{kind}:\n.incbin "{W}/case{i}_{kind}.bin"\n'
 for arm,feature in [('control',EIGHT_OUTPUTS_FEATURE),('candidate',RECTANGULAR_FEATURE)]:
  name=f'case{i}_{arm}';folder=W/name
  text=lower_to_llvm_ir((W/f'case{i}.mlir').read_text(),workdir=folder,features={feature,REDUCTION_UNROLL_FEATURE,IDENTITY}).replace('forward',name).replace('dealloc_helper',name+'_dealloc_helper')
  (folder/'model.ll').write_text(text);run([L/'clang',*flags,'-c',folder/'model.ll','-o',folder/'model.o']);objects.append(folder/'model.o')
  nargs=3 if case['alias'] else 4;protos.append(f'extern void _mlir_ciface_{name}('+','.join(['struct d3*']*nargs)+');')
 protos.append(f'extern float case{i}_a[],case{i}_b[],case{i}_c[];')
 caseinit.append('{'+','.join([str(case[x]) for x in ['m','n','k']])+','+str(int(case['transpose']))+','+str(int(case['alias']))+f',case{i}_a,case{i}_b,case{i}_c,(void*)_mlir_ciface_case{i}_control,(void*)_mlir_ciface_case{i}_candidate'+'}')
(W/'data.S').write_text(assembly);run([L/'clang',*flags,'-c',W/'data.S','-o',W/'data.o']);objects.extend([W/'data.o',B/'mlir_rt.o'])
main=r'''#include <stdint.h>
extern int printf(const char*,...);
struct d3{void*a,*p;long off,size[3],stride[3];};
PROTOTYPES
struct entry{unsigned m,n,k,trans,alias;float*a,*b,*c;void*control,*candidate;};
static struct entry cases[]={CASE_INITIALIZERS};
static union{float f;uint32_t w;} out[2048+128],reference[2048];
static uint8_t arena[1024*1024];static unsigned cursor;static int err;
int *__errno(void){return &err;}
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint32_t checksum(float*p,unsigned n){uint32_t s=0;for(unsigned i=0;i<n;i++){union{float f;uint32_t w;}v={p[i]};s^=v.w;}return s;}
int main(void){for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
struct entry*e=&cases[ci];unsigned an=2*e->m*e->k,bn=2*e->k*e->n,cn=2*e->m*e->n;
uint32_t ca=checksum(e->a,an),cb=checksum(e->b,bn),cc=checksum(e->c,cn);
struct d3 a={e->a,e->a,0,{2,e->m,e->k},{e->m*e->k,e->k,1}},b={e->b,e->b,0,{2,e->trans?e->n:e->k,e->trans?e->k:e->n},{e->k*e->n,e->trans?e->k:e->n,1}},c={e->c,e->c,0,{2,e->m,e->n},{e->m*e->n,e->n,1}},o={out,out,0,{2,e->m,e->n},{e->m*e->n,e->n,1}};
for(unsigned frm=0;frm<5;frm++){asm volatile("csrw frm,%0"::"r"((unsigned long)frm):"memory");unsigned baseflags=0;
for(unsigned arm=0;arm<2;arm++){
for(unsigned i=0;i<cn+128;i++)out[i].w=0x7fc12345;cursor=0;unsigned long seed=8;asm volatile("csrw fflags,%0"::"r"(seed):"memory");
void*call=arm?e->candidate:e->control;if(e->alias)((void(*)(struct d3*,struct d3*,struct d3*))call)(&a,&b,&o);else((void(*)(struct d3*,struct d3*,struct d3*,struct d3*))call)(&a,&b,&c,&o);
unsigned long flags;asm volatile("csrr %0,fflags":"=r"(flags));if(!arm)baseflags=flags;else if(baseflags!=flags){printf("FLAG_FAIL %u %u %u %lu\n",ci,frm,baseflags,flags);return 1;}
for(unsigned i=0;i<cn;i++){if(!arm)reference[i].w=out[i].w;else if(reference[i].w!=out[i].w){printf("VALUE_FAIL %u %u %u %x %x\n",ci,frm,i,reference[i].w,out[i].w);return 2;}}
for(unsigned i=cn;i<cn+128;i++)if(out[i].w!=0x7fc12345){printf("GUARD_FAIL %u\n",ci);return 3;}
if(ca!=checksum(e->a,an)||cb!=checksum(e->b,bn)||cc!=checksum(e->c,cn)){printf("INPUT_FAIL %u\n",ci);return 4;}
}}
printf("RECTANGULAR_INDEPENDENT_CASE PASS %u %u\n",ci,cn);
}asm volatile("csrw frm,zero":::"memory");printf("RECTANGULAR_INDEPENDENT_TARGET PASS\n");return 0;}
'''.replace('PROTOTYPES','\n'.join(protos)).replace('CASE_INITIALIZERS',','.join(caseinit))
(W/'main.c').write_text(main);run([L/'clang',*flags,'-c',W/'main.c','-o',W/'main.o']);objects.append(W/'main.o')
build=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass'
argv=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(build.elf)]
p=subprocess.run(argv,capture_output=True,text=True,timeout=300);log=p.stdout+p.stderr;(W/'spike.log').write_text(log)
assert p.returncode==0 and 'RECTANGULAR_INDEPENDENT_TARGET PASS' in log,log[-5000:]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
record={'schema':'rectangular_scalar_source_independent_actual_rv64gc_v1','status':'pass','cases':[{k:v for k,v in case.items() if k!='arrays'} for case in cases],'all_five_rounding_modes':True,'sticky_flags_exact':True,'all_rv64_words_including_nonfinite_exact':True,'empty_K_source_payloads_preserved':True,'input_seed_live_alias_preserved':True,'tail_refusal_test':True,'poisoned_guards_per_case':128,'commands':commands,'spike_argv':argv,'audit':audit,'elf_sha256':sha(build.elf),'spike_log_sha256':sha(W/'spike.log'),'source_numeric_policy':'Unconstrained ordinary source mul/add, original operand/reduction order. RV64 implementation happens to match all NaN words; generic source makes no arithmetic payload-identity promise. Strict contexts refuse.','token_usage_available':False}
(W/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print('RECTANGULAR_INDEPENDENT_RV64GC_PASS',len(cases),flush=True)
