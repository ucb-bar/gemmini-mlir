"""Complete matched ABBA capsule from original gate tensors and source body."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.scalar_pointwise_packet import FEATURE,BROADCAST_FEATURE
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
from dataclasses import asdict

W=Path(__file__).resolve().parent
C=W/'capture';capture=json.loads((C/'receipt.json').read_text());assert capture['status']=='pass'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
assert sha(W/'source_capsule.mlir')==capture['source_capsule_sha256']
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
case=W/'capsule';case.mkdir(exist_ok=True)
assert not (case/'qualification.json').exists(), 'Do not overwrite qualified evidence'
source=(W/'source_capsule.mlir').read_text()
needle=') -> tensor<1x8x5632xi8> {';assert source.count(needle)==1
source=source.replace(needle,') -> tensor<1x8x5632xi8> attributes {llvm.emit_c_interface} {')
(case/'source.mlir').write_text(source)
names=['a','scale_a','b','scale_b'];inputs=[np.load(C/(name+'.npy')) for name in names]
expected=np.load(C/'expected.npy');assert expected.shape==(1,8,5632)
for name,a in zip(names,inputs):
 assert sha(C/(name+'.npy'))==capture['captured_tensors'][name]['sha256']
 a.tofile(case/(name+'.bin'))
expected.tofile(case/'expected.bin')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
objects=[];receipts=[]
for name,feature in [('baseline',FEATURE),('broadcast',BROADCAST_FEATURE)]:
 ll=lower_to_llvm_ir(source,workdir=case/name,features={feature,'lower_fma_to_intrinsic'})
 ll=ll.replace('forward',name).replace('dealloc_helper',name+'_dealloc_helper')
 native,nr=rewrite(ll,host_isa='portable');target,tr=rewrite(ll,host_isa='rv64gc',combine_clamp=True)
 (case/(name+'.native.ll')).write_text(native);(case/(name+'.target.ll')).write_text(target)
 so=case/(name+'.so');native_argv=[str(LLVM/'clang'),'-O3','-fPIC','-shared',str(case/(name+'.native.ll')),str(mlir_runtime_c()),'-lm','-o',str(so)]
 subprocess.run(native_argv,check=True,capture_output=True)
 out=np.zeros_like(expected)
 HostModel.load(str(so),name=name)([(a.ctypes.data,a.shape) for a in inputs]+[(out.ctypes.data,out.shape)])
 np.save(case/(name+'.npy'),out)
 assert np.array_equal(out,expected),(name,int(np.count_nonzero(out!=expected)))
 obj=case/(name+'.o');argv=[str(LLVM/'clang'),*flags,'-c',str(case/(name+'.target.ll')),'-o',str(obj)]
 subprocess.run(argv,check=True,capture_output=True);objects.append(obj)
 receipts.append({'name':name,'feature':feature,'original45056words_exact':True,
  'target_rne_sites':len(tr['routes']),'native_rne_sites':len(nr['routes']),
  'target_sha256':sha(case/(name+'.target.ll')),'native_sha256':sha(case/(name+'.native.ll')),
  'object_sha256':sha(obj),'compile_argv':argv,'native_argv':native_argv})
 print(name,'NATIVE_ORIGINAL45056_PASS',flush=True)
asm=case/'data.S'
asm.write_text('.section .data\n'+''.join('.balign 64\n.global '+name+'\n'+name+':\n.incbin "'+str(case/(name+'.bin'))+'"\n' for name in [*names,'expected']))
subprocess.run([str(LLVM/'clang'),*flags,'-c',str(asm),'-o',str(case/'data.o')],check=True)
c=r'''
#include <stdint.h>
extern int printf(const char *,...);
extern int32_t a[],b[];extern float scale_a[],scale_b[];extern int8_t expected[];
struct d3 {void *alloc,*ptr;long off;long size[3],stride[3];};
struct d1 {void *alloc,*ptr;long off,size,stride;};
extern void _mlir_ciface_baseline(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
extern void _mlir_ciface_broadcast(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
static int8_t guarded[45056+128];
static uint8_t arena[4*45056+1024];static unsigned cursor;
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void *p=arena+cursor;cursor+=n;return p;}
void free(void *p){(void)p;}
static inline uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t hash(const void *ptr,unsigned size){const uint8_t *p=ptr;uint64_t h=1469598103934665603ul;for(unsigned i=0;i<size;i++){h^=p[i];h*=1099511628211ul;}return h;}
static int validate(unsigned arm){for(unsigned i=0;i<45056;i++)if(guarded[i+64]!=expected[i]){printf("FAIL %u %u %d %d\n",arm,i,guarded[i+64],expected[i]);return 1;}for(unsigned i=0;i<64;i++)if(guarded[i]!=73||guarded[45056+64+i]!=73){printf("GUARD_FAIL\n");return 2;}return 0;}
int main(void){
 struct d3 da={a,a,0,{1,8,5632},{45056,5632,1}},db={b,b,0,{1,8,5632},{45056,5632,1}};
 struct d1 sa={scale_a,scale_a,0,5632,1},sb={scale_b,scale_b,0,5632,1};
 struct d3 out={guarded,guarded+64,0,{1,8,5632},{45056,5632,1}};
 uint64_t input_hash[4]={hash(a,180224),hash(scale_a,22528),hash(b,180224),hash(scale_b,22528)};
 for(unsigned i=0;i<45184;i++)guarded[i]=73;
 for(unsigned arm=0;arm<2;arm++){cursor=0;if(arm==0)_mlir_ciface_baseline(&da,&sa,&db,&sb,&out);else _mlir_ciface_broadcast(&da,&sa,&db,&sb,&out);if(validate(arm))return 1;}
 const unsigned order[4]={0,1,1,0};
 for(unsigned i=0;i<4;i++){
  cursor=0;uint64_t t=tick();if(order[i]==0)_mlir_ciface_baseline(&da,&sa,&db,&sb,&out);else _mlir_ciface_broadcast(&da,&sa,&db,&sb,&out);t=tick()-t;
  if(validate(order[i]))return 2;printf("GATE_PACKET_CYCLES %u %u %lu\n",i,order[i],t);
 }
 if(hash(a,180224)!=input_hash[0]||hash(scale_a,22528)!=input_hash[1]||hash(b,180224)!=input_hash[2]||hash(scale_b,22528)!=input_hash[3]){printf("INPUT_CHANGED\n");return 3;}
 printf("ORIGINAL_GATE_PACKET PASS 45056 128\n");return 0;
}
'''
(case/'main.c').write_text(c)
subprocess.run([str(LLVM/'clang'),*flags,'-c',str(case/'main.c'),'-o',str(case/'main.o')],check=True)
objects.extend([case/'data.o',case/'main.o'])
build=build_program(objects,case/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass'
(case/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
sp=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(build.elf)],text=True,capture_output=True,timeout=600)
(case/'spike.stdout').write_text(sp.stdout);(case/'spike.stderr').write_text(sp.stderr)
assert sp.returncode==0 and 'ORIGINAL_GATE_PACKET PASS' in sp.stdout+sp.stderr
record={'schema':'tiny_original_gate_broadcast_packet_pair_v1','original_source_receipt_sha256':sha(W/'source_receipt.json'),
 'capture_receipt_sha256':sha(C/'receipt.json'),'prepared_source_sha256':sha(case/'source.mlir'),
 'original_outputs':45056,'dirty_guards':128,'immutable_input_bytes':405504,'cases':receipts,
 'elf_path':str(build.elf),'elf_sha256':sha(build.elf),'zero_FSM':True,
 'strict_RV64GC_original_pass':True,'complete_ROI':'Two warm calls, then matched ABBA. Entire compiled source helper includes common bounded allocation and final result copy. Input hash/expected/guard scans outside timing.',
 'scope':'Original1880 first gate source body and its actual captured tensors; no whole-model cycles implied.'}
(case/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print('STRICT_ORIGINAL_GATE_PACKET_PASS',flush=True)
result=run_on_gsim(build.elf,target='gemmini',timeout_s=1800,max_cycles=60000000,stdout_path=case/'gsim.stdout')
(case/'gsim_receipt.json').write_text(json.dumps(asdict(result),indent=2,default=str)+'\n')
print(result,flush=True)
