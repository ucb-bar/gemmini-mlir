from pathlib import Path
import sys,json,subprocess,hashlib
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
from merlin.llvmlower.constant_fma_packet import rewrite
from mlir_oot.rv64gc_scalar_fma import RV64GCScalarFmaCapability
W=Path(__file__).resolve().parent;D=W/'promoted_cases';D.mkdir(exist_ok=True);L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');flags=json.loads((W/'prepared.json').read_text())['flags'];cases=[];src=[];selected=[]
def literal(word):
 sign=word>>31;e=(word>>23)&255;f=word&((1<<23)-1)
 if e:bits=(sign<<63)|((e-127+1023)<<52)|(f<<29)
 else:
  n=f.bit_length();bits=(sign<<63)|((n-1-149+1023)<<52)|((f<<(53-n))&((1<<52)-1))
 return '0x'+format(bits,'016X')
for i,(n,width,mode,word)in enumerate([(3,2,'constant_rhs',0xbf800000),(5,4,'constant_addend',0x40490fdb),(7,4,'constant_lhs_addend',0x3f400000),(6,2,'constant_addend',0x00800000),(5,4,'constant_rhs',1),(7,2,'constant_lhs_addend',0x7f7fffff)]):
 lines=[f'define void @source_case{i}(ptr %a,ptr %o) {{','entry:']
 for j in range(n):lines.extend([f'  %p{j} = getelementptr float,ptr %a,i64 {j}',f'  %a{j} = load float,ptr %p{j}'])
 for j in range(n):
  values=[f'%a{j}',f'%a{(j+1)%n}',literal(word)]if mode=='constant_addend'else([f'%a{j}',literal(word),f'%a{(j+1)%n}']if mode=='constant_rhs'else[literal(word),f'%a{j}',literal(0x3e800000)])
  lines.append(f'  %r{j} = call float @llvm.fma.f32('+', '.join('float '+x for x in values)+')')
 for j in range(n):lines.extend([f'  %q{j} = getelementptr float,ptr %o,i64 {j}',f'  store float %r{j},ptr %q{j}'])
 lines.extend(['  ret void','}']);s='\n'.join(lines)+'\n';draft=s.replace('source_case','selected_case');new,r=rewrite(draft,emitter=RV64GCScalarFmaCapability().emit,width=width,ordinary_nontrapping=True,exception_flags_unobserved=True)
 assert len(r['packets'])==n//width,(i,r);src.append(s);selected.append(new);cases.append(dict(index=i,lanes=n,width=width,mode=mode,constant_word=word,packets=len(r['packets']),untouched_tail_lanes=n%width))
for label,units in [('source',src),('selected',selected)]:
 p=D/f'{label}.ll';p.write_text('\n'.join(units)+'\ndeclare float @llvm.fma.f32(float,float,float)\n');subprocess.run([str(L/'clang'),*flags,'-c',str(p),'-o',str(D/f'{label}.o')],check=True,capture_output=True)
protos='\n'.join(f'extern void {p}_case{i}(float*,float*);'for p in ['source','selected']for i in range(6))
main='''#include <stdint.h>
extern int printf(const char*,...);
PROTOS
typedef void(*fn)(float*,float*);
int main(void){
 fn source[6]={SOURCE};fn selected[6]={SELECTED};unsigned n[6]={3,5,7,6,5,7};
 const unsigned directed[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x3f800001,0x3f7fffff,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f812345,0xff812345};
 unsigned rng=0x714ce813,total=0;float a[8],s[8],t[8];
 for(unsigned frm=0;frm<5;frm++)for(unsigned op=0;op<6;op++)for(unsigned trial=0;trial<278;trial++){
  for(unsigned j=0;j<8;j++){rng=rng*1664525+1013904223;unsigned raw=trial<22?directed[(trial+j)%22]:rng;__builtin_memcpy(a+j,&raw,4);s[j]=73;t[j]=74;}
  unsigned f0,f1;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");source[op](a,s);asm volatile("csrr %0,fflags":"=r"(f0)::"memory");
  asm volatile("csrw fflags,%0"::"r"(8):"memory");selected[op](a,t);asm volatile("csrr %0,fflags":"=r"(f1)::"memory");
  for(unsigned j=0;j<n[op];j++){unsigned x,y;__builtin_memcpy(&x,s+j,4);__builtin_memcpy(&y,t+j,4);if(x!=y){printf("DRAFT_BITS_FAIL %u %u %u %u %x %x\\n",frm,op,trial,j,x,y);return 1;}total++;}
  if(f0!=f1){printf("DRAFT_FLAGS_FAIL %u %u %u %u %u\\n",frm,op,trial,f0,f1);return 2;}
  for(unsigned j=n[op];j<8;j++)if(s[j]!=73||t[j]!=74){printf("TAIL_POISON_FAIL\\n");return 3;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("DRAFT_GENERIC_FMA PASS %u fivefrm stickyflags sharedSSA tails\\n",total);return 0;
}
'''.replace('PROTOS',protos).replace('SOURCE',','.join(f'source_case{i}'for i in range(6))).replace('SELECTED',','.join(f'selected_case{i}'for i in range(6)))
(D/'main.c').write_text(main);subprocess.run([str(L/'clang'),*flags,'-c',str(D/'main.c'),'-o',str(D/'main.o')],check=True,capture_output=True)
b=build_program([D/'source.o',D/'selected.o',D/'main.o'],D/'build',target='gemmini',max_loaded_bytes=None);a=audit_elf(b.elf.read_bytes());assert a['status']=='pass'
s=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,timeout=120);(D/'spike.stdout').write_text(s.stdout);(D/'spike.stderr').write_text(s.stderr);assert s.returncode==0 and'DRAFT_GENERIC_FMA PASS 45870'in s.stdout,s.stdout+s.stderr
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=dict(schema='draft_generic_fma_source_isa_independent_cases_v1',status='pass',cases=cases,words=45870,rounding_modes=5,sticky_flags_exact=True,dirty_output_tails=True,shared_sourceSSA=True,numeric_source_policy='Ordinary nontrapping, exceptionflags-unobserved source rewrite; independent source vs target test additionally proves actualstickyflags all5modes.',no_FSM=True,pins={str(p.relative_to(D)):sha(p)for p in [D/'source.ll',D/'selected.ll',D/'source.o',D/'selected.o',D/'main.c',D/'main.o',b.elf,D/'spike.stdout',D/'spike.stderr']},production_installed=True)
(D/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
