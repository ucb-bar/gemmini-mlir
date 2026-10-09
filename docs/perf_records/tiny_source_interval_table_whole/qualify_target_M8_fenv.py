from pathlib import Path
import re
import hashlib,json,subprocess
import numpy as np
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';case=W/'target_M8_fenv_v3';case.mkdir(exist_ok=False)
H=W/'target_v2/host_llvm';G=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/capture')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(argv):
 commands.append(argv);p=subprocess.run(argv,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p
argv=[str(LLVM/'llvm-extract'),'--recursive','--keep-const-init','--func=forward.extracted.531','--func=forward.extracted.531.__source_interval_original','--glob=source_interval_table','-S',str(H/'expanded.ll'),'-o',str(case/'selected_helpers.ll')];run(argv)
# No arithmetic/body/shape changes: LLVM extracts the actual normal selected
# helper, its original full source clone and reachable immutable functions/data.
from merlin.llvmlower.late_quant_rne import _tokens,_functions,_identity
def bodies(text):
 # LLVM extraction reserializes numeric attribute/metadata IDs. Resolve each
 # reference to its original exact definition; no instruction/semantic drop.
 attrs={m.group(1):m.group(2)for m in re.finditer(r'^attributes (#\d+) = (.*)$',text,re.M)}
 metadata={m.group(1):m.group(2)for m in re.finditer(r'^(!\d+) = (.*)$',text,re.M)}
 def node(name,trail=()):
  if name in trail:return '<recursive-backedge-distance:'+str(len(trail)-trail.index(name)-1)+'>'
  return re.sub(r'!\d+',lambda m:node(m.group(0),trail+(name,)),metadata[name])
 def normalized(body):
  body=re.sub(r'#\d+',lambda m:attrs[m.group(0)],body)
  return re.sub(r'!\d+',lambda m:node(m.group(0)),body)
 token=_tokens(text);result={}
 for body in _functions(token):
  start=max(t.start for t in token if t.text=='define'and t.start<body[0].start)
  stop=next(t.end for t in token if t.text=='}'and t.start>body[-1].end)
  symbol=next(t.text for t in token if start<=t.start<body[0].start and t.text.startswith('@'))
  result[_identity(symbol)]=normalized(text[body[0].start:body[-1].end])
 return result
original=bodies((H/'expanded.ll').read_text());chosen=bodies((case/'selected_helpers.ll').read_text())
assert chosen and all(chosen[name]==original[name]for name in chosen)
for name,dtype in [('a',np.int32),('scale_a',np.float32),('b',np.int32),('scale_b',np.float32),('expected',np.int8)]:
 a=np.load(G/(name+'.npy'));assert a.dtype==dtype;a.tofile(case/(name+'.bin'))
(case/'data.S').write_text('.section .rodata\n'+''.join('.balign 64\n.global '+name+'\n'+name+':\n.incbin "'+str(case/(name+'.bin'))+'"\n'for name in ('a','scale_a','b','scale_b','expected'))+'.section .note.GNU-stack,"",@progbits\n')
(case/'main.c').write_text(r'''#include <stdint.h>
extern int printf(const char*,...);
extern int32_t a[],b[];extern float scale_a[],scale_b[];extern int8_t expected[];
extern void source(int32_t*,float*,int32_t*,float*,int8_t*) __asm__("forward.extracted.531.__source_interval_original");
extern void candidate(int32_t*,float*,int32_t*,float*,int8_t*) __asm__("forward.extracted.531");
static int8_t original[45056+128],changed[45056+128];
static uint64_t hash(const void*ptr,unsigned n){const uint8_t*p=ptr;uint64_t h=1469598103934665603ul;for(unsigned i=0;i<n;i++){h^=p[i];h*=1099511628211ul;}return h;}
int main(void){uint64_t pins[4]={hash(a,180224),hash(scale_a,22528),hash(b,180224),hash(scale_b,22528)};
 const unsigned presets[7]={0,1,2,4,8,16,31};
 for(unsigned mode=0;mode<5;mode++)for(unsigned f=0;f<7;f++){
  for(unsigned i=0;i<45184;i++){original[i]=73;changed[i]=73;}
  unsigned before0,after0,before1,after1;unsigned flags=presets[f];
  __asm__ volatile("csrw frm,%0\ncsrw fflags,%1"::"r"(mode),"r"(flags):"memory");
  __asm__ volatile("csrr %0,fflags":"=r"(before0)::"memory");
  source(a,scale_a,b,scale_b,original+64);__asm__ volatile("csrr %0,fflags":"=r"(after0)::"memory");
  __asm__ volatile("csrw frm,%0\ncsrw fflags,%1"::"r"(mode),"r"(flags):"memory");
  __asm__ volatile("csrr %0,fflags":"=r"(before1)::"memory");
  candidate(a,scale_a,b,scale_b,changed+64);__asm__ volatile("csrr %0,fflags":"=r"(after1)::"memory");
  if(before0!=flags||before1!=flags){printf("FAIL PRESET %u %u %u %u\n",mode,flags,before0,before1);return 1;}
  for(unsigned i=0;i<45056;i++)if(original[i+64]!=changed[i+64]||(!mode&&changed[i+64]!=expected[i])){printf("FAIL WORD %u %u %u\n",mode,flags,i);return 2;}
  for(unsigned i=0;i<64;i++)if(original[i]!=73||changed[i]!=73||original[45120+i]!=73||changed[45120+i]!=73)return 3;
  if(mode&&after0!=after1){printf("FAIL FLAGS %u %u %u %u\n",mode,flags,after0,after1);return 4;}
  printf("M8_FENV_CASE %u %u %u %u\n",mode,flags,after0,after1);
 }
 __asm__ volatile("csrw frm,zero":::"memory");
 if(hash(a,180224)!=pins[0]||hash(scale_a,22528)!=pins[1]||hash(b,180224)!=pins[2]||hash(scale_b,22528)!=pins[3])return 5;
 printf("NORMAL_M8_TARGET_SOURCE_FALLBACK PASS 45056 128 35\n");return 0;}
''')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
sysroot=run([str(GCC),'-print-sysroot']).stdout.strip()
for name in ('selected_helpers.ll','data.S','main.c'):
 run([str(LLVM/'clang'),*flags,'--sysroot='+sysroot,'-isystem',sysroot+'/include','-c',str(case/name),'-o',str(case/(Path(name).stem+'.o'))])
build=build_program([case/'selected_helpers.o',case/'data.o',case/'main.o'],case/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass';(case/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
argv=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(build.elf)]
p=subprocess.run(argv,capture_output=True,text=True,timeout=300);(case/'spike.stdout').write_text(p.stdout);(case/'spike.stderr').write_text(p.stderr)
assert p.returncode==0 and 'NORMAL_M8_TARGET_SOURCE_FALLBACK PASS 45056 128 35'in p.stdout+p.stderr
record=dict(schema='normal_source_interval_actual_M8_target_fenv_v1',status='pass',all45056_originalwords_and128guards_inputhashes=True,target_rounding_modes=5,sticky_presets=7,normal_compiled_helper_instruction_bodies_and_resolved_attributes_metadata_identical=True,all_nonRNE_original_source_flags_exact=True,RNEflags_unobserved=True,qualification_scope='Actual normal selected helper/subordinate source callbacks/table extracted viaLLVM with no body/loop/address arithmetic changes; different subset object/link layout is not performance evidence. FulloriginalM8, all5targetFRMs×7sticky presets, originalsource clone/control words and guards. Wholeoriginal256000targetgate is separate.',commands=commands,spike_argv=argv,elf_sha256=sha(build.elf),pins={str(p):sha(p)for p in [Path(__file__),H/'expanded.ll',*case.glob('*'),*case.glob('build/*'),*(G/(n+'.npy')for n in ('a','scale_a','b','scale_b','expected'))]if p.is_file()})
(case/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print('NORMAL_M8_TARGET_FIVE_FRM_STICKY_GUARDS_PASS',flush=True)
