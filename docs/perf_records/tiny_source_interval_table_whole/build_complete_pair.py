"""Actual same-source complete M2 table pair, OOT ISA/fenv/link closure."""
from pathlib import Path
import ctypes,hashlib,importlib.util,json,re,subprocess,sys
import numpy as np
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.perf.layer_bench import build_program
from mlir_oot.no_fsm_audit import audit_elf

W=Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-table-20261006')
CASE=Path(__file__).resolve().parent/'complete_m2';CASE.mkdir(exist_ok=True)
assert not (CASE/'qualification.json').exists()
BASE=W.parent/'positive-reciprocal-certificate-20261006/complete_m2'
WHOLE=Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006/out/artifacts/probes/tiny-rectangular-whole-20261006/whole')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(argv):
    commands.append(argv);p=subprocess.run(argv,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
sys.path.insert(0,str(W));from llvm_source_binding import bind

# Shipped immutable compiler-generated table. No lazy table generation or
# capture-specific row dictionary is performed at runtime.
(CASE/'table.S').write_text('.section .rodata\n.balign 64\n.global source_interval_table\nsource_interval_table:\n.incbin "'+str(W/'interval_table.bin')+'"\n.section .note.GNU-stack,"",@progbits\n')
(CASE/'lookup.c').write_text('''#include <stdint.h>
extern const float source_interval_table[1048576][2];
extern float source_activation(float);
extern signed char source_quantize(float);
static inline uint32_t bits(float x){uint32_t w;__builtin_memcpy(&w,&x,4);return w;}
__attribute__((always_inline)) float source_lookup_activation(float x,float up,float scale){
 uint32_t w=bits(x);
 if((w&0x7f800000u)==0x7f800000u||(bits(up)&0x7f800000u)==0x7f800000u||(bits(scale)&0x7f800000u)==0x7f800000u)return source_activation(x);
 const float*cell=source_interval_table[w>>12];float lo=cell[0],hi=cell[1];
 if(!(lo<=hi))return source_activation(x);
 float low_product=lo*up,high_product=hi*up;
 float low_scaled=low_product*scale,high_scaled=high_product*scale;
 if(source_quantize(low_scaled)!=source_quantize(high_scaled))return source_activation(x);
 return lo;
}
''')
(CASE/'mode_target.c').write_text('''#include <stdint.h>
extern void baseline(int32_t*,float*,int32_t*,float*,int8_t*);
extern void candidate_rne(int32_t*,float*,int32_t*,float*,int8_t*);
void candidate(int32_t*a,float*sa,int32_t*b,float*sb,int8_t*out){unsigned mode;asm volatile("csrr %0,frm":"=r"(mode)::"memory");if(mode)baseline(a,sa,b,sb,out);else candidate_rne(a,sa,b,sb,out);}
''')
(CASE/'mode_native.c').write_text('''#include <stdint.h>
#include <fenv.h>
extern void baseline(int32_t*,float*,int32_t*,float*,int8_t*);
extern void candidate_rne(int32_t*,float*,int32_t*,float*,int8_t*);
void candidate(int32_t*a,float*sa,int32_t*b,float*sb,int8_t*out){if(fegetround()!=FE_TONEAREST)baseline(a,sa,b,sb,out);else candidate_rne(a,sa,b,sb,out);}
''')
bindings=[]
for route in ('native','target'):
    origin=WHOLE/'host_llvm'/('model.native.ll' if route=='native' else 'model.ll');text=origin.read_text()
    start=text.index('define internal void @forward.extracted.531(');end=text.index('\n}',start)+2
    body=text[start:end];assert body.count('icmp slt i64 %6, 8')==1
    body=body.replace('icmp slt i64 %6, 8','icmp slt i64 %6, 2')
    body=re.sub(r'%(\d+)',r'%s.\1',body);body=re.sub(r'^(\d+):',r's.\1:',body,flags=re.M)
    baseline=body.replace('define internal void @forward.extracted.531','define void @baseline')
    candidate=body.replace('define internal void @forward.extracted.531','define void @candidate_rne')
    matches=bind(candidate,(W/'source_activation.ll').read_text());bindings.append({'route':route,'bindings':matches})
    for m in matches:
        old=re.search(r'^  '+re.escape(m['endpoint'])+r' = .+$',candidate,re.M)[0]
        new=f'  {m["endpoint"]} = call float @source_lookup_activation(float {m["input"]}, float {m["up"]}, float {m["quant_scale"]})'
        candidate=candidate.replace(old,new)
    declarations='\n'.join(line for line in text.splitlines() if line.startswith('declare ') or line.startswith('attributes #'))
    triple='x86_64-unknown-linux-gnu' if route=='native' else 'riscv64-unknown-elf'
    path=CASE/(route+'.ll');path.write_text('target triple = "'+triple+'"\n'+baseline+'\n'+candidate+'\n'+declarations+'\ndeclare float @source_lookup_activation(float,float,float)\n')
    activation=(W/'source_activation.ll').read_text()
    quant=(W/'source_quantize.ll').read_text()
    if route=='target':quant,qreport=rewrite(quant,host_isa='rv64gc',combine_clamp=True)
    # Alwaysinline clones of the ORIGINAL independently compiled source body.
    activation=activation.replace('define float @source_activation(float %0) {','define float @source_activation(float %0) alwaysinline {')
    quant=quant.replace('define i8 @source_quantize(float %0) {','define i8 @source_quantize(float %0) alwaysinline {')
    (CASE/(route+'_activation.ll')).write_text(activation)
    (CASE/(route+'_quant.ll')).write_text(quant)
    cflags=flags if route=='target' else ['-O3','-ffp-contract=off','-fPIC']
    run([str(LLVM/'clang'),*cflags,'-S','-emit-llvm',str(CASE/'lookup.c'),'-o',str(CASE/(route+'_lookup.ll'))])
    run([str(LLVM/'llvm-link'),str(path),str(CASE/(route+'_activation.ll')),str(CASE/(route+'_quant.ll')),str(CASE/(route+'_lookup.ll')),'-o',str(CASE/(route+'_linked.bc'))])
    run([str(LLVM/'opt'),'-passes=always-inline',str(CASE/(route+'_linked.bc')),'-o',str(CASE/(route+'_inline.bc'))])
    run([str(LLVM/'clang'),*cflags,'-c',str(CASE/'table.S'),'-o',str(CASE/(route+'_table.o'))])
    if route=='native':
        run([str(LLVM/'clang'),*cflags,'-shared',str(CASE/(route+'_inline.bc')),str(CASE/'mode_native.c'),str(CASE/(route+'_table.o')),'-lm','-o',str(CASE/'native.so')])
    else:
        run([str(LLVM/'clang'),*flags,'-c',str(CASE/(route+'_inline.bc')),'-o',str(CASE/'target.o')])
        run([str(LLVM/'clang'),*flags,'-c',str(CASE/'mode_target.c'),'-o',str(CASE/'mode_target.o')])

lib=ctypes.CDLL(str(CASE/'native.so'))
inputs=[np.fromfile(BASE/(n+'.bin'),dtype=t) for n,t in [('a',np.int32),('scale_a',np.float32),('b',np.int32),('scale_b',np.float32)]]
expected=np.fromfile(BASE/'expected.bin',np.int8)
for arm in ('baseline','candidate'):
    function=getattr(lib,arm);function.argtypes=[ctypes.c_void_p]*5
    guarded=np.full(11264+128,73,np.int8)
    function(*[ctypes.c_void_p(a.ctypes.data) for a in inputs],ctypes.c_void_p(guarded.ctypes.data+64))
    assert np.array_equal(guarded[64:-64],expected) and np.all(guarded[:64]==73) and np.all(guarded[-64:]==73)
    np.save(CASE/(arm+'.npy'),guarded[64:-64])

main=(BASE/'main.c').read_text().replace('RECIPROCAL_PAIR_CYCLES','SOURCE_TABLE_PAIR_CYCLES').replace('ORIGINAL_RECIPROCAL_PAIR','ORIGINAL_SOURCE_TABLE_PAIR')
# All FIVE FRMs independently compare actual target output against its original
# source arm; only RNE uses the table, other modes route the original helper.
main=main.replace(' const unsigned order[4]', ''' for(unsigned frm=0;frm<5;frm++){
 asm volatile("csrw frm,%0"::"r"(frm):"memory");
 call(0);uint64_t h=hash(guarded+64,11264);call(1);if(hash(guarded+64,11264)!=h)return 4;
 }asm volatile("csrw frm,zero":::"memory");
 const unsigned order[4]''')
(CASE/'main.c').write_text(main)
sysroot=subprocess.check_output([str(GCC),'-print-sysroot'],text=True).strip()
run([str(LLVM/'clang'),*flags,'--sysroot='+sysroot,'-isystem',sysroot+'/include','-c',str(CASE/'main.c'),'-o',str(CASE/'main.o')])
build=build_program([CASE/'target.o',CASE/'mode_target.o',CASE/'target_table.o',BASE/'data.o',CASE/'main.o'],CASE/'build',target='gemmini',max_loaded_bytes=None)
audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass'
(CASE/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
sp=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(build.elf)],capture_output=True,text=True,timeout=300)
(CASE/'spike.stdout').write_text(sp.stdout);(CASE/'spike.stderr').write_text(sp.stderr)
assert sp.returncode==0 and 'ORIGINAL_SOURCE_TABLE_PAIR PASS' in sp.stdout+sp.stderr
record={'schema':'actual_source_interval_table_complete_pair_v1','scope':'Complete original current2004 first pre-down helper M2/full5632cols, original11264i8, same input/output addresses; all source preparations, table reads/addressing, interval finishing, original source fallback, unchanged final2FMUL/clamp/RNE/store, outerRNEguard included','bindings':bindings,'table':'immutable compiler-produced8MiB fixed20leadingrawbits table shippedinELF, no runtime lazy/capture-specific generation','gates':{'native_original11264_exact':True,'strict_actual_target11264_exact':True,'five_actual_FRMs_original_source_pair':True,'inputhashes_and128guards':True,'zeroFSM':True},'cycles':'UNKNOWN','commands':commands,'elf_path':str(build.elf),'elf_sha256':sha(build.elf),'pins':{str(p):sha(p) for p in [*CASE.glob('*'),*CASE.glob('build/*'),W/'source_plan.json',W/'derive_source.py',W/'build_table.py',W/'native_screen.py',W/'native_screen.json',W/'llvm_source_binding.py',W/'interval_table.bin',W/'table_receipt.json',BASE/'main.c',BASE/'data.o',BASE/'expected.bin',WHOLE/'host_llvm/model.ll',WHOLE/'host_llvm/model.native.ll',LLVM/'clang',LLVM/'llvm-link',LLVM/'opt'] if p.is_file()}}
(CASE/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print('SOURCE_TABLE_ACTUAL_TARGET_ORIGINAL11264_PASS',sp.stdout,flush=True)
