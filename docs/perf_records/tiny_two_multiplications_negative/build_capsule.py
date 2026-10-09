"""Complete original dequant/residual capsule with identical immutable inputs."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import MULTIPLY_FOUR_FEATURE, TWO_MULTIPLY_FOUR_FEATURE
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf

W = Path(__file__).resolve().parent
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
FLAGS = ['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def run(argv):
    return subprocess.run([str(x)for x in argv],check=True,capture_output=True)
def save(p,v):
    Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')

capture = json.loads((W/'capture/receipt.json').read_text())
assert capture['status'] == 'pass' and capture['original_compiled_bits_exact']
args = [np.load(W/'capture'/(n+'.npy'))for n in ['residual','accumulation','scale']]
expected = np.load(W/'capture/expected.npy')
assert expected.shape == (1,8,2048) and expected.dtype == np.float32
original = [v.copy()for v in args]
for value in capture['captured_tensors'].values():
    assert sha(value['path']) == value['sha256']
text = (W/'source_capsule.mlir').read_text()
needle = ' -> tensor<1x8x2048xf32> {'
assert text.count(needle) == 1
text = text.replace(needle, ' -> tensor<1x8x2048xf32> attributes {llvm.emit_c_interface} {')
(W/'capsule_abi.mlir').write_text(text)
objects, cases = [], []
for name,features in [('control',{MULTIPLY_FOUR_FEATURE}),('packet',{MULTIPLY_FOUR_FEATURE,TWO_MULTIPLY_FOUR_FEATURE})]:
    raw = lower_to_llvm_ir(text,workdir=W/(name+'_lower'),features=features)
    raw = raw.replace('forward',name).replace('dealloc_helper',name+'_dealloc_helper')
    (W/(name+'.ll')).write_text(raw)
    assert 'fdiv float' not in raw and '@llvm.fma.f32' not in raw
    so = W/(name+'.so')
    run([L/'clang','-O3','-shared','-fPIC','-ffp-contract=off',W/(name+'.ll'),'-lm','-o',so])
    output = np.empty_like(expected)
    HostModel.load(str(so),name=name)([(v.ctypes.data,v.shape)for v in args]+[(output.ctypes.data,output.shape)])
    assert np.array_equal(output.view('u4'),expected.view('u4'))
    assert all(np.array_equal(v.view('u4'),saved.view('u4'))for v,saved in zip(args,original))
    np.save(W/(name+'.npy'),output)
    obj = W/(name+'.o')
    argv = [L/'clang',*FLAGS,'-c',W/(name+'.ll'),'-o',obj]
    run(argv)
    objects.append(obj)
    cases.append({'name':name,'native_original_words_exact':16384,'llvm_sha256':sha(W/(name+'.ll')),'object_sha256':sha(obj),'so_sha256':sha(so),'compile_argv':argv})
    print(name,'ORIGINAL_NATIVE16384_EXACT',flush=True)

data = []
for name,value in zip(['residual','accumulation','scale','expected'],[*args,expected]):
    raw = W/(name+'.bin')
    raw.write_bytes(value.tobytes())
    data.append(f'.section .rodata\n.balign 64\n.global {name}\n{ name}:\n.incbin "{raw}"\n')
(W/'data.S').write_text(''.join(data))
run([L/'clang',*FLAGS,'-c',W/'data.S','-o',W/'data.o'])
main = r'''
#include <stdint.h>
extern int printf(const char*,...);
extern float residual[],scale[],expected[];extern int32_t accumulation[];
struct d3{void*alloc,*ptr;long off,size[3],stride[3];};
struct d1{void*alloc,*ptr;long off,size,stride;};
extern void _mlir_ciface_control(struct d3*,struct d3*,struct d1*,struct d3*);
extern void _mlir_ciface_packet(struct d3*,struct d3*,struct d1*,struct d3*);
#define N 16384
static float guarded[N+32];static uint8_t arena[4*N*4+1024];static unsigned cursor;
void*malloc(unsigned long n){cursor=(cursor+63)&~63u;if(n>sizeof(arena)-cursor)return 0;void*p=arena+cursor;cursor+=n;return p;}
void free(void*p){(void)p;}
static uint32_t bits(const float*p){uint32_t v;__builtin_memcpy(&v,p,4);return v;}
static void word(float*p,uint32_t v){__builtin_memcpy(p,&v,4);}
static void poison(void){for(unsigned i=0;i<N+32;i++)word(guarded+i,0x4f123456);for(unsigned i=0;i<N;i++)word(guarded+16+i,bits(expected+i)^1);for(unsigned i=0;i<sizeof(arena);i++)arena[i]=0xa5;cursor=0;}
static inline uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t hash(const void*ptr,unsigned n){const uint8_t*p=ptr;uint64_t h=1469598103934665603ul;for(unsigned i=0;i<n;i++){h^=p[i];h*=1099511628211ul;}return h;}
static int guards(void){for(unsigned i=0;i<16;i++)if(bits(guarded+i)!=0x4f123456||bits(guarded+16+N+i)!=0x4f123456)return 1;return 0;}
static int validate(unsigned arm){for(unsigned i=0;i<N;i++)if(bits(guarded+16+i)!=bits(expected+i)){printf("TWO_MUL_BITS_FAIL %u %u\n",arm,i);return 1;}return guards();}
STRICT_FUNCTION
int main(void){asm volatile("csrw fflags,zero":::"memory");unsigned frm;asm volatile("csrr %0,frm":"=r"(frm));if(frm)return 9;
 struct d3 r={residual,residual,0,{1,8,2048},{N,2048,1}},a={accumulation,accumulation,0,{1,8,2048},{N,2048,1}};
 struct d1 s={scale,scale,0,2048,1};struct d3 out={guarded,guarded+16,0,{1,8,2048},{N,2048,1}};
 uint64_t before[3]={hash(residual,N*4),hash(accumulation,N*4),hash(scale,2048*4)};
 STRICT_CALL
 for(unsigned arm=0;arm<2;arm++){poison();if(arm)_mlir_ciface_packet(&r,&a,&s,&out);else _mlir_ciface_control(&r,&a,&s,&out);if(validate(arm))return 1;}
 const unsigned order[4]={0,1,1,0};for(unsigned i=0;i<4;i++){poison();uint64_t t=tick();if(order[i])_mlir_ciface_packet(&r,&a,&s,&out);else _mlir_ciface_control(&r,&a,&s,&out);t=tick()-t;if(validate(order[i]))return 2;printf("TWO_MUL_COMPLETE_CYCLES %u %u %lu\n",i,order[i],t);}
 if(before[0]!=hash(residual,N*4)||before[1]!=hash(accumulation,N*4)||before[2]!=hash(scale,2048*4))return 3;
 printf("ORIGINAL_TWO_MUL_COMPLETE PASS 16384 128\n");return 0;
}
'''
strict = r'''
static uint32_t reference[N];
static int five_modes(struct d3*r,struct d3*a,struct d1*s,struct d3*out){
 for(unsigned frm=0;frm<5;frm++){unsigned old_flags,new_flags;
  poison();asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");_mlir_ciface_control(r,a,s,out);asm volatile("csrr %0,fflags":"=r"(old_flags)::"memory");if(guards())return 1;for(unsigned i=0;i<N;i++)reference[i]=bits(guarded+16+i);
  poison();asm volatile("csrw fflags,%0"::"r"(8):"memory");_mlir_ciface_packet(r,a,s,out);asm volatile("csrr %0,fflags":"=r"(new_flags)::"memory");if(guards())return 2;for(unsigned i=0;i<N;i++)if(reference[i]!=bits(guarded+16+i)){printf("TWO_MUL_FRM_FAIL %u %u\n",frm,i);return 3;}if(old_flags!=new_flags){printf("TWO_MUL_FLAGS_FAIL %u %u %u\n",frm,old_flags,new_flags);return 4;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("TWO_MUL_FIVE_MODES_FLAGS PASS 81920\n");return 0;
}
'''
for suffix in ['strict','timing']:
    body = main.replace('STRICT_FUNCTION',strict if suffix=='strict'else'').replace('STRICT_CALL','if(five_modes(&r,&a,&s,&out))return 12;'if suffix=='strict'else'')
    src = W/(suffix+'_main.c')
    src.write_text(body)
    obj = W/(suffix+'_main.o')
    run([L/'clang',*FLAGS,'-c',src,'-o',obj])
    build = build_program([*objects,W/'data.o',obj],W/(suffix+'_build'),target='gemmini',max_loaded_bytes=None)
    audit = audit_elf(build.elf.read_bytes())
    assert audit['status']=='pass'
    save(W/(suffix+'_audit.json'),audit)
    spike = subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc',str(build.elf)],capture_output=True,text=True,timeout=300)
    console = spike.stdout+spike.stderr
    (W/(suffix+'_spike.log')).write_text(console)
    assert spike.returncode==0 and 'ORIGINAL_TWO_MUL_COMPLETE PASS 16384 128' in console
    if suffix=='strict':
        assert 'TWO_MUL_FIVE_MODES_FLAGS PASS 81920' in console
        print('STRICT_FIVE_MODES_FLAGS_PASS',flush=True)
    else:
        record = {'schema':'source_exact_two_multiplications_complete_capsule_v1','shape':[1,8,2048],'native_original_f32_words':16384,'strict_compared_words':81920,'all_five_rounding_modes_and_sticky_flags_match':True,'dirty_output_guard_bytes':128,'dirty_private_arena':True,'immutable_input_bytes':139264,'source_sha256':sha(W/'source_capsule.mlir'),'source_receipt_sha256':sha(W/'source_receipt.json'),'capsule_abi_sha256':sha(W/'capsule_abi.mlir'),'capture_sha256':sha(W/'capture/receipt.json'),'cases':cases,'elf_sha256':sha(build.elf),'zero_FSM':True,'scope':'Complete original dequant/residual body: i32-to-f32 conversion, source constant scale, two separately rounded f32 multiplications, source residual addition, all input reads/stores, temporary allocation and final output copy. Exact1967 source data/control; no source reciprocal or precision change. GSIM capsule allocator/memory regime is separately scoped, not stock whole-model timing.','whole_hardware_prediction':None,'token_usage_available':False}
        save(W/'capsule_qualification.json',record)
        print('TIMING_READY',flush=True)
        result = run_on_gsim(build.elf,target='gemmini',timeout_s=1800,max_cycles=30000000,stdout_path=W/'gsim.stdout')
        save(W/'gsim_receipt.json',asdict(result))
        console = (W/'gsim.stdout').read_text()
        assert result.returncode==0 and result.finish.done and 'ORIGINAL_TWO_MUL_COMPLETE PASS' in console
        rows = [[int(v)for v in line.split()[1:]]for line in console.splitlines()if line.startswith('TWO_MUL_COMPLETE_CYCLES ')]
        assert [(i,arm)for i,arm,c in rows]==[(0,0),(1,1),(2,1),(3,0)]
        old = [c for i,arm,c in rows if not arm]
        new = [c for i,arm,c in rows if arm]
        final = {**record,'paired_complete_gsim_cycles':rows,'before_mean':sum(old)/2,'after_mean':sum(new)/2,'saved_percent':100*(sum(old)-sum(new))/sum(old),'DONE':True,'console_sha256':sha(W/'gsim.stdout'),'decision':'Proceed independent full qualification'if sum(new)<sum(old)else'Reject promotion: complete capsule costs more'}
        save(W/'capsule_result.json',final)
        print('ACTUAL_TWO_MUL_RESULT',rows,final['saved_percent'],flush=True)
