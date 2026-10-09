"""Complete source quantization/layout/init A/B capsule at common addresses."""
from pathlib import Path
import ctypes
import hashlib
import json
import shutil
import struct
import subprocess

import numpy as np

from merlin.perf.layer_bench import build_program
from merlin.llvmlower.codegen import mlir_runtime_c
from mlir_oot.no_fsm_audit import audit_elf
from current_quant_prefix_probe import pin

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'out/current_quant_prefix_v3'
WORK = ROOT/'out/current_quant_prefix_capsule_v4'
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
CORE = Path('/scratch/agustin/tmp/merlin-constant-rne-prefix-20261007')


def run(argv, stem, *, cwd=None, timeout=600):
    (WORK/(stem+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n')
    with (WORK/(stem+'.log')).open('w') as log:
        p=subprocess.run(list(map(str,argv)),stdout=log,stderr=subprocess.STDOUT,cwd=cwd,timeout=timeout)
    if p.returncode:raise ValueError(stem+' failed; retained log')


def qualify_native():
    original = FIXTURE/'candidate/native.o'
    source=(FIXTURE/'lookup_native.c').read_text()+'''
    void run_lookup(const float*x,signed char*y,unsigned n,int admitted){for(unsigned i=0;i<n;++i)y[i]=exact_lookup(x[i],admitted);}
    void run_original(const float*x,signed char*y,unsigned n){for(unsigned i=0;i<n;++i)y[i]=original_scalar(x[i]);}
    '''
    (WORK/'native_cells.c').write_text(source)
    run(['cc','-O2','-fPIC','-fno-fast-math','-ffp-contract=off','-shared',WORK/'native_cells.c',original,mlir_runtime_c(),'-lm','-o',WORK/'native_cells.so'],'native_cells_compile')
    lib=ctypes.CDLL(str(WORK/'native_cells.so'))
    lib.run_lookup.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint,ctypes.c_int]
    lib.run_original.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint]
    prefixes=np.arange(1<<18,dtype=np.uint32)<<np.uint32(14)
    checked=0
    libc=ctypes.CDLL(None);previous=libc.fegetround()
    try:
        for mode in (0,0x400,0x800,0xc00):
            assert libc.fesetround(mode)==0
            for offset in (0,1,8192,16383):
                raw=prefixes+np.uint32(offset)
                # Source fptosi of NaN is poison; no payload/result guarantee.
                raw=raw[((raw>>np.uint32(23))&np.uint32(255))!=255]
                x=raw.view(np.float32);out=np.empty(len(x),np.int8);expected=np.empty_like(out)
                lib.run_lookup(x.ctypes.data,out.ctypes.data,len(x),int(mode==0))
                lib.run_original(x.ctypes.data,expected.ctypes.data,len(x))
                if not np.array_equal(out,expected):raise ValueError('compiled cell/fallback mismatch')
                checked+=len(x)
    finally:libc.fesetround(previous)
    return {'finite_source_values_checked':checked,'rounding_modes':[0,0x400,0x800,0xc00],
            'prefixes':1<<18,'positions_per_prefix':[0,1,8192,16383],
            'proof_scope':'interval certification covers every raw word in each cell; compiled endpoints/interiors independently test executor; unsupported modes retain source',
            'nonfinite_scope':'unsupported nonfinite cells retain source callback; NaN poison adds no result or payload contract'}


MAIN = r'''#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "benchmark_buffer.h"
void htif_puts(const char*s){printf("%s",s);}void htif_putc(char c){printf("%c",c);}
void htif_putd(long x){printf("%ld",x);}void htif_puthex(unsigned long long x){printf("%llx",x);}
__attribute__((noreturn)) void htif_exit(int status){printf("FATAL ALLOCATOR %d\n",status);__asm__ volatile("ebreak");for(;;){}}
extern void merlin_arena_reset(void);
extern const unsigned char captured_input[],captured_input_saved[],captured_expected[],selected;
struct descriptor {void*allocated;void*aligned;int64_t offset,sizes[4],strides[4];};
extern void _mlir_ciface_control_forward(struct descriptor*,struct descriptor*);
extern void _mlir_ciface_candidate_forward(struct descriptor*,struct descriptor*);
struct box {unsigned char prefix[4096],data[158700],suffix[4096];};
static struct box output[2] __attribute__((aligned(1048576)));
static void call(unsigned arm,unsigned out){
 struct descriptor in={(void*)captured_input,(void*)captured_input,0,{1,3,224,224},{150528,50176,224,1}};
 struct descriptor result={output[out].data,output[out].data,0,{1,230,230,3},{158700,690,3,1}};
 if(arm)_mlir_ciface_candidate_forward(&in,&result);else _mlir_ciface_control_forward(&in,&result);
}
static int guards(void){for(unsigned b=0;b<2;b++)for(unsigned i=0;i<4096;i++)if(output[b].prefix[i]!=0xa5||output[b].suffix[i]!=0xa5)return 0;return 1;}
static uint64_t checksum(const unsigned char*p,size_t n){uint64_t x=UINT64_C(14695981039346656037);for(size_t i=0;i<n;i++){x^=p[i];x*=UINT64_C(1099511628211);}return x;}
int main(void){
 if(selected>1){printf("SELECTOR_FAIL\n");return 8;}
 for(unsigned repeat=0;repeat<2;repeat++){
  merlin_arena_reset();merlin_benchmark_fill(output,0xa5,sizeof(output));
  uint64_t start,end,istart,iend,frm,flags;
  __asm__ volatile("csrwi frm,0\ncsrwi fflags,0\nfence rw,rw\ncsrr %0,mcycle\ncsrr %1,minstret":"=r"(start),"=r"(istart)::"memory");
  call(selected,0);
  __asm__ volatile("fence rw,rw\ncsrr %0,minstret\ncsrr %1,mcycle\ncsrr %2,frm\ncsrr %3,fflags":"=r"(iend),"=r"(end),"=r"(frm),"=r"(flags)::"memory");
  size_t d=merlin_benchmark_first_difference(output[0].data,captured_expected,158700);
  if(d!=158700||!guards()){printf("OUTPUT_OR_GUARD_FAIL %lu\n",(unsigned long)d);return 2;}
  if(merlin_benchmark_first_difference(captured_input,captured_input_saved,602112)!=602112){printf("INPUT_FAIL\n");return 3;}
  printf("PREFIX_COUNTER arm=%u repeat=%u cycles=%lu instructions=%lu frm=%lu flags=%lu input=%lu output=%lu elements=150528 padded=158700 digest=%016lx\n",selected,repeat,(unsigned long)(end-start),(unsigned long)(iend-istart),(unsigned long)frm,(unsigned long)flags,(unsigned long)captured_input,(unsigned long)output[0].data,(unsigned long)checksum(output[0].data,158700));
 }
 for(unsigned mode=0;mode<5;mode++)for(unsigned sticky=0;sticky<2;sticky++){
  unsigned initial=sticky?31:0;unsigned long flags0,flags1;
  merlin_benchmark_fill(output,0xa5,sizeof(output));
  for(unsigned arm=0;arm<2;arm++){
   merlin_arena_reset();__asm__ volatile("csrw frm,%0\ncsrw fflags,%1"::"r"(mode),"r"(initial):"memory");
   call(arm,arm);
   unsigned long flags;__asm__ volatile("csrr %0,fflags":"=r"(flags)::"memory");
   if(arm)flags1=flags;else flags0=flags;
  }
  if(!guards()||merlin_benchmark_first_difference(output[0].data,output[1].data,158700)!=158700){printf("MODE_OR_GUARD_FAIL %u %u\n",mode,sticky);return 4;}
  if(merlin_benchmark_first_difference(captured_input,captured_input_saved,602112)!=602112){printf("MODE_INPUT_FAIL\n");return 5;}
  printf("PREFIX_MODE frm=%u sticky=%u control_flags=%lu candidate_flags=%lu exact=158700\n",mode,sticky,flags0,flags1);
 }
 __asm__ volatile("csrwi frm,0\ncsrwi fflags,0":::"memory");
 printf("PREFIX_PASS quant=150528 padded=158700 guards=16384 immutable=602112 modes=10\n");return 0;
}
'''


def main():
    WORK.mkdir(parents=True,exist_ok=False)
    fixture=json.loads((FIXTURE/'prelabel.json').read_text())
    native=qualify_native()
    shutil.copyfile(FIXTURE/'input.bin',WORK/'input.bin');shutil.copyfile(FIXTURE/'expected.bin',WORK/'expected.bin')
    flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off']
    run([LLVM/'clang',*flags,'-S','-emit-llvm',FIXTURE/'lookup_target.c','-o',WORK/'lookup.ll'],'lookup_llvm_compile')
    run([LLVM/'llvm-link','-S',FIXTURE/'candidate/target.ll',WORK/'lookup.ll','-o',WORK/'candidate_linked.ll'],'candidate_llvm_link')
    run([LLVM/'opt','-S','-passes=always-inline',WORK/'candidate_linked.ll','-o',WORK/'candidate_inline.ll'],'candidate_always_inline')
    # The source callback and table are exact linked definitions. No opaque
    # hot lookup remains; neither an annotation nor object membership grants it.
    inlined=(WORK/'candidate_inline.ll').read_text()
    if any('call' in line and '@exact_lookup(' in line for line in inlined.splitlines()):
        raise ValueError('declared lookup inlining did not reach actual caller')
    if 'define i8 @original_scalar(' not in inlined or '@observation_cells = ' not in inlined:
        raise ValueError('source callback/table definition missing from linked IR')
    run([LLVM/'clang',*flags,'-c',WORK/'candidate_inline.ll','-o',WORK/'candidate_inline.o'],'candidate_inline_compile')
    for arm in ('control','candidate'):
        source_object=FIXTURE/'control/kernel.o' if arm=='control' else WORK/'candidate_inline.o'
        run([LLVM/'llvm-objcopy','--redefine-sym',f'forward={arm}_forward','--redefine-sym',f'_mlir_ciface_forward=_mlir_ciface_{arm}_forward',source_object,WORK/(arm+'.o')],arm+'_rename')
    runtime=Path('/scratch/agustin/tmp/merlin-resnet-qualified-runtime-20261005/merlin/runtime')
    shutil.copyfile(runtime/'baremetal/spike/merlin_malloc.c',WORK/'merlin_malloc.c');shutil.copyfile(runtime/'baremetal/spike/htif.h',WORK/'htif.h')
    shutil.copyfile(CORE/'merlin/runtime/c/benchmark_buffer.h',WORK/'benchmark_buffer.h')
    (WORK/'main.c').write_text(MAIN)
    (WORK/'operands.S').write_text('''.section .rodata.capture,"a",@progbits
    .balign 1048576
    .global captured_input
    captured_input:.incbin "input.bin"
    .balign 64
    .global captured_input_saved
    captured_input_saved:.incbin "input.bin"
    .balign 64
    .global captured_expected
    captured_expected:.incbin "expected.bin"
    .section .selector,"a",@progbits
    .global selected
    selected:.byte 0
    ''')
    run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections','-c',mlir_runtime_c(),'-o',WORK/'scoped_runtime.o'],'scoped_runtime_compile')
    program=build_program([WORK/'main.c',WORK/'operands.S',WORK/'control.o',WORK/'candidate.o',WORK/'merlin_malloc.c',WORK/'scoped_runtime.o'],WORK,target='gemmini',max_loaded_bytes=None,support_first=True,
                          extra_cflags=['-O2','-fno-fast-math','-ffp-contract=off','-I',str(WORK),'-DMERLIN_ARENA_BASE_ADDR=0xc0000000ULL','-DMERLIN_ARENA_SIZE_BYTES=0x10000000ULL'],extra_ldflags=['-Wl,--gc-sections'])
    blob=program.elf.read_bytes()
    if blob[:6]!=b'\x7fELF\x02\x01':raise ValueError('ELF64 little-endian selector container required')
    shoff=struct.unpack_from('<Q',blob,40)[0];entsize,count,names_index=struct.unpack_from('<HHH',blob,58)
    if entsize!=64:raise ValueError('unsupported ELF section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',blob,shoff+i*entsize) for i in range(count)]
    names=blob[sections[names_index][4]:sections[names_index][4]+sections[names_index][5]]
    selected=[s for s in sections if names[s[0]:].split(b'\0',1)[0]==b'.selector']
    if len(selected)!=1 or selected[0][5]!=1 or selected[0][1]!=1 or selected[0][2]!=2:
        raise ValueError('one allocated readonly byte selector section required')
    offset=selected[0][4]
    if blob[offset]!=0:raise ValueError('original selector is not zero')
    for arm,value in [('control',0),('candidate',1)]:
        changed=bytearray(blob);changed[offset]=value;(WORK/(arm+'.elf')).write_bytes(changed)
    left=(WORK/'control.elf').read_bytes();right=(WORK/'candidate.elf').read_bytes()
    diffs=[i for i,(a,b) in enumerate(zip(left,right,strict=True)) if a!=b]
    if len(diffs)!=1 or left[diffs[0]]!=0 or right[diffs[0]]!=1:raise ValueError('matched ELF selector closure failed')
    rows={}
    for arm in ('control','candidate'):
        elf=WORK/(arm+'.elf');audit=audit_elf(elf.read_bytes())
        if audit['status']!='pass':raise ValueError('final executable FSM gate refused')
        (WORK/(arm+'_audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
        rows[arm]={'elf':pin(elf),'audit':pin(WORK/(arm+'_audit.json'))}
    record={'schema':'complete_source_integer_quant_prefix_capsule_v1','fixture':pin(FIXTURE/'prelabel.json'),'native_cells':native,
            'arms':rows,'selector_byte_offset':diffs[0],'loaded_bytes':program.loaded_bytes,'producer':pin(__file__),
            'protocol':{'counter_rows':2,'mode_rows':10,'final_marker':'PREFIX_PASS quant=150528 padded=158700 guards=16384 immutable=602112 modes=10'},
            'numeric_contract':fixture['numeric_contract'],'source_work':fixture,
            'scope':'complete quantization+layout+zero padding+allocator+ranked output copy; all table/fallback costs included; checks/UART outside timers',
            'model_status':'UNPRICED mixed gather/branch/source fallback; source cell hit coverage is not a cycle predictor',
            'control_scope':'source-derived ranked complete pre-stem map with current eight-lane scalar schedule; isolated map allocation/copy graph may differ from whole-model bufferization'}
    (WORK/'prelabel.json').write_text(json.dumps(record,indent=2)+'\n')
    for arm in ('control','candidate'):
        run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini','-m0x80000000:0x80000000',WORK/(arm+'.elf')],arm+'_spike',timeout=300)
    print(json.dumps({'loaded_bytes':program.loaded_bytes,'selector_offset':diffs[0],'native_checks':native['finite_source_values_checked'],'arms':rows}),flush=True)


if __name__=='__main__':main()
