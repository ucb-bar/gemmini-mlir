"""Matched CPU mechanism screen; synthetic intervals, no model forecast."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import random
import shutil
import struct
import subprocess

from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf

root = Path.cwd()
work = root / 'out/artifacts/probes/polynomial-table-latency-20261006/attempt2'
work.mkdir(exist_ok=False)
frozen = root / 'out/artifacts/probes/smol-polynomial-table-20261006/table10_shift'
core = Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = next(line for line in (frozen/'provider.c').read_text().splitlines()
            if line.startswith('static const merlin_bit_polynomial_plan plan='))
clang = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang')
base = json.loads((frozen/'build.json').read_text())['commands'][0]
objects = []
commands = []
for name, header in [('division', root/'docs/perf_records/smol_table10_negative_source/monotone_polynomial_table.h'),
                     ('shift', frozen/'monotone_polynomial_table.h')]:
    arm = work/name
    arm.mkdir(exist_ok=False)
    for p in frozen.glob('*.h'):
        shutil.copy2(p, arm/p.name)
    shutil.copy2(header, arm/'monotone_polynomial_table.h')
    unit = arm/'kernel.c'
    unit.write_text('#include "monotone_polynomial_table.h"\n'+plan+f'''
static merlin_monotone_bit_polynomial source;
static merlin_monotone_polynomial_table table;
static struct {{ uint32_t before; int32_t knots[1025]; uint32_t after; }} storage;
int {name}_setup(void) {{
 merlin_fma_bound env=merlin_fma_bound_begin();
 source=merlin_monotone_bit_polynomial_prepare(&env,&plan,0);
 storage.before=0xabc123; storage.after=0x321cba;
 table=merlin_monotone_polynomial_table_prepare(&source,storage.knots,1025,10);
 return table.valid;
}}
int {name}_guards(void) {{return storage.before==0xabc123 && storage.after==0x321cba;}}
void {name}_apply(const float *input,uint32_t *output,int n,int repeat) {{
 for(int r=0;r<repeat;r++) for(int i=0;i<n;i++) {{
  merlin_f32_interval value=merlin_monotone_polynomial_table_apply(
     merlin_interval(input[2*i],input[2*i+1]),&table);
  output[3*i]=merlin_interval_bits(value.lo);
  output[3*i+1]=merlin_interval_bits(value.hi); output[3*i+2]=value.valid;
 }}
}}
uint32_t {name}_source(float x) {{
 return x<plan.cutoff ? 0 : merlin_monotone_polynomial_source_word(x,&plan);
}}
''')
    cmd = [a.replace(str(frozen), str(arm)) for a in base]
    cmd[cmd.index('-c')+4] = str(unit) # -c -MD -MF dep source -o object
    cmd[cmd.index('-MF')+1] = str(arm/'kernel.d')
    cmd[cmd.index('-o')+1] = str(arm/'kernel.o')
    subprocess.run(cmd,check=True,capture_output=True)
    commands.append(cmd)
    objects.append(arm/'kernel.o')
    # Native proof uses the same portable source and explicit numeric hooks;
    # target-specific directed binary64 instructions stay in target builds.
    so=arm/'kernel.so'
    subprocess.run([str(clang),'-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',
                    '-I',str(arm),'-include',str(arm/'numeric_capability.h'),
                    str(unit),'-lm','-o',str(so)],check=True,capture_output=True)

rng=random.Random(731)
values=[]
for i in range(512):
    x=ctypes.c_float(rng.uniform(-87.3,-0.02)).value
    values.extend([x,ctypes.c_float(min(0,x+0.001)).value])
array=(ctypes.c_float*len(values))(*values)
native=[]
for name in ['division','shift']:
    lib=ctypes.CDLL(str(work/name/'kernel.so'))
    assert getattr(lib,name+'_setup')()==1
    apply=getattr(lib,name+'_apply')
    apply.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int]
    source=getattr(lib,name+'_source');source.argtypes=[ctypes.c_float];source.restype=ctypes.c_uint32
    out=(ctypes.c_uint32*1536)();apply(array,out,512,1)
    for i in range(512):
        assert out[3*i+2]==1
        lo,hi=values[2*i:2*i+2]
        for j in range(9):
            original=source(lo+(hi-lo)*j/8)
            assert out[3*i]<=original<=out[3*i+1]
    assert getattr(lib,name+'_guards')()==1
    native.append(list(out))
assert native[0]==native[1]
(work/'input.bin').write_bytes(struct.pack('<1024f',*values))
(work/'inputs.S').write_text('.section .rodata\n.balign 64\n.global inputs\ninputs:\n.incbin "input.bin"\n')
subprocess.run([str(clang),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-c',
                str(work/'inputs.S'),'-o',str(work/'inputs.o')],check=True,capture_output=True,cwd=work)
shutil.copy2(core/'merlin/runtime/c/benchmark_buffer.h',work/'benchmark_buffer.h')
(work/'driver.c').write_text('''#include <stdint.h>
#include <stdio.h>
#include "benchmark_buffer.h"
extern const float inputs[1024];
extern int division_setup(void),shift_setup(void),division_guards(void),shift_guards(void);
extern void division_apply(const float*,uint32_t*,int,int),shift_apply(const float*,uint32_t*,int,int);
static struct { uint32_t before; uint32_t data[1536]; uint32_t after; } output;
static uint32_t reference[1536];
static uint64_t tick(void){uint64_t v;__asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
int main(void){
 if(!division_setup()||!shift_setup())return 1;
 division_apply(inputs,reference,512,1);
 for(int round=0;round<2;round++)for(int step=0;step<2;step++){
  int which=round?1-step:step;
  output.before=0x321123;output.after=0x123321;
  merlin_benchmark_fill(output.data,0xa5,sizeof(output.data));
  uint64_t begin=tick();
  if(which)shift_apply(inputs,output.data,512,2);else division_apply(inputs,output.data,512,2);
  uint64_t elapsed=tick()-begin;
  printf("TABLE_LATENCY %d %d %d\\n",round,which,(int)elapsed);
  if(merlin_benchmark_first_difference(reference,output.data,sizeof(reference))!=sizeof(reference))return 2;
  if(output.before!=0x321123||output.after!=0x123321||!division_guards()||!shift_guards())return 3;
 }
 printf("TABLE_LATENCY PASS\\n");return 0;
}
''')
# API spelling is deliberately taken from the current generic harness header.
text=(work/'driver.c').read_text()
(work/'driver.c').write_text(text)
os.environ.update(MERLIN_TARGET_PATH=str(root/'support/gemmini_gsim'),
  MERLIN_RISCV_GCC='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc',
  MERLIN_GEMMINI_HARNESS_DIR='/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests',
  MERLIN_GEMMINI_LOAD_ADDRESS='0x80000000',
  MERLIN_GEMMINI_GSIM_EMU='/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/emulator')
from dataclasses import replace
from merlin.runtime.backends.base import harness_build_recipe
from merlin.targetgen.runtime_build import derived_link_script
from merlin.perf.layer_bench.build import BuiltProgram,loaded_bytes
recipe=harness_build_recipe('gemmini')
recipe=replace(recipe,cflags=(*recipe.cflags,'-fno-fast-math','-ffp-contract=off'),
               ldflags=('-lm','-lgcc','-lc','-lnosys','-lgcc','-Wl,--defsym,end=_end'))
linkscript=derived_link_script(recipe.load_address,recipe.link_script,work)
harness_objects=[]
for unit in [*recipe.support_sources,work/'driver.c']:
    output=work/(unit.stem+'.o')
    command=recipe.compile_command(source=unit,output=output)
    subprocess.run(command,check=True,capture_output=True,cwd=work)
    commands.append(command);harness_objects.append(output)
elf=work/'layer.elf'
command=recipe.link_command(objects=[*harness_objects,*objects,work/'inputs.o'],output=elf,link_script=linkscript)
subprocess.run(command,check=True,capture_output=True,cwd=work);commands.append(command)
built=BuiltProgram(elf,sha(elf),loaded_bytes(elf))
audit=audit_elf(built.elf.read_bytes());assert audit['status']=='pass'
pins={str(p.resolve()):sha(p) for p in work.rglob('*') if p.is_file()}
(work/'build.json').write_text(json.dumps({'scope':'512 synthetic legal intervals with the actual source polynomial plan; CPU helper mechanism only, not actual attention distribution or whole projection. Same source/word enclosure and complete output/guard comparison outside ROI. Preparation outside ROI.','commands':commands,'elf_sha256':built.elf_sha256,'pins':pins,'native_source_samples':9216,'native_output_words_exact':1536,'nofsm':audit},indent=2)+'\n')
print(json.dumps({'built':built.elf_sha256,'native':'pass'}),flush=True)
run=run_on_gsim(built.elf,target='gemmini',max_cycles=5000000,timeout_s=600,backdoor=True,stdout_path=work/'gsim.stdout')
result={'completed':run.completed,'returncode':run.returncode,'stdout':run.stdout_tail,'stderr':run.stderr_tail,
        'wall_seconds':run.wall_seconds,'engine':run.engine,'command_sha256':run.command_sha256,'stock_cycles':None}
result['pass']=run.completed and run.returncode==0 and 'TABLE_LATENCY PASS' in run.stdout_tail
(work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert result['pass']
