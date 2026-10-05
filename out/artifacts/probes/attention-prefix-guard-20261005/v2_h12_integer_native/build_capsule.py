"""Qualify complete pinned attention with actual primitive device plane readouts."""
from __future__ import annotations
import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from mlir_oot.golden_gemm import GoldenGemm, Shape
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_tuning import tune
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program
from merlin.runtime.backends import base
from merlin.targetgen.runtime_build import derived_link_script

here = Path(__file__).parent
core = Path('/scratch/agustin/tmp/merlin-bf16-prefix-guard-20261005')
fixture = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_vision_layer0_sdpafix/host/torch_sdpa')
llvm = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
compiler = Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
os.environ['MERLIN_RISCV_GCC'] = str(compiler)
os.environ['MERLIN_GEMMINI_HARNESS_DIR'] = '/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests'
os.environ['MERLIN_GEMMINI_LOAD_ADDRESS'] = '0x80000000'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--variant', type=int, choices=(0,1,2,3), default=1)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--inline-outward', action='store_true')
    parser.add_argument('--integer-pack', action='store_true')
    args = parser.parse_args(); assert 1 <= args.heads <= 12
    work = here/(f'v{args.variant}_h{args.heads}'+('_'+args.tag if args.tag else ''))
    # An experiment directory is an immutable source/ELF receipt once built.
    work.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(here/'build_capsule.py', work/'build_capsule.py')
    for name in ['ordered_fma_bounds.h', 'fma_product_norms.h', 'bf16_radix_pack.h']:
        shutil.copyfile(core/'merlin/runtime/c'/name, work/name)
    shutil.copyfile(core/'merlin/runtime/baremetal/spike/output_sha256.h', work/'output_sha256.h')
    shutil.copyfile(core/'out/artifacts/probes/attention-guard-refinement-20261005/selective_absnorm/guard_absnorm.c', work/'reference_guard.c')
    shutil.copyfile(here/'driver.c', work/'driver.c')
    pin_paths = [fixture/f'input{i}.npy' for i in range(3)] + [fixture/'output.npy']
    arrays = [np.load(path)[0,:args.heads].copy().astype('f4') for path in pin_paths]
    labels = ['source_q', 'source_k', 'source_v', 'source_gold']
    for name,array in zip(labels, arrays, strict=True): array.tofile(work/(name+'.bin'))
    asm='\n'.join(f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{work}/{name}.bin"' for name in labels)+'\n'
    (work/'data.S').write_text(asm)
    native_stub='''#include <stdint.h>
extern void cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
static float af[1024*512],bf[1024*512],cf[1024*1024];
static void run(const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k){
 for(int t=0;t<m*k;t++)af[t]=a[t];for(int t=0;t<k*n;t++)bf[t]=b[t];
 cblas_sgemm(101,111,111,m,n,k,1.0f,af,k,bf,n,0.0f,cf,n);
 for(int t=0;t<m*n;t++)c[t]=(int32_t)cf[t];}
void qk_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,1024,64);}
void pv512_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,512);}
void pv192_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,192);}
void pv128_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,128);}
'''
    (work/'native_stub.c').write_text(native_stub)
    native_command=[str(llvm/'clang'), '-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',f'-DHEADS={args.heads}',f'-DVARIANT={args.variant}','-DNATIVE=1','-DBUILD_HASH="native_functional_only"',str(work/'driver.c'),str(work/'native_stub.c'),str(work/'data.S'),'-lblas','-lm','-o',str(work/'native.so')]
    if args.inline_outward:native_command.insert(1,'-DINLINE_OUTWARD=1')
    if args.integer_pack:native_command.insert(1,'-DINTEGER_PACK=1')
    subprocess.run(native_command, check=True, capture_output=True, text=True)
    native = ctypes.CDLL(str(work/'native.so')); native.attention_capsule_output.restype=ctypes.POINTER(ctypes.c_float)
    started=time.monotonic(); rc=native.main(); elapsed=time.monotonic()-started
    assert rc == 0, 'native original elementwise gate'
    output=np.ctypeslib.as_array(native.attention_capsule_output(), shape=(args.heads*1024*64,)).copy().reshape(args.heads,1024,64)
    assert np.array_equal(output.view('u4'), arrays[3].view('u4')), 'native original bit audit'
    np.save(work/'output.npy',output)
    strategies=['absolute_device_original_gamma','holder_metadata_signed_prefix_half_ulp_source_parts','holder_metadata_signed_prefix_gamma_source_parts','holder_metadata_checked_half_ulp_source_parts']
    qualification={'schema':'original_attention_end_to_end_capsule_v1','scope':f'{args.heads} complete original first-vision attention heads; not whole model, other attention layers, or hardware performance','variant':strategies[args.variant],'physical_output_dtype':'bf16','digest_encoding':'lossless bf16 widening to IEEE f32 little endian','element_count':output.size,'original_gate':{'atol':.03125,'rtol':.02,'native_pass':True,'native_original_bit_mismatches':0},'native_functional_seconds':elapsed,'native_reference_sha256':sha(work/'output.npy'),'native_reference_raw_f32le_sha256':hashlib.sha256(output.tobytes()).hexdigest(),'original_fixture_sha256':{str(path):sha(path) for path in pin_paths},'token_usage_available':False,'token_attribution':'Root retains shared campaign snapshots; exact per-agent/experiment attribution unavailable.'}
    numeric_sources=[work/'driver.c',work/'reference_guard.c',work/'ordered_fma_bounds.h',work/'fma_product_norms.h',work/'bf16_radix_pack.h',work/'output_sha256.h',work/'build_capsule.py']
    source_closure={str(path):sha(path) for path in numeric_sources}
    qualification.update({'source_closure_sha256':source_closure,'native_compiler_argv':native_command,'native_compiler_sha256':sha(llvm/'clang'),'native_library_sha256':sha(work/'native.so'),'integer_pack':args.integer_pack})
    (work/'native_qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    print('NATIVE_QUALIFIED',qualification,flush=True)
    if args.native_only:return
    objects=[]; device_records={}
    for name,m,n,k in [('qk_plane',1024,1024,64),('pv512_plane',1024,64,512),('pv192_plane',1024,64,192),('pv128_plane',1024,64,128)]:
        shape, cost=tune(Shape(m,n,k,'i32',wide_b=True,reuse_b=True))
        module=GoldenGemm(shape).build()
        for operation in module.ops:
            if 'sym_name' in operation.properties: operation.properties['sym_name']=type(operation.properties['sym_name'])(name)
        receipt=compile_module(module,llvm,work/name);objects.append(work/name/'kernel.o')
        device_records[name]={'shape':asdict(shape),'analytical_command_cost_not_cycles':cost,'compile':receipt}
    dataobj=work/'data.o'
    target_flags=['-std=gnu11','-O2','-fno-fast-math','-ffp-contract=off','-mcmodel=medany','-march=rv64gc','-mabi=lp64d','-fno-common','-fno-builtin-printf']
    subprocess.run([str(compiler),*target_flags,'-c',str(work/'data.S'),'-o',str(dataobj)],check=True,capture_output=True)
    objects.append(dataobj)
    marker=hashlib.sha256(json.dumps({'source':source_closure,'objects':{str(p):sha(p) for p in objects},'variant':args.variant,'heads':args.heads,'inline_outward':args.inline_outward,'integer_pack':args.integer_pack},sort_keys=True).encode()).hexdigest()[:12]
    mainobj=work/'main.o'
    command=[str(compiler),*target_flags,f'-DHEADS={args.heads}',f'-DVARIANT={args.variant}',f'-DBUILD_HASH="{marker}"','-c',str(work/'driver.c'),'-o',str(mainobj)]
    if args.inline_outward:command.insert(1,'-DINLINE_OUTWARD=1')
    if args.integer_pack:command.insert(1,'-DINTEGER_PACK=1')
    subprocess.run(command,check=True,capture_output=True,text=True)
    objects.append(mainobj)
    # Compile the provider-owned curated startup/syscalls, then link libc for
    # qsort/assert/fenv used by this full certificate capsule. No ISA shortcuts.
    recipe=base.harness_build_recipe('gemmini'); build=work/'build';build.mkdir(exist_ok=True)
    for source in recipe.support_sources:
        obj=build/(source.stem+'.o')
        subprocess.run(recipe.compile_command(source=source,output=obj),check=True,capture_output=True,text=True)
        objects.append(obj)
    linker=derived_link_script(recipe.load_address,recipe.link_script,build)
    elf=build/'model.elf'
    link=recipe.link_command(objects=objects,output=elf,link_script=linker)
    libc_support=['-lc','-lnosys','-lgcc','-Wl,--defsym,end=_end']
    linked=subprocess.run([*link,*libc_support],capture_output=True,text=True)
    (work/'link.log').write_text(linked.stdout+linked.stderr)
    if linked.returncode:raise RuntimeError('target link: '+linked.stderr)
    audit=audit_elf(elf.read_bytes());assert audit['status']=='pass'
    (work/'elf_nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    qualification.update({'elf':str(elf),'elf_sha256':sha(elf),'build_hash_marker':marker,'source_closure_sha256':source_closure,'compiler_sha256':sha(compiler),'compiler_argv':command,'link_argv':[*link,*libc_support],'device_records':device_records,'final_elf_nofsm_status':audit['status'],'target_spike_qualified':False})
    (work/'build_qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    print('TARGET_BUILT',str(elf),qualification['elf_sha256'],flush=True)


if __name__=='__main__':main()
