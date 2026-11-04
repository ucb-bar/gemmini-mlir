"""Compiler command line and selected stock LLVM toolchain bridge."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'compiler_library'))


def toolchain():
    compiler = os.environ.get('MERLIN_CLANG')
    prefix = os.environ.get('MERLIN_MLIR_INSTALL')
    if prefix:
        directory = Path(prefix) / 'bin'
        return Path(compiler) if compiler else directory / 'clang', directory / 'mlir-translate'
    if compiler:
        return Path(compiler), Path(compiler).parent / 'mlir-translate'
    directory = ROOT.parent / 'install' / 'bin'
    return directory / 'clang', directory / 'mlir-translate'


def compile_c(source):
    clang, translate = toolchain()
    with tempfile.TemporaryDirectory(prefix='gm-compile-') as tmp:
        path = Path(tmp)
        (path/'kernel.c').write_text(source)
        command = [str(clang), '--target=riscv64-unknown-elf', '-march=rv64gc_zba_zbb_zbs_zfh',
                   '-mabi=lp64d', '-mcmodel=medany', '-O2', '-ffreestanding', '-fno-builtin',
                   '-fno-vectorize', '-fno-slp-vectorize', '-fno-unroll-loops',
                   '-fno-discard-value-names', '-S', '-emit-llvm', str(path/'kernel.c'), '-o', str(path/'kernel.ll')]
        settings_path = ROOT / 'settings.json'
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        if settings.get('llvm_loop_mode') == 'canonical':
            from llvm_compile import compile_unoptimized_c_to_llvm
            (path/'kernel.ll').write_text(compile_unoptimized_c_to_llvm(source,clang,Path(translate).parent/'opt'))
        else:
            proc = subprocess.run(command, text=True, capture_output=True)
            if proc.returncode:
                raise ValueError('C lowering failed:\n' + proc.stderr)
        # The normalized pointer ABI carries empty argument dictionaries. noundef is
        # only an optimization promise from C, and is unnecessary at this boundary.
        llvm_ir = (path/'kernel.ll').read_text()
        llvm_ir = re.sub(r'\bnoundef\s*', '', llvm_ir)
        # Clang's link metadata repeats this selected target's ABI, ISA and
        # code-model flags. Keep triple/layout/function features; omit redundant
        # module flags so both ordinary islands and whole programs use the
        # same portable LLVM dialect vocabulary.
        flag_list = re.search(r'^!llvm\.module\.flags\s*=\s*(!\{.*\})\s*$', llvm_ir, re.M)
        if flag_list:
            # Only remove the four settings emitted by this pinned Clang for
            # our selected RV64 lp64d/medany build. Other module flags can carry
            # semantic requirements and must never be silently discarded.
            expected_layout = 'e-m:e-p:64:64-i64:64-i128:128-n32:64-S128'
            expected_isa = ('rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_b1p0_zicsr2p0_zifencei2p0_'
                            'zmmul1p0_zaamo1p0_zalrsc1p0_zfh1p0_zfhmin1p0_zca1p0_zcd1p0_'
                            'zba1p0_zbb1p0_zbs1p0')
            if not re.search(r'^target triple = "riscv64-unknown-(?:unknown-)?elf"$',llvm_ir,re.M):
                raise ValueError('module link metadata target differs from selected riscv64 ELF')
            if not re.search(r'^target datalayout = "'+re.escape(expected_layout)+r'"$',llvm_ir,re.M):
                raise ValueError('module link metadata data layout differs from selected RV64 layout')
            definitions = dict(re.findall(r'^!(\d+)\s*=\s*(!\{.*\})\s*$',llvm_ir,re.M))
            if not re.fullmatch(r'!\{!\d+(?:,\s*!\d+)*\}',flag_list[1]):
                raise ValueError('unsupported generated module flag list')
            seen_flags = set()
            for flag_id in re.findall(r'!(\d+)',flag_list[1]):
                definition = definitions.get(flag_id,'')
                match = re.fullmatch(r'!\{i32 (\d+), !"([^"\\]+)", (.*)\}',definition)
                if match is None:
                    raise ValueError('unsupported generated module flag record')
                behavior, key, value = match.groups()
                if key in seen_flags:
                    raise ValueError('duplicate generated module flag: '+key)
                seen_flags.add(key)
                expected = {'target-abi':('1','!"lp64d"'),
                            'Code Model':('1','i32 3'),
                            'SmallDataLimit':('8','i32 0')}
                if key == 'riscv-isa':
                    if behavior != '6' or not re.fullmatch(r'!\d+',value):
                        raise ValueError('unexpected generated RISC-V ISA module flag')
                    if definitions.get(value[1:]) != '!{!"'+expected_isa+'"}':
                        raise ValueError('module flag ISA differs from selected rv64gc_zba_zbb_zbs_zfh')
                elif key not in expected or (behavior,value) != expected[key]:
                    raise ValueError('unknown or incompatible generated module flag: '+key)
            if seen_flags != {'target-abi','riscv-isa','Code Model','SmallDataLimit'}:
                raise ValueError('incomplete generated selected-target module flags')
        elif re.search(r'^!llvm\.module\.flags\s*=',llvm_ir,re.M):
            raise ValueError('unsupported generated module flag syntax')
        llvm_ir = re.sub(r'^!llvm\.module\.flags\s*=.*\n', '', llvm_ir, flags=re.M)
        llvm_ir = '\n'.join(re.sub(r'\bptr\s+.*?(%arg_\d+)','ptr \\1',line)
            if line.lstrip().startswith('define ') and '@kernel_entry(' in line else line
            for line in llvm_ir.split('\n'))
        (path/'kernel.ll').write_text(llvm_ir)
        proc = subprocess.run([str(translate), '--import-llvm', str(path/'kernel.ll')], text=True, capture_output=True)
        if proc.returncode:
            raise ValueError('LLVM dialect import failed:\n' + proc.stderr)
        from program_plan import normalize_llvm_text
        return normalize_llvm_text(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-diagnostics', action='store_true')
    parser.add_argument('--convert-iface-to-gemmini', action='store_true')
    parser.add_argument('--emit-command-buffer')
    parser.add_argument('--emit-target-artifact', action='store_true')
    parser.add_argument('--emit-c', action='store_true')
    parser.add_argument('input')
    args = parser.parse_args()
    try:
        from frontend import load_program
        text = Path(args.input).read_bytes().decode('utf-8')
        cb = load_program(text)
        if args.verify_diagnostics:
            return 0
        rejected = next((cmd for cmd in cb['commands'] if 'requant' in cmd.get('attributes',{}).get('epilogue',[])),None)
        if rejected is not None:
            cb['declined'] = {'op':rejected['opcode'],'reason':'The declared requant stage requires round-half-up; the selected integer shift uses nearest-even and no declared readout stage implements requant.'}
            cb['commands'] = []
            if args.emit_command_buffer:
                Path(args.emit_command_buffer).write_text(json.dumps(cb,indent=2))
            if args.emit_target_artifact or not args.emit_command_buffer:
                print('module {}')
            return 0
        artifact = None
        c = None
        if cb.get('kernel_abi',{}).get('kind') == 'whole_program':
            from codegen import generate_c
            from program_plan import add_plan, tag_and_finalize
            generate_c(cb) # Discover semantic host/device lane choices.
            add_plan(cb,text)
            c = generate_c(cb)
            artifact = tag_and_finalize(cb,compile_c(c))
        if args.emit_command_buffer:
            schema_dir = ROOT / 'contracts'
            # Explicitly bind the declared public schema; no weaker fallback.
            import jsonschema
            jsonschema.Draft7Validator(json.loads((schema_dir/'command_buffer.schema.json').read_text())).validate(cb)
            Path(args.emit_command_buffer).write_text(json.dumps(cb, indent=2))
        if args.emit_target_artifact or args.emit_c:
            from codegen import generate_c
            c = c or generate_c(cb)
            print(c if args.emit_c else (artifact or compile_c(c)))
        elif not args.emit_command_buffer:
            # This is the selected low-level target IR: LLVM device instructions,
            # with the same executable semantics as the artifact endpoint.
            from codegen import generate_c
            print(artifact or compile_c(generate_c(cb)))
        return 0
    except Exception as exc:
        print(f'gemmini compiler: {exc}', file=sys.stderr)
        if os.environ.get('GEMMINI_DEBUG'):
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
