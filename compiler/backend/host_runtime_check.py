"""Public source integration smoke: python -m submission.backend.host_runtime_check SOURCE."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .host_runtime import DeviceArtifact, lower_whole_program, materialize_combined_llvm_dialect, whole_program_command_buffer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--schema', type=Path, default=Path(__file__).parent / 'data/mixed_program_plan.schema.json')
    parser.add_argument('--native-object', action='store_true', help='build the actual RV64 object and report unresolved runtime symbols')
    parser.add_argument('--runtime-object', type=Path, action='append', default=[])
    parser.add_argument('--linker', type=Path)
    parser.add_argument('--linker-script', type=Path)
    parser.add_argument('--runtime-entry')
    args = parser.parse_args()
    from .source_groups import source_to_buffer
    from .compiler import compile_buffer
    from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan
    def compile_device(request):
        command_buffer, text = compile_buffer(source_to_buffer(request.source_text), symbol=request.symbol)
        output = request.workdir / 'device.mlir'
        output.write_text(text)
        return DeviceArtifact(output, request.symbol, command_buffer=command_buffer)
    source = args.source.read_text()
    result = lower_whole_program(source, args.workdir, compile_device)
    module = materialize_combined_llvm_dialect(result)
    buffer = whole_program_command_buffer(result)
    verdict = validate_mixed_program_plan(source, buffer, module.read_text(),
        schema_path=args.schema, lowered_abi=result.stats['adapter_lowered_abi'])
    (args.workdir / 'command_buffer.json').write_text(json.dumps(buffer, indent=2))
    (args.workdir / 'preflight.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps(dict(module=str(module), device_regions=len(result.device_artifacts),
        host_recipe={key: result.stats.get(key) for key in ('tensor_preparation_cse', 'static_arena')},
        **verdict), indent=2))
    if not verdict['ok']:
        raise SystemExit(1)
    if args.native_object or args.runtime_object or args.linker or args.linker_script or args.runtime_entry:
        from .host_native import build_native_product
        from merlin.llvmlower.toolchain import clang
        from .host_runtime import selected_host_llvm_install
        tools = selected_host_llvm_install() / 'bin'
        native = build_native_product(result, module, compiler=clang(), translator=tools / 'mlir-translate',
            nm=tools / 'llvm-nm', cflags=('--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-O2', '-fno-builtin'),
            runtime_objects=args.runtime_object, linker=args.linker, linker_script=args.linker_script,
            entry_symbol=args.runtime_entry)
        print(json.dumps(dict(native_object=native['object_status'], native_link=native['link_status'],
            unresolved_runtime_symbols=native['object_undefined_symbols']), indent=2))


if __name__ == '__main__':
    main()
