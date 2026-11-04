"""Public source integration smoke: python -m submission.backend.host_runtime_check SOURCE."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .host_runtime import (DeviceArtifact, lower_whole_program, materialize_combined_llvm_dialect,
    materialize_combined_object, whole_program_command_buffer)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--schema', type=Path, default=Path('schemas/mixed_program_plan.schema.json'))
    parser.add_argument('--host-optimization', choices=('baseline', 'scalar_loops', 'bounded_rne',
        'bounded_rne_fused', 'source_preparation'), default=None)
    parser.add_argument('--compile-only', action='store_true',
        help='build the complete source object; report runtime imports without schema preflight or execution')
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
    result = lower_whole_program(source, args.workdir, compile_device,
        host_optimization=args.host_optimization)
    module = materialize_combined_llvm_dialect(result)
    buffer = whole_program_command_buffer(result)
    (args.workdir / 'command_buffer.json').write_text(json.dumps(buffer, indent=2))
    if args.compile_only:
        materialize_combined_object(result)
        print(json.dumps(result.stats['adapter_combined_object'], indent=2))
        return
    verdict = validate_mixed_program_plan(source, buffer, module.read_text(),
        schema_path=args.schema, lowered_abi=result.stats['adapter_lowered_abi'])
    (args.workdir / 'preflight.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps(dict(module=str(module), device_regions=len(result.device_artifacts),
        host_recipe={key: result.stats.get(key) for key in ('tensor_preparation_cse', 'static_arena',
            'adapter_host_source_preparation', 'adapter_host_optimization')},
        **verdict), indent=2))
    if not verdict['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
