"""Public source integration smoke: python -m submission.backend.host_runtime_check SOURCE."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .host_runtime import DeviceArtifact, DeviceRegion, lower_whole_program, materialize_combined_llvm_dialect, whole_program_command_buffer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--dense-input-views', action='store_true')
    parser.add_argument('--share-source-preparations', action='store_true')
    parser.add_argument('--sequence-regions', action='store_true')
    parser.add_argument('--merge-independent-regions', action='store_true',
        help='Exercise the multi-result ABI by grouping independent proved source regions')
    parser.add_argument('--schema', type=Path, default=Path('schemas/mixed_program_plan.schema.json'))
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
    regions = None
    if args.sequence_regions:
        from .source_groups import discover_sequence_regions
        regions = discover_sequence_regions
    if args.merge_independent_regions:
        from .source_groups import discover_regions
        def regions(module):
            selected = discover_regions(module)
            if len(selected) < 2:
                raise ValueError('multi-result check requires at least two proved regions')
            members = {op for region in selected for op in region.operations}
            if any(value.owner in members for region in selected for value in region.inputs):
                raise ValueError('multi-result check requires independent regions')
            operations = tuple(op for op in module.walk() if op in members)
            inputs = tuple(dict.fromkeys(value for region in selected for value in region.inputs))
            outputs = tuple(value for region in selected for value in region.outputs)
            return (DeviceRegion(operations, inputs, outputs,
                reason='checked independent source regions with every live output'),)
    result = lower_whole_program(source, args.workdir, compile_device, regions=regions,
        dense_input_views=args.dense_input_views,
        share_source_preparations=args.share_source_preparations)
    module = materialize_combined_llvm_dialect(result)
    buffer = whole_program_command_buffer(result)
    verdict = validate_mixed_program_plan(source, buffer, module.read_text(),
        schema_path=args.schema, lowered_abi=result.stats['adapter_lowered_abi'])
    # Exercise the same frozen command-buffer contract as the public driver.
    # Mixed-plan preflight alone does not validate kernel_abi serialization.
    from merlin.targetgen.oot_starterkit.cmdbuf import CommandBufferBuilder
    builder = CommandBufferBuilder('gemmini',
        schema_path=Path(__file__).with_name('data') / 'command_buffer.schema.json')
    builder._cb = buffer
    builder.write(args.workdir / 'command_buffer.json')
    (args.workdir / 'preflight.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps(dict(module=str(module), device_regions=len(result.device_artifacts),
        host_recipe={key: result.stats.get(key) for key in ('tensor_preparation_cse', 'static_arena')},
        **verdict), indent=2))
    if not verdict['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
