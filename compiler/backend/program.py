"""Source-semantic routing through shared host lowering and target callbacks."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile


def _identity(source):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(source.encode())
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json', '.yaml'):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def compile_program(source):
    """Compile exact source bytes; cache only against all implementation bytes."""
    key = _identity(source)
    work = Path(tempfile.gettempdir()) / 'fresh10-authored-compiler' / key
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cb_path, artifact_path = work / 'commands.json', work / 'module.mlir'
        if cb_path.is_file() and artifact_path.is_file():
            return json.loads(cb_path.read_text()), artifact_path.read_text()
        cb, artifact = _compile(source, work)
        artifact_path.write_text(artifact)
        # Tensor declaration order is part of the logical pointer ABI.
        cb_path.write_text(json.dumps(cb))
        return cb, artifact


def _compile(source, work):
    from .frontend import FrontendError, parse_source, build_direct_program_plan
    from .compiler import compile_buffer
    try:
        cb = parse_source(source)
    except FrontendError:
        if 'merlin_iface.' in source:
            raise
        return _mixed(source, work)
    # Every standalone invocation receives a fresh logical address mapping,
    # including interface projections used to compile owned device kernels.
    # Source syntax does not establish translation-cache residency.
    cb.setdefault('params', {})['address_space_boundary'] = True
    cb, artifact = compile_buffer(cb)
    if 'merlin_iface.' not in source and not cb.get('declined'):
        from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
        from xdsl.parser import Parser
        module = Parser(make_llvm_context(), artifact).parse_module()
        entry = next(op for op in module.body.block.ops if op.name == 'llvm.func' and op.sym_name.data == 'kernel_entry')
        count = sum(op.name == 'llvm.inline_asm' for op in entry.walk())
        cb = build_direct_program_plan(source, cb, schedule_instruction_count=count)
    return cb, artifact


def _mixed(source, work):
    from .host_runtime import DeviceArtifact, lower_whole_program, materialize_combined_llvm_dialect, whole_program_command_buffer
    from .source_groups import source_to_buffer, discover_regions
    from .compiler import compile_buffer
    from merlin.compile.tuning_knobs import knob
    transpose_transfers = knob('TRANSPOSE_TRANSFERS', default='off', values=('off', 'on', 'device_keys'),
        root=Path(__file__).resolve().parent.parent) != 'off'
    repeat_transfers = knob('REPEAT_TRANSFERS', default='off', values=('off', 'on'),
        root=Path(__file__).resolve().parent.parent) == 'on'
    def device(request):
        cb = source_to_buffer(request.source_text, transpose_transfers=transpose_transfers, repeat_transfers=repeat_transfers)
        cb.setdefault('params', {})['address_space_boundary'] = True
        cb['kernel_abi'] = {'kind':'whole_program', 'args':[
            *[{'tensor':f'arg{i}','access':'read'} for i in range(len(request.input_types))],
            {'tensor':'Y0','access':'write'}], 'outputs':['Y0']}
        cb, artifact = compile_buffer(cb, symbol=request.symbol)
        path = request.workdir / 'device.mlir'
        path.write_text(artifact)
        return DeviceArtifact(path, request.symbol, command_buffer=cb)
    result = lower_whole_program(source, work / 'mixed', device,
        regions=lambda module: discover_regions(module, transpose_transfers=transpose_transfers, repeat_transfers=repeat_transfers))
    path = materialize_combined_llvm_dialect(result)
    cb = whole_program_command_buffer(result)
    from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan
    check = validate_mixed_program_plan(source, cb, path.read_text(),
        schema_path=Path(__file__).parent / 'data/mixed_program_plan.schema.json',
        lowered_abi={'entry_symbol':'kernel_entry','completion_symbol':'kernel_complete','completion_asm':'fence'})
    if not check['ok']:
        raise ValueError('mixed source/emitted task preflight: ' + '; '.join(check['findings']))
    return cb, path.read_text()
