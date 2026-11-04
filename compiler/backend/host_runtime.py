"""Caller-owned mixed-program adapter using Merlin's ordinary host lowering.

This module owns routing and ABI glue only. It never implements tensor arithmetic.
Device compilation is supplied explicitly, and a device compilation failure is fatal.
The default region discoverer uses the shared exact integer contraction and
readout proofs; callers can supply other proved regions without changing host code generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Sequence
import hashlib
import json


class WholeProgramError(ValueError):
    pass


def _host_llvm_install(*, module_path=None, environ=None):
    """Find the enclosing workspace install without assuming checkout depth.

    An explicit install remains authoritative, including a missing selection.
    Variants may live several directories below the workspace tool install.
    """
    import os
    from merlin.llvmlower.toolchain import llvm_install
    env = os.environ if environ is None else environ
    if 'MERLIN_MLIR_INSTALL' in env:
        return Path(env['MERLIN_MLIR_INSTALL'])
    origin = Path(__file__ if module_path is None else module_path).resolve()
    for parent in origin.parents:
        candidate = parent / 'install'
        if (candidate / 'bin' / 'mlir-opt').is_file():
            return candidate
    return llvm_install()


def _selected_host_optimization(selection: str | None) -> str:
    from merlin.compile.tuning_knobs import knob
    values = ('baseline', 'scalar_loops', 'bounded_rne', 'bounded_rne_fused')
    if selection is None:
        selection = knob('HOST_OPTIMIZATIONS', default='baseline', values=values,
            root=Path(__file__).resolve().parents[1])
    if selection not in values:
        raise WholeProgramError('unknown selected scalar host optimization')
    return selection


@dataclass(frozen=True)
class DeviceRegion:
    """An exact source region and its pointer ABI, in input then output order.

    ``operations`` must be in one block and contain all producers internal to the
    selected region. ``outputs`` includes every value used outside the region.
    The compiler callback must preserve the complete source function supplied to it.
    """
    operations: tuple[Any, ...]
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]
    reason: str = "supported signed integer contraction"


@dataclass(frozen=True)
class DeviceRequest:
    source_text: str
    symbol: str
    source_sha256: str
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    source_ordinals: tuple[int, ...]
    workdir: Path


@dataclass(frozen=True)
class DeviceArtifact:
    """A real compiled device function with the requested logical pointer ABI.

    The function takes all request inputs then outputs, densely stored, and fully
    writes each output. LLVM dialect, LLVM IR, or an object can be returned as a
    path; final linking remains explicit in WholeProgramResult.link_inputs.
    """
    path: Path
    symbol: str
    format: str = "llvm-dialect"
    command_buffer: dict[str, Any] | None = None


@dataclass(frozen=True)
class WholeProgramResult:
    host_llvm_ir: Path
    host_llvm_dialect: Path | None
    shim_c: Path | None
    device_artifacts: tuple[DeviceArtifact, ...]
    routed_source: Path
    report_path: Path
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def link_inputs(self) -> tuple[Path, ...]:
        return (self.host_llvm_ir,) + ((self.shim_c,) if self.shim_c else ()) + tuple(a.path for a in self.device_artifacts)


def dense_regions(module) -> tuple[DeviceRegion, ...]:
    """Discover exact i8*i8->i32 regions, refusing unsupported integer contractions.

    Floating contractions remain explicit hardware-unsupported host seams. No
    integer contraction is sent to the CPU because its adapter is incomplete.
    """
    from merlin.compile.analysis.contraction_shapes import observe_contractions, zero_initialised
    from merlin.frontends.linalg_patterns import recognize_signed_i8_i32_matmul
    from xdsl.ir import Operation
    regions = []
    for op, shape in observe_contractions(module):
        if shape.dtypes and all(t.startswith(('f', 'bf')) for t in shape.dtypes):
            continue
        if shape.dtypes != ('i8', 'i8', 'i32') or len(shape.parallel) != 2 or len(shape.reduction) != 1:
            raise WholeProgramError(f"integer contraction needs target lowering: {shape}")
        if op.name != 'linalg.generic':
            raise WholeProgramError(f"integer contraction requires a proved scalar-body recognizer: {op.name}")
        try:
            lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
        except ValueError as exc:
            raise WholeProgramError(f"integer contraction body is unsupported: {exc}") from exc
        if not zero_initialised(op):
            raise WholeProgramError('integer contraction needs accumulator preload for its live initializer')
        # Include the source initializer rather than changing it into an assumed zero.
        dependencies = []
        seen = set()
        def collect(value):
            owner = value.owner
            if not isinstance(owner, Operation) or owner.name not in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
                raise WholeProgramError('contraction initializer has unsupported source dependencies')
            if owner in seen:
                return
            for operand in owner.operands:
                collect(operand)
            seen.add(owner)
            dependencies.append(owner)
        collect(init)
        operations = dependencies + [op]
        output = op.results[0]
        users = list(output.uses)
        if len(users) == 1:
            from .frontend import _saturating_cast
            narrowing = users[0].operation
            if _saturating_cast(narrowing) is output:
                # Constants captured by the scalar region belong to this exact
                # source group, including the signed saturation bounds.
                local = set(narrowing.walk())
                for nested in narrowing.walk():
                    for operand in nested.operands:
                        owner = operand.owner
                        if isinstance(owner, Operation) and owner not in local and owner not in operations and owner.name == 'arith.constant':
                            operations.insert(-1, owner)
                operations.append(narrowing)
                output = narrowing.results[0]
        regions.append(DeviceRegion(tuple(operations), (lhs, rhs), (output,)))
    return tuple(regions)


def _source_function(region: DeviceRegion, symbol: str) -> str:
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from merlin.xdsl_dialects._common import text
    block = Block(arg_types=[v.type for v in region.inputs])
    mapping = dict(zip(region.inputs, block.args, strict=True))
    for operation in region.operations:
        block.add_op(operation.clone(value_mapper=mapping))
    block.add_op(func.ReturnOp(*(mapping[v] for v in region.outputs)))
    fn = func.FuncOp(symbol, ([v.type for v in region.inputs], [v.type for v in region.outputs]), Region(block))
    return text(ModuleOp([fn]))


def lower_whole_program(
    source_text: str,
    workdir: str | Path,
    compile_device: Callable[[DeviceRequest], DeviceArtifact],
    *,
    regions: Callable[[Any], Sequence[DeviceRegion]] | None = None,
    retain_llvm_dialect: bool = True,
    host_driver: str = 'stock-cli',
    lower_options: dict[str, Any] | None = None,
    host_optimization: str | None = None,
) -> WholeProgramResult:
    """Route exact device regions, then lower remaining source through Merlin.

    ``compile_device`` receives standalone original linalg text and a unique
    function symbol. It must return the corresponding compiled pointer-ABI
    function. No package loader or reference compiler is consulted. ``regions``
    can use a caller-selected capability oracle to return larger fused groups;
    outputs are checked against all live uses before any source is mutated.

    Returned C glue is shared Merlin descriptor adaptation and must be compiled
    and linked together with both the host LLVM and every device artifact.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from merlin.llvmlower.declaration_access import patch_declaration_arg_attrs, unpatched_declarations
    from merlin.llvmlower.device_shim import LogicalBuffer, emit_logical_translation_unit
    from merlin.llvmlower.lower import lower_model
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType
    from xdsl.ir import Region

    host_optimization = _selected_host_optimization(host_optimization)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    module = parse_mlir_text(source_text)
    source_functions = [op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks]
    candidates = [op for op in source_functions if op.sym_name.data == 'forward'] or source_functions
    if len(candidates) != 1:
        raise WholeProgramError('whole-program source needs one unambiguous public tensor entry')
    entry = candidates[0]
    abi_types = tuple(entry.function_type.inputs) + tuple(entry.function_type.outputs)
    source_abi = dict(entry=entry.sym_name.data, inputs=len(entry.function_type.inputs), slots=[
        dict(shape=list(t.get_shape()), dtype=str(t.element_type)) for t in abi_types])
    original_source_path = work / 'original.mlir'
    original_source_path.write_text(source_text)
    existing_symbols = {getattr(getattr(op, 'sym_name', None), 'data', '') for op in module.body.block.ops}
    if regions is None:
        from .source_groups import discover_regions
        regions = discover_regions
    selected = tuple(regions(module))
    ordinal = {op: i for i, op in enumerate(module.walk())}
    owned = set()
    plans = []
    artifacts = []
    shims = {}
    rows = []
    for index, region in enumerate(selected):
        members = set(region.operations)
        if not members:
            raise WholeProgramError('device region is empty')
        if any(op in owned and op.name not in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill') for op in members):
            raise WholeProgramError('device arithmetic regions overlap')
        if len(region.outputs) != 1:
            raise WholeProgramError('shared logical shim currently requires exactly one returned tensor')
        root = region.outputs[0].owner
        if root not in members or any(op.parent is not root.parent for op in members):
            raise WholeProgramError('device region must be in one block with a retained result producer')
        for op in members:
            for operand in op.operands:
                if operand.owner not in members and operand not in region.inputs:
                    raise WholeProgramError('region omits a live input dependency')
            for result in op.results:
                if any(use.operation not in members for use in result.uses) and result not in region.outputs:
                    # Shared initializer producers may remain in the source; only
                    # the output-producing root is erased below.
                    if op is root:
                        raise WholeProgramError('device region omits a live output')
        for value in (*region.inputs, *region.outputs):
            if not isinstance(value.type, TensorType) or any(d <= 0 for d in value.type.get_shape()):
                raise WholeProgramError('device boundary needs static ranked tensors')
        symbol = f'merlin_device_group_{index}'
        kernel_symbol = f'{symbol}_kernel'
        if symbol in existing_symbols or kernel_symbol in existing_symbols:
            raise WholeProgramError('generated device symbol collides with source symbol')
        source = _source_function(region, kernel_symbol)
        subdir = work / symbol
        subdir.mkdir(exist_ok=True)
        (subdir / 'source.mlir').write_text(source)
        request = DeviceRequest(source, kernel_symbol, hashlib.sha256(source.encode()).hexdigest(),
            tuple(str(v.type) for v in region.inputs), tuple(str(v.type) for v in region.outputs),
            tuple(ordinal[o] for o in region.operations), subdir)
        artifact = compile_device(request)
        if not isinstance(artifact, DeviceArtifact) or artifact.symbol != kernel_symbol or not Path(artifact.path).is_file() or Path(artifact.path).stat().st_size == 0:
            raise WholeProgramError(f'{symbol}: callback did not supply the requested compiled device artifact')
        if artifact.command_buffer is not None and not artifact.command_buffer.get('commands'):
            raise WholeProgramError(f'{symbol}: device callback returned an empty or declined command buffer')
        artifacts.append(artifact)
        buffers = []
        call = []
        for i, value in enumerate((*region.inputs, *region.outputs)):
            kind = 'input' if i < len(region.inputs) else 'output'
            role = f'arg{i}'
            shape, dtype = tuple(value.type.get_shape()), str(value.type.element_type)
            buffers.append(LogicalBuffer(name=role, kind=kind, role=role, shape=shape, dtype=dtype))
            call.append(dict(role=role, shape=list(shape), dtype=dtype))
        shims[symbol] = (buffers, call)
        plans.append((region, root, symbol))
        owned.update(members)
        rows.append(dict(symbol=symbol, kernel_symbol=kernel_symbol, source_ordinals=list(request.source_ordinals),
            source_sha256=request.source_sha256, placement='device', reason=region.reason))

    # Callbacks all succeed before altering source: no partially routed host fallback.
    routed_values = {}
    for region, root, symbol in plans:
        block = root.parent
        destination = tensor.EmptyOp([], region.outputs[0].type)
        operands = (*(routed_values.get(value, value) for value in region.inputs), destination.tensor)
        call = func.CallOp(symbol, operands, [region.outputs[0].type])
        block.insert_op_before(destination, root)
        block.insert_op_before(call, root)
        routed_values[region.outputs[0]] = call.results[0]
        region.outputs[0].replace_all_uses_with(call.results[0])
        # Erase only dead source members; shared initialization remains available.
        for op in reversed(region.operations):
            if op.name not in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill') and all(not result.uses for result in op.results):
                op.detach()
                op.erase()
        accesses = ['read'] * len(region.inputs) + ['write']
        attrs = ArrayAttr([DictionaryAttr({'bufferization.access': StringAttr(a)}) for a in accesses])
        module.body.block.add_op(func.FuncOp(symbol, ([v.type for v in operands], [region.outputs[0].type]),
            Region(), visibility='private', arg_attrs=attrs))
    routed = text(module)
    for region, _, symbol in plans:
        routed = patch_declaration_arg_attrs(routed, [symbol], argument_access=['read'] * len(region.inputs) + ['write'])
    if unpatched_declarations(routed, shims):
        raise WholeProgramError('device declarations lost their required bufferization access')
    numerical_preparation = None
    if host_optimization in ('bounded_rne', 'bounded_rne_fused'):
        # Use the same checked source representation as Merlin's original host
        # numerics route. Extension lowering preserves the original typed
        # quantization; its independent expression checker admits only closed
        # round/clamp/cast chains. Device calls remain opaque boundaries.
        from merlin.llvmlower.passes_xdsl import lower_quant_ext
        from merlin.llvmlower.automatic_host import optimize_host_quantization
        prepared = parse_mlir_text(routed)
        extension_count = lower_quant_ext(prepared)
        prepared, numerical_preparation = optimize_host_quantization(prepared,
            numerical_policy='typed_source:' + hashlib.sha256(source_text.encode()).hexdigest())
        if host_optimization == 'bounded_rne_fused':
            from merlin.llvmlower.host_pointwise_fusion import fuse_live_pointwise
            fusion_rows = fuse_live_pointwise(prepared)
            numerical_preparation['pointwise_fusion'] = dict(
                rewrites=len(fusion_rows), regions=fusion_rows,
                selection='checked private chains and retained live outputs')
        routed = text(prepared, generic=True)
        numerical_preparation['quantization_extensions_lowered'] = extension_count
        numerical_preparation['selected_source_sha256'] = hashlib.sha256(routed.encode()).hexdigest()
        (work / 'host-numerics.json').write_text(json.dumps(numerical_preparation, indent=2))
    source_path = work / 'routed.mlir'
    source_path.write_text(routed)
    shim_path = None
    if shims:
        shim = emit_logical_translation_unit('gemmini', shims, kernel_symbol_for=lambda s: s + '_kernel')
        if shim.skipped or set(shim.symbols) != set(shims):
            raise WholeProgramError(f'device ABI adaptation refused: {shim.skipped}')
        shim_path = work / 'device_shim.c'
        shim_path.write_text(shim.text)
    options = dict(lower_options or {})
    # These shared passes preserve exact source arithmetic and independently
    # check every reuse. No input argument is presumed immutable.
    from merlin.llvmlower.tensor_provenance_cse import FEATURE as preparation_cse
    options['features'] = sorted(set(options.get('features') or ()) | {preparation_cse})
    options.setdefault('static_arena', True)
    if 'targets' in options or 'retain_llvm_dialect' in options:
        raise WholeProgramError('targets and retention are owned by the whole-program adapter')
    if host_driver == 'stock-cli':
        # Configure only the child. Other compiler requests in this process keep
        # their own explicit transport and pass selections.
        import os
        import subprocess
        import sys
        from types import SimpleNamespace
        host_work = (work / 'host').resolve()
        host_work.mkdir(exist_ok=True)
        response = host_work / 'adapter-result.json'
        request_path = host_work / 'adapter-request.json'
        request_path.write_text(json.dumps(dict(source=str(source_path.resolve()), work=str(host_work),
            result=str(response), options=options)))
        env = dict(os.environ)
        env.update(MERLIN_COMPILER_DRIVER='stock-cli', MERLIN_FUSION_GUARD='0',
            MERLIN_SPLIT_FRESH_DEALLOCS='0', MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT='0')
        if host_optimization == 'bounded_rne_fused':
            # Existing scalar-pipeline alternative: expose named pointwise
            # operations to the ordinary tensor fusion before bufferization.
            # This switch is consumed by pipeline._splice in the child, unlike
            # MERLIN_FUSE_POST, which affects only the vector pipeline.
            env['MERLIN_GENERALIZE_BEFORE_FUSE'] = '1'
        install = _host_llvm_install()
        env.setdefault('MERLIN_MLIR_INSTALL', str(install))
        run = subprocess.run([sys.executable, str(Path(__file__).with_name('host_runtime_worker.py')),
            str(request_path)], env=env, capture_output=True, text=True, timeout=900)
        if run.returncode:
            raise WholeProgramError('shared stock host lowering failed: ' + run.stderr[-6000:])
        response_data = json.loads(response.read_text())
        lowered = SimpleNamespace(ll_path=Path(response_data['ll_path']), stats=response_data['stats'])
        if host_optimization == 'bounded_rne_fused':
            recipe_path = host_work / 'lowering_recipe.json'
            recipe = json.loads(recipe_path.read_text())
            pipelines = [token for command in recipe['commands'] for token in command['argv']
                if token.startswith('--pass-pipeline=')]
            if len(pipelines) != 1 or not (
                pipelines[0].find('linalg-generalize-named-ops') >= 0
                and pipelines[0].find('linalg-generalize-named-ops') < pipelines[0].find('linalg-fuse-elementwise-ops')
            ):
                raise WholeProgramError('selected stock scalar fusion order did not reach the actual lowering command')
            lowered.stats['adapter_stock_host_pipeline'] = dict(
                selection='MERLIN_GENERALIZE_BEFORE_FUSE=1',
                pipeline=pipelines[0],
                contract=recipe['sources']['pipeline_driver'],
                recipe=dict(path=str(recipe_path), sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest()),
                source=recipe['sources']['prepared_mlir'],
                returned_llvm_ir=recipe['returned_llvm_ir'],
                cost_evidence='not_measured')
    elif host_driver == 'python':
        if host_optimization == 'bounded_rne_fused':
            raise WholeProgramError('bounded_rne_fused requires the explicitly selected stock scalar transport')
        options['features'] = frozenset(options['features'])
        lowered = lower_model(routed, work / 'host', targets=(), retain_llvm_dialect=retain_llvm_dialect, **options)
    else:
        raise WholeProgramError('host_driver must be stock-cli or python')
    lowered.stats['adapter_abi'] = source_abi
    lowered.stats['adapter_host_source_optimization'] = host_optimization
    if numerical_preparation is not None:
        lowered.stats['adapter_host_numerical_preparation'] = numerical_preparation
    lowered.stats['adapter_original_source'] = str(original_source_path.resolve())
    product = lowered.stats.get('llvm_dialect_product')
    dialect_path = None
    if host_driver == 'stock-cli' and retain_llvm_dialect:
        dialect_path = (work / 'host' / 'model.llvm.mlir').resolve()
    if product:
        receipt = json.loads(Path(product['path']).read_text())
        dialect_path = Path(receipt['llvm_dialect']['path'])
    report_path = work / 'placement.json'
    report_path.write_text(json.dumps(dict(schema='submission.whole_program.v1', device_regions=rows,
        source_sha256=hashlib.sha256(source_text.encode()).hexdigest(), static_boundary_bytes=sum(
            int(__import__('math').prod(v.type.get_shape())) * max(1, int(str(v.type.element_type)[1:]) // 8)
            for r in selected for v in (*r.inputs, *r.outputs)),
        host_policy='Merlin shared host codegen; floating contractions are hardware unsupported',
        measured_host_cycles=None), indent=2))
    return WholeProgramResult(lowered.ll_path, dialect_path, shim_path, tuple(artifacts), source_path, report_path, lowered.stats)


def materialize_combined_llvm_dialect(
    result: WholeProgramResult,
    output: str | Path | None = None,
    *,
    symbol: str = 'kernel_entry',
    cflags: Sequence[str] = ('--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d'),
    host_optimization: str | None = None,
) -> Path:
    """Produce one LLVM-dialect module including host, device and descriptor glue.

    The default C flags bind this task's scalar RV64GC/lp64d host. A different
    caller must explicitly supply its selected CPU flags. Only descriptor glue
    is compiled from C; tensor arithmetic remains in the shared host product and
    callback-supplied target modules. All external runtime calls remain visible.
    """
    import subprocess
    import os
    from math import prod
    from merlin.llvmlower.toolchain import clang, llvm_install, mlir_translate
    selected_install = _host_llvm_install()
    translate = Path(os.environ.get('MERLIN_MLIR_TRANSLATE', str(selected_install / 'bin' / 'mlir-translate')))
    if not symbol.isidentifier() or not symbol.isascii():
        raise WholeProgramError('invalid whole-program kernel symbol')
    if not any(f.startswith('-march=') for f in cflags) or not any(f.startswith('-mabi=') for f in cflags):
        raise WholeProgramError('descriptor glue needs explicit selected host CPU and ABI flags')
    abi = result.stats['adapter_abi']
    entry = abi['entry']
    if not entry.isidentifier() or not entry.isascii():
        raise WholeProgramError('unsupported host entry symbol')
    work = result.routed_source.parent / 'combined'
    work.mkdir(exist_ok=True)
    def run(args):
        completed = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=900)
        if completed.returncode:
            raise WholeProgramError('whole-program materialization failed: ' + completed.stderr[-6000:])
    # The selected bare-metal C runtime supplies memset. Local standard
    # declarations avoid importing an unrelated host libc/sysroot ABI.
    include = work / 'host_runtime_include'
    include.mkdir(exist_ok=True)
    (include / 'string.h').write_text('void *memset(void *, int, __SIZE_TYPE__);\nvoid *memcpy(void *, const void *, __SIZE_TYPE__);\n')
    typedefs, declarations, locals_, addresses = [], [], [], []
    ranks = sorted({len(slot['shape']) for slot in abi['slots']})
    for rank in ranks:
        arrays = f'long sizes[{rank}]; long strides[{rank}];' if rank else ''
        typedefs.append(f'typedef struct {{ void *allocated; void *aligned; long offset; {arrays} }} descriptor_{rank};')
    for i, slot in enumerate(abi['slots']):
        shape = slot['shape']
        rank = len(shape)
        stride = [prod(shape[j+1:]) for j in range(rank)]
        arrays = ', {' + ', '.join(map(str, shape)) + '}, {' + ', '.join(map(str, stride)) + '}' if rank else ''
        locals_.append(f'  descriptor_{rank} d{i} = {{ p{i}, p{i}, 0{arrays} }};')
        addresses.append(f'&d{i}')
        declarations.append(f'descriptor_{rank} *')
    wrapper = '\n'.join(typedefs) + '\n_Static_assert(sizeof(long) == 8 && sizeof(void *) == 8, "ranked ABI requires LP64");\n'
    wrapper += f'extern void _mlir_ciface_{entry}(' + ', '.join(declarations) + ');\n'
    wrapper += f'void {symbol}(' + ', '.join(f'void *p{i}' for i in range(len(abi['slots']))) + ') {\n'
    wrapper += '\n'.join(locals_) + f'\n  _mlir_ciface_{entry}(' + ', '.join(addresses) + ');\n}\n'
    wrapper_path = work / 'entry.c'
    wrapper_path.write_text(wrapper)
    llvm_inputs = [result.host_llvm_ir.resolve()]
    for i, c_source in enumerate((wrapper_path,) + ((result.shim_c,) if result.shim_c else ())):
        ll = work / f'glue_{i}.ll'
        run([clang(), *cflags, '-O1', '-ffreestanding', '-I', include, '-S', '-emit-llvm', c_source, '-o', ll])
        llvm_inputs.append(ll)
    for i, artifact in enumerate(result.device_artifacts):
        if artifact.format == 'llvm-dialect':
            ll = work / f'device_{i}.ll'
            run([translate, '--mlir-to-llvmir', Path(artifact.path).resolve(), '-o', ll])
            llvm_inputs.append(ll)
        elif artifact.format == 'llvm-ir':
            llvm_inputs.append(Path(artifact.path).resolve())
        else:
            raise WholeProgramError('single-module materialization needs LLVM device artifacts')
    linked = work / 'combined.ll'
    run([selected_install / 'bin' / 'llvm-link', '-S', *llvm_inputs, '-o', linked])
    target = Path(output) if output is not None else work / 'combined.mlir'
    # Whole-program artifacts expose one executable entry. Inline the exact
    # linked helpers, retaining external runtime declarations as such.
    import re
    inline_input = work / 'inline.ll'
    linked_text = linked.read_text()
    lines = []
    for line in linked_text.splitlines():
        if line.startswith('define ') and ('@' + symbol + '(') not in line:
            if not line.rstrip().endswith('{'):
                raise WholeProgramError('unsupported multiline LLVM function header')
            line = line.rstrip()[:-1] + ' alwaysinline {'
        lines.append(line)
    completion_symbol = 'kernel_complete'
    completion_asm = 'fence'
    lines.append('define void @kernel_complete() {\n  call void asm sideeffect \"fence\", \"~{memory}\"()\n  ret void\n}')
    inline_input.write_text('\n'.join(lines) + '\n')
    inlined = work / 'inlined.ll'
    # Keep one choice across source preparation, child lowering and native
    # materialization even if another compilation updates the tuning file.
    if host_optimization is None:
        host_optimization = result.stats.get('adapter_host_source_optimization')
    host_optimization = _selected_host_optimization(host_optimization)
    if host_optimization in ('bounded_rne', 'bounded_rne_fused'):
        if result.stats.get('adapter_host_source_optimization') not in ('bounded_rne', 'bounded_rne_fused'):
            raise WholeProgramError('bounded RNE needs its checked source preparation during host lowering')
        if host_optimization == 'bounded_rne_fused' and not result.stats.get('adapter_stock_host_pipeline'):
            raise WholeProgramError('bounded RNE fusion needs its actually consumed stock scalar pipeline')
        if '-march=rv64gc' not in cflags or '-mabi=lp64d' not in cflags:
            raise WholeProgramError('bounded RNE requires the admitted scalar RV64GC/lp64d CPU contract')
    native_pipeline = 'internalize,always-inline,sroa,instcombine,simplifycfg,globaldce'
    if host_optimization in ('scalar_loops', 'bounded_rne_fused'):
        # Native LLVM constructs loop/dominance/MemorySSA analyses after the
        # linked descriptor and target functions have been inlined. LICM uses
        # its ordinary alias and speculative-execution checks. Memory-clobbering
        # device asm remains opaque; no FP reassociation or vector pass is added.
        native_pipeline = ('internalize,always-inline,function(sroa,instcombine,simplifycfg,'
            'loop-simplify,lcssa,loop-mssa(licm),instcombine,simplifycfg),globaldce')
    optimization_command = [selected_install / 'bin' / 'opt', '-S',
        '--passes=' + native_pipeline, '--internalize-public-api-list=' + symbol + ',kernel_complete',
        inline_input, '-o', inlined]
    optimization_remarks = work / 'host-loop-remarks.yaml'
    optimization_remarks.unlink(missing_ok=True)
    if host_optimization in ('scalar_loops', 'bounded_rne_fused'):
        optimization_command += ['--pass-remarks=licm', '--pass-remarks-output=' + str(optimization_remarks)]
    run(optimization_command)
    numerical_legalization = None
    if host_optimization in ('bounded_rne', 'bounded_rne_fused'):
        from merlin.llvmlower import late_quant_rne
        original_native = work / 'before-bounded-rne.ll'
        original_native.write_bytes(inlined.read_bytes())
        rewritten, numerical_legalization = late_quant_rne.rewrite(
            original_native.read_text(), host_isa='rv64gc', combine_clamp=True)
        numerical_legalization['legalizer_sha256'] = hashlib.sha256(Path(late_quant_rne.__file__).read_bytes()).hexdigest()
        numerical_legalization['combine_clamp'] = True
        (work / 'host-bounded-rne.json').write_text(json.dumps(numerical_legalization, indent=2))
        selected_native = work / 'bounded-rne.ll'
        selected_native.write_text(rewritten)
        # Remove the now-dead source DAG through ordinary LLVM DCE. No vector
        # flags, floating reassociation or source input assumptions are added.
        numerical_cleanup_command = [selected_install / 'bin' / 'opt', '-S',
            '--passes=function(instcombine,simplifycfg,loop-simplify,lcssa,loop-mssa(licm),instcombine,simplifycfg),globaldce'
            if host_optimization == 'bounded_rne_fused' else
            '--passes=function(instcombine,simplifycfg),globaldce', selected_native, '-o', inlined]
        run(numerical_cleanup_command)
    run([translate, '--import-llvm', inlined, '-o', target])
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import IntegerAttr
    from merlin.xdsl_dialects._common import text as llvm_text
    generic_import = work / 'generic-import.mlir'
    run([selected_install / 'bin' / 'mlir-opt', target, '--mlir-print-op-generic', '-o', generic_import])
    module = Parser(make_llvm_context(), generic_import.read_text()).parse_module()
    functions = [op for op in module.body.block.ops if op.name == 'llvm.func' and op.body.blocks]
    if {fn.sym_name.data for fn in functions} != {symbol, completion_symbol}:
        raise WholeProgramError('whole-program inlining left unselected executable helpers')
    # Native LLVM import spells a void inline asm as one !llvm.void result.
    # The public completion ABI requires the equivalent zero-result spelling.
    from xdsl.dialects import llvm
    completion_function = next(fn for fn in functions if fn.sym_name.data == completion_symbol)
    completion_block = completion_function.body.blocks[0]
    old_asm = next(op for op in completion_block.ops if op.name == 'llvm.inline_asm')
    completion_block.insert_op_before(llvm.InlineAsmOp(completion_asm, '~{memory}', [], has_side_effects=True), old_asm)
    completion_block.erase_op(old_asm)
    selected_function = next(fn for fn in functions if fn.sym_name.data == symbol)
    # A trap cannot return. Use a return as its unreachable continuation so
    # the public CFG index can represent this terminal error path; the trap
    # itself and every runtime boundary check remain executable and unchanged.
    for block in selected_function.body.blocks:
        terminal = block.last_op
        if terminal is not None and terminal.name == 'llvm.unreachable':
            if terminal.prev_op is None or (terminal.prev_op.name != 'llvm.intr.trap' and
                    getattr(terminal.prev_op.attributes.get('op_name__'), 'data', None) != 'llvm.intr.trap'):
                raise WholeProgramError('unsupported non-trap unreachable host continuation')
            block.insert_op_before(llvm.ReturnOp(), terminal)
            block.erase_op(terminal)
    task_operations = [op for op in selected_function.walk() if op is not selected_function]
    result.stats['adapter_lowered_abi'] = dict(entry_symbol=symbol, completion_symbol=completion_symbol, completion_asm=completion_asm)
    for operation in module.walk():
        # LLVM import carries optional optimization metadata that the public
        # xDSL LLVM verifier does not model. Dropping alias hints is conservative.
        for metadata in ('alias_scopes', 'noalias_scopes', 'tbaa', 'access_groups'):
            operation.properties.pop(metadata, None)
    for operation in task_operations:
        operation.attributes['merlin.global_task'] = IntegerAttr(0, 64)
    result.stats['adapter_schedule_instruction_count'] = len(task_operations)
    result.stats['adapter_combined_llvm_dialect'] = str(target.resolve())
    alias_prefix = '\n'.join(line for line in generic_import.read_text().splitlines() if line.startswith('#'))
    target.write_text(alias_prefix + '\n' + llvm_text(module, generic=True))
    # Parsing the actual resulting module catches unresolved MLIR dialect or
    # signature issues before handing the artifact to the caller.
    run([selected_install / 'bin' / 'mlir-opt', target, '-o', work / 'verified.mlir'])
    def identity(path):
        payload = path.read_bytes()
        return dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    host_recipe = dict(schema='submission.scalar_host_optimization.v1', selection=host_optimization,
        stage='linked LLVM after exact descriptor and device inlining',
        selected_cpu_flags=list(cflags), pipeline=native_pipeline,
        command=list(map(str, optimization_command)), tool=identity(selected_install / 'bin' / 'opt'),
        input=identity(inline_input), selected_native_llvm=identity(inlined),
        emitted_llvm_dialect=identity(target),
        source_arithmetic='unchanged LLVM semantics; ordinary alias and speculative-execution checks',
        cost_evidence='not_measured')
    if optimization_remarks.is_file():
        host_recipe['loop_optimization_remarks'] = identity(optimization_remarks)
        host_recipe['loop_optimization_remark_count'] = len(re.findall(r'^Pass:\s+licm$', optimization_remarks.read_text(), re.MULTILINE))
    if numerical_legalization is not None:
        host_recipe['source_numerical_preparation'] = result.stats['adapter_host_numerical_preparation']
        host_recipe['bounded_rne'] = numerical_legalization
        host_recipe['before_bounded_rne'] = identity(original_native)
        host_recipe['selected_bounded_rne'] = identity(selected_native)
        host_recipe['bounded_rne_cleanup_command'] = list(map(str, numerical_cleanup_command))
        host_recipe['source_arithmetic'] = 'Merlin checked bounded ties-even quantization; scalar RV64GC explicit RNE; original unsupported chains retained'
        if 'adapter_stock_host_pipeline' in result.stats:
            host_recipe['stock_host_lowering'] = result.stats['adapter_stock_host_pipeline']
    (work / 'host-optimization.json').write_text(json.dumps(host_recipe, indent=2))
    result.stats['adapter_host_optimization'] = host_recipe
    (work / 'materialization.json').write_text(json.dumps(dict(
        schema='submission.whole_program_materialization.v1', symbol=symbol,
        host_cpu_flags=list(cflags), sources=[str(p) for p in llvm_inputs],
        output=str(target), runtime_symbols='external; supplied by selected host runtime'), indent=2))
    return target


def whole_program_command_buffer(result: WholeProgramResult) -> dict[str, Any]:
    """Describe the exact public pointer boundary and complete source task.

    The command stream describes only accelerator work; host semantics reside in
    the submitted compiled module. Device buffers remain attached as evidence
    under device_regions rather than inventing commands for host operations.
    """
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    inventory = source_operation_inventory(Path(result.stats['adapter_original_source']).read_bytes())
    ninputs = len(inventory['arguments'])
    entry_bindings = [f'arg{i}' for i in range(ninputs)]
    output_bindings = [f'output{i}' for i in range(len(inventory['returns']))]
    tensors = {name: dict(**typ, role='input') for name, typ in zip(entry_bindings, inventory['arguments'])}
    source_values = []
    for name, returned in zip(output_bindings, inventory['returns']):
        origin = returned['source']
        if origin is None or 'op_index' not in origin:
            raise WholeProgramError('whole-program metadata currently requires output operation results')
        tensors[name] = dict(shape=returned['shape'], dtype=returned['dtype'], role='output')
        source_values.append(dict(op_index=origin['op_index'], result_index=origin['result_index'], tensor=name))
    count = result.stats.get('adapter_schedule_instruction_count')
    if not count:
        raise WholeProgramError('materialize combined LLVM before building the actual emitted task plan')
    plan = dict(schema='mixed_program_plan_v1', source_sha256=inventory['source_sha256'],
        source_op_count=inventory['source_op_count'], tasks=[dict(task_index=0, kind='compiled_complete_program',
            source_op_indices=list(range(inventory['source_op_count'])), instruction_start=0,
            instruction_end=count, reads=entry_bindings, writes=output_bindings)],
        schedule_instruction_count=count, prologue_instruction_range=[0, 0],
        epilogue_instruction_range=[count, count], entry_bindings=entry_bindings,
        source_values=source_values, output_bindings=output_bindings)
    return dict(abi_version='0.1', target='gemmini', backend='shared_host_target_callbacks',
        interface='linalg_positional', operand_naming='positional', tensors=tensors,
        kernel_abi=dict(kind='whole_program', args=[dict(tensor=name, access='read') for name in entry_bindings]
            + [dict(tensor=name, access='write') for name in output_bindings], outputs=output_bindings),
        commands=[], params=dict(global_program_plan=plan,
            host_optimization=result.stats.get('adapter_host_optimization'),
            device_regions=[dict(symbol=a.symbol, artifact=str(a.path), command_buffer=a.command_buffer)
                for a in result.device_artifacts]))
