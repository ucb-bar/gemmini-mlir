"""Stock MLIR host-region lowering composed into the common device kernel.

No target instruction emission, placement inference, reference evaluation or C
compute implementation lives here. The frontend must authorize each HOST_REGION.
"""
from __future__ import annotations

from functools import lru_cache
from io import StringIO
from pathlib import Path
import subprocess

MLIR_OPT = Path('/work/merlin/toolchains/llvm-a47bddcc/install/bin/mlir-opt')


class HostLoweringError(ValueError):
    """The selected host compiler cannot yet lower this exact source region."""


def _print(op):
    from xdsl.printer import Printer
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(op)
    return stream.getvalue()


def _buffer_wrapper(text):
    """Keep source operations unchanged; supply explicit nonaliasing destinations."""
    from xdsl.dialects import builtin, bufferization, func
    from xdsl.ir import Block, Region, OpResult
    from xdsl.parser import Parser
    from merlin.targetgen.contract.linalg_iface import make_linalg_context

    module = Parser(make_linalg_context(), text).parse_module()
    from .host_support_quant import expand_quantization
    expand_quantization(module)
    # Splat is exactly a fill. Expose that source semantics before upstream
    # bufferization, avoiding unnecessary read-only global tensor objects.
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.dialects.linalg.ops import FillOp
    for op in list(module.walk()):
        if op.name == 'tensor.splat':
            empty = EmptyOp([], op.results[0].type)
            fill = FillOp([op.operands[0]], [empty.results[0]], [op.results[0].type])
            fill.attributes.update(op.attributes)
            op.parent.insert_op_before(empty, op)
            op.parent.insert_op_before(fill, op)
            op.results[0].replace_all_uses_with(fill.results[0])
            op.parent.erase_op(op)
    functions = [o for o in module.ops if isinstance(o, func.FuncOp)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise HostLoweringError('host region requires one closed single-block tensor function')
    original = functions[0]
    block = original.body.blocks[0]
    ret = block.last_op
    if not isinstance(ret, func.ReturnOp):
        raise HostLoweringError('host region requires explicit tensor returns')
    inputs, outputs = list(block.args), list(ret.arguments)
    types = [v.type for v in inputs + outputs]
    if any(not isinstance(t, builtin.TensorType) or any(d < 0 for d in t.get_shape()) for t in types):
        raise HostLoweringError('host pointer ABI currently requires ranked static tensors')
    memrefs = [builtin.MemRefType(t.element_type, t.get_shape()) for t in types]
    target = Block(arg_types=memrefs)
    value_map = {}
    for old, arg in zip(inputs, target.args):
        tensor = bufferization.ToTensorOp(arg, restrict=True)
        target.add_op(tensor)
        value_map[old] = tensor.tensor
    clones = {}
    for op in block.ops:
        if op is ret:
            break
        new = op.clone(value_mapper=value_map)
        target.add_op(new)
        clones[op] = new
    # Redirect only initialization chains ending in tensor.empty. Reads of real
    # source tensors are never discarded. In particular a linalg.fill on the
    # path remains and initializes its exact original scalar value.
    redirected = set()
    for i, result in enumerate(outputs):
        value = result
        seen = set()
        while isinstance(value, OpResult) and value.owner not in seen:
            owner = value.owner
            seen.add(owner)
            if owner.name == 'tensor.empty':
                cloned = clones[owner]
                if len(list(value.uses)) != 1:
                    break
                tensor = bufferization.ToTensorOp(target.args[len(inputs)+i], restrict=True, writable=True)
                target.insert_op_before(tensor, cloned)
                cloned.results[0].replace_all_uses_with(tensor.tensor)
                redirected.add(i)
                break
            destinations = getattr(owner, 'outputs', ())
            if len(destinations) != len(owner.results):
                break
            value = destinations[value.index]
    # A materialization also handles movement/identity regions and an output
    # whose initialization chain could not safely be redirected.
    for i, result in enumerate(outputs):
        dest = target.args[len(inputs)+i]
        materialize = bufferization.MaterializeInDestinationOp(
            operands=[value_map[result], dest], result_types=[[]], properties={'writable': builtin.UnitAttr()})
        target.add_op(materialize)
    target.add_op(func.ReturnOp())
    wrapper = func.FuncOp('host_region', (memrefs, []), Region(target))
    return _print(builtin.ModuleOp([wrapper]))


@lru_cache(maxsize=512)
def _compile_canonical_host_region(text):
    """Return a verified, closed LLVM func with pointer-only input/output ABI.

    All static compiler temporaries use explicit caller-provided pointers.
    This region is composed into a caller CFG, where a local stack allocation
    could otherwise become a nonentry dynamic allocation. External calls fail.
    """
    source = _buffer_wrapper(text)
    pipeline = [
        '--convert-tensor-to-linalg', '--eliminate-empty-tensors', '--empty-tensor-to-alloc-tensor',
        '--one-shot-bufferize=bufferize-function-boundaries',
        '--promote-buffers-to-stack=max-alloc-size-in-bytes=4096',
        '--convert-linalg-to-loops', '--lower-affine', '--convert-scf-to-cf',
        '--convert-math-to-llvm', '--convert-arith-to-llvm',
        '--expand-strided-metadata', '--memref-expand', '--finalize-memref-to-llvm',
        '--convert-func-to-llvm=use-bare-ptr-memref-call-conv',
        '--convert-cf-to-llvm', '--reconcile-unrealized-casts', '--canonicalize',
    ]
    # Stock memref-to-LLVM leaves noncontiguous copies as runtime memrefCopy
    # calls. Express the identical copy with the stock linalg copy operation,
    # whose existing loop lowering keeps this region closed and self-contained.
    first = pipeline.index('--convert-linalg-to-loops')
    buffered = subprocess.run([str(MLIR_OPT), *pipeline[:first], '--mlir-print-op-generic'],
                              input=source, text=True, capture_output=True, timeout=120)
    if buffered.returncode:
        raise HostLoweringError('stock MLIR host bufferization failed: ' + buffered.stderr[-6000:])
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from xdsl.dialects.memref import MemRef, CopyOp as MemrefCopyOp, AllocOp, AllocaOp
    from xdsl.dialects.linalg.ops import CopyOp as LinalgCopyOp
    from xdsl.parser import Parser
    context = make_linalg_context()
    context.load_dialect(MemRef)
    intermediate = Parser(context, buffered.stdout).parse_module()
    from xdsl.dialects.builtin import FunctionType, ArrayAttr, IntegerAttr, i64
    import math
    host_function = next(o for o in intermediate.ops if o.name == 'func.func')
    entry = host_function.body.block
    scratch_bytes = []
    for allocation in list(intermediate.walk()):
        if not isinstance(allocation, (AllocOp, AllocaOp)):
            continue
        typ = allocation.results[0].type
        shape = typ.get_shape()
        element = str(typ.element_type)
        widths = {'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8, 'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8}
        if any(d < 0 for d in shape) or element not in widths:
            raise HostLoweringError('host scratch requires static known scalar storage')
        alignment = allocation.properties.get('alignment') or allocation.attributes.get('alignment')
        if alignment is not None and alignment.value.data > 64:
            raise HostLoweringError('host scratch alignment exceeds the shared allocator guarantee of 64 bytes')
        scratch_bytes.append(math.prod(shape) * widths[element])
        pointer = entry.insert_arg(typ, len(entry.args))
        allocation.results[0].replace_all_uses_with(pointer)
        allocation.parent.erase_op(allocation)
    host_function.properties['function_type'] = FunctionType.from_lists(
        [a.type for a in entry.args], [])
    for op in list(intermediate.walk()):
        if isinstance(op, MemrefCopyOp):
            copy = LinalgCopyOp([op.operands[0]], [op.operands[1]], res=[])
            copy.attributes.update(op.attributes)
            op.parent.insert_op_before(copy, op)
            op.parent.erase_op(op)
    run = subprocess.run([str(MLIR_OPT), *pipeline[first:], '--mlir-print-op-generic'],
                         input=_print(intermediate), text=True, capture_output=True, timeout=120)
    if run.returncode:
        raise HostLoweringError('stock MLIR host lowering failed: ' + run.stderr[-6000:])
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from xdsl.parser import Parser
    from xdsl.dialects import llvm
    try:
        module = Parser(make_llvm_context(), run.stdout).parse_module()
        function = next(o for o in module.ops if isinstance(o, llvm.FuncOp)
                        and o.sym_name.data == 'host_region')
        function.verify()
    except Exception as exc:
        raise HostLoweringError('generated host LLVM could not be represented exactly: ' + str(exc)) from exc
    for op in list(function.walk()):
        # The stock bufferizer may retain a copy from a destination to itself.
        # Remove only this exact identity operation; all other copies survive.
        if op.name == 'builtin.unregistered' and getattr(op, 'op_name', None) is not None and op.op_name.data == 'llvm.intr.memcpy' and op.operands[0] is op.operands[1]:
            op.parent.erase_op(op)
            continue
        if isinstance(op, (llvm.CallOp, llvm.AddressOfOp)):
            raise HostLoweringError('host region requires unresolved external call/global: ' + _print(op))
        actual_name = op.op_name.data if op.name == 'builtin.unregistered' else op.name
        if not actual_name.startswith('llvm.'):
            raise HostLoweringError('host region retains an unlowered operation: ' + _print(op))
    function.attributes['prov.host_scratch_bytes'] = ArrayAttr([IntegerAttr(n, i64) for n in scratch_bytes])
    return function


@lru_cache(maxsize=1024)
def _canonical_host_source(text):
    """Factor only provenance out of native compilation, with exact replay slots.

    Numeric policies, arithmetic flags, types, layouts, constants, effects and
    non-provenance attributes remain part of the cache key. Slots are assigned
    by structural operation order and attribute key, independent of source IDs.
    """
    from merlin.targetgen.contract.linalg_iface import make_linalg_context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import StringAttr
    module = Parser(make_linalg_context(), text).parse_module()
    metadata = {}
    for op in module.walk():
        for key in sorted(op.attributes):
            if key.startswith('prov.'):
                slot = '__merlin_host_provenance_slot_' + str(len(metadata))
                metadata[slot] = op.attributes[key]
                op.attributes[key] = StringAttr(slot)
        # Name hints are printing conveniences, not SSA identity. Canonical
        # numbering makes alpha-renamed source functions share the same key.
        for result in op.results:
            result.name_hint = None
        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    arg.name_hint = None
    return _print(module), metadata


def compile_host_region(text):
    """Compile once per typed semantic source, restoring exact caller metadata.

    Always return a fresh function: tagging or composing one caller's task must
    never mutate the cached template or a different caller's source evidence.
    """
    from xdsl.dialects.builtin import StringAttr
    canonical, metadata = _canonical_host_source(text)
    function = _compile_canonical_host_region(canonical).clone()
    for op in function.walk():
        for key, value in list(op.attributes.items()):
            if key.startswith('prov.') and isinstance(value, StringAttr) and value.data in metadata:
                op.attributes[key] = metadata[value.data]
    return function


def host_compile_cache_info():
    """Construction diagnostics; no execution/timing authority is implied."""
    return _compile_canonical_host_region.cache_info()


def lower_host_task(builder, task, pointers, allocate=None):
    """Append a frontend HOST_REGION into FnBuilder, preserving task/source tags.

    task: {attributes: {host_module_ir, source_op_index, task_index, ...},
           operands: {inputs: [tensor names], outputs: [tensor names]}}.
    pointers maps those names to live LLVM pointer SSA values. Device ordering
    and lifetime are obligations of the common caller, as for device callbacks.
    """
    from merlin.llvmlower.host_builder import append_void_region
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
    attrs = task.get('attributes', task.get('params', {}))
    if attrs.get('eligible_device') or attrs.get('mandatory_device'):
        raise HostLoweringError('mandatory device region cannot be lowered on the host')
    source = attrs.get('host_module_ir')
    if not source:
        raise HostLoweringError('HOST_REGION has no closed source module')
    operands = task.get('operands', {})
    inputs = attrs.get('input_tensors', operands.get('inputs', []))
    outputs = attrs.get('output_tensors', operands.get('outputs', []))
    function = compile_host_region(source)
    for op in function.walk():
        for key in ('source_op_index', 'task_index'):
            if key in attrs:
                op.attributes['prov.' + key] = IntegerAttr(int(attrs[key]), i64)
        if op is not function:
            if 'task_index' in attrs:
                op.attributes['merlin.global_task'] = IntegerAttr(int(attrs['task_index']), i64)
            if 'source_op_index' in attrs:
                op.attributes['merlin.source_op_index'] = IntegerAttr(int(attrs['source_op_index']), i64)
        op.attributes['prov.placement'] = StringAttr('host')
        if attrs.get('host_reason'):
            op.attributes['prov.host_reason'] = StringAttr(str(attrs['host_reason']))
    arguments = [pointers[n] for n in inputs + outputs]
    scratch = function.attributes.get('prov.host_scratch_bytes')
    scratch_report = []
    if scratch:
        if allocate is None:
            raise HostLoweringError('host task needs shared scratch allocator for static temporary buffers')
        from merlin.compile.tuning_knobs import knob
        reuse = knob('storage_reuse', default=False, values=(False, True),
                     root=Path(__file__).resolve().parents[1])
        # These are private buffers introduced by stock bufferization, never
        # tensor operands/results. The closed void region has no external
        # calls or globals, and its scalar memrefs cannot contain pointers.
        # append_void_region completes all CPU work synchronously. The common
        # caller also owns device completion at its host/device boundaries.
        # Keep different allocations within one region distinct, including
        # equal-sized allocations; only sequential regions share a slot.
        pool = getattr(builder, '_host_private_scratch_pool', None)
        if pool is None:
            pool = {}
            builder._host_private_scratch_pool = pool
        occurrences = {}
        for i, n in enumerate(scratch):
            size = n.value.data
            occurrence = occurrences.get(size, 0)
            occurrences[size] = occurrence + 1
            key = (size, occurrence)
            reused = bool(reuse and key in pool)
            if reused:
                name, pointer = pool[key]
            else:
                name = ('host_scratch_reuse_' + str(size) + '_' + str(occurrence)
                        if reuse else 'host_scratch_' + str(attrs.get('task_index', 'task')) + '_' + str(i))
                # Source ABI names are unrestricted. Hidden storage must never
                # accidentally alias a source-named input/output or constant.
                while name in pointers:
                    name += '_private'
                pointer = allocate(name, [size], 'i8')
                if reuse:
                    pool[key] = (name, pointer)
            arguments.append(pointer)
            scratch_report.append({'slot': name, 'bytes': size, 'reused_across_regions': reused})
    report = append_void_region(builder, function, arguments)
    report.update(compiler='stock_mlir', source_op_index=attrs.get('source_op_index'),
                  task_index=attrs.get('task_index'), placement='host',
                  private_scratch=scratch_report)
    return report
