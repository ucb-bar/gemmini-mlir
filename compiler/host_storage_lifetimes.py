"""Bounded, per-invocation storage through Merlin's stock promotion recipe.

Two hooks are required: route the host stock pipeline through run_stock_pipeline,
then wrap the COMPLETE kernel compilation with compile_with_storage_policy. The
latter owns the opt-in context and admits the actual target object, including
existing stack slots, alignment and register spills. An island-only compilation
cannot activate promotion. This creates no global workspace or shared lifetime.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import re
from pathlib import Path
import subprocess
import tempfile

from merlin.llvmlower import impr_features
from merlin.llvmlower.impr_features import _promote_buffers_to_stack
from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy
from host_completion_policy import run_stock_pipeline as _incumbent


@dataclass
class _Compilation:
    stack_alignment: int
    records: list = field(default_factory=list)


_active = ContextVar('host_storage_complete_compilation', default=None)
_CAP = 1024


def _sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def run_stock_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    """Incumbent-compatible hook; activation requires the complete-kernel driver."""
    context = _active.get()
    if context is None:
        return _incumbent(source, lowered, passes, tool=tool, invoke=invoke,
                          print_module=print_module, report=report)
    anchor = 'func.func(buffer-hoisting,buffer-loop-hoisting)'
    anchors = [i for i, name in enumerate(passes) if name == anchor]
    deallocs = [i for i, name in enumerate(passes) if 'ownership-based-buffer-deallocation' in name]
    loops = [i for i, name in enumerate(passes) if 'convert-linalg-to-loops' in name]
    if not (len(anchors) == len(deallocs) == len(loops) == 1
            and anchors[0] < deallocs[0] < loops[0]):
        raise ValueError('stack promotion requires hoist, promotion, ownership, loop order')
    original = Path(source).read_bytes()
    # This stock option changes only newly allocated private buffer alignment.
    # Explicit source alignments remain intact. Use the compiler's target ABI
    # stack alignment, rather than an arbitrary host default (normally 64).
    if context.stack_alignment <= 0:
        raise ValueError('whole-kernel context lacks target stack alignment')
    aligned = list(passes)
    bufferizers = [i for i, name in enumerate(aligned) if name.startswith('one-shot-bufferize{')]
    if len(bufferizers) != 1 or 'buffer-alignment=' in aligned[bufferizers[0]]:
        raise ValueError('private buffer alignment requires one unmodified stock bufferizer')
    i = bufferizers[0]
    aligned[i] = aligned[i][:-1] + ' buffer-alignment=' + str(context.stack_alignment) + '}'
    selected = _promote_buffers_to_stack(aligned, cap=_CAP)
    _incumbent(source, lowered, selected, tool=tool, invoke=invoke,
               print_module=print_module, report=report)
    if Path(source).read_bytes() != original:
        raise ValueError('stack promotion modified original source')
    hoisting = _hoist_static_allocas(Path(lowered), print_module)
    observation = {
        'source_sha256': _sha(original), 'lowered_sha256': _sha(Path(lowered).read_bytes()),
        'per_buffer_cap_bytes': _CAP, 'private_buffer_alignment': context.stack_alignment,
        'pass': selected[anchors[0] + 1],
        'library_sha256': _sha(Path(impr_features.__file__).read_bytes()),
        'position': 'after buffer hoisting; before ownership-based deallocation',
        'lifetime': 'per invocation; stock promotion proves allocation does not escape',
        'entry_alloca_adapter': hoisting,
    }
    context.records.append(observation)
    if report is not None:
        report['storage_lifetimes'] = observation



def _hoist_static_allocas(path, print_module):
    """Bridge stock promotion to LLVM's actual static-entry-allocation contract.

    No bytes are shared. Only constant primitive slots bounded by the promotion
    cap, whose blocks cannot execute twice in an invocation, move to entry.
    The allocation's existing users and all source memory operations stay put.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.llvmlower.stack_arena import element_bytes_of, StackArenaError
    module = parse_mlir_text(path.read_text(), ctx=make_llvm_context())
    moved, refused = [], []
    for function in module.body.block.ops:
        if function.name != 'llvm.func' or not function.body.blocks:
            continue
        blocks = tuple(function.body.blocks)
        entry = blocks[0]
        edges = {block: tuple(block.last_op.successors) for block in blocks}
        reachable = set()
        pending = [entry]
        while pending:
            block = pending.pop()
            if block in reachable:
                continue
            reachable.add(block)
            pending.extend(edges[block])
        def cyclic(block):
            seen, pending = set(), list(edges[block])
            while pending:
                successor = pending.pop()
                if successor is block:
                    return True
                if successor not in seen:
                    seen.add(successor)
                    pending.extend(edges[successor])
            return False
        barriers = any('stacksave' in op.name or 'stackrestore' in op.name
                       or 'stacksave' in str(op.properties) or 'stackrestore' in str(op.properties)
                       for op in function.walk())
        for op in tuple(function.walk()):
            if op.name != 'llvm.alloca' or op.parent is entry:
                continue
            block = op.parent
            if barriers or block not in reachable or cyclic(block) or len(op.operands) != 1:
                refused.append({'function': function.sym_name.data, 'reason': 'repeated, unreachable or scoped allocation'})
                continue
            count = op.operands[0].owner
            try:
                if count.name != 'llvm.mlir.constant':
                    raise ValueError('nonconstant allocation count')
                number = count.properties['value'].value.data
                typ = str(op.properties['elem_type'])
                aliases = {'f16': 'half', 'bf16': 'bfloat', 'f32': 'float', 'f64': 'double'}
                size = number * element_bytes_of(aliases.get(typ, typ))
                if number < 0 or size > _CAP:
                    raise ValueError('outside promoted allocation cap')
            except (ValueError, StackArenaError, AttributeError, KeyError):
                refused.append({'function': function.sym_name.data, 'reason': 'unproved primitive constant-size allocation'})
                continue
            bound = count.clone()
            block.detach_op(op)
            op.operands = (bound.results[0],)
            entry.insert_ops_before([bound, op], entry.first_op)
            moved.append({'function': function.sym_name.data, 'element_type': typ,
                          'count': number, 'bytes': size})
    module.verify()
    if moved:
        path.write_text(print_module(module))
    return {'moved': moved, 'refused': refused,
            'premises': 'reachable acyclic allocation block; constant bounded size; distinct slot per site'}

def _contract(path):
    import yaml
    document = yaml.safe_load(Path(path).read_text())
    def collect(node):
        if isinstance(node, dict):
            if isinstance(node.get('kernel_stack_frame'), dict):
                yield node
            for value in node.values():
                yield from collect(value)
        elif isinstance(node, list):
            for value in node:
                yield from collect(value)
    recipes = list(collect(document))
    if len(recipes) != 1:
        raise ValueError('storage admission requires one target-owned stack/toolchain recipe')
    recipe = recipes[0]
    policy = KernelStackFramePolicy(**recipe['kernel_stack_frame'])
    flags = [str(f) for f in recipe.get('cflags', ())
             if str(f).startswith(('-march=', '-mabi=', '-mcmodel=', '-O'))
             or f in ('-ffreestanding', '-fno-builtin')]
    if not any(f.startswith('-march=') for f in flags) or not any(f.startswith('-mabi=') for f in flags):
        raise ValueError('target must declare ISA and ABI for stack admission')
    compiler = Path(recipe['compiler']['path']).name
    triple = compiler.rpartition('-')[0]
    if not triple:
        raise ValueError('target compiler does not declare a cross target')
    return policy, flags, triple



def _stack_alignment(contract_path, toolchain):
    _, flags, triple = _contract(contract_path)
    query = subprocess.run([str(Path(toolchain)/'clang'), '--target='+triple, *flags,
                            '-x', 'c', '-S', '-emit-llvm', '/dev/null', '-o', '-'],
                           capture_output=True, text=True, timeout=30)
    if query.returncode:
        raise ValueError('cannot read target compiler stack alignment: '+query.stderr[-2000:])
    layouts = re.findall(r'^target datalayout = "([^"\n]+)"', query.stdout, re.M)
    if len(layouts) != 1:
        raise ValueError('target compiler did not provide one data layout')
    alignments = [part[1:] for part in layouts[0].split('-') if part.startswith('S')]
    if len(alignments) != 1 or not alignments[0].isdigit():
        raise ValueError('target compiler data layout lacks natural stack alignment')
    bits = int(alignments[0])
    if bits < 8 or bits % 8 or bits & (bits-1):
        raise ValueError('unsupported target stack alignment')
    return bits//8

def _invoke(arguments, cwd):
    result = subprocess.run(list(map(str, arguments)), cwd=cwd, capture_output=True,
                            text=True, timeout=240)
    if result.returncode:
        raise ValueError('storage candidate object inspection failed: ' + result.stderr[-4000:])


def _entry_type(text, entry):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    module = parse_mlir_text(text, ctx=make_llvm_context())
    functions = [op for op in module.body.block.ops if op.name == 'llvm.func'
                 and op.sym_name.data == entry and op.body.blocks]
    if len(functions) != 1:
        raise ValueError('storage admission requires exactly one defined entry')
    # Refuse recursion: sum of static defined frames bounds acyclic call stacks.
    definitions = {op.sym_name.data: op for op in module.body.block.ops
                   if op.name == 'llvm.func' and op.body.blocks}
    edges = {name: set() for name in definitions}
    for name, function in definitions.items():
        for operation in function.walk():
            if operation.name == 'llvm.call':
                callee = operation.properties.get('callee')
                if callee is None:
                    raise ValueError('storage admission refuses indirect kernel calls')
                target = callee.root_reference.data
                if target in definitions:
                    edges[name].add(target)
    visiting, complete = set(), set()
    def visit(name):
        if name in visiting:
            raise ValueError('storage admission refuses recursive kernel calls')
        if name in complete:
            return
        visiting.add(name)
        for target in edges[name]:
            visit(target)
        visiting.remove(name)
        complete.add(name)
    for name in definitions:
        visit(name)
    # The stock recipe changes only storage. Independently require identical logical ABI.
    return str(functions[0].function_type)



def _alloca_observation(text):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.llvmlower.stack_arena import element_bytes_of, StackArenaError
    module = parse_mlir_text(text, ctx=make_llvm_context())
    records, unsized = [], []
    for op in module.walk():
        if op.name != 'llvm.alloca':
            continue
        typ = str(op.properties['elem_type'])
        aliases = {'f16': 'half', 'bf16': 'bfloat', 'f32': 'float', 'f64': 'double'}
        try:
            count = op.operands[0].owner.properties.get('value')
            width = element_bytes_of(aliases.get(typ, typ))
            number = count.value.data
            if number < 0:
                raise ValueError('negative alloca size')
            alignment = op.properties.get('alignment')
            records.append({'element_type': typ, 'count': number, 'bytes': number * width,
                            'alignment': alignment.value.data if alignment else None})
        except (StackArenaError, AttributeError):
            unsized.append(typ)
    return {'primitive_bytes': sum(row['bytes'] for row in records),
            'primitive_allocations': records, 'other_alloca_types': unsized,
            'note': 'actual machine stack report includes ALL types, spills and alignment'}

def inspect_stack(text, *, contract_path, toolchain, work):
    """Build real target object and conservatively sum every defined static frame.

    Sum rather than max: no reuse between inlined tasks or caller/helper frames
    is assumed. Unknown or dynamic rows refuse admission. External runtime frames
    are unchanged by this recipe and are outside the declared kernel-frame ABI.
    """
    policy, flags, triple = _contract(contract_path)
    work, tools = Path(work), Path(toolchain)
    work.mkdir(parents=True, exist_ok=True)
    source, ll, obj = work/'kernel.mlir', work/'kernel.ll', work/'kernel.o'
    source.write_text(text)
    _invoke([tools/'mlir-translate', '--mlir-to-llvmir', source, '-o', ll], work)
    _invoke([tools/'clang', '--target='+triple, *flags, '-fstack-usage', '-c', ll, '-o', obj], work)
    rows, refusals = [], []
    for line in (work/'kernel.su').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) != 3 or not fields[1].isdigit():
            raise ValueError('malformed actual stack usage row')
        identity, size, allocation = fields
        rows.append({'function': identity.rpartition(':')[2], 'bytes': int(size), 'allocation': allocation})
        if allocation != 'static':
            refusals.append('non-static frame: '+identity)
    if not rows:
        refusals.append('empty stack usage')
    entries = [r for r in rows if r['function'] == policy.entry_symbol]
    if len(entries) != 1:
        refusals.append('entry frame must occur exactly once')
    total = sum(row['bytes'] for row in rows)
    if total > policy.max_static_bytes:
        refusals.append('sum of all defined frames exceeds target kernel budget')
    return {'policy': policy.record(), 'rows': rows, 'sum_static_frame_bytes': total,
            'entry_static_frame_bytes': entries[0]['bytes'] if len(entries) == 1 else None,
            'llvm_allocations': _alloca_observation(text),
            'llvm_sha256': _sha(ll.read_bytes()), 'object_sha256': _sha(obj.read_bytes()),
            'refusals': refusals}


def compile_with_storage_policy(compile_kernel, *, get_llvm=lambda product: product,
                                mode='off', contract_path=None, toolchain=None, report=None):
    """Compile actual whole kernel; accept small_stack only after target preflight.

    ``compile_kernel`` must return a fresh independent complete product on each
    invocation; ``get_llvm`` extracts its final inlined LLVM-dialect module. The
    caller receives that same product, including its original source plan. No
    source tensor, final writer, arithmetic, or global allocation is rewritten.
    """
    if mode not in ('off', 'small_stack'):
        raise ValueError('unknown host storage mode')
    if _active.get() is not None:
        raise ValueError('nested complete-kernel storage compilation is unsupported')
    incumbent = compile_kernel()
    if mode == 'off':
        return incumbent
    if contract_path is None or toolchain is None:
        raise ValueError('small_stack requires actual target contract and LLVM toolchain')
    baseline = get_llvm(incumbent)
    policy, _, _ = _contract(contract_path)
    context = _Compilation(stack_alignment=_stack_alignment(contract_path, toolchain))
    token = _active.set(context)
    try:
        candidate = compile_kernel()
    finally:
        _active.reset(token)
    text = get_llvm(candidate)
    if text == baseline:
        if report is not None:
            report.update({'schema': 'merlin.host_stock_small_stack.v1', 'mode': mode,
                           'accepted': False, 'reason': 'actual LLVM unchanged',
                           'source_lowerings': context.records,
                           'baseline_mlir_sha256': _sha(baseline),
                           'candidate_mlir_sha256': _sha(text),
                           'runtime_cycles': 'not_measured'})
        return incumbent
    if _entry_type(baseline, policy.entry_symbol) != _entry_type(text, policy.entry_symbol):
        raise ValueError('storage recipe changed whole-kernel ABI')
    with tempfile.TemporaryDirectory(prefix='gemmini-storage-frame-') as directory:
        baseline_stack = inspect_stack(baseline, contract_path=contract_path,
                                       toolchain=toolchain, work=Path(directory)/'baseline')
        candidate_stack = inspect_stack(text, contract_path=contract_path,
                                        toolchain=toolchain, work=Path(directory)/'candidate')
    accepted = not candidate_stack['refusals']
    if report is not None:
        report.update({'schema': 'merlin.host_stock_small_stack.v1', 'mode': mode,
                       'accepted': accepted, 'source_lowerings': context.records,
                       'baseline_mlir_sha256': _sha(baseline), 'candidate_mlir_sha256': _sha(text),
                       'baseline_stack': baseline_stack, 'candidate_stack': candidate_stack,
                       'runtime_cycles': 'not_measured',
                       'numerical_validation': 'caller_required'})
    return candidate if accepted else incumbent


def attribute_storage_lifetimes(lowered_mlir):
    """Preserve actual stock-inliner lifetime ownership for private stack slots.

    Upstream inserts start/end intrinsics after our callee instructions have
    been tagged. Their sole pointer identifies the existing allocation's actual
    task. No instruction is removed or given new memory/timing semantics.
    """
    if 'llvm.intr.lifetime.' not in lowered_mlir:
        return lowered_mlir
    from io import StringIO
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.common.dense_float_ir import DenseFloatPrinter
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.llvmlower.stack_arena import element_bytes_of, StackArenaError
    module = parse_mlir_text(lowered_mlir, ctx=make_llvm_context())
    changes, groups = [], {}
    for op in module.walk():
        name = op.attributes.get('op_name__')
        name = name.data if name is not None else op.name
        if name not in ('llvm.intr.lifetime.start', 'llvm.intr.lifetime.end'):
            continue
        if len(op.operands) != 1:
            raise ValueError('unexpected stock lifetime signature')
        allocation = op.operands[0].owner
        if allocation.name != 'llvm.alloca' or len(allocation.operands) != 1:
            continue
        tag = allocation.attributes.get('merlin.global_task')
        if tag is None or tag.value.data < 0:
            continue
        try:
            count = allocation.operands[0].owner.properties['value'].value.data
            typ = str(allocation.properties['elem_type'])
            aliases = {'f16': 'half', 'bf16': 'bfloat', 'f32': 'float', 'f64': 'double'}
            size = count * element_bytes_of(aliases.get(typ, typ))
            if count < 0 or size > _CAP:
                continue
        except (AttributeError, KeyError, StackArenaError):
            continue
        previous = op.attributes.get('merlin.global_task')
        if previous is not None:
            if previous != tag:
                raise ValueError('stock lifetime contradicts actual allocation owner')
            continue
        groups.setdefault(allocation, []).append(name)
        changes.append((op, tag))
    for names in groups.values():
        if sorted(names) != ['llvm.intr.lifetime.end', 'llvm.intr.lifetime.start']:
            raise ValueError('private stock allocation requires one actual start/end pair')
    if not changes:
        return lowered_mlir
    for op, tag in changes:
        op.attributes['merlin.global_task'] = tag
    module.verify()
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()
