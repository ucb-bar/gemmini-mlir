"""Original-SSA host seams composed with independently lowered device regions.

Tensor/scalar evaluation is owned by Merlin's host code generator. This adapter
only chooses boundaries, transports pointers, and retains source provenance.
"""
from __future__ import annotations

import copy
import hashlib
from io import StringIO
from pathlib import Path

from xdsl.dialects import func, llvm
from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerAttr, i64
from xdsl.ir import Block, Region
from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_builder import FnBuilder
from merlin.llvmlower.host_scalar_builder import FpBuilder
from merlin.llvmlower.host_tensor_codegen import HostLinalg, LoweringDeclined, estimate_cost, HOST_LINALG_ELEMENT_BUDGET


def _text(op):
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(op)
    return stream.getvalue()


def _owner(module):
    owners = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == 'forward']
    if len(owners) != 1 or len(owners[0].body.blocks) != 1:
        raise ValueError('host source requires one closed forward block')
    return owners[0]


def _attr(op, name):
    return getattr(op.attributes.get(name), 'data', None)


def _host_op(op):
    if _attr(op, 'prov.placement') == 'host_required' or _attr(op, 'prov.family') == 'host_scalar':
        return True
    if op.name.startswith('math.'):
        return True
    return any(isinstance(v.type, TensorType) and str(v.type.element_type) in ('f32', 'bf16', 'f16', 'f64')
               for v in (*op.operands, *op.results))


def needs_host(source):
    """Inspect typed source, without treating a device lowering refusal as host eligibility."""
    return any(_host_op(op) for op in _owner(parse_mlir_text(source)).body.block.ops
               if op.name != 'func.return')


def _scaffold(op):
    return op.name in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill')


def _spec(value, role):
    ty = value.type
    if not isinstance(ty, TensorType) or any(d < 1 for d in ty.get_shape()):
        raise ValueError('host/device boundary requires static nonempty tensors')
    return {'shape': list(ty.get_shape()), 'dtype': str(ty.element_type), 'role': role}


def _source_layout_reuse_requested():
    """Consume explicit composition requests as well as the tuning selection."""
    import json
    from merlin.compile.tuning_knobs import knob
    root = Path(__file__).resolve().parent
    selected = knob('source_layout_reuse', default='off', values=('off', 'cse'), root=root) == 'cse'
    path = root / 'source_composition.json'
    if not path.exists() and not path.is_symlink():
        return selected
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError('source_composition.json must be a regular file of at most 8 KB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate source composition option: ' + key)
            result[key] = value
        return result
    request = json.loads(path.read_text(), object_pairs_hook=unique)
    allowed = {'shared_input_siblings', 'prepack_weights', 'share_exact_tensor_preparations',
               'dense_input_views', 'propagate_layouts', 'reuse_intermediate_storage',
               'independent_heads', 'whole_head_batches', 'op_profile', 'host_elementwise_fusion'}
    if (not isinstance(request, dict) or request.get('schema') != 'merlin.source-composition-preferences.v1'
            or not {'shared_input_siblings', 'prepack_weights'}.issubset(request)
            or set(request) - allowed - {'schema'}
            or any(type(value) is not bool for key, value in request.items() if key != 'schema')):
        raise ValueError('invalid source composition preferences')
    return selected or request.get('share_exact_tensor_preparations', False)


class _ExactHostLinalg(HostLinalg):
    """Complete scalar compare/select transport with direct typed LLVM ops."""

    def _scalar_op(self, op, ins):
        if op.name == 'arith.cmpf':
            predicate = op.properties['predicate']
            return self.fp.fb.add(llvm.FCmpOp(ins[0], ins[1], predicate)).results[0]
        if op.name == 'arith.cmpi':
            return self.fp.fb.add(llvm.ICmpOp(ins[0], ins[1], op.properties['predicate'])).results[0]
        if op.name == 'arith.select':
            return self.fp.fb.add(llvm.SelectOp(ins[0], ins[1], ins[2])).results[0]
        return super()._scalar_op(op, ins)


def _host_function(module, symbol, task_index=0, preparation_report=None, optimization_report=None):
    from source_preparation import prepare_source_layouts
    module = prepare_source_layouts(module).module
    if _source_layout_reuse_requested():
        from merlin.llvmlower.tensor_provenance_cse import deduplicate_tensor_preparation
        report = deduplicate_tensor_preparation(module)
        if preparation_report is not None:
            preparation_report.update(report)
    from source_quantization import needs_quantization_preparation, prepare_host_quantization
    if needs_quantization_preparation(module):
        from host_stock import lower_stock_host
        return lower_stock_host(prepare_host_quantization(module), symbol, task_index=task_index,
                                optimization_report=optimization_report)
    owner = _owner(module)
    ops = tuple(owner.body.block.ops)
    # Preserve the exact math primitive through the stock scalar pipeline.
    hit = next((op.name for op in owner.walk() if op.name.startswith('math.')), None)
    if hit or estimate_cost(ops) > 2048:
        from host_stock import lower_stock_host
        return lower_stock_host(module, symbol, task_index=task_index, optimization_report=optimization_report)
    count = len(owner.function_type.inputs)
    builder = FnBuilder([llvm.LLVMPointerType()] * (count + len(owner.function_type.outputs)))
    _ExactHostLinalg(FpBuilder(builder), list(builder.entry.args[:count]), list(builder.entry.args[count:]), row_alignment=1).run(owner)
    function = llvm.FuncOp(symbol, llvm.LLVMFunctionType([llvm.LLVMPointerType()] * len(builder.entry.args), llvm.LLVMVoidType()),
                           linkage=llvm.LinkageAttr('external'), body=builder.finish())
    for operation in function.walk():
        if operation is not function and operation.name != 'llvm.return':
            operation.attributes['merlin.global_task'] = IntegerAttr(task_index, i64)
    emitted = ModuleOp([function])
    emitted.verify()
    return _text(emitted)


def _outline(group, available):
    """Clone original operations and their scalar/storage scaffolding only."""
    selected = set(group)
    external = []
    def require(value):
        if value in available:
            if value not in external:
                external.append(value)
            return
        owner = value.owner
        if owner in selected:
            return
        if not hasattr(owner, 'name') or not _scaffold(owner):
            raise ValueError('unavailable source value at host/device boundary')
        selected.add(owner)
        for operand in owner.operands:
            require(operand)
    for op in group:
        for nested in op.walk():
            for value in nested.operands:
                if value.owner in selected:
                    continue
                # Region block arguments and values produced inside a selected
                # operation are internal to its clone.
                block = value.owner if isinstance(value.owner, Block) else value.owner.parent_block()
                if block is not None and block.parent_op() in selected:
                    continue
                require(value)
    original_block = group[0].parent_block()
    ordered = [op for op in original_block.ops if op in selected]
    outputs = [value for op in group for value in op.results
               if isinstance(value.type, TensorType) and any(use.operation not in selected and use.operation.parent_op() not in selected
                                                            for use in value.uses)]
    block = Block(arg_types=[value.type for value in external])
    mapper = dict(zip(external, block.args))
    for op in ordered:
        block.add_op(op.clone(value_mapper=mapper))
    block.add_op(func.ReturnOp(*(mapper[value] for value in outputs)))
    owner = func.FuncOp('forward', ([v.type for v in external], [v.type for v in outputs]), Region(block))
    module = ModuleOp([owner])
    module.verify()
    return module, external, outputs, ordered


def _module_body(text):
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('expected emitted LLVM module')
    return text[start + 1:end]


def _strip_completion(product):
    """Nested emitters own device work; the complete entry owns its fence."""
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    module = parse_mlir_text(product, ctx=make_llvm_context())
    for operation in tuple(module.body.block.ops):
        if isinstance(operation, llvm.FuncOp) and operation.sym_name.data == 'kernel_complete':
            module.body.block.erase_op(operation)
    return _text(module)


def _stock_tool(name):
    import os
    import shutil
    configured = os.environ.get('MERLIN_MLIR_INSTALL')
    candidates = ([Path(configured) / 'bin' / name] if configured else [])
    candidates += [Path(__file__).resolve().parent.parent / 'install' / 'bin' / name,
                   Path('/work/merlin/toolchains/llvm-a47bddcc/install/bin') / name]
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError('selected stock tool unavailable: ' + name)


def inline_host_product(product, *, symbol='kernel_entry'):
    """Run the stock LLVM inliner, then erase only unreferenced outlined owners."""
    import subprocess
    import tempfile
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    # Each independently lowered island may declare the same libm primitive.
    # Resolve these declarations before stock symbol verification/inlining.
    combined = parse_mlir_text(product, ctx=make_llvm_context())
    symbols = {}
    for operation in tuple(combined.body.block.ops):
        if not isinstance(operation, llvm.FuncOp):
            continue
        name = operation.sym_name.data
        previous = symbols.get(name)
        if previous is None:
            symbols[name] = operation
            continue
        if previous.function_type != operation.function_type:
            raise ValueError('inconsistent merged LLVM function type: ' + name)
        if previous.body.blocks and operation.body.blocks:
            raise ValueError('duplicate merged LLVM function definition: ' + name)
        discarded = operation if previous.body.blocks or not operation.body.blocks else previous
        combined.body.block.erase_op(discarded)
        if discarded is previous:
            symbols[name] = operation
    product = _text(combined)
    with tempfile.TemporaryDirectory(prefix='gemmini-source-inline-') as work:
        src, out = Path(work) / 'source.mlir', Path(work) / 'inlined.mlir'
        src.write_text(product)
        process = subprocess.run([str(_stock_tool('mlir-opt')), str(src), '--inline', '--mlir-print-op-generic', '--mlir-print-local-scope', '-o', str(out)],
                                 capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise ValueError('stock LLVM source inlining failed: ' + process.stderr[-6000:])
        lowered = parse_mlir_text(out.read_text(), ctx=make_llvm_context())
    for operation in tuple(lowered.body.block.ops):
        if isinstance(operation, llvm.FuncOp) and operation.body.blocks and operation.sym_name.data not in (symbol, 'kernel_complete'):
            lowered.body.block.erase_op(operation)
    for operation in lowered.walk():
        if operation.name == 'llvm.call':
            callee = str(operation.properties.get('callee', ''))
            if 'source_host_' in callee or 'source_device_' in callee:
                raise ValueError('stock inliner left an outlined source call')
    # The stock inliner creates real dispatch/continuation edges. Bind each
    # such edge to the adjacent executed source scope, rather than inventing
    # marker operations or leaving the final CFG unattributed.
    for function in lowered.body.block.ops:
        if not isinstance(function, llvm.FuncOp) or not function.body.blocks or function.sym_name.data != symbol:
            continue
        for nested in function.walk():
            if nested.name == 'llvm.return' and 'merlin.global_task' not in nested.attributes:
                nested.attributes['merlin.global_task'] = IntegerAttr(-2, i64)
        def next_owner(block, seen):
            if block in seen:
                return None
            seen.add(block)
            for operation in block.ops:
                if operation.name != 'llvm.mlir.constant' and 'merlin.global_task' in operation.attributes:
                    return operation.attributes['merlin.global_task']
            terminal = block.last_op
            for successor in (() if terminal is None else terminal.successors):
                owner = next_owner(successor, seen)
                if owner is not None:
                    return owner
            return None
        predecessors = {block: [] for block in function.body.blocks}
        for predecessor in function.body.blocks:
            if predecessor.last_op is not None:
                for successor in predecessor.last_op.successors:
                    predecessors[successor].append(predecessor)
        def previous_owner(block, seen):
            if block in seen:
                return None
            seen.add(block)
            for predecessor in predecessors[block]:
                for operation in reversed(tuple(predecessor.ops)):
                    tag = operation.attributes.get('merlin.global_task')
                    if operation.name not in ('llvm.mlir.constant', 'llvm.return') and tag is not None and int(tag.value.data) >= 0:
                        return tag
                tag = previous_owner(predecessor, seen)
                if tag is not None:
                    return tag
            return None
        for block in function.body.blocks:
            previous = None
            for operation in block.ops:
                if operation.name == 'llvm.mlir.constant':
                    continue
                tag = operation.attributes.get('merlin.global_task')
                if tag is not None:
                    previous = tag
                elif operation.name in ('llvm.br', 'llvm.cond_br', 'llvm.switch'):
                    tag = previous if previous is not None else previous_owner(block, set())
                    if tag is None:
                        tag = next_owner(block, set())
                    if tag is None:
                        tag = IntegerAttr(-1, i64)
                    operation.attributes['merlin.global_task'] = tag
                elif operation.name == 'llvm.return':
                    operation.attributes['merlin.global_task'] = IntegerAttr(-2, i64)
    # The LLVM inliner may append a callee's loop blocks after the caller's
    # return continuation. Preserve CFG edges while placing final return-only
    # continuations after the source work they follow in the static inventory.
    for function in lowered.body.block.ops:
        if not isinstance(function, llvm.FuncOp) or not function.body.blocks or function.sym_name.data != symbol:
            continue
        blocks = tuple(function.body.blocks)
        def block_scope(block):
            for operation in block.ops:
                if operation.name == 'llvm.mlir.constant':
                    continue
                attr = operation.attributes.get('merlin.global_task')
                if attr is not None:
                    value = int(attr.value.data)
                    return value if value >= 0 else (10**12 if value == -2 else -1)
            return 10**12
        ordered = [blocks[0], *sorted(blocks[1:], key=block_scope)]
        for block in blocks:
            function.body.detach_block(block)
        function.body.add_block(ordered)
    lowered.verify()
    return _text(lowered)


def lower_host_source(source, *, device_lower=None, symbol='kernel_entry'):
    """Return ``(command_buffer, llvm_dialect)`` for an actual mixed kernel.

    ``device_lower(source, symbol=...)`` returns its complete command buffer and
    LLVM dialect. It is called only on integer device regions. Failure propagates;
    required device work is never retried on the CPU.
    """
    original_module = parse_mlir_text(source)
    original = _text(original_module)
    original_direct = [op for op in _owner(original_module).body.block.ops if op.name != 'func.return']
    original_index = {op: index for index, op in enumerate(original_direct)}
    module = original_module
    preparation = None
    preparation_report = None
    layout_reuse = _source_layout_reuse_requested()
    from host_preparation_policy import requested as prepack_requested, prepare_host_preparations
    prepack_reuse = prepack_requested()
    if layout_reuse or prepack_reuse:
        from source_preparation import prepare_source_layouts
        if prepack_reuse:
            candidate = prepare_host_preparations(
                module, prepack_weights=True, share_exact_tensor_preparations=layout_reuse)
            preparation_report = candidate.report
        else:
            candidate = prepare_source_layouts(module)
        original_returns = next(op for op in _owner(module).body.block.ops if op.name == 'func.return')
        returned = {original_index[value.owner] for value in original_returns.operands if value.owner in original_index}
        # Preserve distinct original output writers and the public pointer ABI.
        if not any(alias['source_ordinal'] in returned for alias in candidate.aliases):
            preparation = candidate
            module = candidate.module
    owner = _owner(module)
    block = owner.body.block
    returns = next(op for op in block.ops if op.name == 'func.return')
    direct = [op for op in block.ops if op.name != 'func.return']
    ordinals = preparation.source_ordinals[:len(direct)] if preparation is not None else tuple(range(len(direct)))
    direct_index = dict(zip(direct, ordinals, strict=True))
    forced_device = set()
    epilogue_admissions, epilogue_refusals = [], []
    import source_epilogue
    if source_epilogue.enabled():
        epilogues = source_epilogue.prepare_source_epilogues(module)
        for refusal in epilogues.refusals:
            refusal = copy.deepcopy(refusal)
            refusal['root_ordinal'] = direct_index[direct[refusal['root_ordinal']]]
            epilogue_refusals.append(refusal)
        for group in epilogues.groups:
            members = {group['root'], *group['members']}
            positions = [direct.index(operation) for operation in members]
            # This adapter keeps sequential source regions. An independent
            # floating seam between chain stages remains an explicit boundary.
            intervening = direct[min(positions):max(positions) + 1]
            if any(operation not in members and not _scaffold(operation) and _host_op(operation)
                   for operation in intervening):
                epilogue_refusals.append({
                    'root_ordinal': direct_index[group['root']],
                    'reason': 'an independent host seam separates admitted readout stages'})
                continue
            forced_device.update(members)
            admission = {key: copy.deepcopy(value) for key, value in group.items()
                         if key not in ('root', 'members', 'result')}
            admission['root_ordinal'] = direct_index[group['root']]
            admission['source_op_indices'] = sorted(direct_index[operation] for operation in members)
            epilogue_admissions.append(admission)
    groups = []
    for op in block.ops:
        if op.name == 'func.return':
            break
        # A returned literal tensor is observable source work, rather than
        # storage scaffolding captured by a downstream computation.
        if _scaffold(op) and not any(result in returns.operands for result in op.results):
            continue
        host = op not in forced_device and _host_op(op)
        if groups and groups[-1][0] == host:
            groups[-1][1].append(op)
        else:
            groups.append((host, [op]))
    if not groups:
        raise ValueError('source contains no computational region')
    removed = set(range(len(original_direct))) - set(ordinals)
    attributed = set()
    attributed_indices = set()
    task_records = []
    extra_source_map = {}
    cb = {'abi_version': '0.1', 'target': 'gemmini', 'tensors': {}, 'commands': [], 'params': {}}
    if source_epilogue.enabled():
        cb['params'].update(source_epilogue_admissions=epilogue_admissions,
                            source_epilogue_refusals=epilogue_refusals)
    if preparation_report is not None:
        cb['params']['host_preparation_policy'] = preparation_report
    if preparation is not None:
        cb['params']['source_layout_preparation'] = {
            'witnesses': list(preparation.witnesses), 'aliases': list(preparation.aliases),
            'source_ordinals': list(ordinals), 'scope': 'whole_connected_source'}
    names = {}
    for index, value in enumerate(block.args):
        names[value] = 'arg' + str(index)
        cb['tensors'][names[value]] = _spec(value, 'input')
    output_names = []
    for index, value in enumerate(returns.operands):
        name = 'output' + str(index)
        if value in names:
            raise ValueError('direct input return requires explicit source transport')
        names[value] = name
        cb['tensors'][name] = _spec(value, 'output')
        output_names.append(name)
    available = set(block.args)
    functions = []
    calls = []
    placement = []
    for index, (host, group) in enumerate(groups):
        from model_regions import outline_region
        outlined, inputs, outputs, selected = outline_region(group, available)
        if not outputs:
            continue
        for value in outputs:
            if value not in names:
                name = 'boundary_' + str(len(names))
                names[value] = name
                cb['tensors'][name] = _spec(value, 'intermediate')
        segment_symbol = 'source_host_' + str(index) if host else 'source_device_' + str(index)
        owned = []
        def own_original(operation):
            ordinal = original_index.get(operation)
            if ordinal is None or ordinal in attributed_indices:
                return
            # Removed immutable preparations retain their source identity in
            # the real consumer task, along with their original scalar/storage
            # closure. No execution marker or extra materialization is emitted.
            for nested in operation.walk():
                for operand in nested.operands:
                    producer = operand.owner
                    if producer in original_index and (original_index[producer] in removed or _scaffold(producer)):
                        own_original(producer)
            attributed_indices.add(ordinal)
            owned.append(ordinal)
        for operation in selected:
            own_original(original_direct[direct_index[operation]])
        owned.sort()
        attributed.update(selected)
        first_command = len(cb['commands'])
        if host:
            host_preparation = {}
            host_optimization = {}
            product = _host_function(outlined, segment_symbol, index, host_preparation, host_optimization)
            if host_optimization:
                host_optimization.update(task_index=index, source_op_indices=owned)
                cb['params'].setdefault('host_optimizations', []).append(host_optimization)
            if host_preparation:
                cb['params'].setdefault('host_preparation_reuse', []).append({
                    'task_index': index, 'recipe': 'merlin.llvmlower.tensor_provenance_cse',
                    'removed_operations': host_preparation})
            argument_names = [names[v] for v in inputs + outputs]
        else:
            if device_lower is None:
                raise ValueError('mixed source needs an explicit accelerator emitter')
            part, product = device_lower(_text(outlined), symbol=segment_symbol)
            from merlin.compile.logical_abi import logical_interface, logical_abi
            abi = logical_abi(Path(__file__).resolve().parent / 'contracts' / 'mlir_oot_backend_contract.yaml')
            boundary = logical_interface(part, abi)
            in_names = [row.name for row in boundary if part['tensors'][row.name]['role'] in ('input', 'weight', 'bias', 'scale')]
            out_names = [row.name for row in boundary if row.kind == 'output']
            if len(in_names) != len(inputs) or len(out_names) != len(outputs):
                raise ValueError('device emitter changed outlined pointer boundary')
            mapping = dict(zip(in_names + out_names, [names[v] for v in inputs + outputs]))
            for name, spec in part.get('tensors', {}).items():
                mapped = mapping.setdefault(name, 'region_' + str(index) + '_' + name)
                if mapped not in cb['tensors']:
                    cb['tensors'][mapped] = dict(spec, role='intermediate')
            for name, origin in part.get('params', {}).get('tensor_source_map', {}).items():
                if 'op_index' in origin and origin['op_index'] < len(selected):
                    extra_source_map[mapping[name]] = {'op_index': direct_index[selected[origin['op_index']]],
                                                       'result_index': origin['result_index']}
            for temporary in part.get('params', {}).get('compiler_temporaries', ()):
                temporary = copy.deepcopy(temporary)
                temporary['tensor'] = mapping[temporary['tensor']]
                local_index = temporary['source_op_index']
                if not 0 <= local_index < len(selected):
                    raise ValueError('device temporary has no original source owner')
                temporary['source_op_index'] = direct_index[selected[local_index]]
                cb['params'].setdefault('compiler_temporaries', []).append(temporary)
            for fusion in part.get('params', {}).get('source_epilogue_fusions', ()):
                fusion = copy.deepcopy(fusion)
                fusion['source_op_indices'] = [direct_index[selected[local]]
                                               for local in fusion['source_op_indices']]
                fusion['source_task_index'] = index
                cb['params'].setdefault('source_epilogue_fusions', []).append(fusion)
            for command in part['commands']:
                command = copy.deepcopy(command)
                command['operands'] = {k: mapping.setdefault(v, 'region_' + str(index) + '_' + v)
                                       for k, v in command.get('operands', {}).items()}
                command.setdefault('attributes', {}).update({'source_task_index': index, 'source_op_indices': owned})
                cb['commands'].append(command)
            argument_names = [mapping[row.name] for row in boundary]
        import re
        source_indices = {i: direct_index[op] for i, op in enumerate(selected)}
        product = re.sub(r'merlin\.source_op_index = (\d+) : i64',
                         lambda m: 'merlin.source_op_index = ' + str(source_indices.get(int(m[1]), owned[0])) + ' : i64', product)
        product = re.sub(r'merlin\.global_task = -?\d+ : i64', 'merlin.global_task = ' + str(index) + ' : i64', product)
        # Stock generic globals store their symbol in properties, whereas the
        # direct device printer uses the custom @name form. Normalize both.
        from xdsl.dialects.builtin import StringAttr
        from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
        part_module = parse_mlir_text(product, ctx=make_llvm_context())
        global_names = []
        for operation in part_module.body.block.ops:
            if operation.name == 'llvm.mlir.global':
                old = operation.properties['sym_name'].data
                new = 'region_' + str(index) + '_' + old
                operation.properties['sym_name'] = StringAttr(new)
                global_names.append((old, new))
        product = _text(part_module)
        for old, new in global_names:
            product = re.sub(r'@' + re.escape(old) + r'\b', '@' + new, product)
        task_records.append({'task_index': index, 'kind': 'host_source' if host else 'accelerator_source',
                             'source_op_indices': owned, 'command_indices': list(range(first_command, len(cb['commands']))),
                             'symbol': segment_symbol, 'reads': [names[v] for v in inputs], 'writes': [names[v] for v in outputs]})
        functions.append(_module_body(_strip_completion(product)))
        calls.append((segment_symbol, argument_names))
        available.update(outputs)
        for op in group:
            placement.append({'source_region_id': _attr(op, 'prov.region_id'), 'operation': op.name,
                              'placement': 'host' if host else 'accelerator',
                              'host_reason': ('typed floating source or explicit host-required seam' if host else None)})
    cb['kernel_abi'] = {'kind': 'whole_program', 'args': [
        {'tensor': name, 'access': 'read' if spec['role'] in ('input', 'weight', 'bias', 'scale') else 'write'}
        for name, spec in cb['tensors'].items()], 'outputs': output_names}
    from isa import FunctionBuilder
    entry = FunctionBuilder(list(cb['tensors']), symbol=symbol)
    for callee, arguments in calls:
        pointers = ', '.join(entry.args[name] for name in arguments)
        types = ', '.join('!llvm.ptr' for _ in arguments)
        entry.emit('llvm.call @' + callee + '(' + pointers + ') : (' + types + ') -> ()')
    product = 'module {\n' + '\n'.join(functions) + _module_body(entry.finish()) + '\n}\n'
    product = inline_host_product(product, symbol=symbol)
    from target_storage import normalize_target_storage
    target_storage = {}
    product = normalize_target_storage(product, report=target_storage)
    cb['params']['target_storage_finalization'] = target_storage
    from host_storage_lifetimes import attribute_storage_lifetimes
    product = attribute_storage_lifetimes(product)
    from attributed_emit import normalize_compiler_prologue
    product = normalize_compiler_prologue(product)
    if _text(original_module) != original:
        raise ValueError('host route mutated original source SSA')
    tensor_sources = dict(extra_source_map)
    for value, name in names.items():
        if value in block.args:
            tensor_sources[name] = {'arg_index': block.args.index(value)}
        else:
            tensor_sources[name] = {'op_index': direct_index[value.owner], 'result_index': list(value.owner.results).index(value)}
    cb['params'].update({'tensor_source_map': tensor_sources, 'source_task_definitions': task_records,
                         'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                         'source_placement': placement, '_host_llvm': product, '_host_symbol': symbol})
    return cb, product
