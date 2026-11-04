"""Source-bound provenance for authored whole-program kernels.

The source inventory is structural and includes erased initializers.  Plan ranges
are finalized from the actual LLVM entry, after inlining and task attribution.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

_LIBRARY = Path(__file__).resolve().parent / 'compiler_library'
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))


def inspect_source(source_text):
    """Read complete generic source op order, without changing source bytes."""
    from xdsl.parser import Parser
    from merlin.targetgen.contract.linalg_iface import (
        make_linalg_context, _op_name, _prov, _initialization_only, _fn_name, _shape, _dtype,
    )
    text = source_text.decode('utf-8') if isinstance(source_text, bytes) else source_text
    module = Parser(make_linalg_context(), text).parse_module()
    definitions = [o for o in module.walk() if o.name == 'func.func' and o.body.blocks]
    forwards = [o for o in definitions if _fn_name(o) == 'forward']
    if len(forwards) > 1 or (not forwards and len(definitions) != 1):
        raise ValueError('whole-program provenance requires a unique source function')
    fn = forwards[0] if forwards else definitions[0]
    if len(fn.body.blocks) != 1:
        raise ValueError('whole-program provenance requires a single-block source function')
    block = fn.body.blocks[0]
    operations = [o for o in block.ops if _op_name(o) != 'func.return']
    keys = {v: ('arg', i, 0) for i, v in enumerate(block.args)}
    for i, op in enumerate(operations):
        for j, value in enumerate(op.results):
            keys[value] = ('op', i, j)
    records = []
    payload_indices = []
    for i, op in enumerate(operations):
        if not _initialization_only(op) and _op_name(op) != 'linalg.yield':
            payload_indices.append(i)
        records.append({'index': i, 'kind': _op_name(op), 'prov': _prov(op),
                        'operands': [keys[v] for v in op.operands if v in keys],
                        'results': [('op', i, j) for j in range(len(op.results))],
                        'result_specs': [{'shape':_shape(v.type),'dtype':_dtype(v.type)} for v in op.results]})
    terminator = block.ops.last
    if terminator is None or _op_name(terminator) != 'func.return':
        raise ValueError('source entry lacks a return')
    return {'ops': records, 'payload_source_indices': payload_indices,
            'entry_keys': [('arg', i, 0) for i in range(len(block.args))],
            'entry_specs': [{'shape':_shape(v.type),'dtype':_dtype(v.type)} for v in block.args],
            'return_keys': [keys[v] for v in terminator.operands]}


def _source_bytes(source_text):
    return source_text if isinstance(source_text, bytes) else source_text.encode('utf-8')


def _command_reads_writes(command, tensors):
    operands = command.get('operands', {})
    reads, writes = [], []
    for key, name in operands.items():
        if name not in tensors:
            continue
        target = writes if key == 'dst' else reads
        if name not in target:
            target.append(name)
    # Scalar region captures and seeded contractions can read additional
    # declared tensors through typed attribute records rather than operands.
    def attribute_reads(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in ('tensor', 'seed_tensor') and isinstance(nested, str) and nested in tensors:
                    if nested not in reads:
                        reads.append(nested)
                else:
                    attribute_reads(nested)
        elif isinstance(value, (list,tuple)):
            for nested in value:
                attribute_reads(nested)
    attribute_reads(command.get('attributes', {}))
    return reads, writes


def add_plan(cb, source_text, inventory=None):
    """Add exact source ownership and ordered ABI to a linalg command buffer.

    Optional inventory keys: ``value_tensors`` maps complete source value keys to
    materialized names; ``tasks`` supplies explicit emission groups.  Otherwise
    payload results use the frontend's __value_N naming and return ABI names.
    Each command receives source_op_index and global_task attributes.  Initial
    ranges describe command scheduling; finalize_plan replaces them with actual
    LLVM instruction ranges before the final artifact is published.
    """
    source = inspect_source(source_text)
    extra = inventory or {}
    tensors = cb.get('tensors', {})
    params = cb.setdefault('params', {})
    payload_inv = params.get('source_inventory', {})
    if extra.get('level') == 'linalg-on-tensors':
        payload_inv = extra
    payload_indices = source['payload_source_indices']
    payload_ops = payload_inv.get('ops', [])
    if payload_ops and len(payload_ops) != len(payload_indices):
        raise ValueError('payload and complete source inventories disagree')
    abi = cb.get('kernel_abi')
    if not abi or abi.get('kind') != 'whole_program':
        raise ValueError('generic provenance requires explicit whole_program ABI')
    outputs = list(abi['outputs'])
    if len(outputs) != len(source['return_keys']):
        raise ValueError('whole_program outputs differ from source return arity')
    input_names = [n for n, spec in tensors.items() if spec.get('role') in ('input','weight','bias','scale')]
    if len(input_names) != len(source['entry_keys']):
        raise ValueError('source argument count differs from materialized input ABI')
    values = dict(zip(source['entry_keys'], input_names))
    for i, name in enumerate(outputs):
        key = source['return_keys'][i]
        if key[0] != 'op' or key in values:
            raise ValueError('whole_program output requires a distinct source operation result')
        values[key] = name
    for payload_i, source_i in enumerate(payload_indices):
        for result_key in source['ops'][source_i]['results']:
            if result_key in values:
                continue
            suffix = '' if result_key[2] == 0 else f'_{result_key[2]}'
            name = f'__value_{payload_i}{suffix}'
            if name in tensors:
                values[result_key] = name
    values.update({tuple(key): name for key, name in extra.get('value_tensors', {}).items()})
    for key, name in values.items():
        expected = source['entry_specs'][key[1]] if key[0] == 'arg' else source['ops'][key[1]]['result_specs'][key[2]]
        actual = tensors[name]
        if actual['shape'] != expected['shape'] or actual['dtype'] != expected['dtype']:
            raise ValueError(f'materialized tensor {name} differs from its source shape/dtype')
    if len(set(values.values())) != len(values):
        raise ValueError('multiple source values may not share materialized storage')
    unbound = set(tensors) - set(values.values())
    temporaries = list(extra.get('compiler_temporaries', []))
    unbound -= {t['tensor'] for t in temporaries}
    if unbound:
        raise ValueError(f'materialized tensors lack source provenance: {sorted(unbound)}')
    tasks = extra.get('tasks')
    if tasks is None:
        tasks = []
        previous = 0
        commands = cb.get('commands', [])
        command_cursor = 0
        for payload_i, source_i in enumerate(payload_indices):
            names = [values[k] for k in source['ops'][source_i]['results'] if k in values]
            end = command_cursor
            found = set()
            while end < len(commands) and set(names) != found:
                _, written = _command_reads_writes(commands[end], tensors)
                found.update(n for n in written if n in names)
                end += 1
            if not names or set(names) != found:
                raise ValueError(f'payload {payload_i} lacks an emitted materialized writer')
            indices = list(range(previous, source_i + 1))
            if payload_i == len(payload_indices)-1:
                indices.extend(range(source_i+1, len(source['ops'])))
            reads, writes = [], []
            for command in commands[command_cursor:end]:
                r, w = _command_reads_writes(command, tensors)
                reads.extend(n for n in r if n not in reads and n not in writes)
                writes.extend(n for n in w if n not in writes)
                attrs = command.setdefault('attributes', {})
                attrs['merlin.global_task'] = payload_i
                attrs['merlin.source_op_index'] = source_i
                region = source['ops'][source_i]['prov'].get('region_id')
                if region:
                    attrs['prov.region_id'] = region
            lanes = [c.get('attributes', {}).get('lowering_lane', c.get('attributes', {}).get('lower_lane', 'device')) for c in commands[command_cursor:end]]
            kind = 'host' if 'host' in lanes else 'device'
            reasons = list(dict.fromkeys(c.get('attributes', {}).get('host_reason') for c in commands[command_cursor:end] if c.get('attributes', {}).get('host_reason')))
            if kind == 'host' and not reasons:
                raise ValueError('host lowering requires an explicit source-owned reason')
            tasks.append({'task_index': payload_i, 'kind': kind,
                          'source_op_indices': indices, 'source_op_index': source_i,
                          'source_region_id': source['ops'][source_i]['prov'].get('region_id'),
                          'instruction_start': command_cursor, 'instruction_end': end,
                          'reads': reads, 'writes': writes})
            if reasons:
                tasks[-1]['host_reason'] = '; '.join(reasons)
            command_cursor, previous = end, source_i+1
        if command_cursor != len(commands):
            raise ValueError('trailing commands lack source task ownership')
    all_indices = []
    for task_i, task in enumerate(tasks):
        task['task_index'] = task_i
        indices = task['source_op_indices']
        all_indices.extend(indices)
        task['source_region_ids'] = list(dict.fromkeys(source['ops'][i]['prov']['region_id']
            for i in indices if source['ops'][i]['prov'].get('region_id')))
    if sorted(all_indices) != list(range(len(source['ops']))):
        raise ValueError('tasks must cover every source operation exactly once')
    source_values = [{'op_index': k[1], 'result_index': k[2], 'tensor': name}
                     for k, name in values.items() if k[0] == 'op']
    for key, name in zip(source['return_keys'], outputs):
        if values.get(key) != name:
            raise ValueError('output bindings differ from source return order')
        writers = [t for t in tasks if name in t['writes']]
        if len(writers) != 1 or key[1] not in writers[0]['source_op_indices']:
            raise ValueError(f'output {name} lacks its unique source-owned writer')
    # Preserve existing argument order, and include every materialized tensor once.
    old_args = abi.get('args', [])
    order = [a['tensor'] for a in old_args]
    order.extend(n for n in tensors if n not in order)
    if len(set(order)) != len(order) or set(order) != set(tensors):
        raise ValueError('whole_program args must cover every materialized tensor exactly once')
    abi['args'] = [{'tensor': n, 'access': 'read' if n in input_names else
                    'write' if n in outputs else 'readwrite'} for n in order]
    params['global_program_plan'] = {
        'schema':'mixed_program_plan_v1',
        'source_sha256':hashlib.sha256(_source_bytes(source_text)).hexdigest(),
        'source_op_count':len(source['ops']), 'tasks':tasks,
        'schedule_instruction_count': max(t['instruction_end'] for t in tasks),
        'prologue_instruction_range':[0, tasks[0]['instruction_start']],
        'epilogue_instruction_range':[tasks[-1]['instruction_end'], tasks[-1]['instruction_end']],
        'entry_bindings':input_names, 'source_values':source_values,
        'output_bindings':outputs, 'compiler_temporaries':temporaries,
    }
    params['lane_placement'] = [
        {'task_index':t['task_index'], 'source_op_indices':t['source_op_indices'],
         'source_region_ids':t['source_region_ids'], 'lane':'host', 'route':'H',
         'region_id':region, 'reason':t['host_reason']}
        for t in tasks if t['kind'] == 'host'
        for region in (t['source_region_ids'] or [None])]
    cb['compiler_pointer_abi'] = {'version':1,'arguments':order}
    return cb


def finalize_plan(cb, lowered_llvm_text):
    """Set exact instruction ranges from the final, task-tagged LLVM entry."""
    module = _parse_llvm_module(lowered_llvm_text)
    entries = [op for op in module.walk() if op.name == 'llvm.func'
               and getattr(getattr(op, 'sym_name', None), 'data', None) == 'kernel_entry']
    if len(entries) != 1:
        raise ValueError('expected one lowered kernel_entry')
    operations = [op for region in entries[0].regions for block in region.blocks
                  for op in block.ops]
    plan = cb['params']['global_program_plan']
    owned = {t['task_index']: [] for t in plan['tasks']}
    for index, op in enumerate(operations):
        attrs = {**op.attributes, **getattr(op,'properties',{})}
        tag = attrs.get('merlin.global_task')
        if tag is None:
            continue
        task_i = getattr(getattr(tag, 'value', None), 'data', None)
        if task_i not in owned:
            raise ValueError('LLVM operation names an unknown global task')
        owned[task_i].append(index)
    previous_end = 0
    for task in plan['tasks']:
        indices = owned[task['task_index']]
        if not indices or indices[0] < previous_end:
            raise ValueError('LLVM task instructions are absent or interleaved')
        if indices != list(range(indices[0],indices[-1]+1)):
            raise ValueError('LLVM task range contains unowned instructions')
        task['instruction_start'], task['instruction_end'] = indices[0], indices[-1]+1
        previous_end = indices[-1]+1
    plan['schedule_instruction_count'] = len(operations)
    plan['prologue_instruction_range'] = [0,plan['tasks'][0]['instruction_start']]
    plan['epilogue_instruction_range'] = [previous_end,len(operations)]
    return cb


def tag_and_finalize(cb, llvm_text):
    """Replace authored asm delimiters with structural LLVM task attributes.

    Task ownership is a forward CFG dataflow fact, not an instruction-name
    heuristic.  Every path reaching an operation must agree on its current
    task.  Block reordering changes no CFG edges or SSA uses.
    """
    import io
    import re
    from xdsl.dialects.builtin import IntegerAttr, StringAttr
    from xdsl.printer import Printer
    module = _parse_llvm_module(llvm_text)
    entries = [op for op in module.walk() if op.name == 'llvm.func'
               and getattr(getattr(op, 'sym_name', None), 'data', None) == 'kernel_entry']
    if len(entries) != 1:
        raise ValueError('expected one lowered kernel_entry')
    entry = entries[0]
    # C's inferred nocapture/readonly/writeonly are optional optimization
    # promises, while this runtime's normalized pointer boundary requires
    # empty argument dictionaries. The declared pointer types stay unchanged.
    entry.properties.pop('arg_attrs', None)
    region = entry.regions[0]
    blocks = list(region.blocks)
    plan = cb['params']['global_program_plan']
    known_tasks = {t['task_index'] for t in plan['tasks']}
    marker_pattern = re.compile(r'^\s*#\s*merlin_task_(start|end)\s+(\d+)\s*$')

    def marker(op):
        if op.name != 'llvm.inline_asm':
            return None
        attrs = {**op.attributes, **getattr(op, 'properties', {})}
        asm = attrs.get('asm_string')
        match = marker_pattern.match(getattr(asm, 'data', ''))
        if not match:
            return None
        task_i = int(match.group(2))
        if task_i not in known_tasks:
            raise ValueError(f'unknown task delimiter {task_i}')
        return match.group(1), task_i

    markers = {op: marker(op) for block in blocks for op in block.ops}
    starts = {v[1] for v in markers.values() if v and v[0] == 'start'}
    ends = {v[1] for v in markers.values() if v and v[0] == 'end'}
    if starts != known_tasks or ends != known_tasks:
        raise ValueError('each planned task needs start and end asm delimiters')
    incoming = {block: set() for block in blocks}
    incoming[blocks[0]].add(None)
    operation_states = {}
    pending = [blocks[0]]
    while pending:
        block = pending.pop()
        states = set(incoming[block])
        for op in block.ops:
            mark = markers[op]
            if mark:
                kind, task_i = mark
                if kind == 'start':
                    if any(s is not None and s != task_i for s in states):
                        raise ValueError('nested global task delimiters')
                    states = {task_i}
                else:
                    if states != {task_i}:
                        raise ValueError('task end delimiter has inconsistent CFG ownership')
                    states = {None}
            else:
                operation_states.setdefault(op, set()).update(states)
        term = block.last_op
        for successor in term.successors if term is not None else ():
            updated = incoming[successor] | states
            if updated != incoming[successor]:
                incoming[successor] = updated
                pending.append(successor)
    for block in blocks:
        for op in list(block.ops):
            if markers[op]:
                block.erase_op(op)
                continue
            states = operation_states.get(op, set())
            if not states:
                raise ValueError('unreachable LLVM block has no task provenance')
            if len(states) != 1:
                raise ValueError('LLVM operation has ambiguous CFG task ownership')
            task_i = next(iter(states))
            if task_i is not None:
                op.attributes['merlin.global_task'] = IntegerAttr(task_i, 64)
                task = plan['tasks'][task_i]
                if op.name == 'llvm.inline_asm' and 'source_op_index' in task:
                    op.attributes['merlin.source_op_index'] = IntegerAttr(task['source_op_index'],64)
                    if task.get('source_region_id'):
                        op.attributes['prov.region_id'] = StringAttr(task['source_region_id'])
    # Place detached continuation blocks in task order, preserving entry block.
    # A start/end marker pair may occur in one straight-line block; its operations
    # already have the correct source order and require no splitting.
    def block_rank(block):
        tags = [getattr(getattr(op.attributes.get('merlin.global_task'), 'value', None), 'data', None)
                for op in block.ops]
        owners = [tag for tag in tags if tag is not None]
        return (min(owners), len(set(owners)) > 1) if owners else (len(known_tasks), False)
    ordered = [blocks[0], *sorted(blocks[1:], key=block_rank)]
    if ordered != blocks:
        for block in blocks[1:]:
            region.detach_block(block)
        for block in ordered[1:]:
            region.add_block(block)
    operations = [op for block in region.blocks for op in block.ops]
    task_positions = {t: [] for t in known_tasks}
    for i, op in enumerate(operations):
        tag = op.attributes.get('merlin.global_task')
        if tag is not None:
            task_positions[tag.value.data].append(i)
    previous_end = 0
    for task in plan['tasks']:
        positions = task_positions[task['task_index']]
        if not positions or positions[0] < previous_end:
            raise ValueError('LLVM CFG layout interleaves planned tasks')
        # Pure unowned address/control instructions between task operations are
        # assigned to that same task. No tensor work can escape a delimiter.
        start = positions[0] if task['task_index'] == 0 else previous_end
        for op in operations[start:positions[-1]+1]:
            existing = op.attributes.get('merlin.global_task')
            if existing is not None and existing.value.data != task['task_index']:
                raise ValueError('task instruction ranges overlap')
            op.attributes['merlin.global_task'] = IntegerAttr(task['task_index'],64)
        task['instruction_start'], task['instruction_end'] = start, positions[-1]+1
        previous_end = positions[-1]+1
    plan['schedule_instruction_count'] = len(operations)
    plan['prologue_instruction_range'] = [0,plan['tasks'][0]['instruction_start']]
    plan['epilogue_instruction_range'] = [previous_end,len(operations)]
    return normalize_llvm_text(module)


def normalize_llvm_text(module_or_text):
    """Serialize ordinary and task-owned LLVM with the same portable vocabulary.

    Parsing and serialization preserve computations, CFG edges, source/task
    attributes and target features. Only checked redundant link metadata,
    absent-property defaults, and inferred entry pointer promises are omitted.
    """
    import io
    from xdsl.printer import Printer
    module = _parse_llvm_module(module_or_text) if isinstance(module_or_text,str) else module_or_text
    for op in module.walk():
        if op.name == 'llvm.func' and getattr(getattr(op,'sym_name',None),'data',None) == 'kernel_entry':
            op.properties.pop('arg_attrs',None)
    # These Clang module flags repeat the selected build's RISC-V target
    # ABI/ISA, medany model and small-data threshold. The runner selects the
    # same ABI in its toolchain; target triple, data layout and per-function
    # target features remain in this module. The source-program reader lacks
    # llvm.module_flags, so omit this known redundant link metadata operation.
    import re
    allowed_flags = {'target-abi','riscv-isa','Code Model','SmallDataLimit'}
    for op in list(module.body.block.ops):
        if op.name != 'llvm.module_flags':
            continue
        triple = module.attributes.get('llvm.target_triple')
        if not getattr(triple,'data','').startswith('riscv64-'):
            raise ValueError('cannot omit target link metadata for an unselected target')
        for flag in op.properties['flags']:
            spelling = str(flag)
            match = re.search(r',\s*"([^"\\]+)"\s*,',spelling)
            if match is None or match.group(1) not in allowed_flags:
                raise ValueError('cannot omit unknown LLVM module link metadata')
            if match.group(1) == 'target-abi' and not re.search(r'"target-abi",\s*"lp64d"',spelling):
                raise ValueError('module ABI differs from selected lp64d toolchain')
        module.body.block.erase_op(op)
    # Native LLVM materializes explicit empty fastmath on FP casts, while
    # the selected xDSL release omits this optional default. Elide only the
    # empty set so the emitted artifact also parses with the public reader.
    for op in module.walk():
        # ATT is LLVM's absent-property default. Omitting its explicit zero
        # also avoids the native/xDSL inline-asm optional-field order mismatch.
        if op.name == 'llvm.inline_asm':
            dialect = op.properties.get('asm_dialect')
            if dialect is not None and dialect.value.data == 0:
                op.properties.pop('asm_dialect')
        if op.name in ('llvm.fpext','llvm.fptrunc'):
            flags = op.properties.get('fastmathFlags')
            if flags is not None and not flags.data:
                op.properties.pop('fastmathFlags')
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _parse_llvm_module(text):
    """Preserve native FP cast flags absent from the selected xDSL release.

    Native parsing/verifying runs before this compatibility reader.  Its local
    operation definitions add only the upstream optional fastmathFlags property;
    all original SSA/type and custom cast verification still runs.
    """
    from merlin.targetgen.oot_starterkit import llvm_context as support
    # xDSL accepts native attribute aliases, but unregistered nested LLVM
    # metadata stores their names as opaque text instead of resolving them.
    # Reprinting then loses the alias declarations. Native local-scope generic
    # serialization inlines aliases recursively without changing the IR. Use
    # it before xDSL reads any module containing attribute alias declarations.
    import re
    if re.search(r'^\s*#[A-Za-z_][\w$.-]*\s*=', text, re.M):
        import subprocess
        serialized = subprocess.run(
            [str(support._native_mlir_opt()), '--mlir-print-op-generic',
             '--mlir-print-local-scope', '-o', '-'], input=text,
            capture_output=True, text=True, timeout=30)
        if serialized.returncode:
            raise ValueError('native LLVM alias expansion failed:\n'+serialized.stderr)
        text = serialized.stdout
    try:
        return support.parse_llvm_module(text)
    except Exception as original:
        if 'fastmathFlags' not in str(original):
            raise
    import subprocess
    from copy import copy
    from dataclasses import replace
    from xdsl.context import Context
    from xdsl.dialects import llvm, builtin, func
    from xdsl.ir import Dialect
    from xdsl.irdl import BaseAttr, OptPropertyDef, ParsePropInAttrDict
    from xdsl.irdl.operations import get_accessors_from_op_def
    from xdsl.parser import Parser

    def extend(base):
        original_def = base.get_irdl_definition()
        definition = replace(original_def, properties={**original_def.properties,
            'fastmathFlags':OptPropertyDef(BaseAttr(llvm.FastMathAttr))},
            options=[*original_def.options, ParsePropInAttrDict()])
        def verify(op):
            view = copy(op)
            view.properties = dict(op.properties)
            view.properties.pop('fastmathFlags', None)
            base.verify_(view)
        return type('_NativeFlags'+base.__name__, (base,), get_accessors_from_op_def(definition, verify))
    additions = {base:extend(base) for base in (llvm.FPExtOp,llvm.FPTruncOp)}
    operations = [additions.get(base, support._OPERATIONS.get(base,base)) for base in llvm.LLVM.operations]
    operations.append(support._ModuleFlagsOp)
    context = Context(allow_unregistered=True)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(func.Func)
    context.load_dialect(Dialect('llvm',operations,list(llvm.LLVM.attributes)))
    tool = support._native_mlir_opt()
    serialized = subprocess.run([str(tool),'--mlir-print-op-generic','--mlir-print-local-scope','-o','-'],
                                input=text,capture_output=True,text=True,timeout=30)
    if serialized.returncode:
        raise ValueError('native LLVM compatibility verification failed:\n'+serialized.stderr)
    module = Parser(context,serialized.stdout).parse_module()
    module.verify()
    return module
