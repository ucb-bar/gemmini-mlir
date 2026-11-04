"""Source-bound placement accounting for actually compiled mixed programs.

These records describe source work, not a flattened execution stream. Device
ownership comes from successful callback artifacts; remaining arithmetic keeps
its existing numerical/capability diagnostic without acquiring host permission.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path


class PlacementError(ValueError):
    pass


def logical_placement(command_buffer):
    """Return placement for successfully emitted logical compute commands.

    Invoke only after emission succeeds. Deferred product handles and resident
    preparation do not constitute another computation region. This function
    never traverses child buffers or projects across mixed-program host seams.
    """
    from merlin.targetgen import semantic_families as families
    operations = {'COMMIT': 'matmul', 'CONV2D': 'conv2d',
        'ATTENTION_QK': 'matmul', 'ATTENTION_PV': 'matmul',
        'BATCHED_MATMUL': 'matmul', 'MOVEMENT': 'movement',
        'STRIDED_MOVEMENT': 'transpose', 'ACCUMULATOR_READOUT': 'requantize',
        'RESIDUAL_ADD': 'residual_add', 'BIAS_ADD': 'bias_add', 'MAXPOOL2D': 'maxpool2d'}
    definitions = set()
    rows = []
    for index, command in enumerate(command_buffer.get('commands', ())):
        code = command['opcode']
        if code in ('MATMUL', 'MATMUL_RESIDENT'):
            definitions.add(command['operands']['dst'])
        if code not in operations:
            if code not in ('RES_PACK', 'EVICT', 'MATMUL', 'MATMUL_RESIDENT'):
                raise PlacementError('emitted logical command has no placement classification: ' + code)
            continue
        if code == 'COMMIT' and command['operands']['src'] not in definitions:
            raise PlacementError('commit placement lacks its deferred product definition')
        attrs = command.get('attributes', {})
        # Closed dispatch above proves the actual emitted operation. Retain its
        # canonical name if the shared reporting taxonomy lacks that operation.
        row = dict(lane='systolic_mesh', family=families.from_op(operations[code]) or operations[code],
            operation=code, command_index=index, status='compiled_device_command',
            reason='successfully emitted selected device lowering',
            epilogue=list(attrs.get('epilogue', ())))
        if 'merlin.source_op_index' in attrs:
            row['source_op_index'] = attrs['merlin.source_op_index']
        rows.append(row)
    return rows


def attach_source_placement(source, command_buffer, artifact, report, *,
                            transpose_transfers=False, prefer_quantized_readout=False):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.common.mlir_query import op_name
    from merlin.compile.analysis.contraction_shapes import observe_contractions
    from merlin.targetgen import semantic_families as families
    from xdsl.dialects.builtin import TensorType
    from .source_groups import placement_diagnostics

    digest = hashlib.sha256(source.encode()).hexdigest()
    cb = deepcopy(command_buffer)
    params = cb.setdefault('params', {})
    plan = params.get('global_program_plan', {})
    if report.get('source_sha256') != digest or plan.get('source_sha256') != digest:
        raise PlacementError('placement evidence differs from exact source bytes')
    module = parse_mlir_text(source)
    operations = list(module.walk())
    ordinals = {op: i for i, op in enumerate(operations)}
    functions = [op for op in operations if op.name == 'func.func' and op.body.blocks]
    functions = [op for op in functions if op.sym_name.data == 'forward'] or functions
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise PlacementError('placement requires the compiled single-block entry')
    top_indices = {op: i for i, op in enumerate(functions[0].body.block.ops)
                   if op.name != 'func.return'}
    compiled = {row['symbol']: row for row in params.get('device_regions', ())}
    owned = {}
    groups = report.get('device_regions', ())
    if len(groups) != len(compiled):
        raise PlacementError('placement report and compiled callback population differ')
    for group in groups:
        symbol = group['kernel_symbol']
        child = compiled.get(symbol)
        if child is None or not child.get('command_buffer', {}).get('commands'):
            raise PlacementError('source placement lacks its compiled device callback')
        if child['command_buffer'].get('declined'):
            raise PlacementError('declined callback cannot establish device placement')
        child_path = Path(child['artifact'])
        if not child_path.is_file() or not child_path.stat().st_size:
            raise PlacementError('device placement artifact is absent')
        for ordinal in group['source_ordinals']:
            if type(ordinal) is not int or not 0 <= ordinal < len(operations):
                raise PlacementError('device placement references an absent source operation')
            if ordinal in owned and operations[ordinal].name not in (
                    'arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
                raise PlacementError('source arithmetic belongs to two device callbacks')
            owned[ordinal] = group

    diagnostics = placement_diagnostics(module, transpose_transfers=transpose_transfers,
        prefer_quantized_readout=prefer_quantized_readout)
    diagnosed = {row['source_index']: row for row in diagnostics['regions']}
    contractions = {op for op, _ in observe_contractions(module)}
    rows = []
    excluded = Counter()
    tensor_movement = {'tensor.extract_slice', 'tensor.insert_slice', 'tensor.concat',
                       'tensor.pad', 'tensor.generate'}
    for op, top_index in top_indices.items():
        if not any(isinstance(v.type, TensorType) for v in op.results):
            continue
        ordinal = ordinals[op]
        name = op_name(op)
        diagnostic = diagnosed.get(ordinal)
        if diagnostic is None and name not in tensor_movement:
            excluded[name] += 1
            continue
        def attr(key):
            value = op.attributes.get(key)
            return getattr(value, 'data', None)
        family = (families.from_prov(attr('prov.family'), attr('prov.op'))
                  or families.from_op(name))
        if op in contractions:
            family = 'contraction'
        elif name in tensor_movement:
            family = 'movement'
        elif family is None and name == 'linalg.generic':
            family = ('reduction' if any(a.data.value == 'reduction' for a in op.iterator_types)
                      else 'elementwise_map')
        row = dict(lane='host', family=family or name, operation=name,
            source_sha256=digest, source_op_index=top_index, source_walk_ordinal=ordinal,
            source_ids={k: str(v) for k, v in op.attributes.items()
                        if k.startswith(('prov.', 'merlin.source'))},
            result_types=[str(v.type) for v in op.results],
            source_semantics=[op_name(nested) for nested in op.walk() if nested is not op])
        group = owned.get(ordinal)
        if group is not None:
            row.update(lane='systolic_mesh', status='compiled_device_callback',
                device_kernel=group['kernel_symbol'],
                device_source_sha256=group['source_sha256'],
                reason='exact source operation belongs to a successfully emitted device callback',
                device_form=(diagnostic or {}).get('device_form'))
        elif diagnostic is not None:
            if diagnostic['placement'] == 'device':
                # Re-running discovery is a capability diagnostic, not evidence
                # that this source operation reached an emitted callback. Keep
                # a successful executable even if those two views differ; the
                # callback report remains the sole device-ownership authority.
                row.update(status='unresolved_proof',
                    reason='source discovery admits a device form, but no successfully emitted '
                           'callback owns this exact source operation; device attribution remains unresolved',
                    source_discovery=deepcopy(diagnostic),
                    standalone={'status': 'unresolved_proof',
                                'reason': 'source admission does not establish emitted standalone ownership'},
                    fused={'status': 'unresolved_proof',
                           'reason': 'no emitted callback establishes fused source ownership'})
            else:
                for key in ('status', 'reason', 'standalone', 'fused'):
                    if key in diagnostic:
                        row[key] = deepcopy(diagnostic[key])
        else:
            row.update(status='unresolved_proof',
                reason='typed source tensor movement remains in shared host lowering; '
                       'no exact selected device layout/boundary composition was established',
                standalone={'status': 'unresolved_proof'},
                fused={'status': 'unresolved_proof'})
        rows.append(row)
    params['lane_placement'] = rows
    params['lane_placement_summary'] = dict(schema='submission.source_placement.v1',
        source_sha256=digest, emitted_module_sha256=hashlib.sha256(artifact.encode()).hexdigest(),
        scope='computation-carrying tensor source operations; source counts, not executed instruction counts',
        source_operations=plan['source_op_count'], placed_source_operations=len(rows),
        device_kernel_regions=len(groups),
        device_source_operations=sum(row['lane'] == 'systolic_mesh' for row in rows),
        host_source_operations=sum(row['lane'] == 'host' for row in rows),
        host_status_counts=dict(Counter(row.get('status', 'unresolved_proof') for row in rows if row['lane'] == 'host')),
        unowned_discovered_device_source_operations=sum('source_discovery' in row for row in rows),
        excluded_tensor_preparation=dict(excluded),
        static_device_boundary_bytes=report.get('static_boundary_bytes'),
        measured_host_cycles=None, measured_transfer_bytes=None,
        host_permission='not established by placement metadata')
    return cb
