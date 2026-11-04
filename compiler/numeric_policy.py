"""Source-bound Gemmini numeric admission and lossless readout planning.

These diagnostics describe semantic domains, not which compiler adapters happen to
exist. They never reroute an eligible contraction to host for convenience.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any


class NumericDomainError(ValueError):
    """The source command itself has an inconsistent or malformed numeric domain."""


def _f32(value: Any) -> tuple[float, int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericDomainError('scale must be a finite numeric f32 value')
    try:
        raw = struct.pack('<f', float(value))
    except (OverflowError, ValueError, struct.error) as exc:
        raise NumericDomainError('scale is not representable as finite f32') from exc
    number = struct.unpack('<f', raw)[0]
    if not math.isfinite(number):
        raise NumericDomainError('scale must be finite f32')
    return number, struct.unpack('<I', raw)[0]


def normalize_readout(attributes: dict[str, Any], columns: int) -> dict[str, Any]:
    """Return an ordered numeric plan without changing the source attributes.

    ``column_groups`` identifies legal independently configured readouts. Distinct
    column gains remain distinct; no gain is inferred from shape or tensor values.
    Group boundaries also respect the 16-column accumulator bank geometry.
    """
    if not isinstance(columns, int) or columns <= 0:
        raise NumericDomainError('readout requires a positive static column extent')
    stages = list(attributes.get('epilogue', []))
    allowed = {'bias', 'bias_add', 'requant', 'acc_scale', 'relu', 'maxpool'}
    if any(s not in allowed for s in stages) or len(stages) != len(set(stages)):
        raise NumericDomainError(f'invalid or duplicated epilogue stages {stages!r}')
    output_dtype = attributes.get('output_dtype')
    if output_dtype not in ('i8', 'i32'):
        raise NumericDomainError(f'unsupported declared readout dtype {output_dtype!r}')
    plan: dict[str, Any] = {'output_dtype': output_dtype, 'stages': stages,
        'bias': attributes.get('bias'), 'scale': None, 'column_groups': [],
        'rounding': None, 'saturation': [-128, 127] if output_dtype == 'i8' else None,
        'diagnostics': []}
    if ('bias' in stages or 'bias_add' in stages) and not isinstance(plan['bias'], str):
        raise NumericDomainError('declared bias stage must identify its bias tensor')
    if 'requant' in stages:
        shift = attributes.get('requant_shift')
        if isinstance(shift, bool) or not isinstance(shift, int) or shift < 0:
            raise NumericDomainError('requant requires an explicit nonnegative integer requant_shift')
        plan['requant_shift'] = shift
        plan['diagnostics'].append({'category': 'hardware_numeric_mismatch',
            'code': 'requant_round_half_up',
            'reason': 'Source requant rounds half up; Gemmini scaled readout rounds half even. Replacing requant with an acc_scale changes tie results.'})
    if 'acc_scale' in stages:
        scalar = 'acc_scale' in attributes
        vector = 'acc_scale_by_column' in attributes
        if scalar == vector:
            raise NumericDomainError('acc_scale stage requires exactly one of acc_scale and acc_scale_by_column')
        values = attributes['acc_scale_by_column'] if vector else [attributes['acc_scale']] * columns
        if not isinstance(values, list) or len(values) != columns:
            raise NumericDomainError('acc_scale_by_column requires one explicit scale per final-axis column')
        rounded = [_f32(v) for v in values]
        bits = [b for _, b in rounded]
        payload = b''.join(struct.pack('<I', b) for b in bits)
        # Identity includes granularity, preserving a declared column-vector even
        # if its entries happen to be equal to a scalar source's value.
        granularity = 'per_column' if vector else 'per_tensor'
        plan['scale'] = {'identity': granularity + ':' + hashlib.sha256(payload).hexdigest(),
            'granularity': granularity, 'dtype': 'f32', 'values': [v for v, _ in rounded],
            'bits': bits, 'source_attribute': 'acc_scale_by_column' if vector else 'acc_scale'}
        plan['rounding'] = 'half_even'
        start = 0
        while start < columns:
            end = start + 1
            bank_end = min(columns, (start // 16 + 1) * 16)
            while end < bank_end and bits[end] == bits[start]:
                end += 1
            plan['column_groups'].append({'start': start, 'columns': end - start,
                'scale': rounded[start][0], 'scale_bits': bits[start]})
            start = end
        # Hardware activation precedes scale. Moving a source post-scale ReLU
        # ahead is valid for nonnegative gains, not for a negative multiplier.
        if 'relu' in stages and stages.index('relu') > stages.index('acc_scale') and any(v < 0 for v, _ in rounded):
            plan['diagnostics'].append({'category': 'requires_unfused_device_path',
                'code': 'negative_gain_postscale_relu',
                'reason': 'Source applies ReLU after a negative scale; the single readout applies activation before scale. Keep the ordered source stages in separate legal operations.'})
    elif 'acc_scale' in attributes or 'acc_scale_by_column' in attributes:
        raise NumericDomainError('scale parameter is present without an acc_scale stage')
    bias_positions = [i for i, s in enumerate(stages) if s in ('bias', 'bias_add')]
    if len(bias_positions) > 1:
        raise NumericDomainError('bias and bias_add cannot name two applications of one bias')
    if bias_positions and any(s not in ('bias', 'bias_add') for s in stages[:bias_positions[0]]):
        plan['diagnostics'].append({'category': 'requires_unfused_device_path',
            'code': 'bias_after_numeric_stage',
            'reason': 'Accumulator seeding applies bias before contraction readout; source places bias after another stage. Preserve its position with a separate operation.'})
    if output_dtype == 'i32' and any(s not in ('bias', 'bias_add') for s in stages):
        plan['diagnostics'].append({'category': 'requires_unfused_device_path',
            'code': 'raw_i32_readout_has_stages',
            'reason': 'Full-width i32 readout copies the accumulator without activation, scaling or pooling. Preserve the requested stages separately.'})
    if 'maxpool' in stages:
        for key in ('pool_in_dims', 'pool_size', 'pool_stride'):
            val = attributes.get(key)
            if not isinstance(val, list) or len(val) != 2 or any(type(v) is not int or v <= 0 for v in val):
                raise NumericDomainError(f'maxpool requires two positive integers in {key}')
        padding = attributes.get('pool_padding', [0, 0, 0, 0])
        if not isinstance(padding, list) or len(padding) != 4 or any(type(v) is not int or v < 0 for v in padding):
            raise NumericDomainError('pool_padding requires four nonnegative integers')
        if any(padding) and 'pool_pad_value' not in attributes:
            raise NumericDomainError('padded maxpool requires an explicit pool_pad_value')
        plan['pool'] = {k: attributes[k] for k in ('pool_in_dims', 'pool_size', 'pool_stride', 'pool_padding', 'pool_pad_value') if k in attributes}
        if stages[-1] != 'maxpool':
            plan['diagnostics'].append({'category': 'requires_unfused_device_path',
                'code': 'pool_before_later_stage', 'reason': 'Single readout pooling is the final stage; preserve later source arithmetic separately.'})
    return plan


def analyze_admission(cb: dict[str, Any]) -> dict[str, Any]:
    """Classify semantic placement and exact numeric mismatches per source command.

    Malformed source domains raise NumericDomainError. Valid unsupported numerical
    functions return explicit refusals. Unknown compiler/proof gaps remain separate
    from hardware limitations. ``readouts`` is keyed by semantic command index.
    """
    tensors = cb.get('tensors', {})
    diagnostics = []
    readouts = {}
    rows = []
    device_ops = {'MATMUL_RESIDENT', 'CONV2D', 'BATCHED_MATMUL', 'ATTENTION_QK', 'ATTENTION_PV', 'RESIDUAL_ADD', 'BIAS_ADD', 'MOVEMENT', 'MAXPOOL2D'}
    structural = {'RES_PACK', 'EVICT'}
    for i, cmd in enumerate(cb.get('commands', [])):
        opcode = cmd.get('opcode')
        attrs = cmd.get('attributes', {})
        operands = cmd.get('operands', {})
        source = cmd.get('source_record', {})
        row = {'command_index': i, 'source_index': cmd.get('source_index', i),
            'opcode': opcode, 'route': 'device', 'category': 'supported_domain',
            'source_provenance': source.get('source_provenance', source.get('attributes', {}))}
        if opcode in structural:
            row['route'] = 'structural'
        elif opcode in device_ops or opcode == 'COMMIT':
            dst = tensors.get(operands.get('dst'), {})
            dtype = attrs.get('output_dtype', dst.get('dtype'))
            if dst and dtype is not None and dtype != dst.get('dtype'):
                raise NumericDomainError(f'{opcode}: output_dtype differs from declared tensor dtype')
            # Readout attributes carry explicit stages only on the opcodes whose
            # ABI permits them. RESIDUAL_ADD has separate operand rounding rules.
            if opcode in {'COMMIT', 'CONV2D', 'ATTENTION_QK', 'ATTENTION_PV', 'BIAS_ADD', 'MAXPOOL2D'} and dst:
                if not dst.get('shape'):
                    raise NumericDomainError(f'{opcode}: missing static output shape')
                plan = normalize_readout({**attrs, 'output_dtype': dtype}, dst['shape'][-1])
                readouts[str(i)] = plan
                for d in plan['diagnostics']:
                    diagnostics.append({**d, 'command_index': i, 'source_index': row['source_index']})
                if any(d['category'] == 'hardware_numeric_mismatch' for d in plan['diagnostics']):
                    row.update(route='refuse', category='hardware_numeric_mismatch')
                elif plan['diagnostics']:
                    row.update(route='device_unfused', category='compiler_or_proof_gap')
                if plan['bias']:
                    bias = tensors.get(plan['bias'])
                    if bias is None or bias.get('dtype') != 'i32' or bias.get('shape') != [dst['shape'][-1]]:
                        raise NumericDomainError(f'{opcode}: bias must name an i32 vector with one element per output column')
            for key, name in operands.items():
                tensor = tensors.get(name)
                if tensor is not None and key not in ('dst', 'bias') and opcode in {'MATMUL_RESIDENT', 'CONV2D', 'BATCHED_MATMUL', 'ATTENTION_QK', 'ATTENTION_PV'}:
                    if tensor.get('dtype') != 'i8':
                        row.update(route='host_required', category='hardware_unsupported_domain')
                        diagnostics.append({'command_index': i, 'source_index': row['source_index'],
                            'category': 'hardware_unsupported_domain', 'code': 'non_i8_contraction_operand',
                            'reason': f'Selected mesh multiplies signed i8 operands; {name!r} is {tensor.get("dtype")!r}. Preserve the source numeric boundary.'})
        elif opcode in ('LINALG_OP', 'LINALG_RETURN'):
            kind = source.get('kind', '')
            dtypes = {r.get('dtype') for r in source.get('ins', []) + source.get('results', [])}
            nonlinear = any(n.startswith('math.') for n in source.get('body_ops', [])) or source.get('family') in ('normalization', 'softmax', 'nonlinear')
            if nonlinear or any(t and (t.startswith('f') or t.startswith('bf')) for t in dtypes):
                row.update(route='host_required', category='hardware_unsupported_domain')
            elif source.get('family') == 'contraction' or kind in ('linalg.matmul', 'linalg.batch_matmul'):
                row.update(route='device_required', category='compiler_or_proof_gap')
            elif dtypes and dtypes <= {'i8', 'i32', 'index', 'i64'}:
                row.update(route='device_or_host_proof_required', category='compiler_or_proof_gap')
            else:
                row.update(route='host_required', category='compiler_or_proof_gap')
        elif opcode in ('RMSNORM', 'ROPE', 'SOFTMAX'):
            row.update(route='refuse', category='unresolved_source_numeric_contract')
            diagnostics.append({'command_index': i, 'source_index': row['source_index'],
                'category': row['category'], 'code': 'legacy_nonlinear_missing_contract',
                'reason': f'{opcode} needs host nonlinear arithmetic with an exact typed source contract; this legacy mnemonic has no numerical definition in command_buffer_abi.yaml. A parser mnemonic does not define its arithmetic.'})
        else:
            row.update(route='unknown', category='compiler_or_proof_gap')
        rows.append(row)
    routes = {r['route'] for r in rows}
    route = 'refuse' if 'refuse' in routes else ('mixed' if any(r.startswith('host') for r in routes) and any(r.startswith('device') or r == 'device' for r in routes) else ('host' if any(r.startswith('host') for r in routes) else 'device'))
    return {'route': route, 'commands': rows, 'diagnostics': diagnostics, 'readouts': readouts,
        'refusals': [d for d in diagnostics if d['category'] in ('hardware_numeric_mismatch', 'unresolved_source_numeric_contract')],
        'compiler_gaps': [r for r in rows if r['category'] == 'compiler_or_proof_gap'],
        'counts': {r: sum(x['route'] == r for x in rows) for r in sorted(routes)}}
