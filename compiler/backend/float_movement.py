"""Exact f32 representation movement over the selected raw i32 DMA route.

The logical boundary remains f32. Only command-local pointer views are i32;
there is no floating conversion, arithmetic, host tensor copy or NaN decoding.
"""
from __future__ import annotations

from math import prod
from pathlib import Path


class FloatMovementError(ValueError):
    pass


def enabled():
    from merlin.compile.tuning_knobs import knob
    return knob('FLOAT_MOVEMENT', default='off', values=('off', 'on'),
                root=Path(__file__).resolve().parents[1]) == 'on'


def _spec(value):
    from xdsl.dialects.builtin import NoneAttr, TensorType
    ty = value.type
    if (not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr)
            or str(ty.element_type) != 'f32' or not ty.get_shape()
            or any(d <= 0 for d in ty.get_shape())):
        return None
    shape = list(ty.get_shape())
    if prod(shape) * 4 >= 1 << 63:
        return None
    return {'shape': shape, 'dtype': 'f32'}


def admit(path):
    """Require raw, unrounded accumulator preload and full-width exit."""
    preloads = [s for s in path.get('accumulator_stages', ()) if s.get('kind') == 'preload']
    if (path.get('accumulator', {}).get('dtype') != 'i32'
            or not path.get('exits', {}).get('accumulator')
            or not any(s.get('dtype') == 'i32' and s.get('rounding') == 'none'
                       and s.get('range') == [-(1 << 31), (1 << 31) - 1] for s in preloads)):
        raise FloatMovementError('f32 bit transport requires exact full-width i32 preload and raw exit')


def match(op):
    """Recognize a literal scalar identity or a proven contiguous dense view."""
    if len(op.results) != 1 or not op.operands:
        return None
    src = op.operands[0]
    a, b = _spec(src), _spec(op.results[0])
    if a is None or b is None or prod(a['shape']) != prod(b['shape']):
        return None
    form = None
    if op.name in ('linalg.copy', 'linalg.generic'):
        if len(op.operands) != 2 or _spec(op.operands[1]) != b or a != b:
            return None
        if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
            return None
        body = op.regions[0].block
        ops = list(body.ops)
        if (len(body.args) != 2 or any(str(v.type) != 'f32' for v in body.args)
                or len(ops) != 1 or ops[0].name != 'linalg.yield'
                or list(ops[0].operands) != [body.args[0]]):
            return None
        if op.name == 'linalg.generic':
            from xdsl.ir.affine import AffineMap
            maps, iters = op.properties.get('indexing_maps'), op.properties.get('iterator_types')
            if (maps is None or tuple(m.data for m in maps) != (AffineMap.identity(len(a['shape'])),) * 2
                    or iters is None or [v.data.value for v in iters] != ['parallel'] * len(a['shape'])):
                return None
        form = 'literal_scalar_identity'
    elif op.name in ('tensor.collapse_shape', 'tensor.expand_shape', 'linalg.transpose'):
        from .quantized_boundary import _contiguous_view, QuantizedBoundaryError
        try:
            if _contiguous_view(op, op.results[0], dtypes=('f32',)) is not src:
                return None
        except QuantizedBoundaryError:
            return None
        form = 'contiguous_dense_view'
    if form is None:
        return None
    return src, dict(
        form='exact_f32_bit_movement', source_operation=op.name, source_form=form,
        source_shape=a['shape'], result_shape=b['shape'], element_count=prod(a['shape']),
        logical_dtype='f32', transport_dtype='i32', bits_per_element=32,
        address_projection='identical row-major linear element index',
        numeric_operations=[],
        transport='full-width accumulator MVIN shrink=false; raw full=true MVOUT',
        representation='all 32 bits unchanged, including NaN payloads and signed zero',
        abi='dense row-major floats stored as uint32 bit patterns; distinct output pointer',
        completion='load drain before readout; store drain before local reuse and return')


def bind_buffer(buffer):
    """Bind proved f32 movement commands to same-address i32 storage aliases.

    Run after composition so independent and connected source plans retain their
    original logical tensor declarations and return ordering. Alias views flatten
    only command addressing, without changing the dense boundary shape.
    """
    from .float_transpose import bind_buffer as bind_transposes
    commands = buffer.get('commands', [])
    tensors = buffer.get('tensors', {})
    if not any(c['opcode'] == 'MOVEMENT'
               and any(tensors.get(c['operands'].get(k), {}).get('dtype') == 'f32' for k in ('src', 'dst'))
               for c in commands):
        return bind_transposes(buffer)
    params = buffer.setdefault('params', {})
    aliases = params.setdefault('tensor_aliases', {})
    records = []
    for index, command in enumerate(commands):
        if command['opcode'] != 'MOVEMENT':
            continue
        operands = command['operands']
        src, dst = operands['src'], operands['dst']
        if src not in tensors or dst not in tensors:
            continue
        a, b = tensors[src], tensors[dst]
        if a['dtype'] != 'f32' and b['dtype'] != 'f32':
            continue
        if (a['dtype'] != 'f32' or b['dtype'] != 'f32' or src == dst
                or src in aliases or dst in aliases or prod(a['shape']) != prod(b['shape'])
                or command.get('attributes') != {'semantic': 'mvin_mvout', 'output_dtype': 'i32'}):
            raise FloatMovementError('f32 movement lost its exact source/type/output proof')
        count = prod(a['shape'])
        if not count or count * 4 >= 1 << 63:
            raise FloatMovementError('f32 movement footprint outside selected address domain')
        names = []
        for label, source in (('src', src), ('dst', dst)):
            name = f'_f32_bits_{index}_{label}'
            if name in tensors or name in aliases:
                raise FloatMovementError('f32 movement alias name collision')
            aliases[name] = {'source': source, 'shape': [count], 'dtype': 'i32'}
            names.append(name)
        command['operands'] = {'src': names[0], 'dst': names[1]}
        records.append(dict(command_index=index, source=src, destination=dst,
                            source_shape=list(a['shape']), result_shape=list(b['shape']),
                            elements=count, source_alias=names[0], destination_alias=names[1],
                            original_logical_dtype='f32', alias_dtype='i32',
                            storage_bytes=count * 4, host_tensor_operations=0))
    if records:
        params['float_movement_alias_proof'] = records
        # Alias-only commands must never add physical pointer arguments.
        if not buffer.get('kernel_abi'):
            inputs = [name for name, spec in tensors.items() if spec.get('role') == 'input']
            outputs = [name for name, spec in tensors.items() if spec.get('role') == 'output']
            buffer['kernel_abi'] = {'kind': 'whole_program',
                'args': [*({'tensor': n, 'access': 'read'} for n in inputs),
                         *({'tensor': n, 'access': 'write'} for n in outputs)],
                'outputs': outputs}
    if not aliases:
        params.pop('tensor_aliases', None)
    return bind_transposes(buffer)
