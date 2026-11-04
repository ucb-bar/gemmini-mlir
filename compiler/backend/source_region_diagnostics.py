"""Conservative descriptors for typed non-linalg source operations.

Only source structure is described. These helpers do not decide placement,
rewrite source, prove a device implementation, or equate a tensor operation
with a physical copy/view. Uncertain forms return None.
"""
from __future__ import annotations

from math import isfinite, prod

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    DenseArrayBase, DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr,
    NoneAttr, StringAttr, TensorType,
)
from xdsl.ir import Operation
from xdsl.utils.exceptions import VerifyException

from merlin.common import mlir_query as mq
from merlin.perf.lane_cost import dtype_bits
from merlin.targetgen import semantic_families as sf
from merlin.targetgen.eligibility import RegionDescriptor
from merlin.targetgen.model_coverage import _ELEM_DTYPE


_QUANTIZE = 'quant_ext.quantize_per_tensor'
_DEQUANTIZE = 'quant_ext.dequantize_per_tensor'
_MOVEMENT = frozenset(('tensor.extract_slice', 'tensor.insert_slice', 'tensor.concat'))


def _typed(value, role):
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr):
        return None
    shape, dtype = list(ty.get_shape()), str(ty.element_type)
    bits = dtype_bits(dtype)
    if any(type(d) is not int or d <= 0 for d in shape) or bits is None or bits % 8:
        return None
    return dict(role=role, shape=shape, dtype=dtype, bytes=prod(shape) * (bits // 8))


def _format(dtype):
    # Machine float literals independently preserve the floating/integer
    # distinction even when a named-format registry is unavailable.
    if dtype.startswith('f') and dtype[1:].isdigit():
        return dtype
    return _ELEM_DTYPE.get(dtype, 'unsupported_mlir:' + dtype)


def _constant(value):
    """Observe a rank-zero original constant, without evaluating a tensor DAG."""
    owner = value.owner
    if isinstance(owner, tensor.SplatOp):
        if len(owner.operands) != 1 or len(owner.results) != 1:
            return None
        owner = owner.operands[0].owner
    if not isinstance(owner, arith.ConstantOp) or owner.operands or owner.regions or owner.successors:
        return None
    attr = owner.value
    if isinstance(attr, (FloatAttr, IntegerAttr)):
        return attr.value.data
    if isinstance(attr, DenseIntOrFPElementsAttr):
        values = tuple(attr.get_values())
        return values[0] if len(values) == 1 else None
    return None


def _integer(properties, key):
    attr = properties.get(key)
    return attr.value.data if isinstance(attr, IntegerAttr) else None


def _array(properties, key):
    attr = properties.get(key)
    if not isinstance(attr, DenseArrayBase) or str(attr.elt_type) not in ('i32', 'i64'):
        return None
    values = list(attr.get_values())
    return values if all(type(v) is int for v in values) else None


def _quantization(op, name):
    if len(op.operands) != 3 or len(op.results) != 1:
        return None
    inputs = [_typed(v, role) for v, role in zip(op.operands, ('data', 'scale', 'zero_point'), strict=True)]
    output = _typed(op.results[0], 'result')
    if output is None or any(row is None for row in inputs):
        return None
    data, scale, zero = inputs
    # Deliberately narrow: this is the observed f32/signed-i8 scalar affine
    # source contract, not a claim about arbitrary quant_ext extensions.
    quantizing = name == _QUANTIZE
    if (data['dtype'] != ('f32' if quantizing else 'i8')
            or output['dtype'] != ('i8' if quantizing else 'f32')
            or output['shape'] != data['shape']
            or scale['shape'] != [] or scale['dtype'] != 'f32'
            or zero['shape'] != [] or zero['dtype'] != 'i64'):
        return None
    fields = op.properties
    if set(fields) - {'quant_min', 'quant_max', 'output_dtype'}:
        return None
    qmin, qmax = _integer(fields, 'quant_min'), _integer(fields, 'quant_max')
    if qmin is None or qmax is None or not -128 <= qmin <= qmax <= 127:
        return None
    output_attr = fields.get('output_dtype')
    if quantizing:
        if not isinstance(output_attr, StringAttr) or output_attr.data != 'int8':
            return None
    elif output_attr is not None:
        # The shared per-tensor dequantization uses the actual result type;
        # reject an additional output-type promise we have not interpreted.
        return None
    scale_value, zero_value = _constant(op.operands[1]), _constant(op.operands[2])
    if (type(scale_value) not in (float, int) or not isfinite(scale_value) or scale_value <= 0
            or type(zero_value) is not int or not qmin <= zero_value <= qmax):
        return None
    numerics = dict(scale=scale_value, zero_point=zero_value, quant_min=qmin, quant_max=qmax,
                    scale_dtype='f32', zero_point_dtype='i64', granularity='per_tensor',
                    semantics_owner='merlin.llvmlower.passes_xdsl.lower_quant_ext')
    if quantizing:
        numerics.update(order=['f32_reciprocal_scale', 'f32_multiply', 'round_even',
                               'zero_point_to_f32', 'f32_add', 'clamp', 'signed_integer_convert'],
                        integer_accumulator_input=False)
    else:
        numerics.update(order=['signed_data_to_f32', 'zero_point_to_f32', 'f32_subtract', 'f32_multiply'],
                        floating_result_preserved=True)
    return inputs, output, dict(kind='typed_scalar_affine_quantization', numeric=numerics), (
        'exact device numeric order and source-to-device domain mapping; '
        'f32 source quantization is not an i32 accumulator readout; '
        'fusion must independently prove every original floating boundary')


def _movement(op, name):
    if len(op.results) != 1:
        return None
    roles = ('source', 'destination_initial') if name == 'tensor.insert_slice' else (
        ('source',) if name == 'tensor.extract_slice' else tuple('source_' + str(i) for i in range(len(op.operands))))
    if len(op.operands) != len(roles):
        return None
    inputs = [_typed(v, role) for v, role in zip(op.operands, roles, strict=True)]
    output = _typed(op.results[0], 'result')
    if not inputs or output is None or any(row is None for row in inputs):
        return None
    if any(row['dtype'] != output['dtype'] or len(row['shape']) != len(output['shape']) for row in inputs):
        return None
    fields = op.properties
    if name == 'tensor.concat':
        if set(fields) != {'dim'}:
            return None
        axis = _integer(fields, 'dim')
        rank = len(output['shape'])
        if axis is None or not 0 <= axis < rank:
            return None
        if (any(any(row['shape'][i] != output['shape'][i] for i in range(rank) if i != axis) for row in inputs)
                or sum(row['shape'][axis] for row in inputs) != output['shape'][axis]):
            return None
        proof = dict(kind='static_concatenation_coordinates', axis=axis,
                     input_extents=[row['shape'][axis] for row in inputs],
                     order='original operand order')
    else:
        if set(fields) != {'static_offsets', 'static_sizes', 'static_strides', 'operandSegmentSizes'}:
            return None
        offsets, sizes, strides, segments = (_array(fields, key) for key in (
            'static_offsets', 'static_sizes', 'static_strides', 'operandSegmentSizes'))
        expected_segments = [1, 0, 0, 0] if name == 'tensor.extract_slice' else [1, 1, 0, 0, 0]
        if segments != expected_segments or any(v is None for v in (offsets, sizes, strides)):
            return None
        source_shape = inputs[0]['shape']
        container = source_shape if name == 'tensor.extract_slice' else inputs[1]['shape']
        sliced = output['shape'] if name == 'tensor.extract_slice' else source_shape
        if (len(offsets) != len(container) or len(sizes) != len(container) or len(strides) != len(container)
                or sizes != sliced or any(o < 0 for o in offsets) or any(s <= 0 for s in strides)
                or any(o + (n - 1) * s >= dim for o, n, s, dim in zip(offsets, sizes, strides, container, strict=True))
                or name == 'tensor.insert_slice' and output['shape'] != container):
            return None
        proof = dict(kind='static_rank_preserving_slice_coordinates', offsets=offsets,
                     sizes=sizes, strides=strides,
                     index_relation='container_index = offset + slice_index * stride')
        if name == 'tensor.insert_slice':
            proof['unwritten_result_elements'] = 'preserved from destination_initial'
    proof.update(storage_alias=None, physical_copy=None, storage_realization='unresolved')
    return inputs, output, proof, 'source storage/alias binding and exact movement realization; not established as copy or free view'


def describe_source_operation(op):
    """Return ``(RegionDescriptor, proof_metadata)`` or None for an uncertain form.

    The descriptor remains the same for standalone and fused oracle queries:
    fusion alone cannot change the observed data/result dtype or erase source
    operations. Metadata records remaining realization gaps, not eligibility.
    It contains no model identity, invented source ordinal or device decision.
    """
    if not isinstance(op, Operation) or op.regions or op.successors:
        return None
    name = mq.op_name(op)
    if name not in (_QUANTIZE, _DEQUANTIZE) and name not in _MOVEMENT:
        return None
    if any(key != 'op_name__' and not key.startswith('prov.') for key in op.attributes):
        return None
    try:
        op.verify()
        observed = _quantization(op, name) if name in (_QUANTIZE, _DEQUANTIZE) else _movement(op, name)
    except (ValueError, TypeError, KeyError, IndexError, VerifyException):
        return None
    if observed is None:
        return None
    inputs, output, proof, gap = observed
    short = name.rsplit('.', 1)[-1]
    family = sf.from_op(short) if name in (_QUANTIZE, _DEQUANTIZE) else sf.from_prov('layout', short)
    if family not in sf.FAMILIES:
        return None
    # Parameter tensors are retained as exact source roles in metadata. They
    # are not separate activation or contraction-weight compute domains.
    data_roles = inputs[:1] if name in (_QUANTIZE, _DEQUANTIZE) else inputs
    descriptor = RegionDescriptor(source=name, op=short, family=family,
        in_dtype=_format(data_roles[0]['dtype']), out_dtype=_format(output['dtype']),
        captured_input_formats=tuple(_format(row['dtype']) for row in data_roles),
        rank=len(output['shape']), form=sf.operation_form(short, carrier_op=name),
        scale_granularity='per_tensor' if name in (_QUANTIZE, _DEQUANTIZE) else None,
        config={k: str(v) for k, v in op.properties.items()})
    metadata = dict(schema='merlin.authored_source_region_descriptor.v1', operation=name,
        inputs=inputs, outputs=[output], structural_proof=proof,
        source_attributes={k: str(v) for k, v in op.attributes.items()},
        source_properties={k: str(v) for k, v in op.properties.items()},
        boundary_bytes=dict(inputs=sum(row['bytes'] for row in inputs), outputs=output['bytes']),
        byte_basis='static logical source boundaries; not measured transfers',
        standalone_form='original typed operation',
        fused_form=dict(descriptor_domain_unchanged=True, equivalence_proven=False,
                        remaining_proof='complete original producer/consumer numeric and effect preservation'),
        remaining_proof=gap, storage_realization='unresolved' if name in _MOVEMENT else None,
        target_eligibility=None, device_execution=False)
    return descriptor, metadata


def source_region_descriptors(module):
    """Map actual parsed operations to descriptors/proofs; never change the IR."""
    result = {}
    for op in module.walk():
        description = describe_source_operation(op)
        if description is not None:
            result[op] = description
    return result
