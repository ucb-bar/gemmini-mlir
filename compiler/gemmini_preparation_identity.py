"""Exact flat-order identities in immutable host tensor preparation.

The source operations and every observer remain in the original graph. This
adapter borrows only an invocation-local host descriptor: it neither publishes
an alias nor extends allocation lifetime or grants an opaque call permission.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import builtin
from merlin.llvmlower.tensor_provenance_cse import _supported
from merlin.transforms.certificates import ir_digest


def _shape(value):
    typ = value.type
    if (not isinstance(typ, builtin.TensorType)
            or not isinstance(typ.encoding, builtin.NoneAttr)
            or any(n <= 0 for n in typ.get_shape())):
        return None
    return tuple(typ.get_shape())


@dataclass(frozen=True)
class PreparationIdentity:
    operation: object
    kind: str
    source_sha256: str
    source_type: object
    result_type: object
    source_value: object
    result_value: object
    owner_block: object

    def revalidate(self):
        if (self.operation.parent is not self.owner_block
                or self.operation.operands[0] is not self.source_value
                or self.operation.results[0] is not self.result_value
                or ir_digest(self.operation) != self.source_sha256
                or self.operation.operands[0].type != self.source_type
                or self.operation.results[0].type != self.result_type):
            raise ValueError('host preparation identity changed after analysis')
        fresh = recognize(self.operation)
        if fresh is None or fresh.kind != self.kind:
            raise ValueError('host preparation no longer preserves flat order')
        return self


def recognize(operation):
    """Prove full slices or transposes that move only unit-length dimensions.

    The shared preparation domain provides immutable tensor/copy admission.
    Additional exact coordinates establish equality of the row-major sequence;
    same shape, provenance, and equal element counts alone never suffice.
    """
    if (operation.name not in ('tensor.extract_slice', 'linalg.transpose')
            or not _supported(operation) or len(operation.results) != 1
            or any(not key.startswith('prov.') for key in operation.attributes)):
        return None
    parent = operation.parent_op()
    while parent is not None:
        if any('strictfp' in key or 'strictfp' in str(value)
               for key, value in (*parent.attributes.items(), *parent.properties.items())):
            return None
        parent = parent.parent_op()
    source, result = operation.operands[0], operation.results[0]
    source_shape, result_shape = _shape(source), _shape(result)
    if (source_shape is None or result_shape is None
            or source.type.element_type != result.type.element_type):
        return None
    if operation.name == 'tensor.extract_slice':
        if (len(operation.operands) != 1 or operation.regions or operation.successors
                or set(operation.properties) != {'static_offsets', 'static_sizes',
                                                 'static_strides', 'operandSegmentSizes'}
                or source.type != result.type
                or tuple(operation.static_offsets.get_values()) != (0,)*len(source_shape)
                or tuple(operation.static_sizes.get_values()) != source_shape
                or tuple(operation.static_strides.get_values()) != (1,)*len(source_shape)):
            return None
        kind = 'complete_static_slice'
    else:
        if (set(operation.properties) != {'permutation'}
                or len(operation.operands) != 2 or operation.successors):
            return None
        permutation = tuple(operation.permutation.get_values())
        if (sorted(permutation) != list(range(len(source_shape)))
                or result_shape != tuple(source_shape[axis] for axis in permutation)
                or tuple(axis for axis in permutation if source_shape[axis] != 1)
                   != tuple(axis for axis, extent in enumerate(source_shape) if extent != 1)):
            return None
        kind = 'unit_dimension_transpose'
    return PreparationIdentity(operation, kind, ir_digest(operation), source.type, result.type,
                               source, result, operation.parent)


def try_apply(host, operation):
    """Consume one proved identity through the existing host descriptor owner.

    Caller explicitly selects this under host_tensor_views. Original source SSA
    values each retain a host mapping; ordinary store_result materializes every
    published/device boundary. No mutable storage or memref is admitted.
    """
    identity = recognize(operation)
    if identity is None:
        return False
    identity.revalidate()
    source = host.get(operation.operands[0])
    unit_axes_removed = 0
    if identity.kind == 'complete_static_slice':
        host.vals[operation.results[0]] = source
    else:
        # The existing reshape owner preserves flat reads, pointer and traversal.
        # The proof above establishes why this transpose has the same sequence.
        host._reshape(operation)
        descriptor = host.vals[operation.results[0]]
        # Unit axes have exactly one legal coordinate. Remove their loop
        # dimensions from the existing traversal, then restore literal zero
        # when reading. This changes counted host emission even when terminal
        # arithmetic folding already removes the identity slice expressions.
        traversal = descriptor.traversal
        if traversal is not None:
            domain, reader = traversal
            axes = tuple(axis for axis, extent in enumerate(domain) if extent != 1)
            if len(axes) != len(domain):
                unit_axes_removed = len(domain) - len(axes)
                def compact_read(indices):
                    coordinates = [host._v(0)] * len(domain)
                    for axis, value in zip(axes, indices, strict=True):
                        coordinates[axis] = value
                    return reader(tuple(coordinates))
                descriptor.traversal = (tuple(domain[axis] for axis in axes), compact_read)
    if not hasattr(host, 'host_view_reports'):
        host.host_view_reports = []
    host.host_view_reports.append(dict(
        schema='merlin.gemmini.host_preparation_identity.v1',
        operation=operation.name, source_operation_sha256=identity.source_sha256,
        source_shape=list(identity.source_type.get_shape()),
        result_shape=list(identity.result_type.get_shape()),
        recipe=identity.kind, realized='original immutable flat reader reused',
        traversal_unit_axes_removed=unit_axes_removed,
        source_region=str(operation.attributes.get('prov.region_id', '')),
        publication='original store_result; independent outputs and device boundaries',
        lifetime='original host descriptor within this synchronous invocation',
        arithmetic='none; original scalar bit patterns retained'))
    return True
