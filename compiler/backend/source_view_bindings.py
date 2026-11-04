"""Prove read-only dense tensor views for ordinary device pointer bindings.

Static contiguous slices, rank-two pitched slices and verified row-major reshapes
are admitted through explicit read-only selections.
The original root storage, dtype and axes determine every element offset; a
permutation or dynamic stride retains its unsupported boundary. Pitched rows
are admitted only when the consumer explicitly accepts a physical row stride.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod


class SourceViewError(ValueError):
    pass


@dataclass(frozen=True)
class DenseSourceView:
    root: object
    shape: tuple[int, ...]
    dtype: str
    element_offset: int
    operations: tuple
    row_stride: int | None = None
    element_span: int | None = None

    def record(self):
        record = dict(shape=list(self.shape), dtype=self.dtype,
                    element_offset=self.element_offset, access='read',
                    source_shape=list(self.root.type.get_shape()),
                    proof='static contiguous slice/row-major reshape chain',
                    view_operations=[op.name for op in self.operations])
        if self.row_stride is not None:
            record.update(row_stride=self.row_stride, element_span=self.element_span,
                proof='static rank-two rectangular slices with exact physical row pitch')
        return record


def _type(value):
    from xdsl.dialects.builtin import TensorType, NoneAttr
    typ = value.type
    if (not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr)
            or any(extent <= 0 for extent in typ.get_shape())):
        raise SourceViewError('dense view requires a static unencoded positive tensor')
    return tuple(typ.get_shape()), str(typ.element_type)


def read_dense_source_view(value, *, maximum_depth=64, allow_row_pitch=False):
    """Return exact base-plus-offset storage facts, without granting effects."""
    target_shape, target_dtype = _type(value)
    current, operations = value, []
    for _ in range(maximum_depth):
        op = current.owner
        name = getattr(op, 'name', '')
        if name not in ('tensor.collapse_shape', 'tensor.expand_shape', 'tensor.extract_slice'):
            break
        if len(op.operands) != 1 or len(op.results) != 1:
            raise SourceViewError('dense view requires one original static source/result')
        before, dtype = _type(op.operands[0])
        after, result_dtype = _type(current)
        if dtype != result_dtype:
            raise SourceViewError('dense view cannot reinterpret element dtype')
        allowed = {'tensor.collapse_shape': {'reassociation'},
                   'tensor.expand_shape': {'reassociation', 'static_output_shape'},
                   'tensor.extract_slice': {'static_offsets', 'static_sizes', 'static_strides', 'operandSegmentSizes'}}[name]
        if any(key not in allowed and not key.startswith('prov.')
               for key in (*op.attributes, *op.properties)):
            raise SourceViewError('dense view has an unsupported semantic attribute')
        op.verify()
        if name != 'tensor.extract_slice' and prod(before) != prod(after):
            raise SourceViewError('dense reshape changes original element count')
        operations.append(op)
        current = op.operands[0]
    else:
        raise SourceViewError('dense view exceeds the bounded chain depth')
    root_shape, root_dtype = _type(current)
    shape = root_shape
    physical = tuple(prod(shape[axis+1:]) for axis in range(len(shape)))
    offset = 0
    def contiguous(shape, strides):
        return all(extent == 1 or step == prod(shape[axis+1:])
                   for axis,(extent,step) in enumerate(zip(shape,strides,strict=True)))
    for op in reversed(operations):
        after, _ = _type(op.results[0])
        if op.name == 'tensor.extract_slice':
            def values(key):
                attr = op.properties.get(key, op.attributes.get(key))
                return tuple(attr.get_values()) if hasattr(attr, 'get_values') else ()
            offsets, sizes, steps = (values(key) for key in ('static_offsets', 'static_sizes', 'static_strides'))
            if not len(offsets) == len(sizes) == len(steps) == len(shape) == len(after):
                raise SourceViewError('dense slice requires complete original source axes')
            if sizes != after:
                raise SourceViewError('dense slice result differs from its static sizes')
            if any(start < 0 or count <= 0 or step <= 0 or start+(count-1)*step >= extent
                   for start,count,step,extent in zip(offsets,sizes,steps,shape,strict=True)):
                raise SourceViewError('dense slice is dynamic or outside its original source')
            if len(shape) == 2 and steps[1] != 1:
                raise SourceViewError('dense source views require unit column stride')
            offset += sum(start*stride for start,stride in zip(offsets,physical,strict=True))
            physical = tuple(stride*step for stride,step in zip(physical,steps,strict=True))
            if not contiguous(after, physical) and not (
                    allow_row_pitch and len(after) == 2 and physical[1] == 1
                    and steps[1] == 1 and physical[0] >= after[1]):
                raise SourceViewError('dense slice has a noncontiguous row or column pitch')
        else:
            if not contiguous(shape, physical):
                raise SourceViewError('reshape cannot flatten physical row gaps in a pitched source view')
            physical = tuple(prod(after[axis+1:]) for axis in range(len(after)))
        shape = after
    span = 1 + sum((extent-1)*stride for extent,stride in zip(shape,physical,strict=True))
    if target_dtype != root_dtype or offset+span > prod(root_shape):
        raise SourceViewError('dense view storage range differs from its original root')
    pitched = not contiguous(shape,physical)
    if pitched and not (allow_row_pitch and len(shape) == 2 and physical[1] == 1
                        and physical[0] >= shape[1]):
        raise SourceViewError('only ordinary dense operands support physical row pitch')
    return DenseSourceView(current, target_shape, target_dtype, offset, tuple(operations),
                          physical[0] if pitched else None, span if pitched else None)
