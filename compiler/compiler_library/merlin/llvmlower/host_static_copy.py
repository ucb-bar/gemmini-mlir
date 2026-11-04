"""Realize static logical copies through shared scalar and bulk-copy lowering.

This is the stock-tool adapter for ``expand_memref_copy``. It introduces no
arithmetic or pointer assumptions: original logical indices and memref layouts
are retained. Only distinct fresh allocation roots, or a fresh allocation and
an original function argument, establish disjointness. Unknown aliases stay on
the original runtime path. Contiguous copies retain upstream memcpy lowering;
common contiguous suffixes reuse the existing bounded row-copy contract.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from math import prod

from .copy_expand import FEATURE


def _root(value):
    from xdsl.ir import BlockArgument, Operation
    views = {'memref.subview', 'memref.cast', 'memref.collapse_shape',
             'memref.expand_shape', 'memref.reinterpret_cast', 'memref.view'}
    seen = set()
    while value not in seen:
        seen.add(value)
        if isinstance(value, BlockArgument):
            block = value.owner
            owner = block.parent_op()
            return ('argument', value) if owner is not None and owner.name == 'func.func' else None
        owner = value.owner
        if not isinstance(owner, Operation):
            return None
        if owner.name in ('memref.alloc', 'memref.alloca'):
            return ('allocation', owner)
        if owner.name not in views or not owner.operands:
            return None
        value = owner.operands[0]
    return None


def _contiguous_suffix(shape, source_type, destination_type):
    """Select the existing contiguous-suffix contract, with 64-bit bounds."""
    ss, ds = source_type.get_strides(), destination_type.get_strides()
    if ss is None or ds is None or any(s is None or s <= 0 for s in (*ss, *ds)):
        return None
    axis, span = len(shape), 1
    for i in reversed(range(len(shape))):
        if shape[i] != 1 and (ss[i] != span or ds[i] != span):
            break
        span *= shape[i]
        axis = i
    bits = getattr(source_type.element_type, 'bitwidth', None)
    if (axis == 0 or axis == len(shape) or span <= 1 or prod(shape) >= (1 << 63)
            or not isinstance(bits, int) or bits <= 0 or bits % 8
            or span * (bits // 8) >= (1 << 63)):
        return None
    return axis, span


def _suffix_copy(source, destination, shape, axis, location, attributes):
    """Emit outer logical loops and original-descriptor contiguous row copies."""
    from xdsl.dialects import arith, memref, scf
    from xdsl.dialects.builtin import DYNAMIC_INDEX, IndexType, MemRefType, StridedLayoutAttr
    from xdsl.ir import Block

    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    bounds = [arith.ConstantOp.from_int_and_width(n, IndexType()) for n in shape[:axis]]

    def emit(level, offsets):
        if level < axis:
            if shape[level] == 1:
                return emit(level + 1, [*offsets, zero.result])
            body = Block(arg_types=[IndexType()])
            body.add_ops(emit(level + 1, [*offsets, body.args[0]]))
            body.add_op(scf.YieldOp())
            return [scf.ForOp(zero, bounds[level], one, [], body)]
        rows = []
        for value in (source, destination):
            typ = value.type
            row_type = MemRefType(typ.element_type, shape[axis:],
                StridedLayoutAttr(typ.get_strides()[axis:], None), typ.memory_space)
            rows.append(memref.SubviewOp(value, offsets, [], [],
                [DYNAMIC_INDEX] * axis + [0] * (len(shape) - axis),
                [1] * axis + list(shape[axis:]), [1] * len(shape), row_type))
        copy = memref.CopyOp(rows[0], rows[1])
        copy.attributes.update(attributes)
        return [*rows, copy]

    result = [zero, one, *bounds, *emit(0, [])]
    for op in result:
        for child in op.walk():
            child.location = location
    return result


def expand_text(source: str):
    """Lower proved-disjoint static strided copies into bounded logical copies.

    Bufferized memrefs carry the actual source view offsets/strides, including
    dynamically computed offsets. The replacement visits exactly the same
    bounded logical shape and performs only a same-type load and store. No
    signed/unsigned or floating conversion is introduced. Unknown roots and
    potentially overlapping views remain unchanged.
    """
    from xdsl.dialects.builtin import MemRefType
    from xdsl.dialects.linalg.ops import CopyOp
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    module = parse_mlir_text(source)
    counts, rows = Counter(), []
    for op in tuple(module.walk()):
        if op.name != 'memref.copy':
            continue
        counts['copies_seen'] += 1
        a, b = op.operands
        at, bt = a.type, b.type
        if (not isinstance(at, MemRefType) or not isinstance(bt, MemRefType)
                or at.get_shape() != bt.get_shape() or not at.get_shape()
                or any(n <= 0 for n in at.get_shape())
                or at.element_type != bt.element_type or at.memory_space != bt.memory_space):
            counts['refused_shape_or_type'] += 1
            continue
        # A fully contiguous copy already becomes a stock memcpy. Preserve that
        # bulk route rather than replacing it with a scalar element loop.
        def contiguous(typ):
            strides = typ.get_strides()
            if strides is None:
                return False
            span = 1
            for extent, stride in reversed(tuple(zip(typ.get_shape(), strides))):
                if extent != 1 and stride != span:
                    return False
                span *= extent
            return True
        if contiguous(at) and contiguous(bt):
            counts['retained_contiguous'] += 1
            continue
        roots = _root(a), _root(b)
        if (None in roots or roots[0] == roots[1]
                or not any(root[0] == 'allocation' for root in roots)):
            counts['refused_alias'] += 1
            continue
        suffix = _contiguous_suffix(at.get_shape(), at, bt)
        if suffix is not None:
            axis, span = suffix
            replacement = _suffix_copy(a, b, at.get_shape(), axis, op.location, op.attributes)
            Rewriter.replace_op(op, replacement)
            counts['copies_specialized_contiguous_suffix'] += 1
            mode = dict(kind='contiguous_suffix', suffix_axis=axis,
                elements_per_bulk_copy=span, bulk_copy_count=prod(at.get_shape()[:axis]))
        else:
            replacement = CopyOp([a], [b], res=[])
            replacement.attributes.update(op.attributes)
            replacement.location = op.location
            Rewriter.replace_op(op, replacement)
            counts['copies_scalarized'] += 1
            mode = dict(kind='scalar_logical_copy')
        counts['copies_expanded'] += 1
        rows.append(dict(shape=list(at.get_shape()), dtype=str(at.element_type), mode=mode,
            elements=prod(at.get_shape()), source_type=str(at), destination_type=str(bt),
            alias_basis=[root[0] for root in roots],
            semantics='same logical indices, original offset/stride descriptors, same-type load/store'))
    selected = text(module, generic=True)
    return selected, dict(schema='merlin.stock_static_copy.v1', feature=FEATURE,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        selected_sha256=hashlib.sha256(selected.encode()).hexdigest(), counts=dict(counts),
        copies=rows, contiguous_suffix_contract='merlin.llvmlower.contiguous_suffix_copy',
        formal_proof='not_issued', timing='UNKNOWN')
