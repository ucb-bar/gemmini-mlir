"""Exact i8 and opaque 32-bit layout changes using accelerator DMA copies.

The frontend proves reshape reassociation, static extraction, or transpose
indexing. CPU operations compute addresses and control only; every output value
is loaded into the accumulator and read back by Gemmini.
"""
from __future__ import annotations

from math import prod
from xdsl.dialects.builtin import IntegerAttr, i8, i32, i64
from xdsl.dialects import llvm
from . import isa


def _extent(value, label, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{label} must be a static {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _shape(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a static tensor shape")
    return tuple(_extent(x, label) for x in value)


def _transfer(f, source, dest, source_index, dest_index, ty, cols, rows):
    isa.mvin(f, f.gep(source, source_index, ty), 0x80000000, cols, rows)
    isa.mvout(f, f.gep(dest, dest_index, ty),
              0x80000000 if ty == i8 else 0xA0000000, cols, rows)


def _contiguous(f, source, dest, source_base, dest_base, ty, count):
    """Copy a contiguous interval; caller configures 16-element DMA strides."""
    def tile(index, cols, rows):
        _transfer(f, source, dest, f.add_i(source_base, index),
                  f.add_i(dest_base, index), ty, cols, rows)
    full = count // 256 * 256
    f.for_loop(0, full, 256, lambda index: tile(index, 16, 16), retain_loop=True)
    rest = count - full
    if rest // 16:
        tile(f.const(full), 16, rest // 16)
    if rest % 16:
        tile(f.const(full + rest // 16 * 16), rest % 16, 1)


def _row_strides(shape):
    return tuple(prod(shape[i + 1:]) for i in range(len(shape)))


def _indexed(f, source, dest, source_offset, dest_offset, shape,
             source_strides, dest_strides, ty, width):
    prefix = len(shape) - 1
    inner = shape[-1] if shape else 1
    ss = source_strides[-1] if source_strides else 1
    ds = dest_strides[-1] if dest_strides else 1
    while (prefix > 0 and source_strides[prefix - 1] == ss * inner
           and dest_strides[prefix - 1] == ds * inner):
        prefix -= 1
        inner *= shape[prefix]
    if ss * width >= 1 << 64 or ds * width >= 1 << 32:
        raise ValueError("layout DMA byte stride exceeds its register width")
    contiguous = ss == ds == 1
    isa.config_ld(f, 16 * width if contiguous else ss * width,
                  state=0, shrink=ty == i8)
    isa.config_st(f, 16 * width if contiguous else ds * width, scale=1.0, activation=0)
    outer_shape = shape[:prefix]
    def outer(index):
        source_base, dest_base = f.const(source_offset), f.const(dest_offset)
        for axis, extent in enumerate(outer_shape):
            divisor = prod(outer_shape[axis + 1:])
            coordinate = f.urem_i(f.udiv_i(index, f.const(divisor)), f.const(extent))
            source_base = f.add_i(source_base, f.mul_i(coordinate, f.const(source_strides[axis])))
            dest_base = f.add_i(dest_base, f.mul_i(coordinate, f.const(dest_strides[axis])))
        if contiguous:
            _contiguous(f, source, dest, source_base, dest_base, ty, inner)
        else:
            def chunk(position, rows):
                si = f.add_i(source_base, f.mul_i(position, f.const(ss)))
                di = f.add_i(dest_base, f.mul_i(position, f.const(ds)))
                _transfer(f, source, dest, si, di, ty, 1, rows)
            full = inner // 16 * 16
            f.for_loop(0, full, 16, lambda position: chunk(position, 16), retain_loop=True)
            if inner % 16:
                chunk(f.const(full), inner % 16)
    f.for_loop(0, prod(outer_shape), 1, outer, retain_loop=True)


def _strides(desc, key, shape):
    strides = desc[key]
    if not isinstance(strides, (list, tuple)) or len(strides) != len(shape):
        raise ValueError(f"{key} must align with iteration axes")
    return tuple(_extent(x, key, allow_zero=True) for x in strides)


def _bounded(offset, shape, strides, capacity, label):
    if offset + sum((n - 1) * s for n, s in zip(shape, strides)) >= capacity:
        raise ValueError(f"layout {label} accesses outside its declared tensor")


def _disjoint(shape, strides):
    # Sufficient exact separation proof for row-major, permuted, and sliced
    # rectangular destinations. Overlapping writes cannot claim layout purity.
    span = 1
    for step, extent in sorted((s, n) for n, s in zip(shape, strides) if n > 1):
        if step < span:
            raise ValueError("layout destination mapping has overlapping or unproved writes")
        span += (extent - 1) * step


def _literal(ctx, bits, value):
    """Intern one immutable raw scalar; no invocation-time literal stores."""
    raw = value & ((1 << bits) - 1)
    key = bits, raw
    cache = getattr(ctx, "_layout_literal_globals", None)
    if cache is None:
        cache = ctx._layout_literal_globals = {}
    if key in cache:
        return cache[key]
    ty = i8 if bits == 8 else i32
    # i32 and f32 descriptors share this raw-bit storage when their payloads
    # match. The full-width DMA path does not interpret its arithmetic type.
    symbol = f"layout_literal_{bits}_{raw:x}"
    ctx.globals.append(llvm.GlobalOp(ty, symbol, "internal", constant=True,
                                    value=IntegerAttr(value, ty), alignment=64))
    address = llvm.AddressOfOp(symbol, llvm.LLVMPointerType())
    address.attributes["merlin.global_task"] = IntegerAttr(-1, i64)
    entry = ctx.f.entry
    if entry.last_op is not None and entry.last_op.has_trait(llvm.IsTerminator):
        entry.insert_op_before(address, entry.last_op)
    else:
        entry.add_op(address)
    cache[key] = address.results[0]
    return cache[key]


def _fill(ctx, command, desc):
    if not isinstance(desc, dict) or desc.get("schema") != "constant_layout_fill_v1":
        raise ValueError("unsupported MOVEMENT layout_fill schema")
    shape = _shape(desc["result_shape"], "result_shape")
    dtype, value = desc["dtype"], desc["value"]
    if dtype not in ("i8", "i32", "f32") or type(value) is not int:
        raise ValueError("constant layout fill requires a known signed integer payload")
    bits = 8 if dtype == "i8" else 32
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError("layout fill scalar payload is outside the declared storage width")
    attrs = command.get("attributes") or {}
    if attrs.get("epilogue") or attrs.get("output_dtype", dtype) != dtype:
        raise ValueError("layout fill cannot apply a numeric epilogue or conversion")
    dst = command["operands"]["dst"]
    existing = ctx.cb["tensors"].get(dst)
    if existing is not None and (tuple(existing["shape"]) != shape or existing["dtype"] != dtype):
        raise ValueError("layout fill destination differs from tensor declaration")
    f, ty = ctx.f, i8 if dtype == "i8" else i32
    dest = ctx.alloc(dst, list(shape), dtype)
    # f32 literals arrive as signed i32 raw bits, shared with identical
    # i32 literals. Only real DMA reads this immutable scalar at runtime.
    scalar = _literal(ctx, bits, value)
    _indexed(f, scalar, dest, 0, 0, (prod(shape),), (0,), (1,), ty, bits // 8)
    isa.fence(f)
    return True


def lower_layout_movement(ctx, command):
    """Consume same-dtype i8/i32/f32 MOVEMENT attributes.layout_copy descriptors.

    contiguous_layout_copy_v1: source_offset_elements, elements, source_shape,
      result_shape; output is one contiguous source storage interval.
    strided_layout_copy_v1: source_offset_elements, source_shape, result_shape,
      source_strides_elements; output coordinate maps to offset + sum(i*s).
    indexed_layout_copy_v1 adds iteration_shape, destination_offset_elements,
      destination_strides_elements for exact insert/concat pieces.
    constant_layout_fill_v1 under attributes.layout_fill supplies one literal
      value, result_shape, and dtype; source storage is a known scalar.
    Strides are in elements and aligned with result/iteration axes.
    """
    if command.get("opcode") != "MOVEMENT":
        return False
    attrs = command.get("attributes") or {}
    if "layout_fill" in attrs:
        return _fill(ctx, command, attrs["layout_fill"])
    desc = attrs.get("layout_copy")
    if command.get("opcode") != "MOVEMENT" or desc is None:
        return False
    schemas = ("contiguous_layout_copy_v1", "strided_layout_copy_v1", "indexed_layout_copy_v1")
    if not isinstance(desc, dict) or desc.get("schema") not in schemas:
        raise ValueError("unsupported MOVEMENT layout_copy schema")
    if attrs.get("epilogue"):
        raise ValueError("layout copies cannot carry arithmetic epilogues")
    src, dst = command["operands"]["src"], command["operands"]["dst"]
    source = ctx.cb["tensors"][src]
    source_shape = _shape(desc["source_shape"], "source_shape")
    result_shape = _shape(desc["result_shape"], "result_shape")
    if source_shape != _shape(source["shape"], "declared source shape"):
        raise ValueError("layout copy source shape differs from tensor declaration")
    offset = _extent(desc["source_offset_elements"], "source_offset_elements", allow_zero=True)
    count = prod(result_shape)
    if desc["schema"] == schemas[0]:
        if _extent(desc["elements"], "elements") != count or offset + count > prod(source_shape):
            raise ValueError("layout copy interval is out of bounds or result size differs")
    else:
        iteration = (_shape(desc["iteration_shape"], "iteration_shape")
                     if desc["schema"] == schemas[2] else result_shape)
        strides = _strides(desc, "source_strides_elements", iteration)
        destination_offset = (_extent(desc["destination_offset_elements"], "destination offset", allow_zero=True)
                              if desc["schema"] == schemas[2] else 0)
        destination_strides = (_strides(desc, "destination_strides_elements", iteration)
                               if desc["schema"] == schemas[2] else _row_strides(result_shape))
        _bounded(offset, iteration, strides, prod(source_shape), "source")
        _bounded(destination_offset, iteration, destination_strides, count, "destination")
        _disjoint(iteration, destination_strides)
    dtype = source["dtype"]
    if dtype not in ("i8", "i32", "f32") or attrs.get("output_dtype", dtype) != dtype:
        raise ValueError("layout copies require identical i8, i32, or opaque f32 containers")
    dest = ctx.cb["tensors"].get(dst)
    if dest is not None and (dest["dtype"] != dtype or tuple(dest["shape"]) != result_shape):
        raise ValueError("layout copy destination differs from the declared result")
    f = ctx.f
    source_ptr, dest_ptr = ctx.ptr(src), ctx.alloc(dst, list(result_shape), dtype)
    # f32 storage uses i32 address elements and the exact full-width
    # accumulator load/readout path: no FP conversion, scale, or activation.
    ty, width = (i8, 1) if dtype == "i8" else (i32, 4)
    if desc["schema"] == schemas[0]:
        isa.config_ld(f, 16 * width, state=0, shrink=dtype == "i8")
        isa.config_st(f, 16 * width, scale=1.0, activation=0)
        _contiguous(f, source_ptr, dest_ptr, f.const(offset), f.const(0), ty, count)
    else:
        _indexed(f, source_ptr, dest_ptr, offset, destination_offset, iteration,
                 strides, destination_strides, ty, width)
    isa.fence(f)
    return True
