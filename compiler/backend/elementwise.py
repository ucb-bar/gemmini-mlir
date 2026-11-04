"""Standalone tensor movement and additions using Gemmini's load adder.

Scalar LLVM work is address/control arithmetic only. All data values travel
through the accumulator, including the broadcast bias addition.
"""
from __future__ import annotations

from functools import lru_cache
from math import isfinite, prod
import struct

from xdsl.dialects.builtin import i8, i32

from .isa import DIM, config_ld, config_st, fence, mvin, mvout

_ACC = 0x80000000
_ACC_ADD = 0xC0000000
_ACC_FULL = 0xA0000000
_TYPES = {"i8": i8, "i32": i32}


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _sat8(x):
    return min(127, max(-128, x))


@lru_cache(maxsize=256)
def _check_residual(lhs_scale, rhs_scale, bound, relu):
    """Choose a numerically qualified scale factorization over all i8 pairs.

    Factoring a common scale into readout keeps operand loads from saturating
    when a source multiplier exceeds one. Every candidate is independently
    checked against the source's once-rounded f32 reference.
    """
    if not all(isfinite(x) for x in (lhs_scale, rhs_scale)) or bound < 0:
        raise ValueError("residual scales must be finite and bound_lsb nonnegative")
    lp = [_f32(x * lhs_scale) for x in range(-128, 128)]
    rp = [_f32(x * rhs_scale) for x in range(-128, 128)]
    common = max(1.0, abs(lhs_scale), abs(rhs_scale))
    candidates = [(lhs_scale, rhs_scale, 1.0)]
    if common != 1.0:
        candidates.append((_f32(lhs_scale / common), _f32(rhs_scale / common), common))
    for lhs_load, rhs_load, readout in candidates:
        ll = [_sat8(round(_f32(x * lhs_load))) for x in range(-128, 128)]
        rl = [_sat8(round(_f32(x * rhs_load))) for x in range(-128, 128)]
        admitted = True
        for ai, a in enumerate(lp):
            for bi, b in enumerate(rp):
                ref = _sat8(round(_f32(a + b)))
                acc = ll[ai] + rl[bi]
                # The target applies ReLU before readout scaling. Readout is
                # positive, so this equals the source's post-saturation ReLU.
                if relu:
                    acc = max(0, acc)
                    ref = max(0, ref)
                got = _sat8(round(_f32(acc * readout)))
                if abs(ref - got) > bound:
                    admitted = False
                    break
            if not admitted:
                break
        if admitted:
            return lhs_load, rhs_load, readout
    raise ValueError("residual operand-load path exceeds declared bound_lsb")


def _tiles(f, rows, cols, body):
    """Use compact ordinary loops for complete tiles and static tail extents."""
    def row_block(row, nr):
        def full_col(col):
            body(row, col, nr, DIM)
        f.for_loop(0, cols // DIM * DIM, DIM, full_col, retain_loop=True)
        if cols % DIM:
            body(row, f.const(cols // DIM * DIM), nr, cols % DIM)
    f.for_loop(0, rows // DIM * DIM, DIM,
               lambda row: row_block(row, DIM), retain_loop=True)
    if rows % DIM:
        row_block(f.const(rows // DIM * DIM), rows % DIM)


def lower_elementwise(ctx, command):
    """Lower MOVEMENT, RESIDUAL_ADD, or BIAS_ADD; return True if handled.

    ctx supplies f: FnBuilder, cb: command buffer, ptr(name), and
    alloc(name, shape, dtype). All tensor pointers use contiguous row-major ABI.
    """
    opcode = command.get("opcode")
    if opcode not in ("MOVEMENT", "RESIDUAL_ADD", "BIAS_ADD"):
        return False
    if opcode == "MOVEMENT" and any(key in (command.get("attributes") or {})
                                    for key in ("layout_copy", "layout_fill")):
        from .layout_movement import lower_layout_movement
        return lower_layout_movement(ctx, command)
    ops, attrs = command.get("operands", {}), command.get("attributes", {})
    tensors = ctx.cb["tensors"]
    src = ops["lhs"] if opcode == "RESIDUAL_ADD" else ops["src"]
    spec = tensors[src]
    shape = tuple(int(x) for x in spec["shape"])
    if not shape or any(x <= 0 for x in shape):
        raise ValueError("elementwise requires a nonempty positive shape")
    cols, rows = shape[-1], prod(shape[:-1])
    src_dtype = spec["dtype"]
    dst = ops["dst"]
    out_dtype = attrs.get("output_dtype", tensors.get(dst, {}).get("dtype", src_dtype))
    if src_dtype not in _TYPES or out_dtype not in _TYPES:
        raise ValueError("elementwise target supports i8 and i32 containers")
    if dst in tensors:
        if tuple(tensors[dst]["shape"]) != shape or tensors[dst]["dtype"] != out_dtype:
            raise ValueError("elementwise destination type differs from source contract")
    stages = list(attrs.get("epilogue", []) or [])
    if opcode == "MOVEMENT":
        if stages or (src_dtype, out_dtype) not in (("i8", "i8"), ("i8", "i32"), ("i32", "i32")):
            raise ValueError("movement must preserve values without narrowing")
    elif opcode == "RESIDUAL_ADD":
        rhs = ops["rhs"]
        if src_dtype != "i8" or out_dtype != "i8" or tensors[rhs]["dtype"] != "i8" or tuple(tensors[rhs]["shape"]) != shape:
            raise ValueError("residual target path requires equal-shaped i8 tensors")
        if stages not in ([], ["relu"]):
            raise ValueError("residual admits only an optional relu")
        ls, rs = _f32(attrs["lhs_scale"]), _f32(attrs["rhs_scale"])
        bound = attrs["bound_lsb"]
        if not isinstance(bound, int):
            raise ValueError("bound_lsb must be an integer")
        ls, rs, readout = _check_residual(ls, rs, bound, bool(stages))
    else:
        bias = ops["bias"]
        if src_dtype != "i32" or out_dtype != "i32" or tensors[bias]["dtype"] != "i32" or tuple(tensors[bias]["shape"]) != (cols,):
            raise ValueError("bias addition requires i32 source and trailing-axis i32 vector")
        if any(x != "bias_add" for x in stages):
            raise ValueError("standalone bias_add does not apply requantization or activation")
    f = ctx.f
    src_ptr, dst_ptr = ctx.ptr(src), ctx.alloc(dst, list(shape), out_dtype)
    src_bytes = 1 if src_dtype == "i8" else 4
    out_bytes = 1 if out_dtype == "i8" else 4
    config_ld(f, cols * src_bytes, state=0,
              scale=ls if opcode == "RESIDUAL_ADD" else 1.0,
              shrink=src_dtype == "i8")
    config_st(f, cols * out_bytes, scale=readout if opcode == "RESIDUAL_ADD" else 1.0,
              activation=1 if opcode == "RESIDUAL_ADD" and stages else 0)
    if opcode == "RESIDUAL_ADD":
        other_ptr = ctx.ptr(rhs)
        config_ld(f, cols, state=1, scale=rs, shrink=True)
    elif opcode == "BIAS_ADD":
        other_ptr = ctx.ptr(bias)
        config_ld(f, 0, state=1, shrink=False)
    def tile(row, col, nr, nc):
        index = f.add_i(f.mul_i(row, f.const(cols)), col)
        mvin(f, f.gep(src_ptr, index, _TYPES[src_dtype]), _ACC, nc, nr)
        if opcode in ("RESIDUAL_ADD", "BIAS_ADD"):
            # Both transfers use the load queue. Its younger entries may
            # issue before older accumulator writes complete, so overwrite
            # and accumulate must have an explicit completion boundary.
            fence(f)
        if opcode == "RESIDUAL_ADD":
            mvin(f, f.gep(other_ptr, index, i8), _ACC_ADD, nc, nr, state=1)
        elif opcode == "BIAS_ADD":
            mvin(f, f.gep(other_ptr, col, i32), _ACC_ADD, nc, nr, state=1)
        mvout(f, f.gep(dst_ptr, index, _TYPES[out_dtype]),
              _ACC_FULL if out_dtype == "i32" else _ACC, nc, nr)
    _tiles(f, rows, cols, tile)
    fence(f)
    return True
