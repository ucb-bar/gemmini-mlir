"""Scalar IR for an explicitly selected CPU flash-attention numerical policy.

This reproduces PyTorch v2.10.0 ``Vectorized<float>::fexp_u20`` on the
nonpositive softmax domain. It is not a general exponential approximation or
an automatic optimization. The policy describes source-backend arithmetic;
the emitted scalar IR has no dependence on the compilation target's ISA.

Source: pytorch/pytorch v2.10.0,
aten/src/ATen/cpu/vec/vec256/vec256_float.h, ``fexp_u20``.
Callers must preserve binary32 round-to-nearest-even, explicit FMA semantics,
and the absence of fast-math reassociation. NaN and positive inputs are outside
this builder's contract; finite nonpositive values and negative infinity are
supported. Underflow is selected before float-to-int conversion to avoid LLVM
poison from the source SIMD instruction's otherwise masked invalid conversion.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdsl.ir import Operation, SSAValue


CPU_FLASH_BF16_AVX2_POLICY = "torch_cpu_flash_bf16_avx2"


def build_cpu_flash_exp_nonpositive(
    value: SSAValue,
    *,
    backend_policy: str,
    domain: str,
) -> tuple[list[Operation], SSAValue]:
    """Return detached scalar operations and their f32 result.

    ``domain='nonpositive'`` is the caller's explicit proof obligation, normally
    established by subtracting a finite row maximum from softmax logits. This
    routine does not infer runtime ranges or silently change ``math.exp``.
    """
    from xdsl.dialects import arith, math
    from xdsl.dialects.builtin import FloatAttr, f32, i32

    if backend_policy != CPU_FLASH_BF16_AVX2_POLICY:
        raise ValueError("CPU flash exponential requires the explicit bf16 AVX2 source policy")
    if domain != "nonpositive":
        raise ValueError("CPU flash exponential requires the nonpositive domain contract")
    if value.type != f32:
        raise TypeError("CPU flash exponential requires scalar f32 input")

    ops: list[Operation] = []

    def emit(op):
        ops.append(op)
        return op.results[0]

    def constant(x: float):
        return emit(arith.ConstantOp(FloatAttr(x, f32)))

    def from_bits(bits: int):
        return constant(struct.unpack("!f", struct.pack("!I", bits))[0])

    zero = constant(0.0)
    underflow = emit(arith.CmpfOp(value, from_bits(0xC2AEAC50), "olt"))
    safe = emit(arith.SelectOp(underflow, zero, value))
    scaled = emit(arith.MulfOp(safe, from_bits(0x3FB8AA3B)))
    fraction = emit(arith.SubfOp(scaled, emit(math.FloorOp(scaled))))
    polynomial = emit(math.FmaOp(fraction, constant(-0.079204240219773236), constant(-0.22433836478672356)))
    polynomial = emit(math.FmaOp(fraction, polynomial, constant(0.30354260500649682)))
    polynomial = emit(math.FmaOp(fraction, polynomial, constant(0.00010703434948458272)))
    adjusted = emit(arith.SubfOp(scaled, polynomial))
    encoded = emit(math.FmaOp(constant(8388608.0), adjusted, constant(1065353216.0)))
    bits = emit(arith.FPToSIOp(encoded, i32))
    approximation = emit(arith.BitcastOp(bits, f32))
    result = emit(arith.SelectOp(underflow, zero, approximation))
    return ops, result
