"""Typed xDSL linalg.matmul for i8×i8→i32 integer rewrites.

xDSL's stock named matmul synthesizes its hidden body at the input type: an
i8 product is added directly to the i32 accumulator, failing verification.
This local dialect replacement widens both signed inputs before multiply and
add, matching the integer contraction's stated result type.  Other matmul
types keep xDSL's ordinary construction and verifier.
"""

from __future__ import annotations

from xdsl.dialects import arith, builtin, linalg
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, Dialect, Region


class MixedMatmulOp(L.MatmulOp):
    name = "linalg.matmul"

    @classmethod
    def get_hidden_region(cls, inputs, outputs) -> Region:
        types = cls.body_arg_types((*inputs, *outputs))
        if (len(types) != 3 or types[0] != builtin.i8 or
                types[1] != builtin.i8 or types[2] != builtin.i32):
            return super().get_hidden_region(inputs, outputs)
        block = Block(arg_types=types)
        lhs = arith.ExtSIOp(block.args[0], builtin.i32)
        rhs = arith.ExtSIOp(block.args[1], builtin.i32)
        product = arith.MuliOp(lhs, rhs, builtin.i32)
        summation = arith.AddiOp(product, block.args[2], builtin.i32)
        block.add_ops((lhs, rhs, product, summation, L.YieldOp(summation)))
        return Region(block)


LINALG_WITH_MIXED_MATMUL = Dialect(
    "linalg",
    [MixedMatmulOp if op is L.MatmulOp else op for op in linalg.Linalg.operations],
    list(linalg.Linalg.attributes),
)
