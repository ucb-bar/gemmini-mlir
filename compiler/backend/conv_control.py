"""Typed convolution control including statically peeled source tails.

Shared gather and source-order emitters compose comparisons with logical AND.
Comparisons can fold to Python bools for a statically peeled tail; the generic
LLVM adapter deliberately accepts only SSA i1 operands. Resolve Boolean
identities before delegating genuine SSA conjunctions unchanged.
"""
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter


class ConvolutionControlAdapter(FnBuilderControlAdapter):
    def op(self, opcode, lhs, rhs, dtype="i64"):
        if opcode == "and" and dtype == "i1":
            if type(lhs) is bool:
                return rhs if lhs else False
            if type(rhs) is bool:
                return lhs if rhs else False
        return super().op(opcode, lhs, rhs, dtype)
