"""Selected RISC-V completion transport, shared by kernel and ABI shim.

The public target's RISC-V toolchain and harness ABI select the architectural
``fence`` spelling (merlin.development.targetgen.contract.harness_abi.lowered_plan_abi).
The operand-free RISC-V pseudoinstruction assembles as ``fence iorw, iorw``:
all predecessor memory/device operations precede all successor operations.
It therefore includes the ordering of ``fence rw, rw`` while retaining the
selected structural decoder's explicit FENCE identity. The compiler memory
clobber and side-effect marker prevent motion or deletion at LLVM lowering.

This module emits the real instruction at each original completion site. It
neither hides effects in an uninspected call nor changes the checker contract.
"""
from xdsl.dialects import llvm

COMPLETION_SYMBOL = 'kernel_complete'
COMPLETION_ASM = 'fence'


def emit_fence(builder):
    """Append the selected side-effecting, memory-clobbering completion."""
    return builder.add(llvm.InlineAsmOp(
        COMPLETION_ASM, '~{memory}', [], has_side_effects=True,
    ))


def completion_function():
    """The exact external no-argument, fence-only harness completion shim."""
    from merlin.compiler.lower.model.host_builder import FnBuilder
    builder = FnBuilder([])
    emit_fence(builder)
    return llvm.FuncOp(COMPLETION_SYMBOL, llvm.LLVMFunctionType([]),
                       linkage=llvm.LinkageAttr('external'), body=builder.finish())
