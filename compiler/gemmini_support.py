"""Module assembly and explicit private storage for the authored target."""
from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8
from xdsl.ir import Block, Region
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.xdsl_dialects._common import text


def scratch_buffer(fn, size: int, name='scratch'):
    """A disjoint private buffer for a synchronous single-hart invocation.

    Each site gets its own BSS object, rather than an unbounded kernel stack.
    All consumers initialize the bytes they read on every invocation. The
    selected harness calls kernels sequentially on one hart; this allocation
    does not promise a reentrant interface.
    """
    if type(size) is not int or size < 1 or size >= 1 << 40:
        raise ValueError('private storage needs a positive bounded byte count')
    globals_ = getattr(fn, '_gemmini_globals', None)
    if globals_ is None:
        globals_ = fn._gemmini_globals = []
    symbol = f'gemmini_private_{len(globals_)}_{name}'
    ty = llvm.LLVMArrayType(size, i8)
    zero = llvm.ZeroOp(result_types=[ty])
    body = Region(Block([zero, llvm.ReturnOp(zero.results[0])]))
    glob = llvm.GlobalOp(ty, symbol, 'internal', alignment=64, body=body)
    globals_.append(glob)
    return fn.add(llvm.AddressOfOp(symbol, PTR)).results[0]


def selected_entry_symbol():
    import yaml
    from pathlib import Path
    contract = yaml.safe_load((Path(__file__).resolve().parent/'data/target_contract.yaml').read_text())
    return contract['harness_abi']['entry_symbol']


def completion_function():
    """Export the selected harness completion ABI with the common target fence."""
    import yaml
    from pathlib import Path
    import gemmini_encode as enc
    contract = yaml.safe_load((Path(__file__).resolve().parent/'data/target_contract.yaml').read_text())
    symbol = contract['harness_abi']['fence_symbol']
    fn = FnBuilder([])
    enc.fence(fn)
    return llvm.FuncOp(symbol, llvm.LLVMFunctionType([]),
                      linkage=llvm.LinkageAttr('external'), body=fn.finish())



def ensure_entry_fence(fn):
    """Start every invocation with an unconditional accelerator completion fence.

    This separates reused accelerator storage from effects of the preceding
    invocation. Return the actual entry operation so scheduling adapters can
    retain its provenance and instruction-count ownership. Repeated calls are
    idempotent when the entry already starts with the canonical fence.
    """
    import gemmini_encode as enc
    first = fn.entry.first_op
    if (isinstance(first, llvm.InlineAsmOp)
            and first.asm_string.data.strip() == 'fence'):
        return first
    operation = enc.fence_op()
    if first is None:
        fn.entry.add_op(operation)
    else:
        fn.entry.insert_op_before(operation, first)
    return operation


def finish_module(fn, symbol=None):
    ensure_entry_context(fn)
    symbol = symbol or selected_entry_symbol()
    function = llvm.FuncOp(symbol, llvm.LLVMFunctionType([a.type for a in fn.entry.args]),
                           linkage=llvm.LinkageAttr('external'), body=fn.finish())
    module = ModuleOp([*getattr(fn, '_gemmini_globals', []), function, completion_function()])
    from gemmini_command_expr import simplify_command_expressions
    simplify_command_expressions(module)
    module.verify()
    return text(module)


def ensure_entry_context(fn):
    """Complete old DMA and refresh translations for this call's tensor pointers.

    External tensor pointers belong to each invocation's address context. A
    completion fence precedes FLUSH because FLUSH bypasses the device queues.
    Neither operation evicts scratchpad data; resident weights remain live for
    all consumers inside this invocation. Return both actual instructions for
    the scheduling adapter's provenance and command-count ownership.

    Encoding: selected target_contract instructions FLUSH_CMD/funct=7 and
    rtl_facts funct_decode_table. Operand semantics: the public Gemmini ISA,
    https://github.com/ucb-bar/gemmini#flush-flushes-the-tlb (rs1=0 retries).
    """
    import gemmini_encode as enc
    first = ensure_entry_fence(fn)
    second = first.next_op
    spelling = '.insn r 0x7b, 3, 7, x0, x0, x0'
    if (isinstance(second, llvm.InlineAsmOp)
            and second.asm_string.data.strip() == spelling):
        return first, second
    operation = enc.flush_op()
    fn.entry.insert_op_after(operation, first)
    return first, operation
