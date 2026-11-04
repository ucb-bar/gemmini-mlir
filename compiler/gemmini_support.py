"""Module assembly and explicit private storage for the authored target."""
from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, ArrayAttr, StringAttr, SymbolRefAttr, i8
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
    module = ModuleOp([*getattr(fn, '_gemmini_globals', []),
                       *getattr(fn, '_gemmini_helpers', []), function, completion_function()])
    from gemmini_command_expr import simplify_command_expressions
    simplify_command_expressions(module)
    module.verify()
    return text(module)


def register_internal_helper(parent, helper, *, symbol_prefix='horizontal_phase'):
    """Keep a caller-selected void region out of the parent's register lifetime.

    This only assembles functions; the caller retains responsibility for source
    ownership, argument order and device completions. No entry FLUSH or fence is
    inserted in the helper. Its original effects and CFG survive, and noinline
    prevents upstream optimization from recreating the oversized parent frame.
    Private globals are renamed into this helper's namespace before registration.
    """
    import re
    if parent is helper or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol_prefix):
        raise ValueError('internal helper requires a distinct builder and symbol prefix')
    if getattr(helper, '_gemmini_registered', False) or getattr(helper, '_gemmini_helpers', []):
        raise ValueError('internal helper must be fresh and cannot contain registered helpers')
    if (helper.region.parent is not None
            or parent.blk.last_op is not None and parent.blk.last_op.has_trait(llvm.IsTerminator)
            or helper.blk.last_op is not None and helper.blk.last_op.has_trait(llvm.IsTerminator)):
        raise ValueError('internal helper registration requires detached region and open caller')
    operations = tuple(helper.region.walk())
    defined = {arg for block in helper.region.blocks for arg in block.args}
    defined.update(result for op in operations for result in op.results)
    if any(op.regions for op in operations) or any(
            value not in defined for op in operations for value in op.operands):
        raise ValueError('internal helper must be a closed flat CFG without captured SSA')
    helpers = getattr(parent, '_gemmini_helpers', [])
    symbol = f'gemmini_{symbol_prefix}_{len(helpers)}'
    globals_ = tuple(getattr(helper, '_gemmini_globals', []))
    names = {g.sym_name.data: f'{symbol}_private_{i}' for i, g in enumerate(globals_)}
    if len(names) != len(globals_):
        raise ValueError('internal helper has duplicate private globals')
    for g in globals_:
        if g.linkage.linkage.data != 'internal':
            raise ValueError('internal helper globals must have private linkage')
    # Finish exactly once, preserving the original closed CFG and effects.
    region = helper.finish()
    function = llvm.FuncOp(symbol, llvm.LLVMFunctionType([a.type for a in helper.entry.args]),
        linkage=llvm.LinkageAttr('internal'), body=region,
        other_props={'passthrough': ArrayAttr([StringAttr('noinline')])})
    for g in globals_:
        g.properties['sym_name'] = StringAttr(names[g.sym_name.data])
    for op in function.walk():
        if isinstance(op, llvm.AddressOfOp) and op.global_name.root_reference.data in names:
            op.properties['global_name'] = SymbolRefAttr(names[op.global_name.root_reference.data])
    # Calls to an earlier registered helper resolve in finish_module's symbol
    # table. Verify the function shell here and the complete body in that module.
    function.verify(verify_nested_ops=False)
    parent._gemmini_helpers = [*helpers, function]
    parent._gemmini_globals = [*getattr(parent, '_gemmini_globals', []), *globals_]
    helper._gemmini_registered = True
    return symbol


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
