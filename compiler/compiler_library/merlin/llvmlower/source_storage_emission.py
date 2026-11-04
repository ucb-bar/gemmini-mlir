"""Actual LLVM address closure for private storage lifetime proofs.

Address arithmetic is followed transitively, including ptrtoint -> integer
arithmetic -> inttoptr. Control-flow forwarding and opaque pointer escapes are
refused. This is deliberately separate from source SSA liveness: lazy views
and reordered device groups have already chosen their actual execution here.
"""
from dataclasses import dataclass


_ADDRESS_OPERATIONS = frozenset({
    'llvm.getelementptr', 'llvm.ptrtoint', 'llvm.inttoptr', 'llvm.bitcast',
    'llvm.add', 'llvm.sub', 'llvm.mul', 'llvm.and', 'llvm.or', 'llvm.xor',
    'llvm.shl', 'llvm.lshr', 'llvm.ashr', 'llvm.zext', 'llvm.sext', 'llvm.trunc',
})


@dataclass(frozen=True)
class EmittedPointerUses:
    terminals: tuple
    refusal: str | None


def emitted_pointer_uses(function, argument, terminal_effects, *, operation_indices=None):
    """Read every terminal of one actual ABI pointer, without mutating IR.

    ``terminal_effects`` are implementation-bound hashes witnessing no escape,
    no release and synchronous completion at the supplied enclosing emission
    boundary. An opaque terminal that returns SSA values is refused: its return
    may carry an address and a bare noescape effect does not rule that out.
    """
    if operation_indices is None:
        operation_indices = {operation: i for i, operation in enumerate(
            operation for block in function.body.blocks for operation in block.ops)}
    ordinal = operation_indices
    terminals, pending, seen = set(), [argument], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for use in current.uses:
            operation = use.operation
            reason = None
            if operation not in ordinal:
                reason = 'pointer_escapes_function'
            elif operation.name in _ADDRESS_OPERATIONS:
                pending.extend(operation.results)
            elif operation.name == 'llvm.load':
                # Pointer-valued loads need a separately proved memory alias
                # relation, which this typed tensor storage adapter does not own.
                from xdsl.dialects.llvm import LLVMPointerType
                if any(isinstance(result.type, LLVMPointerType) for result in operation.results):
                    reason = 'pointer_loaded_from_private_tensor'
                else:
                    terminals.add(operation)
            elif operation.name == 'llvm.store':
                if use.index != 1:
                    reason = 'pointer_stored_as_data'
                else:
                    terminals.add(operation)
            elif operation in terminal_effects and not operation.results:
                terminals.add(operation)
            else:
                reason = 'unknown_pointer_terminal_or_CFG_forwarding'
            if reason:
                return EmittedPointerUses((), reason)
    return EmittedPointerUses(tuple(sorted(terminals, key=ordinal.__getitem__)), None)
