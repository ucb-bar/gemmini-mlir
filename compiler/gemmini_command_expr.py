"""Exact terminal LLVM integer folding and block-local upstream CSE.

Runs after source/effect scheduling and before printing LLVM. This is a terminal
representation cleanup, not a scheduling pass: no memory operation, command,
control edge, completion, or task instruction ownership can be removed or moved.
Unknown properties and poison-producing arithmetic flags are outside the domain.
"""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.transforms.common_subexpression_elimination import CSEDriver
from merlin.llvmlower.host_builder import iconst

_BINARY = (llvm.AddOp, llvm.SubOp, llvm.MulOp, llvm.AndOp, llvm.OrOp, llvm.XOrOp,
           llvm.ShlOp, llvm.LShrOp, llvm.AShrOp)


def _constant(value):
    owner = value.owner
    if getattr(owner, 'name', None) != 'llvm.mlir.constant':
        return None
    literal = owner.properties.get('value')
    if not isinstance(literal, IntegerAttr) or not isinstance(value.type, IntegerType):
        return None
    return literal.value.data & ((1 << value.type.width.data) - 1)


def _eligible(op):
    if type(op) not in (*_BINARY, llvm.ICmpOp, llvm.SelectOp):
        return False
    if op.regions or op.successors or len(op.results) != 1:
        return False
    if any(not isinstance(v.type, IntegerType) for v in (*op.operands, *op.results)):
        return False
    if any(not k.startswith('merlin.') for k in op.attributes):
        return False
    allowed = {'overflowFlags'} if type(op) in _BINARY else {'predicate'} if type(op) is llvm.ICmpOp else {'fastmathFlags'}
    if set(op.properties) - allowed:
        return False
    overflow = op.properties.get('overflowFlags')
    if overflow is not None and (not isinstance(overflow, IntegerAttr) or overflow.value.data != 0):
        return False
    fastmath = op.properties.get('fastmathFlags')
    return fastmath is None or not fastmath.data


class _IntegerCSE(CSEDriver):
    def _simplify_operation(self, op):
        if _eligible(op):
            # Upstream equivalence includes types, properties, attributes and
            # SSA operands. A fresh driver per block proves local dominance.
            super()._simplify_operation(op)


def _fold(op):
    values = tuple(_constant(v) for v in op.operands)
    width = op.results[0].type.width.data
    mask = (1 << width) - 1
    if type(op) is llvm.SelectOp:
        cond, a, b = op.operands
        return (a if values[0] else b) if values[0] is not None else None
    if type(op) is llvm.ICmpOp:
        if any(v is None for v in values):
            return None
        a, b = values
        operand_width = op.operands[0].type.width.data
        signed = lambda x: x - (1 << operand_width) if x >> (operand_width - 1) else x
        predicate = op.properties['predicate'].value.data
        tests = {0: a == b, 1: a != b, 2: signed(a) < signed(b),
                 3: signed(a) <= signed(b), 4: signed(a) > signed(b),
                 5: signed(a) >= signed(b), 6: a < b, 7: a <= b,
                 8: a > b, 9: a >= b}
        return int(tests[predicate]) if predicate in tests else None
    a, b = op.operands
    x, y = values
    cls = type(op)
    if cls in (llvm.AddOp, llvm.OrOp, llvm.XOrOp):
        if x == 0: return b
        if y == 0: return a
    if cls is llvm.SubOp and y == 0: return a
    if cls is llvm.MulOp:
        if x == 1: return b
        if y == 1: return a
    if cls is llvm.AndOp:
        if x == mask: return b
        if y == mask: return a
    if cls in (llvm.ShlOp, llvm.LShrOp, llvm.AShrOp) and y == 0: return a
    # Constant-only folding preserves modular width. No x*0 or x&0 rule:
    # such rules could erase poison from an unknown SSA producer.
    if x is None or y is None: return None
    if cls is llvm.AddOp: return (x + y) & mask
    if cls is llvm.SubOp: return (x - y) & mask
    if cls is llvm.MulOp: return (x * y) & mask
    if cls is llvm.AndOp: return x & y
    if cls is llvm.OrOp: return x | y
    if cls is llvm.XOrOp: return x ^ y
    if y >= width: return None  # LLVM overshift is poison, never invent a value.
    if cls is llvm.ShlOp: return (x << y) & mask
    if cls is llvm.LShrOp: return x >> y
    if cls is llvm.AShrOp:
        signed = x - (1 << width) if x >> (width - 1) else x
        return (signed >> y) & mask
    return None


def simplify_command_expressions(module):
    """Fold total integer expressions and CSE within each original CFG block.

    Preconditions: verified terminal LLVM SSA. Postconditions: same CFG and
    ordered effect operations, with equivalent command operand bitvectors.
    No cached analysis survives this final representation-only cleanup.
    """
    module.verify()
    folded = 0
    for parent in tuple(module.walk()):
        for region in parent.regions:
            for block in region.blocks:
                constants = {}
                for op in tuple(block.ops):
                    for value in op.operands:
                        literal = _constant(value)
                        if literal is not None:
                            constants.setdefault((value.type, literal), value)
                    if not _eligible(op): continue
                    replacement = _fold(op)
                    if replacement is None and type(op) is llvm.OrOp:
                        # Associativity combines constant packing suffixes:
                        # (x | C1) | C2 == x | (C1 | C2), with the same x.
                        # Dynamic masks and poison propagation stay intact.
                        outer, literal = op.operands
                        suffix = _constant(literal)
                        inner = outer.owner
                        if suffix is not None and type(inner) is llvm.OrOp and _eligible(inner):
                            base, previous = inner.operands
                            prefix = _constant(previous)
                            if prefix is not None:
                                combined = prefix | suffix
                                key = (literal.type, combined)
                                if key not in constants:
                                    bits = literal.type.width.data
                                    signed = combined - (1 << bits) if combined >= 1 << (bits - 1) else combined
                                    constant = iconst(signed, literal.type)
                                    constant.attributes.update(op.attributes)
                                    block.insert_op_before(constant, op)
                                    constants[key] = constant.results[0]
                                merged = llvm.OrOp(base, constants[key], attributes=dict(op.attributes))
                                block.insert_op_before(merged, op)
                                replacement = merged.results[0]
                    if replacement is None: continue
                    if isinstance(replacement, int):
                        width = op.results[0].type.width.data
                        signed = replacement - (1 << width) if replacement >= 1 << (width - 1) else replacement
                        key = (op.results[0].type, replacement)
                        if key in constants:
                            replacement = constants[key]
                        else:
                            constant = iconst(signed, op.results[0].type)
                            constant.attributes.update(op.attributes)
                            block.insert_op_before(constant, op)
                            replacement = constants[key] = constant.results[0]
                    op.results[0].replace_all_uses_with(replacement)
                    block.erase_op(op)
                    folded += 1
                _IntegerCSE().simplify(block)
                # Erase only newly dead closed integer chains; every effect,
                # branch and unknown operation is retained as a liveness root.
                for op in reversed(tuple(block.ops)):
                    if _eligible(op) and not op.results[0].uses:
                        block.erase_op(op)
    module.verify()
    return {'folded_integer_expressions': folded,
            'scope': 'terminal LLVM exact integer folding and block-local upstream CSE'}
