"""Emit source configuration words as constants and index selection, without loads.

The words are immutable compiler inputs, never tensor payload. Keep their exact
integer bits through selection; the ordinary target encoder consumes the result.
The caller proves that the source index is inside the original word vector.
"""
from functools import lru_cache


@lru_cache(maxsize=128)
def _selection_plan(words):
    """Compress exact affine subranges only within the old tree's node budget.

    Shared index expressions supply canonical zero/unit arithmetic and exact
    evaluation. The convolution gather-run planner requires spatial geometry,
    so immutable word vectors need this small source-domain adapter instead.
    A candidate pays for every arithmetic operation and distinct constant;
    the old comparison/select tree pays only for its nonconstant operations.
    This conservative admission cannot increase the emitted scalar node count.
    """
    from merlin.sched.ir.expr import Const, Var, Sum, Scaled, evaluate

    def expression_cost(expr):
        constants = set()
        def visit(e):
            if isinstance(e, Const):
                constants.add(e.value)
                return 0
            if isinstance(e, Var):
                return 0
            if isinstance(e, Scaled):
                constants.add(e.factor)
                return 1 + visit(e.expr)
            if isinstance(e, Sum):
                return len(e.terms) - 1 + sum(visit(t) for t in e.terms)
            raise ValueError('unsupported source word affine expression')
        operations = visit(expr)
        return operations + len(constants)

    def plan(first, stop):
        value = words[first]
        if all(v == value for v in words[first:stop]):
            return ('constant', value), 0
        middle = (first + stop) // 2
        left, left_cost = plan(first, middle)
        right, right_cost = plan(middle, stop)
        fallback = ('select', middle, left, right)
        fallback_cost = 2 + left_cost + right_cost
        step = words[first + 1] - value
        if all(words[i] - words[i - 1] == step for i in range(first + 1, stop)):
            offset = value - first * step
            # All branches evaluate eagerly. Bound even inactive affine values
            # over the entire selected prefix, preserving total signed i64 IR.
            if abs(step) * (len(words) - 1) + abs(offset) < 1 << 63:
                expr = Var('index') * step + offset
                cost = expression_cost(expr)
                if cost <= fallback_cost:
                    if any(evaluate(expr, {'index': i}) != words[i] for i in range(first, stop)):
                        raise ValueError('source affine word plan changed an original word')
                    return ('affine', expr), cost
        return fallback, fallback_cost

    return plan(0, len(words))[0]


def _emit_word_plan(control, index, plan):
    from merlin.sched.ir.expr import Const, Var, Sum, Scaled

    def affine(expr):
        if isinstance(expr, Const):
            return control.value(expr.value)
        if isinstance(expr, Var):
            return index
        if isinstance(expr, Scaled):
            return control.op('mul', affine(expr.expr), expr.factor)
        if isinstance(expr, Sum):
            values = [affine(t) for t in expr.terms]
            result = values[0]
            for value in values[1:]:
                result = control.op('add', result, value)
            return result
        raise ValueError('unsupported source word affine expression')

    if plan[0] == 'constant':
        return control.value(plan[1])
    if plan[0] == 'affine':
        return affine(plan[1])
    return control.select(control.cmp('ult', index, plan[1]),
        _emit_word_plan(control, index, plan[2]), _emit_word_plan(control, index, plan[3]))


def _period(words):
    """Smallest repeating prefix, including a partially repeated final prefix."""
    prefix = [0] * len(words)
    for i in range(1, len(words)):
        j = prefix[i - 1]
        while j and words[i] != words[j]:
            j = prefix[j - 1]
        if words[i] == words[j]:
            j += 1
        prefix[i] = j
    return len(words) - prefix[-1]


def source_configuration_word(control, index, source_words):
    """Select exact source bits using only scalar command configuration IR.

    Static schedule indices fold before IR emission. Dynamic indices use a
    repeated source prefix where present and a balanced tree with exact affine
    source runs where their scalar node bound beats constant word selection.
    No address, global table or host memory operation is needed.
    """
    words = tuple(source_words)
    if not words or any(type(v) is not int or not 0 <= v < 1 << 32 for v in words):
        raise ValueError('source configuration needs nonempty unsigned i32 words')
    static = index if type(index) is int else None
    if static is None:
        from xdsl.dialects.builtin import IntegerAttr
        owner = getattr(index, 'owner', None)
        # FnBuilder's TypedConstantOp deliberately shares LLVM's operation name
        # without subclassing its constant class (it preserves integer widths).
        value = getattr(owner, 'value', None)
        if getattr(owner, 'name', None) == 'llvm.mlir.constant' and isinstance(value, IntegerAttr):
            static = value.value.data
    if static is not None:
        if not 0 <= static < len(words):
            raise ValueError('source configuration index is outside source words')
        return control.value(words[static])
    period = _period(words)
    if period < len(words):
        words = words[:period]
        if period > 1:
            index = control.op('urem', index, period)

    return _emit_word_plan(control, index, _selection_plan(words))
