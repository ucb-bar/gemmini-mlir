"""Must-state configuration cleanup on ordinary LLVM control-flow graphs.

The target supplies decoded persistent effects.  The analysis starts with no
state at function entry and intersects incoming facts, including loop edges.
Only an idempotent setter already established on every path may disappear.
No instructions move, and no synchronization or data operations are rewritten.
"""
from __future__ import annotations

from .configuration import ConfigurationEffects, _advance


def _states(blocks, effects, omitted=frozenset(), *, iteration_limit=4096):
    entry = blocks[0]
    owned = set(blocks)
    predecessors = {block: [] for block in blocks}
    for block in blocks:
        for successor in block.last_op.successors if block.last_op else ():
            if successor not in owned:
                raise ValueError('configuration CFG has an external successor')
            predecessors[successor].append(block)
    reachable = set()
    pending = [entry]
    while pending:
        block = pending.pop()
        if block in reachable:
            continue
        reachable.add(block)
        pending.extend(block.last_op.successors if block.last_op else ())
    incoming, outgoing = {}, {}
    for iteration in range(iteration_limit):
        changed = False
        for block in blocks:
            if block not in reachable:
                continue
            if block is entry:
                state = {}
            else:
                ready = [outgoing[p] for p in predecessors[block] if p in outgoing]
                if not ready:
                    continue
                state = {key: value for key, value in ready[0].items()
                         if all(other.get(key) == value for other in ready[1:])}
            live = dict(state)
            for op in block.ops:
                if op not in omitted:
                    _advance(live, effects[op])
            if incoming.get(block) != state or outgoing.get(block) != live:
                changed = True
                incoming[block], outgoing[block] = state, live
        if not changed:
            return incoming, outgoing, iteration + 1
    raise ValueError('configuration CFG did not converge within its work budget')


def eliminate_llvm_configuration(function, project, *, max_blocks=4096, max_operations=200000):
    """Delete proved redundant setters and independently check the omissions.

    ``project(op)`` returns ConfigurationEffects or None (an unknown clobber).
    The adapter must describe actual target semantics and ordering; matching
    operands or a declaration alone does not establish those properties.
    """
    blocks = tuple(function.body.blocks)
    if not blocks:
        return {'applied': False, 'removed_static_setters': 0}
    if len(blocks) > max_blocks or sum(len(tuple(b.ops)) for b in blocks) > max_operations:
        return {'applied': False, 'removed_static_setters': 0, 'refusal': 'CFG work budget'}
    effects = {op: project(op) for block in blocks for op in block.ops}
    if any(effect is not None and type(effect) is not ConfigurationEffects for effect in effects.values()):
        raise ValueError('configuration CFG needs typed effects')
    before, final_before, iterations = _states(blocks, effects)
    omitted = set()
    for block, state in before.items():
        live = dict(state)
        for op in block.ops:
            effect = effects[op]
            if effect is not None and effect.pure_write and all(
                    key in live and live[key] == value for key, value in effect.writes):
                omitted.add(op)
            _advance(live, effect)
    # Recompute from the actual retained effects.  Every omitted setter must
    # already hold in that program too, including at every loop backedge.
    after, final_after, verification_iterations = _states(blocks, effects, frozenset(omitted))
    if final_after != final_before:
        raise ValueError('configuration CFG rewrite changed a block exit state')
    for block, state in after.items():
        live = dict(state)
        for op in block.ops:
            effect = effects[op]
            if op in omitted:
                if effect is None or not effect.pure_write or any(
                        key not in live or live[key] != value for key, value in effect.writes):
                    raise ValueError('configuration omission lacks retained must-state')
            else:
                _advance(live, effect)
    for op in omitted:
        op.parent.erase_op(op)
    return {'applied': bool(omitted), 'removed_static_setters': len(omitted),
            'analysis_iterations': iterations, 'verification_iterations': verification_iterations,
            'entry_state': 'unknown', 'meet': 'intersection of all reachable predecessors',
            'scope': 'exact CFG persistent state; no motion or cross-invocation assumption'}
