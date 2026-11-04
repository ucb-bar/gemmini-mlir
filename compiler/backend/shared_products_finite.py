"""Exact bounded row-write noninterference for actual shared staging epochs.

This is an epoch refinement of an independently source-checked group, not a
substitute source or hardware law. It changes no data calls, addresses or loop
bounds. Distinct writes in the same load epoch must occupy disjoint rows for
*every* outer environment; each epoch still ends with the original full drain.
"""
from itertools import product
from merlin.sched.ir import Call, Loop, evaluate

MAX_COMMANDS = 2_000_000
MAX_STAGE_WRITES = 100_000


def counted_commands(kernel, *, limit=MAX_COMMANDS):
    """Check the entire integer domain before allocating/enumerating any row set."""
    if type(limit) is not int or limit < 1:
        raise ValueError('shared finite proof requires a positive command bound')
    def count(body, names=(), depth=0):
        if type(body) is not tuple or depth > 32:
            raise ValueError('shared finite proof structural domain exceeded')
        total = 0
        for node in body:
            if type(node) is Loop:
                if (type(node.extent) is not int or node.extent <= 0 or node.var in names
                    or node.iter_args or node.yields or node.results):
                    raise ValueError('shared finite proof requires unique bounded positive integer loops')
                total += node.extent * count(node.body, (*names, node.var), depth+1)
            elif type(node) is Call:
                if node.produces is not None or node.awaits or node.unit or node.stages or node.sets or node.assumes:
                    raise ValueError('shared finite proof cannot erase command effects')
                total += 1
            else:
                raise ValueError('shared finite proof has an unknown statement')
            if total > limit:
                raise ValueError('shared finite actual-command proof budget exceeded')
        return total
    return count(kernel.body)


def disjoint_staging(item, outer, *, maximum_rows, maximum_columns,
                     limit=MAX_STAGE_WRITES):
    """Exhaustively check actual instantiated DMA row intervals, including tails.

    Outer indices are shared by all competing writes. Local loop indices range
    independently across all writes, exactly the original pairwise SMT domain.
    Sorted half-open row intervals decide disjointness without a solver.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError('shared finite staging requires an explicit positive bound')
    names = [name for name, _ in outer]
    if len(names) != len(set(names)) or any(type(e) is not int or e <= 0 for _,e in outer):
        raise ValueError('shared finite staging outer domain is ambiguous')
    def writes(node, env):
        if type(node) is Loop:
            if node.var in env or type(node.extent) is not int or node.extent <= 0:
                raise ValueError('shared finite staging local domain is ambiguous')
            for value in range(node.extent):
                yield from (entry for child in node.body for entry in writes(child,{**env,node.var:value}))
        elif type(node) is Call:
            args = dict(node.args)
            if len(args) != len(node.args) or not {'addr','rows','cols'} <= args.keys():
                raise ValueError('shared finite staging lacks actual unique DMA operands')
            address, rows, columns = (evaluate(args[k],env) for k in ('addr','rows','cols'))
            if (any(type(v) is not int for v in (address,rows,columns)) or address < 0
                or not 1 <= rows <= maximum_rows or not 1 <= columns <= maximum_columns):
                raise ValueError('shared finite staging has an invalid primitive interval')
            yield address,address+rows
        else:
            raise ValueError('shared finite staging has an unknown statement')
    def count(node):
        if type(node) is Loop:
            if type(node.extent) is not int or node.extent <= 0:
                raise ValueError('shared finite staging has an unbounded local loop')
            return node.extent*sum(count(child) for child in node.body)
        if type(node) is not Call:
            raise ValueError('shared finite staging has an unknown statement')
        return 1
    local=count(item)
    if local > limit:
        raise ValueError('shared finite staging row-set budget exceeded')
    environments=1
    for _,extent in outer: environments*=extent
    if environments*local > MAX_COMMANDS:
        raise ValueError('shared finite staging actual-command budget exceeded')
    checked=0
    for values in product(*(range(extent) for _,extent in outer)):
        intervals=sorted(writes(item,dict(zip(names,values,strict=True))))
        if len(intervals)!=local:
            raise ValueError('shared finite staging instantiated domain changed')
        if any(right[0] < left[1] for left,right in zip(intervals,intervals[1:])):
            raise ValueError('shared staging epoch has overlapping actual DMA row writes')
        checked+=local
    return checked
