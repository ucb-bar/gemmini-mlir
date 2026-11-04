"""Conservative completion receipts from the actual emitted leaf CFG.

This analysis grants no overlap. It can only reuse a real selected full fence
that already completes every path before the enclosing original boundary.
"""
from hashlib import sha256
import json

from .completion import COMPLETION_ASM


def leaf_start(builder):
    return builder.blk, builder.blk.last_op


def completed_leaf(builder, start):
    first, previous = start
    endpoint = builder.blk
    blocks = tuple(builder.region.blocks)
    indices = {block:i for i,block in enumerate(blocks)}
    if first not in indices or endpoint not in indices:
        return {'complete':False,'reason':'leaf endpoints lack current CFG ownership'}
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.ir import BlockArgument
    def value(v,env):
        if isinstance(v,BlockArgument):return env.get(v)
        op=v.owner
        if op.name=='llvm.mlir.constant' and isinstance(op.value,IntegerAttr):
            return op.value.value.data
        args=[value(a,env) for a in op.operands]
        if any(a is None for a in args):return None
        if op.name=='llvm.add':result=args[0]+args[1]
        elif op.name=='llvm.sub':result=args[0]-args[1]
        elif op.name=='llvm.mul':result=args[0]*args[1]
        elif op.name=='llvm.icmp':
            from xdsl.dialects.llvm import ICmpPredicateFlag
            flag=ICmpPredicateFlag.from_int(op.predicate.value.data).value
            a,b=args
            if flag.startswith('u'):
                width=op.lhs.type.width.data; a&=(1<<width)-1;b&=(1<<width)-1
            return int({'eq':a==b,'ne':a!=b,'slt':a<b,'sle':a<=b,'sgt':a>b,'sge':a>=b,
                        'ult':a<b,'ule':a<=b,'ugt':a>b,'uge':a>=b}[flag])
        else:return None
        width=v.type.width.data
        result&=(1<<width)-1
        return result-(1<<width) if result&(1<<(width-1)) else result
    work = [(first,True,{})]
    visited = set()
    exits = set()
    fences = set()
    effects = []
    while work:
        block,dirty,env = work.pop()
        state = block,dirty,frozenset(env.items())
        if state in visited:continue
        visited.add(state)
        if len(visited)>4096:
            return {'complete':False,'reason':'bounded CFG completion analysis exhausted'}
        operations = tuple(block.ops)
        if block is first and previous is not None:
            if previous not in operations:
                return {'complete':False,'reason':'leaf entry marker changed'}
            operations = operations[operations.index(previous)+1:]
        for position,op in enumerate(operations):
            if op.regions:
                return {'complete':False,'reason':'unanalysed nested leaf control'}
            if op.name=='llvm.inline_asm':
                asm = op.asm_string.data
                constraints = op.constraints.data
                effects.append((indices[block],position,asm,constraints))
                if (asm==COMPLETION_ASM and constraints=='~{memory}'
                        and not op.operands and op.has_side_effects):
                    dirty=False
                    fences.add((indices[block],position))
                else:
                    dirty=True
            elif op.name not in (
                    'llvm.mlir.constant','llvm.add','llvm.sub','llvm.mul','llvm.and','llvm.or',
                    'llvm.xor','llvm.shl','llvm.lshr','llvm.ashr','llvm.icmp','llvm.zext',
                    'llvm.sext','llvm.trunc','llvm.getelementptr','llvm.ptrtoint','llvm.inttoptr',
                    'llvm.select','llvm.mlir.undef','llvm.alloca','llvm.br','llvm.cond_br'):
                # Unknown effects and host memory operations cannot certify a
                # completed boundary. Only ordinary pure index/control ops
                # may follow the witnessed full fence.
                dirty=True
        if block is endpoint:
            exits.add(dirty)
            continue
        terminator=operations[-1] if operations else None
        successors = tuple(terminator.successors) if terminator is not None else ()
        if not successors or any(target not in indices for target in successors):
            return {'complete':False,'reason':'leaf control has an unowned exit'}
        if terminator.name=='llvm.br':
            edges=[(terminator.successor,terminator.arguments)]
        elif terminator.name=='llvm.cond_br':
            condition=value(terminator.cond,env)
            edges=[]
            if condition is None or condition:edges.append((terminator.then_block,terminator.then_arguments))
            if condition is None or not condition:edges.append((terminator.else_block,terminator.else_arguments))
        else:
            return {'complete':False,'reason':'unanalysed leaf branch semantics'}
        for target,arguments in edges:
            # Widen every backward edge to unknown loop indices. Initial
            # constant loop entries remain exact, so a positive counted loop
            # cannot gain a spurious zero-iteration bypass. No trip expansion.
            backward=indices[target]<=indices[block]
            bindings={} if backward else dict(env)
            bindings.update((arg,None if backward else value(v,env))
                for arg,v in zip(target.args,arguments,strict=True))
            work.append((target,dirty,{k:v for k,v in bindings.items() if v is not None}))
    complete = exits=={False} and bool(fences)
    return {'complete':complete,'method':'actual LLVM CFG all-path full-fence completion',
        'entry_pending':True,'exit_pending_states':sorted(exits),
        'reachable_control_states':len(visited),'real_full_fence_sites':len(fences),
        'target_effect_projection_sha256':sha256(json.dumps(sorted(set(effects))).encode()).hexdigest(),
        'scope':'only emitted leaf CFG; no motion, overlap or inferred storage lifetime'}
