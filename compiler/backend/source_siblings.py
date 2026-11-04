"""Source-bound shared-LHS regions using the library's immutable tensor proof."""
from dataclasses import dataclass
import json


@dataclass
class Siblings:
    operations: tuple
    inputs: tuple
    outputs: tuple
    diagnostic: dict
    insertion_anchor: object


def checked(module, plans):
    """Prove raw siblings without changing source or moving an observer.

    The host inserts the plural call before the first root. Every external
    input must already dominate it. Existing observers stay at their source sites.
    Nonzero initialization and numeric epilogues retain their original regions.
    """
    from merlin.llvmlower.source_region import _plan_sibling_contractions, _sibling_tensor_operation_is_pure
    if not 2 <= len(plans) <= 8:
        raise ValueError('two to eight sibling products required')
    for p in plans:
        owner=p.root
        while owner is not None and owner is not module:
            owner=owner.parent_op()
        if owner is not module:
            raise ValueError('sibling root is absent from the current source module')
    if any(p.attributes != {'epilogue': [], 'output_dtype': 'i32'} or
           p.output is not p.root.results[0] for p in plans):
        raise ValueError('only unchanged raw zero-seeded products can share')
    proof=None
    try:
        proof = _plan_sibling_contractions(module, tuple(p.root for p in plans),
                                         symbol='candidate_source_shared_siblings')
        block=proof.block
        receipt=proof.receipt
    except ValueError:
        # The outline API accepts splats only. Public sources also spell the
        # identical exact seed as linalg.fill. Use its existing strict source
        # reader and effect checker on the original operations, without any
        # source rewrite or normalization of initializer semantics.
        from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
        block=plans[0].root.parent
        if block is None or any(p.root.parent is not block for p in plans):
            raise ValueError('siblings must belong to one sequential block')
        positions={op:i for i,op in enumerate(block.ops)}
        roots=tuple(p.root for p in plans)
        if sorted(roots,key=positions.get)!=list(roots):
            raise ValueError('siblings must preserve source root order')
        geometries=[read_zero_seeded_integer_matmul(op) for op in roots]
        lhs=roots[0].operands[0]
        if any(op.operands[0] is not lhs or shape[:2]!=geometries[0][:2]
               for op,shape in zip(roots,geometries)):
            raise ValueError('siblings require identical LHS SSA and reduction geometry')
        for op in tuple(block.ops)[positions[roots[0]]:positions[roots[-1]]+1]:
            if op not in roots and not _sibling_tensor_operation_is_pure(op):
                raise ValueError('siblings cross an unproved operation effect')
        receipt={'source_root_ordinals':[positions[op] for op in roots],
                 'source_geometries':[list(g) for g in geometries],
                 'library_checks':['read_zero_seeded_integer_matmul','_sibling_tensor_operation_is_pure']}
    try:
        positions={op:i for i,op in enumerate(block.ops)}
        owned={op for p in plans for op in p.operations}
        anchor=plans[0].root
        outputs=tuple(p.output for p in plans)
        for p in plans:
            for op in p.operations:
                for value in op.results:
                    for use in value.uses:
                        if use.operation in owned:
                            continue
                        if value not in outputs:
                            # Shared literals/seeds remain in host source; only
                            # contraction outputs are replaced by the runtime.
                            continue
                        observer=use.operation
                        while observer.parent is not block and observer.parent_op() is not None:
                            observer=observer.parent_op()
                        if observer.parent is not block or positions[observer] < positions[anchor]:
                            raise ValueError('sibling output has an earlier external observer')
        inputs=tuple(dict.fromkeys(v for p in plans for v in p.inputs))
        if any(v in {r for op in owned for r in op.results} for v in inputs):
            raise ValueError('sibling region contains dependent products')
        from xdsl.ir import Operation
        if any(isinstance(v.owner,Operation) and
               (v.owner.parent is not block or positions[v.owner]>=positions[anchor]) for v in inputs):
            raise ValueError('sibling external inputs do not dominate first root')
        return Siblings(tuple(op for op in block.ops if op in owned), inputs, outputs,
                        {'form':'shared_lhs_siblings', 'library_source_proof':receipt,
                         'same_lhs_ssa':True, 'all_outputs_preserved':True,
                         'insertion_anchor':'first_root', 'earlier_observers':False},anchor)
    finally:
        if proof is not None:
            proof.core.drop_all_references()
            proof.callee.drop_all_references()


def regions(module, plans, region_type):
    """Replace only admitted disjoint source groups; preserve incumbent refusals."""
    from .source_groups import _Plan, _root_inputs
    candidates={}
    for p in plans:
        if (type(p) is _Plan and p.attributes == {'epilogue': [], 'output_dtype':'i32'}
                and p.output is p.root.results[0]):
            lhs,_,_=_root_inputs(p.root)
            candidates.setdefault((p.root.parent,lhs),[]).append(p)
    grouped={}; consumed=set()
    for candidates_for_lhs in candidates.values():
        for start in range(0,len(candidates_for_lhs),8):
            members=candidates_for_lhs[start:start+8]
            try:
                group=checked(module,members)
            except ValueError:
                continue
            grouped[id(members[0])]=group
            consumed.update(id(p) for p in members[1:])
    result=[]
    for p in plans:
        if id(p) in consumed:
            continue
        g=grouped.get(id(p))
        result.append(region_type(g.operations,g.inputs,g.outputs,reason=json.dumps(g.diagnostic),insertion_anchor=g.insertion_anchor) if g
                      else region_type(p.operations,p.inputs,(p.output,),reason=json.dumps(p.diagnostic,default=str)))
    return tuple(result)
