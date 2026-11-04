"""Bind shared source zero proposals to the existing dense zero accumulator path."""
from dataclasses import dataclass

# Shared exact source proofs and selected dense realization guard every omission.
ENABLED = True


@dataclass(frozen=True)
class VirtualInitializer:
    root: object
    initializer: object
    source_op_index: int
    root_source_op_index: int
    proposal: object
    storage_operations: tuple = ()

    def record(self):
        return dict(source_op_index=self.source_op_index,
            root_source_op_index=self.root_source_op_index,
            omitted_result_positions=[list(position) for position in self.proposal.virtual_result_positions
                if position[0] in {index for index, _ in self.storage_operations}],
            realization='existing dense exact zero accumulator; seed pointer never read',
            shared_source_proposal=self.proposal.record())


def proposals(module, operations):
    """Read the original module before quantization or storage alias rewrites."""
    if not ENABLED:
        return ()
    from merlin.compile.scheduling.zero_initializer_groups import find_zero_initializer_groups
    operations=tuple(operations)
    result=[]
    for group in find_zero_initializer_groups(module):
        root=operations[group.contraction_ordinal]
        initializer=operations[group.initializer_ordinal]
        # The shared proposal owns source geometry, scalar recurrence, single
        # use, positivity and exact zero. Bind its original objects explicitly.
        if root.operands[2] is not initializer.results[0]:
            raise ValueError('zero initializer source position mapping differs')
        storage = tuple((index,operations[index]) for index,result_index in group.virtual_result_positions
            if hasattr(operations[index].results[result_index].type,'get_shape'))
        result.append(VirtualInitializer(root,initializer,group.initializer_ordinal,
            group.contraction_ordinal,group,storage))
    return tuple(result)


def select(candidates, *, is_device, excluded_roots=(), aliased_values=()):
    """Do not virtualize a seed unless its exact existing terminal path owns it."""
    excluded=set(excluded_roots)
    aliases=set(aliased_values)
    return tuple(candidate for candidate in candidates
        if candidate.root not in excluded and is_device(candidate.root)
        and all(op.results[0] not in aliases for _,op in candidate.storage_operations))
