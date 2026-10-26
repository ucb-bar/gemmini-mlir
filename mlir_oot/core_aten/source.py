"""Queries shared by source-bound target admission."""


def module_of(op):
    while op.parent is not None:
        op = op.parent
    return op
