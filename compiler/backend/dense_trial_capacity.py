"""Explicit capacity-segmented trial; incumbent selection remains caller-owned.

The same counted schedule and terminal codec own source and completion proofs.
The trial uses the complete physical LHS allocation plus two disjoint bounded
RHS slots; its N geometry is an explicit alternative rather than a shape rule.
"""
from .dense_streaming import streaming_schedule


def capacity_segmented_schedule(m,k,n,bias,column_scales=False):
    return streaming_schedule(m,k,n,bias,column_scales,geometry=(8,8),
                              batch_readouts=True)
