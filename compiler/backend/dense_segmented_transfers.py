"""Exact wide RHS mapping checked under the original two-slot segmented law.

Only adjacent source N tiles inside a real segment/slot are joined. The raw
software move-in-B law is shared with retained transfers; the execution owner
and its source/epoch verifier remain segmented, including slot reuse drains.
"""
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, verify_dense_segmented_pipeline,
)
from merlin.compile.scheduling.panel_tile_loads import _admit_dense_tile_loads
from .dense_rhs_transfers import select_rhs_block_transfers


def _admit_segmented(original, proposal, problem, block, resources, storage,
                     adapter, *, project, encode_band, timeout_ms):
    from .dense_segmented_frontier import DenseGroupedFrontierResult
    if type(original) not in (DenseSegmentedPipelineResult,DenseGroupedFrontierResult):
        raise ValueError('segmented wide transfer requires its original typed owner')
    return _admit_dense_tile_loads(
        original, proposal, problem, block, resources, storage, adapter,
        verifier=verify_dense_segmented_pipeline, project=project,
        encode_band=encode_band, static_budget=128, timeout_ms=timeout_ms,
        max_static_nodes=4096,
    )


def select_segmented_block_transfers(original, problem, block, resources,
                                     storage, adapter):
    return select_rhs_block_transfers(original, problem, block, resources,
                                     storage, adapter, consumer=_admit_segmented)
