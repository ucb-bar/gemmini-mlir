"""Verify an exact column-broadcast source view before loaded-sum scheduling.

The rank-one bias remains the sole physical bias allocation. A checked view maps
logical rhs[r,c] to bias[c]; normalization creates no tensor or data movement.
Existing compact loaded-sum verification then proves the complete output cover.
"""
from dataclasses import dataclass

from merlin.sched.ir import Const, Kernel, TensorArg
from .compact_loaded_sum import LoadedSumEffect, verify_compact_loaded_sum


@dataclass(frozen=True)
class BiasSumEffect:
    logical: LoadedSumEffect
    source_row: object
    source_column: object
    source_pitch_bytes: int


def normalize_bias_sum(kernel, layout, contract, project, *, scalar_proof):
    """Check actual broadcast operands and return a verified logical view.

    The caller must retain/recheck this original source, use the returned
    projector for epoch planning, and implement the checked source-coordinate
    mapping at final issue. This establishes no target arithmetic or ordering.
    """
    layout.validate()
    contract.validate()
    shape = (layout.rows, layout.columns)
    if (type(kernel) is not Kernel or kernel.attrs
        or layout.input_dtype != 'i32' or layout.accumulator_dtype != 'i32'
        or layout.output_dtype != 'i32'
        or tuple((a.name, a.shape, a.dtype, a.access) for a in kernel.args) != (
            ('lhs', shape, 'i32', 'read'), ('bias', (layout.columns,), 'i32', 'read'),
            ('output', shape, 'i32', 'write'))):
        raise ValueError('bias view requires complete i32 matrix, column vector and output')
    original_digest = kernel.digest()
    normalized = Kernel(kernel.name + '_column_broadcast_view', (
        TensorArg('lhs', shape, 'i32', 'read'), TensorArg('rhs', shape, 'i32', 'read'),
        TensorArg('output', shape, 'i32', 'write')), kernel.body)

    def logical_project(call):
        effect = project(call)
        if type(effect) is not BiasSumEffect or type(effect.logical) is not LoadedSumEffect:
            raise ValueError('bias view requires actual decoded physical and logical operands')
        logical = effect.logical
        if logical.kind not in ('initialize', 'accumulate', 'store'):
            raise ValueError('unknown bias effect')
        bias = logical.kind == 'accumulate'
        if (effect.source_row != (Const(0) if bias else logical.row)
            or effect.source_column != logical.column
            or type(effect.source_pitch_bytes) is not int
            or effect.source_pitch_bytes != (0 if bias else layout.columns * 4)):
            raise ValueError('actual bias source differs from exact column broadcast view')
        return logical

    report = verify_compact_loaded_sum(normalized, layout, contract, logical_project,
        scalar_proof=scalar_proof)
    if kernel.digest() != original_digest:
        raise ValueError('bias original source changed while checking the broadcast view')
    report = dict(report, schema='merlin.bias_sum_view.v1',
        original_kernel_sha256=original_digest, bias_elements=layout.columns,
        physical_source_cover='lhs/output[r,c]; bias[c] repeated exactly once per source row',
        normalization='read-only column broadcast view; no materialization')
    return normalized, logical_project, report
