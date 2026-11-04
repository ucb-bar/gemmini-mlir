"""Complete ranked independent products through checked source batch offsets.

The original source owner selects/refolds heads and supplies the canonical batch
interface. This adapter preserves that interface and emits every original
matrix product through the same target dense lowerer. No source selection,
inter-head concurrency, storage retention or numerical conversion is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from merlin.compile.scheduling.batched_dense import (
    BatchMatrixView, BatchedDenseProblem, counted_batched_dense_products,
)
from merlin.compile.scheduling.llvm_program import (
    FnBuilderIndexAdapter, emit_index_effect_program,
)
from merlin.sched.ir import Call, Loop

_PRODUCT = 'gemmini_independent_head_product'
_COMPLETE = 'gemmini_independent_head_complete'


class HeadBatchError(ValueError):
    pass


def _program(problem):
    original = counted_batched_dense_products(problem,
        name='gemmini_ranked_head_batch', instruction=_PRODUCT)
    def complete(body):
        result = []
        for op in body:
            if isinstance(op, Loop):
                result.append(replace(op, body=complete(op.body)))
            else:
                result.extend((op, Call(_COMPLETE, ())))
        return tuple(result)
    return replace(original, body=complete(original.body))


@dataclass(frozen=True)
class HeadBatchPlan:
    problem: BatchedDenseProblem
    kernel: object
    output_dtype: str

    def revalidate(self):
        self.problem.__post_init__()
        # Compare every actual loop bound and affine offset to the selected
        # constructor, including the real completion at each product boundary.
        if self.output_dtype not in ('i8', 'i32') or self.kernel != _program(self.problem):
            raise HeadBatchError('ranked batch indices, completion or output format changed')
        return dict(schema='submission.ranked_head_batch.v1',
            kernel_sha256=self.kernel.digest(), problem=self.problem.record(),
            matrix_products=self.problem.products, logical_output_shape=list(self.problem.output_shape),
            output_dtype=self.output_dtype, offset_unit='elements',
            output_intervals='product * M*N through (product+1) * M*N; disjoint complete partition',
            storage='same private dense scratch reused only after synchronous full completion',
            numerical_pipeline='unchanged target dense product and declared source readout')


def plan_head_batches(lhs_shape, rhs_shape, output_shape, *, output_dtype='i32'):
    """Plan canonical dense row-major batches without flattening the tensor ABI."""
    lhs_shape, rhs_shape, output_shape = map(tuple, (lhs_shape, rhs_shape, output_shape))
    if (not 3 <= len(lhs_shape) <= 8 or len(rhs_shape) != len(lhs_shape)
            or lhs_shape[:-2] != rhs_shape[:-2]):
        raise HeadBatchError('canonical head batches need identical explicit leading batch axes')
    axes = tuple(range(len(lhs_shape)-2))
    problem = BatchedDenseProblem(lhs_shape[:-2],
        BatchMatrixView(lhs_shape, axes), BatchMatrixView(rhs_shape, axes))
    if output_shape != problem.output_shape:
        raise HeadBatchError('ranked batch output must preserve all source batch and matrix axes')
    plan = HeadBatchPlan(problem, _program(problem), output_dtype)
    plan.revalidate()
    return plan


def lower_head_batches(ctx, operands, attributes):
    """Emit canonical BATCHED_MATMUL through the selected counted batch API.

    ``operands`` contains a/w/dst. Dense readout attributes pass through exactly.
    An optional bias is shared [N] or explicitly batch-ranked [...,N]; an optional
    i32 initializer is shared [M,N] or has the complete ranked output shape.
    Batch geometry and alias checks finish before any device command is emitted.
    """
    from .dense import lower_dense
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    from .shared_completion import leaf_start, completed_leaf
    completion_mode=knob('BATCH_COMPLETION',default='explicit',
        values=('explicit','completed_leaf'),root=Path(__file__).resolve().parent.parent)
    lhs, rhs, dst = (operands[key] for key in ('a', 'w', 'dst'))
    a, b, c = (ctx.spec(name) for name in (lhs, rhs, dst))
    attrs = dict(attributes)
    if (a['dtype'] != 'i8' or b['dtype'] != 'i8'
            or c['dtype'] not in ('i8', 'i32')
            or attrs.get('output_dtype', c['dtype']) != c['dtype']):
        raise HeadBatchError('head batches require signed i8 operands and source-declared i8/i32 output')
    plan = plan_head_batches(a['shape'], b['shape'], c['shape'], output_dtype=c['dtype'])
    problem = plan.problem
    if 'batch' in attrs and attrs['batch'] != problem.products:
        raise HeadBatchError('declared batch count differs from the complete ranked operands')
    if attrs.get('block_scale') is not None or 'pool' in attrs.get('epilogue', ()):
        raise HeadBatchError('ranked head consumer has no block-scale or pooled-output contract')
    aliases = ctx.cb.get('params', {}).get('tensor_aliases', {})
    def base(name):
        seen = set()
        while name in aliases:
            if name in seen or not isinstance(aliases[name], dict):
                raise HeadBatchError('unresolved tensor alias in ranked batch boundary')
            seen.add(name)
            name = aliases[name]['source']
        return name
    auxiliary = {}
    for key, matrix_shape, ranked_shape in (
            ('bias', (problem.n,), (*problem.batch_shape, problem.n)),
            ('accumulator_init', (problem.m, problem.n), problem.output_shape)):
        name = attrs.get(key)
        if name is None:
            continue
        spec = ctx.spec(name)
        shape = tuple(spec['shape'])
        if spec['dtype'] != 'i32' or shape not in (matrix_shape, ranked_shape):
            raise HeadBatchError(f'{key} has no exact shared or ranked i32 projection')
        auxiliary[key] = (name, matrix_shape, shape == ranked_shape)
    if base(dst) in {base(name) for name in (lhs, rhs, *(row[0] for row in auxiliary.values()))}:
        raise HeadBatchError('ranked batch output must not alias a live input')
    # Dense readout validates its numeric contract. Shapes and bounds needed for
    # every projected pointer are established here, before allocating aliases.
    for shape, width in ((a['shape'], 1), (b['shape'], 1), (c['shape'], 1 if c['dtype']=='i8' else 4)):
        if prod(shape)*width >= 1 << 63:
            raise HeadBatchError('ranked batch byte range exceeds signed address arithmetic')
    proof = plan.revalidate()
    bridge = FnBuilderIndexAdapter(ctx.fb)
    source_index = ctx.current_source_index
    members = []
    completion_receipts = []
    pending_completion = [None]
    def alias(name, offset, shape):
        label = f'_head_batch_{name}_{len(ctx.tensors)}'
        ctx.register(label, ctx.ptr(name, bridge.value(offset)), shape, ctx.spec(name)['dtype'])
        return label
    def emit(instruction, args):
        if instruction != _PRODUCT:
            raise HeadBatchError('unexpected instruction in checked ranked batch')
        ctx.current_source_index = source_index
        aa = alias(lhs, args['lhs_offset'], (problem.m, problem.k))
        bb = alias(rhs, args['rhs_offset'], (problem.k, problem.n))
        cc = alias(dst, args['output_offset'], (problem.m, problem.n))
        member_attrs = dict(attrs)
        for key, (name, shape, ranked) in auxiliary.items():
            if ranked:
                offset = bridge.op('mul', args['product'], prod(shape))
                member_attrs[key] = alias(name, offset, shape)
        start=leaf_start(ctx.fb) if completion_mode=='completed_leaf' else None
        receipt = lower_dense(ctx, aa, bb, cc, m=problem.m, k=problem.k, n=problem.n, attributes=member_attrs)
        pending_completion[0]=completed_leaf(ctx.fb,start) if completion_mode=='completed_leaf' else None
        members.append(receipt)
    def complete():
        witness=pending_completion[0]
        if witness is not None and witness['complete']:
            completion_receipts.append({**witness,'boundary':'reused completed dense leaf'})
        else:
            ctx.fence()
            completion_receipts.append({'boundary':'explicit original batch fence','leaf_witness':witness})
        pending_completion[0]=None
    emission = emit_index_effect_program(plan.kernel, bridge, emit,
        complete_epoch=complete, completion_instruction=_COMPLETE)
    mapping=emission['completion_mapping']
    mapping['checked_static_boundaries']=mapping['emitted_static_drains']
    mapping['reused_completed_leaf_boundaries']=sum(row.get('complete',False) for row in completion_receipts)
    mapping['emitted_static_drains']=sum(row['boundary']=='explicit original batch fence' for row in completion_receipts)
    mapping['kind']='checked_full_epoch_boundaries_with_completed_leaf_reuse' if completion_mode=='completed_leaf' else mapping['kind']
    ctx.current_source_index = source_index
    report = {**proof, 'emission': emission, 'dense_leaf_schedules': members,
        'external_tensors': {'lhs':lhs, 'rhs':rhs, 'output':dst},
        'batch_axes': list(range(len(problem.batch_shape))),
        'completion': 'actual full target fence after every original matrix product',
        'completion_mode':completion_mode,'completion_receipts':completion_receipts,
        'native_source_selection': 'owned by original source router; not inferred from interface name'}
    ctx.metadata.setdefault('source_head_batches', []).append(report)
    return report
