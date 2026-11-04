"""Isolated storage-derived M blocking for checked complete-K RHS reuse.

No production selector is changed. Capacity defines legal alternatives, while
completed ordinary feedback alone can determine whether smaller M is useful.
Source views, numeric identity, RHS retention, causal readout, wide admission and
terminal encoding remain owned by the existing checked pipeline.
"""
from dataclasses import asdict
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
from merlin.transforms.dense_planning import DenseResources
from gemmini_dense_readout import ReadoutUnavailable

M_CAPACITY_DIVISORS=(1,2,4)


def plan_capacity_rhs(incumbent,adapter,problem,geometry,facts_path,*,m_divisor=1,partition='half',
        bias=False,initial=False,attrs=None,output_dtype='i32',outer_order=('m','n')):
    from gemmini_dense_rhs_experiment import scalar_terminal_source_supported
    from gemmini_dense_capacity_transfers import plan_capacity_transfers
    if m_divisor not in M_CAPACITY_DIVISORS:
        raise ReadoutUnavailable('unregistered capacity M divisor')
    if not scalar_terminal_source_supported(problem,attrs=dict(attrs or {}),bias=bias,
            initial=initial,output_dtype=output_dtype,outer_order=outer_order):
        raise ReadoutUnavailable('capacity RHS alternative requires original contiguous scalar dense source')
    incumbent.revalidate();old=incumbent.choice.block
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    if partition not in ('half','rhs_min'):
        raise ReadoutUnavailable('unregistered operand partition')
    split=(operand_rows//2 if partition=='half' else operand_rows-old.n*problem.tiles[1]*edge)
    rows_per_m_block=problem.tiles[1]*edge
    fitting_m=split//rows_per_m_block
    accumulator_m=accumulator_rows//(old.n*edge)
    m_limit=min(problem.tiles[0]//2,accumulator_m,fitting_m)//m_divisor
    if m_limit<1:raise ReadoutUnavailable('capacity divisor leaves no full-K LHS block')
    if problem.tiles[0]//m_limit<2:
        raise ReadoutUnavailable('capacity RHS alternative needs repeated complete M groups')
    if old.n*rows_per_m_block>operand_rows-split:
        raise ReadoutUnavailable('complete RHS N panel exceeds original disjoint half')
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=m_limit,max_k_blocks=1,max_n_blocks=old.n,
        partial_output_tiles=True,partial_output_blocks=True)
    seed=schedule_compact_dense(problem,resources,storage,adapter,name='capacity_rhs_seed',
        dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,
        outer_order=outer_order,verification_timeout_ms=20000)
    result,receipt=plan_capacity_transfers(seed,adapter,problem,geometry,facts_path,
        bias=bias,initial=initial,outer_order=outer_order,lhs_partition_rows=split)
    result.revalidate()
    receipt['capacity_m_alternative']=dict(m_divisor=m_divisor,partition=partition,maximum_fitting_m_blocks=fitting_m,
        original_block=asdict(old),bounded_source_block=asdict(seed.choice.block),
        rows_per_m_block=rows_per_m_block,lhs_half_rows=split,rhs_half_rows=operand_rows-split,
        lhs_full_k_rows=seed.choice.block.m*rows_per_m_block,
        rhs_full_k_rows=seed.choice.block.n*rows_per_m_block,
        source_problem=asdict(problem),numerical_identity=adapter.numerical_policy_identity,
        terminal='unchanged consume_checked_dense with panel_epochs',timing='UNKNOWN until matched ordinary feedback')
    return result,receipt
