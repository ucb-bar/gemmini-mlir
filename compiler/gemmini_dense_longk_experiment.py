"""Bounded-output alternative for retained-LHS segmented dense schedules.

The selected library owns full-K input residency, physical RHS segments, causal
ready readout and final wide transfer admission. This module changes only the
bounded N blocking supplied to those owners. Production selects it through the
``dense_longk_columns`` knob only for automatic single-M-tile sources whose
complete K exceeds a physical RHS segment; its returned artifact must reach
the ordinary terminal. No timing or case identity participates in construction.
"""
from dataclasses import asdict
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
from merlin.transforms.dense_planning import DenseResources
from gemmini_dense_readout import ReadoutUnavailable

OUTPUT_BLOCK_LIMITS=(8,16,32)


def plan_longk_output(incumbent,adapter,problem,geometry,facts_path,*,column_blocks=16,
        bias=False,initial=False,attrs=None,output_dtype='i32',outer_order=('m','n'),k_tile_limit=None):
    """Select bounded N and optionally cap K through the same segmented owner.

    None retains the owner's selected default; explicit limits receive its
    existing validation before any target instruction is emitted.
    """
    from gemmini_dense_coalesced_policy import _raw_source_supported
    from gemmini_dense_transfers import plan_transfers
    if column_blocks not in OUTPUT_BLOCK_LIMITS:
        raise ReadoutUnavailable('long-K output experiment requires a registered bounded N alternative')
    if not _raw_source_supported(problem,attrs=dict(attrs or {}),bias=bias,initial=initial,
            output_dtype=output_dtype,outer_order=outer_order):
        raise ReadoutUnavailable('long-K output experiment requires original contiguous raw signed dense source')
    incumbent.revalidate();old=incumbent.choice.block
    if problem.tiles[0]!=old.m:
        raise ReadoutUnavailable('long-K output experiment requires a single complete M output group')
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry;split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=old.m,max_k_blocks=1,max_n_blocks=column_blocks,
        partial_output_tiles=True,partial_output_blocks=True)
    seed=schedule_compact_dense(problem,resources,storage,adapter,name='longk_bounded_output',
        dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,
        outer_order=outer_order,verification_timeout_ms=20000)
    k_options = {} if k_tile_limit is None else {'k_tile_limit': k_tile_limit}
    result,receipt=plan_transfers(seed,adapter,problem,geometry,facts_path,
        outer_order=outer_order,**k_options)
    result.revalidate()
    receipt['longk_output_experiment']=dict(requested_column_blocks=column_blocks,
        incumbent_block=asdict(old),bounded_source_block=asdict(seed.choice.block),
        source_problem=asdict(problem),numerical_identity=adapter.numerical_policy_identity,
        terminal='unchanged consume_checked_dense with panel_epochs',timing='UNKNOWN until ordinary matched check')
    return result,receipt
