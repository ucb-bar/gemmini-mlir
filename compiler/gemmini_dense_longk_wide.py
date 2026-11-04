"""Isolated hardware-relative wide-N retained-LHS long-K alternative.

No production policy or installation lives here. Selected physical accumulator
rows bound output panels independently of finite compiler issue bounds and
physical queue slots. Existing source, segmented, ready-readout and final-wide
transfer owners produce the artifact consumed by the ordinary dense terminal.
"""
from dataclasses import asdict
import hashlib,json
from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
from merlin.transforms.dense_planning import DenseResources
from gemmini_dense_readout import ReadoutUnavailable


def plan_wide(incumbent,adapter,problem,geometry,facts_path,*,column_blocks=64,
        bias=False,initial=False,attrs=None,output_dtype='i32',outer_order=('m','n'),k_tile_limit=None):
    """Build a 48/64/128-column-tile candidate, or refuse before target mutation.

    The selected device currently fits 64 primitive output tiles per M tile;
    128 is explicitly refused by capacity, rather than silently capped. The
    segmented owner derives the largest physically legal K segment and retains
    original K order. An explicit k_tile_limit bounds that same shared planner;
    None preserves its selected default. Transfer aggregation retains its selected
    DMA-width cap.
    """
    from gemmini_dense_coalesced_policy import _raw_source_supported
    from gemmini_dense_scalar import qualify_scalar_commit
    from gemmini_dense_transfers import plan_transfers
    if column_blocks not in (48,64,128):
        raise ReadoutUnavailable('wide long-K requires an explicit bounded 48/64/128 N alternative')
    attrs=dict(attrs or {});incumbent.revalidate()
    if problem.tiles[0]!=1 or incumbent.choice.block.m!=1:
        raise ReadoutUnavailable('wide long-K requires one primitive source M tile')
    qualification=None
    if output_dtype in ('i8','int8'):
        qualification=qualify_scalar_commit(adapter,problem,attrs=attrs,output_dtype=output_dtype,
            bias=bias,initial=initial,outer_order=outer_order)
    elif not _raw_source_supported(problem,attrs=attrs,bias=bias,initial=initial,
            output_dtype=output_dtype,outer_order=outer_order):
        raise ReadoutUnavailable('wide long-K requires original contiguous zero-seeded signed dense source')
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry
    if column_blocks*edge>accumulator_rows:
        raise ReadoutUnavailable('requested complete wide N panel exceeds physical accumulator rows')
    if problem.tiles[2]<column_blocks:
        raise ReadoutUnavailable('source N extent cannot realize requested complete wide output panel')
    lhs_rows=problem.tiles[1]*edge
    # One complete primitive RHS K tile must fit in each disjoint window.
    if lhs_rows>=operand_rows or 2*column_blocks*edge>operand_rows-lhs_rows:
        raise ReadoutUnavailable('retained full-K LHS leaves no two physical wide RHS segments')
    split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=1,max_k_blocks=1,max_n_blocks=column_blocks,
        partial_output_tiles=True,partial_output_blocks=True)
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    source_identity=hashlib.sha256(json.dumps(dict(source='row-major-source-ordered-dense',
        problem=asdict(problem),numerical_policy=adapter.numerical_policy_identity),sort_keys=True).encode()).hexdigest()
    decision=make_dense_schedule_plan(problem,resources,storage,adapter,source_identity=source_identity,
        dtypes=('i8','i8','i32'),static_budget=128,block_tiles=(1,1,column_blocks),outer_order=outer_order)
    seed=schedule_compact_dense(problem,resources,storage,adapter,name='wide_longk_bounded_output',
        dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,
        outer_order=outer_order,verification_timeout_ms=20000,
        source_identity=source_identity,schedule_plan=decision)
    if seed.choice.block.n!=column_blocks:
        raise ReadoutUnavailable('source N extent cannot realize requested complete wide output panel')
    if column_blocks*problem.tiles[1]*edge<=(operand_rows-lhs_rows)//2:
        raise ReadoutUnavailable('complete RHS K fits one physical segment; preserve incumbent')
    k_options = {} if k_tile_limit is None else {'k_tile_limit': k_tile_limit}
    result,receipt=plan_transfers(seed,adapter,problem,geometry,facts_path,
        outer_order=outer_order,**k_options)
    result.revalidate()
    receipt['longk_output_experiment']=dict(requested_column_blocks=column_blocks,
        incumbent_block=asdict(incumbent.choice.block),bounded_source_block=asdict(seed.choice.block),
        source_problem=asdict(problem),numerical_identity=adapter.numerical_policy_identity,
        terminal='ordinary dense_terminal mode panel_epochs; original scalar readout unchanged')
    receipt['wide_longk']=dict(schema='gemmini.wide_longk.v1',requested_column_blocks=column_blocks,
        physical_accumulator_rows=accumulator_rows,required_accumulator_rows=column_blocks*edge,
        retained_lhs_rows=lhs_rows,remaining_rhs_rows=operand_rows-lhs_rows,
        physical_rhs_windows=2,queue_slots_source='existing selected segmented admission',
        logical_issue_bound=4096,scalar_qualification=qualification,
        transformed_effect_sha256=result.kernel.digest(),timing='UNKNOWN')
    return result,receipt


def plan_wide_or_fallback(ordinary_planner,*args,column_blocks=32,wide_column_blocks=64,**kwargs):
    """Pure opt-in parent hook preserving existing alternatives and refusals.

    Caller supplies its selected original raw/scalar planner. Explicit 16-tile
    alternatives remain unchanged; an unsupported 48/64/128 geometry calls that
    exact ordinary planner with the original block argument before emission.
    """
    if column_blocks!=32:
        return ordinary_planner(*args,column_blocks=column_blocks,**kwargs)
    try:
        return plan_wide(*args,column_blocks=wide_column_blocks,**kwargs)
    except ReadoutUnavailable:
        return ordinary_planner(*args,column_blocks=column_blocks,**kwargs)
