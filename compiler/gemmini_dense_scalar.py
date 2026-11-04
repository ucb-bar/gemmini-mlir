"""Opt-in segmented dense scheduling for explicit fused scalar i8 commits.

This does not fuse or delete a source chain: interface COMMIT already requests
hardware scalar readout. The missing bridge is schedule eligibility for that
explicit source epilogue. Shared selected-path and complete-i32 scale proofs
qualify the original commit; existing source/epoch owners realize the actual
segmented schedule and unchanged terminal realizes the original readout.

The target selects this pure planner at its existing long-K boundary through
the declared ``dense_scalar_longk`` choice; explicit schedule overrides remain
caller-owned.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gemmini_dense_readout import ReadoutUnavailable


def qualify_scalar_commit(adapter,problem,*,attrs,output_dtype,bias=False,initial=False,outer_order=('m','n')):
    """Qualify the exact original interface commit, without normalizing it away."""
    import yaml
    from merlin.llvmlower.requantization import f32
    from merlin.verify.epilogue_applicability import ReadoutCapability,selectors_applying
    from gemmini_source_readout import selected_path,prove_scalar_readout
    from gemmini_readout import source_scales,_relu
    if (bias or initial or problem.bias or problem.tiles[0]!=1
            or output_dtype not in ('i8','int8') or tuple(outer_order)!=('m','n')
            or (problem.lhs_bytes,problem.rhs_bytes,problem.accumulator_bytes)!=(1,1,4)):
        raise ReadoutUnavailable('fused scalar schedule requires one source M tile and zero-seeded signed i8 dense')
    stages=tuple(attrs.get('epilogue',()))
    if not stages or len(set(stages))!=len(stages) or any(x not in ('acc_scale','relu') for x in stages):
        raise ReadoutUnavailable('fused scalar schedule requires explicit scalar scale/ReLU readout stages')
    if any(int(attrs.get(key,expected))!=expected for key,expected in (
            ('lhs_stride',problem.k),('rhs_stride',problem.n),('dst_stride',problem.n))):
        raise ReadoutUnavailable('fused scalar schedule requires original contiguous source axes')
    if int(attrs.get('requant_shift',attrs.get('shift',0))) or attrs.get('zero_point',0)!=0:
        raise ReadoutUnavailable('fused scalar schedule does not absorb shift or a nonzero zero point')
    scale=source_scales(attrs)
    if type(scale) not in (int,float):
        raise ReadoutUnavailable('fused scalar schedule requires an original constant scalar binary32 coefficient')
    relu=_relu(attrs);scale=f32(float(scale))
    if scale<=0:raise ReadoutUnavailable('fused scalar readout requires a positive coefficient')
    path,selected=selected_path()
    contract_path=Path(__file__).resolve().parent/'data/target_contract.yaml'
    declared=yaml.safe_load(contract_path.read_text())['runner']['chipyard_rocc']['readout']['epilogue_capability']
    readouts=tuple(ReadoutCapability(row['selector'],frozenset(row['applies']),row['evidence']) for row in declared)
    selectors=selectors_applying(readouts,stages)
    if 'i8' not in selectors:raise ReadoutUnavailable('selected readout does not apply every original epilogue stage')
    try:numeric=prove_scalar_readout((scale,),relu=relu)
    except (ValueError,ZeroDivisionError) as error:
        raise ReadoutUnavailable('original scalar readout proof refused: '+str(error)) from error
    readout=json.dumps(dict(dtype=output_dtype,scale=float(source_scales(attrs)),relu=relu),sort_keys=True)
    if adapter.numerical_policy_identity!='source-ordered-i8-i8-i32:'+readout:
        raise ReadoutUnavailable('fused scalar planner lost the original terminal numeric identity')
    return dict(schema='gemmini.explicit_scalar_commit.v1',original_stages=stages,
        original_attrs_sha256=hashlib.sha256(json.dumps(attrs,sort_keys=True).encode()).hexdigest(),
        original_readout_identity=readout,selected_path=selected,applying_selectors=selectors,
        full_i32_numeric_proof=numeric,source_operation_deletions=0,source_operations_preserved=True)


def plan_fused_dense(incumbent,adapter,problem,geometry,facts_path,*,column_blocks=32,
        attrs=None,output_dtype='i8',bias=False,initial=False,outer_order=('m','n'),k_tile_limit=None):
    """Realize an admitted scalar commit through the existing segmented owners.

    Explicit K limits receive the segmented owner's validation. None preserves
    its selected default and the existing source qualification and receipt.
    The eight-column alternative binds a shared explicit seed decision; the
    retained-LHS owner then selects its legal K segment at that exact N width.
    """
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
    from merlin.transforms.dense_planning import DenseResources
    from gemmini_dense_transfers import plan_transfers
    if column_blocks not in (8,16,32):raise ReadoutUnavailable('unregistered bounded fused-output alternative')
    attrs=dict(attrs or {});incumbent.revalidate()
    qualification=qualify_scalar_commit(adapter,problem,attrs=attrs,output_dtype=output_dtype,
        bias=bias,initial=initial,outer_order=outer_order)
    edge,operand,operand_rows,accumulator,accumulator_rows=geometry;split=operand_rows//2
    storage=DenseLoopStorage(DenseMemoryView('lhs',operand.name,operand.row_bytes,0,split),
        DenseMemoryView('rhs',operand.name,operand.row_bytes,split,operand_rows-split),
        DenseMemoryView('acc',accumulator.name,accumulator.row_bytes,0,accumulator_rows))
    resources=DenseResources(operand_rows*operand.row_bytes,accumulator_rows*accumulator.row_bytes,
        max_m_blocks=1,max_k_blocks=1,max_n_blocks=column_blocks,
        partial_output_tiles=True,partial_output_blocks=True)
    explicit_options={}
    if column_blocks==8:
        from merlin.compile.scheduling.schedule_ir import (
            make_dense_schedule_plan,dense_schedule_text,parse_dense_schedule_plan)
        source_identity=hashlib.sha256(json.dumps(dict(
            source='row-major-source-ordered-dense',problem=asdict(problem),
            qualification=qualification,numerical_policy=adapter.numerical_policy_identity),
            sort_keys=True).encode()).hexdigest()
        try:
            plan=make_dense_schedule_plan(problem,resources,storage,adapter,
                source_identity=source_identity,dtypes=('i8','i8','i32'),static_budget=128,
                block_tiles=(1,1,8),outer_order=outer_order,name='fused_scalar_bounded_output')
            seed_plan_text=dense_schedule_text(plan)
            explicit_options=dict(schedule_plan=parse_dense_schedule_plan(seed_plan_text),
                source_identity=source_identity)
        except ValueError as error:
            raise ReadoutUnavailable('explicit eight-column seed declined: '+str(error)) from error
    seed=schedule_compact_dense(problem,resources,storage,adapter,name='fused_scalar_bounded_output',
        dtypes=('i8','i8','i32'),static_budget=128,external_buffers_disjoint=True,
        outer_order=outer_order,verification_timeout_ms=20000,**explicit_options)
    block=seed.choice.block;lhs_rows=block.m*problem.tiles[1]*edge
    if block.n*problem.tiles[1]*edge<=(operand_rows-lhs_rows)//2:
        raise ReadoutUnavailable('complete fused RHS K fits one physical segment; preserve incumbent')
    k_options = {} if k_tile_limit is None else {'k_tile_limit': k_tile_limit}
    result,receipt=plan_transfers(seed,adapter,problem,geometry,facts_path,
        outer_order=outer_order,**k_options)
    result.revalidate()
    receipt['longk_output_experiment']=dict(requested_column_blocks=column_blocks,
        incumbent_block=asdict(incumbent.choice.block),bounded_source_block=asdict(block),
        source_problem=asdict(problem),numerical_identity=adapter.numerical_policy_identity,
        bridge='explicit scalar COMMIT; readout unchanged')
    receipt['explicit_scalar_fused_schedule']=dict(qualification=qualification,
        requested_column_blocks=column_blocks,bounded_source_block=asdict(block),
        original_compact_effect_sha256=incumbent.kernel.digest(),transformed_effect_sha256=result.kernel.digest(),
        terminal='unchanged original i8 scalar readout through dense_terminal',timing='UNKNOWN until matched ordinary check')
    if column_blocks==8:
        receipt['explicit_scalar_fused_schedule']['bounded_seed_schedule']=dict(
            source_identity=source_identity,schedule_ir=seed_plan_text,
            schedule_ir_sha256=hashlib.sha256(seed_plan_text.encode()).hexdigest(),
            realized_block=asdict(block))
    return result,receipt


