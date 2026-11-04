"""Counted, checked dense schedules consumed by the target effect encoder.

All arithmetic below computes addresses and loop bounds. Matrix products are
issued to the accelerator, including ragged tiles and reduction continuations.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from functools import lru_cache
from math import isqrt
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import (
    DenseBiasSource, DenseLoopAdapter, DenseLoopStorage, DenseMemoryView,
    emit_compact_dense, _static_calls,
)
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.sched.ir import Call, as_expr
from merlin.transforms.dense_planning import DenseProblem, DenseResources, choose_dense_block


def dense_adapter():
    """Lossless actual-operand projection, shared by every schedule alternative."""
    def call(name, names, values):
        return Call(name, tuple(zip(names, map(as_expr, values))))

    def encode(op):
        if isinstance(op, Load):
            return call(op.role + "_load",
                        ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step"),
                        (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step))
        if isinstance(op, InitializeAccumulator):
            return call("bias_load", ("m", "n", "col", "rows", "cols", "acc"),
                        (*op.block, op.dram_col, op.rows, op.cols, op.row))
        if isinstance(op, Preload):
            return call("preload", ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
                        (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate,
                         op.rows, op.cols, op.weight_rows))
        if isinstance(op, Compute):
            return call("compute", ("lhs", "m", "k", "rows", "depth"),
                        (op.input_row, *op.input_block, op.rows, op.cols))
        if isinstance(op, Store):
            return call("store", ("row", "col", "rows", "cols", "acc"),
                        (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
        raise ValueError(f"unsupported dense effect: {type(op).__name__}")

    def project(c):
        a = dict(c.args)
        if c.instr in ("lhs_load", "weight_load"):
            return Load(c.instr.removesuffix("_load"), (a["a"], a["b"]),
                        a["src_row"], a["src_col"], a["rows"], a["cols"], a["dst"], a["step"])
        if c.instr == "preload":
            return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"],
                           a["rows"], a["cols"], a["depth"])
        if c.instr == "compute":
            return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
        if c.instr == "store":
            return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
        if c.instr == "bias_load":
            return InitializeAccumulator((a["m"], a["n"]), a["col"], a["rows"], a["cols"], a["acc"])
        raise ValueError(f"unknown dense instruction: {c.instr}")

    # Every ordinary data effect is completed by emit_dense below. Preload and
    # compute form an inseparable command pair, completed after compute.
    return DenseLoopAdapter(encode, project, "gemmini-ws-i8-v1", "i8-i8-modular-i32-v1", True, True, 64)


@lru_cache(maxsize=1)
def _capacities():
    from merlin.targetgen import address_space as AS
    path = Path(__file__).resolve().parent / "data" / "facts.json"
    facts = json.loads(path.read_text())
    space = AS.derive_address_space(facts["inputs"]["target"], facts=facts)
    operand, accumulator = AS.operand_store(space), AS.accumulator_store(space)
    if operand.store is None or accumulator.store is None or not space.array_rows or not space.array_cols:
        raise ValueError("dense lowering needs declared array and both storage capacities")
    return (space.array_rows, space.array_cols, operand.store.name, operand.capacity_rows(),
            operand.store.row_bytes, accumulator.store.name, accumulator.capacity_rows(),
            accumulator.store.row_bytes)


def _execution_laws(adapter):
    """Bind queue/order premises to the packaged selected RTL evidence."""
    root = Path(__file__).resolve().parent / "data"
    evidence = json.loads((root / "command_ordering.json").read_text())
    ordering = evidence["ordering"]
    facts = json.loads((root / "facts.json").read_text())
    classes = facts.get("facts", facts)["timing_structure"]["command_classes"]["classes"]
    pending_execute = classes["execute"]["entries"]["value"] // 2
    pending_load = classes["load"]["entries"]["value"]
    if ordering["status"] != "derived" or ordering["completion_order"]["execute"]["order"] != "in_order":
        raise ValueError("dense pipeline needs derived ordered accumulator commits")
    cross = {(r["older"], r["younger"], r["space"]): r for r in ordering["cross_queue"]}
    loads = ("MVIN.spad_addr", "MVIN2.spad_addr")
    readers = ("COMPUTE_PRELOADED.a_spad", "PRELOAD.weight_spad")
    def waits(a, b, space):
        return cross.get((a, b, space), {}).get("overlap") == "completion"
    raw = all(waits(w, rd, "scratchpad") for w in loads for rd in readers)
    war = all(waits(rd, w, "scratchpad") for w in loads for rd in readers)
    visibility = waits("PRELOAD.c_addr", "MVOUT.acc_addr", "accumulator")
    if not pending_execute or not pending_load:
        raise ValueError("dense pipeline requires finite native backpressure queues")
    # Load/execute/store configuration selects the same ordered queue as the
    # subsequent instruction; each accepted command captures its configuration.
    by_code = next(r["by_code"] for r in ordering["funct_queue"] if r.get("funct") == 0)
    config = by_code == {"0": "execute", "1": "load", "2": "store", "3": "store"}
    identity = "selected-ordering:" + evidence["ordering_sha256"]
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission, DenseRetainedReductionAdmission
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
    ids = (adapter.configuration_identity, adapter.numerical_policy_identity, identity)
    overlap = DensePanelOverlapAdmission(*ids, 65536, 65536, pending_execute, pending_load)
    reduction = DenseRetainedReductionAdmission(*ids, 65536, 65536, pending_execute, pending_load)
    epoch = DensePanelEpochLaw(*ids, local_read_after_write=raw, local_write_after_read=war,
        ordered_accumulator_writes=True, accumulator_result_visibility=visibility,
        ordered_configuration_capture=config, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    return overlap, reduction, epoch


@lru_cache(maxsize=128)
def select_dense_alternative(m, k, n, *, resident_rhs=False, bias=False, wide_transfers=False, ready_readout=False):
    """Frozen hardware-relative policy over the existing shared alternatives.

    The paired boundary panel covers K tile counts one and two with M=1 and
    multi-row tiles; three geometries were held out before timing. Small
    one-tile reductions retain compact. This is a performance guard only:
    plan_dense still derives storage and ordering legality independently.
    """
    from merlin.transforms.pipeline_state import PipelineContext, PipelineSession, PassSpec
    from merlin.transforms.pipeline_policy import (
        DimensionRef, ResourceRef, TileFeature, FeaturePredicate, PolicyFacts,
        RegisteredPass, PipelineVariant, PipelinePolicy, execute_policy, PropertyPredicate,
    )
    shape = (m, k, n)
    def verify(value):
        if len(value) != 3 or any(type(v) is not int or v <= 0 for v in value):
            raise ValueError("dense schedule policy requires positive source dimensions")
    verify(shape)
    capacities = _capacities()
    context = PipelineContext(hardware=hashlib.sha256(
        (Path(__file__).parent/"data/facts.json").read_bytes()).hexdigest(),
        resources=hashlib.sha256(repr(capacities).encode()).hexdigest(),
        execution="frozen-dense-regional-policy-v1")
    session = PipelineSession(shape, context=context, ir_level="dense_geometry",
        fingerprint=lambda value: hashlib.sha256(repr(value).encode()).hexdigest(), verify_ir=verify)
    kr,edge,_,operand_rows,_,_,_,_ = capacities
    full_k = (k+kr-1)//kr
    rhs_rows = full_k*((n+edge-1)//edge)*kr
    resident_fit = (not bias and rhs_rows <= operand_rows-operand_rows//2
                    and full_k*edge <= operand_rows//2)
    # Empirical narrow-output frontier region, frozen from matched complete-call
    # comparisons. The half-window cutoff is not a proven optimal crossover;
    # below-cutoff narrow timing remains unknown. Shared planning still proves
    # segmented placement, source effects, readiness, and DMA legality.
    lhs_window_rows = operand_rows // 2
    narrow_frontier_fit = (ready_readout and not bias and m <= edge and n <= edge
                           and lhs_window_rows <= 2*full_k*edge
                           and full_k*edge <= lhs_window_rows)
    facts = PolicyFacts(session.stamp(), shapes=((m,k),(k,n)),
                        resources=(("array_k",kr),
                                   ("m_panel_rows",isqrt(capacities[6]//edge)*edge)),
                        capabilities=(("full_rhs_fit", resident_fit),
                                      ("narrow_frontier_fit", narrow_frontier_fit)))
    predicate = FeaturePredicate(TileFeature(DimensionRef(0,1),ResourceRef("array_k"),"count"),
                                 at_least=(2,1))
    registry = {name: RegisteredPass(PassSpec(name,0,produces=("dense_alternative",),
                    mutates_ir=False,ir_level="dense_geometry"),
                    lambda current,name=name: {"dense_alternative": name})
                for name in ("compact_mn","epochs","epochs_wide_dma","epochs_ready_dma","shared_rhs_epochs","segmented_frontier_dma")}
    variants = []
    if resident_rhs:
        rows = FeaturePredicate(TileFeature(DimensionRef(0,0),ResourceRef("m_panel_rows"),"count"),
                                at_least=(2,1))
        variants.append(PipelineVariant("shared_rhs_epochs",("shared_rhs_epochs",),
                        (predicate, rows, PropertyPredicate("capability","full_rhs_fit"))))
    variants.append(PipelineVariant("segmented_frontier_dma",("segmented_frontier_dma",),
                    (PropertyPredicate("capability","narrow_frontier_fit"),)))
    epochs = "epochs_ready_dma" if ready_readout else "epochs_wide_dma" if wide_transfers else "epochs"
    variants.extend((PipelineVariant(epochs,(epochs,),(predicate,)),
                     PipelineVariant("compact_mn",("compact_mn",))))
    policy = PipelinePolicy(tuple(variants),"compact_mn")
    selected = execute_policy(session,policy,registry,facts,output="dense_alternative")
    return selected.artifact, selected.receipt


@lru_cache(maxsize=128)
def _plan_dense(m, k, n, *, bias=False, alternative="compact_mn", panel_shape="balanced"):
    """Two reusable loop orders, with shared capacity planning and exact tails."""
    if alternative in ("regional_epochs", "regional_resident_rhs", "regional_transfers", "regional_ready_transfers"):
        alternative, _ = select_dense_alternative(m,k,n,
            resident_rhs=alternative != "regional_epochs", bias=bias,
            wide_transfers=alternative in ("regional_transfers", "regional_ready_transfers"),
            ready_readout=alternative == "regional_ready_transfers")
    resident_mode = {"resident_ready": "ready", "resident_input_pipeline": "input_pipeline"}.get(alternative)
    if resident_mode:
        alternative = "retained"
    ready_readout = alternative == "epochs_ready_dma"
    if ready_readout:
        alternative = "epochs_wide_dma"
    retained_transfers = alternative == "retained_wide_dma"
    request_bound = alternative in ("epochs_wide_dma", "retained_wide_dma")
    wide_transfers = alternative in ("epochs_wide", "epochs_wide_dma", "retained_wide_dma")
    if wide_transfers:
        alternative = "retained" if retained_transfers else "epochs"
    if alternative not in ("compact_mn", "compact_nm", "retained", "prefetch", "epochs", "shared_rhs_epochs", "compact_unit", "segmented_ready_dma", "segmented_frontier_dma"):
        raise ValueError("unknown dense schedule alternative")
    kr, edge, operand_name, operand_rows, operand_bytes, acc_name, acc_rows, acc_bytes = _capacities()
    p = DenseProblem(m, k, n, edge, kr, edge, 1, 1, 4, bias=bias)
    if edge > operand_bytes or edge * 4 > acc_bytes:
        raise ValueError("primitive dense tile does not fit a hardware row")
    lhs_rows = operand_rows // 2
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", operand_name, operand_bytes, 0, lhs_rows),
        DenseMemoryView("rhs", operand_name, operand_bytes, lhs_rows, operand_rows-lhs_rows),
        DenseMemoryView("acc", acc_name, acc_bytes, 0, acc_rows),
    )
    # Bound the geometric search by store capacities. Runtime iteration remains
    # counted even when dimensions exceed these limits by many orders.
    if panel_shape not in ("balanced", "rectangular"):
        raise ValueError("unknown dense panel resource geometry")
    # Rectangular enumeration offers the full accumulator product domain. The
    # shared planner still checks actual m*n accumulator bytes, separate operand
    # views and both live pipeline windows. Residency keeps its frozen resources.
    panel_tiles = (acc_rows // edge if panel_shape == "rectangular" and
                   alternative in ("epochs", "prefetch") else isqrt(acc_rows // edge))
    r = DenseResources(operand_rows * operand_bytes, acc_rows * acc_bytes,
                       max_m_blocks=min(p.tiles[0], panel_tiles), max_n_blocks=min(p.tiles[2], panel_tiles),
                       max_k_blocks=p.tiles[1], partial_output_tiles=True, partial_output_blocks=True)
    if alternative == "compact_unit":
        r = replace(r, max_m_blocks=1, max_k_blocks=1, max_n_blocks=1)
    adapter = dense_adapter()
    bias_source = DenseBiasSource(n, "i32") if bias else None
    source = hashlib.sha256(repr((m, k, n, bias, "row-major-i8-product-i32")).encode()).hexdigest()
    if alternative in ("segmented_ready_dma", "segmented_frontier_dma"):
        from merlin.compile.scheduling.dense_segmented_pipeline import (
            dense_segmented_blocks, DenseSegmentedAdmission,
        )
        from dense_readout import resident_readout_laws
        overlap, reduction, epoch = _execution_laws(adapter)
        if epoch.unknown():
            raise ValueError("segmented ready needs selected combined load/compute/readout ordering")
        rr = replace(r, max_k_blocks=p.tiles[1]-1)
        legal = [b for b in dense_segmented_blocks(p, rr, storage)
                 if b.k < p.tiles[1] and (alternative == "segmented_frontier_dma"
                    or (p.tiles[2]+b.n-1)//b.n >= 2)]
        if not legal:
            raise ValueError("segmented ready needs retained full LHS and two legal RHS segments")
        block = min(legal, key=lambda b: (b.loads, -b.m*b.n, b.m, b.n))
        reduction, readout = resident_readout_laws(adapter, block, reduction, epoch)
        overlap = replace(overlap, max_issued_readouts=reduction.max_issued_readouts,
            pending_readouts=reduction.pending_readouts,
            readout_configuration=reduction.readout_configuration)
        admission = DenseSegmentedAdmission(overlap, reduction,
            "selected-combined-segmented:" + epoch.qualification_identity)
        decision = make_dense_schedule_plan(p, rr, storage, adapter, source_identity=source,
            dtypes=("i8","i8","i32"), static_budget=256,
            block_tiles=(block.m,block.k,block.n), outer_order=("m","n"),
            bias_source=bias_source, pipeline_prefetch=True,
            pipeline_kind="retained_lhs_segmented_k",
            pipeline_issue_order=("prefetch_rhs_between_k_tiles"
                if alternative == "segmented_frontier_dma" else "compute_then_prefetch"))
        result = schedule_dense_pipeline(p, rr, storage, adapter, name="dense",
            dtypes=("i8","i8","i32"), external_buffers_disjoint=True,
            schedule_plan=decision, source_identity=source, pipeline_admission=admission,
            readout_mode="ready", readout_admission=readout,
            static_budget=256, timeout_ms=20000, bias_source=bias_source)
        from dense_transfer import coalesce_dense_epochs
        return p, coalesce_dense_epochs(result, p, storage, adapter, request_bound=True)
    if alternative == "shared_rhs_epochs":
        # Partition the original row domain into two adjacent, disjoint views.
        # Both source products use the identical immutable RHS; scratch windows
        # are reused only across the checked closed member epochs.
        if bias or p.tiles[0] < 2:
            raise ValueError("shared RHS requires two nonempty row views and zero seed")
        from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
        from merlin.compile.scheduling.dense_shared_rhs_check import DenseRHSReuseLaw
        adapter = replace(adapter, qualification_identity="gemmini-dense-actual-fields:" +
            hashlib.sha256((Path(__file__).parent / "encoding.py").read_bytes()).hexdigest())
        split = (p.tiles[0] // 2) * p.tile_m
        members = (replace(p, m=split), replace(p, m=m-split))
        def encode_member(member, op):
            from merlin.sched.ir import Ptr
            call = adapter.encode(op)
            args = call.args + (("member", as_expr(-1 if member is None else member)),)
            if isinstance(op, Load):
                weight = op.role == "weight"
                args += (("ptr", Ptr("rhs" if weight else f"lhs{member}",
                          as_expr(op.dram_row)*(n if weight else k)+op.dram_col)),)
            elif isinstance(op, Store):
                args += (("ptr", Ptr(f"output{member}",
                          (as_expr(op.dram_row)*n+op.dram_col)*4)),)
            return replace(call, args=args)
        def project_member(call):
            fields = dict(call.args)
            owner = fields["member"].value
            return (None if owner == -1 else owner,
                    adapter.project(replace(call, args=tuple((a,b) for a,b in call.args if a != "member"))))
        _, reduction, epoch = _execution_laws(adapter)
        retention = DenseRHSReuseLaw(adapter.configuration_identity,
            adapter.numerical_policy_identity, adapter.qualification_identity,
            "row-partition-disjoint-private-output-immutable-rhs-v1",
            rhs_unchanged_by_member_operations=True, independent_member_interleaving=True,
            suppressed_loads_no_faults=True, suppressed_loads_no_extra_observations=True)
        result = schedule_dense_shared_rhs(members, r, (storage,storage), adapter,
            encode_member=encode_member, project_member=project_member,
            source_identity=source, name="dense", dtypes=("i8","i8","i32"),
            static_budget=256, external_buffers_disjoint=True, retention_law=retention,
            epoch_admission=reduction, epoch_law=epoch, verification_timeout_ms=20000)
        if result is None:
            raise ValueError("complete resident RHS or full-K staging does not fit")
        return p, result
    order = ("m", "n") if alternative in ("compact_mn", "compact_unit") else ("n", "m")
    if alternative in ("retained", "prefetch", "epochs"):
        from merlin.transforms.dense_planning import dense_blocks
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_compute_order, dense_retained_operand_hoists
        from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
        overlap, reduction, epoch = _execution_laws(adapter)
        rr = r if alternative == "retained" else replace(r, max_k_blocks=1)
        blocks = dense_blocks(p, rr, k_limit=lambda mb, nb: storage.k_limit(p, rr, mb, nb))
        if alternative == "retained":
            legal = [b for b in blocks if b.k == p.tiles[1] and storage.admits(p, b, rr)]
        else:
            legal = [b for b in blocks if storage.admits(p, b, rr)
                     and 2*b.m*p.tile_m <= storage.lhs.capacity
                     and 2*b.n*p.tile_k <= storage.rhs.capacity
                     and 2*b.operand_bytes <= rr.operand_bytes]
        if not legal:
            raise ValueError("requested dense pipeline has no capacity-legal block")
        block = min(legal, key=lambda b: (b.loads, -b.m*b.n, b.m, b.n))
        transforms = ()
        if alternative == "retained":
            raw = emit_compact_dense(p, block, storage, adapter, name="dense", dtypes=("i8", "i8", "i32"),
                                     outer_order=order, bias_source=bias_source)
            transforms = dense_retained_operand_hoists(raw, adapter, role="weight")
            transforms += dense_retained_compute_order(propose_loop_transforms(raw, transforms), adapter,
                                                       order=("k", "n", "m"))
        decision = make_dense_schedule_plan(p, rr, storage, adapter, source_identity=source,
                    dtypes=("i8", "i8", "i32"), static_budget=256,
                    block_tiles=(block.m, block.k, block.n), outer_order=order, bias_source=bias_source,
                    loop_transforms=transforms, pipeline_prefetch=alternative != "retained",
                    pipeline_kind="retained_full_k" if alternative == "retained" else "two_window")
        readout_admission = None
        if resident_mode:
            from dense_readout import resident_readout_laws
            reduction, readout_admission = resident_readout_laws(
                adapter, block, reduction, epoch, input_pipeline=resident_mode == "input_pipeline")
        result = schedule_dense_pipeline(p, rr, storage, adapter, name="dense", dtypes=("i8", "i8", "i32"),
                    readout_mode="ready" if resident_mode else "separate",
                    readout_admission=readout_admission,
                    input_pipeline_row_offset=(block.m*block.k*p.tile_m
                        if resident_mode == "input_pipeline" else None),
                    external_buffers_disjoint=True, schedule_plan=decision, source_identity=source,
                    pipeline_admission=reduction if alternative == "retained" else overlap,
                    epoch_law=epoch if alternative == "epochs" else None,
                    static_budget=256, timeout_ms=20000, bias_source=bias_source)
        if ready_readout:
            from dense_readout import schedule_epoch_ready_readout
            result = schedule_epoch_ready_readout(result, p, rr, storage, adapter)
        if wide_transfers:
            from dense_transfer import coalesce_dense_epochs
            result = coalesce_dense_epochs(result, p, storage, adapter, request_bound=request_bound)
        return p, result
    choice = choose_dense_block(p, r, static_budget=256, require_reuse=False,
                    k_limit=lambda mb, nb: storage.k_limit(p, r, mb, nb),
                    admits=lambda b: storage.admits(p, b, r),
                    price=lambda b: sum(1 for _ in _static_calls(emit_compact_dense(p, b, storage, adapter,
                        name="dense", dtypes=("i8", "i8", "i32"), outer_order=order,
                        bias_source=bias_source).body)))
    if choice is None:
        raise ValueError("no capacity-legal dense block fits the static command budget")
    block = choice.block
    decision = make_dense_schedule_plan(p, r, storage, adapter, source_identity=source,
                    dtypes=("i8", "i8", "i32"), static_budget=256,
                    block_tiles=(block.m, block.k, block.n), outer_order=order,
                    bias_source=bias_source)
    result = schedule_dense_pipeline(p, r, storage, adapter, name="dense", dtypes=("i8", "i8", "i32"),
                    external_buffers_disjoint=True, schedule_plan=decision, source_identity=source,
                    static_budget=256, timeout_ms=20000, bias_source=bias_source)
    return p, result


@lru_cache(maxsize=128)
def plan_dense(m, k, n, *, bias=False, alternative="compact_mn", panel_shape="balanced"):
    """Finish a source proof before emitting; use a simpler checked plan on UNKNOWN.

    Only an exhausted/unavailable symbolic source proof triggers the fallback.
    An invariant violation is an error. The fallback has one primitive block per
    dimension, retains counted loops and runs the complete ordinary checker.
    If that bounded proof also fails, compilation fails before target emission.
    """
    from merlin.compile.scheduling.compact_verification import DenseVerificationUnavailable
    try:
        return _plan_dense(m, k, n, bias=bias, alternative=alternative, panel_shape=panel_shape)
    except DenseVerificationUnavailable as exc:
        problem, result = _plan_dense(m, k, n, bias=bias, alternative="compact_unit", panel_shape="balanced")
        result = replace(result, realization=dict(result.realization,
            verification_fallback={"requested_schedule": alternative,
                "selected_schedule": "compact_unit", "reason": str(exc),
                "proof": "complete checked counted primitive-block program"}))
        result.revalidate()
        return problem, result



@lru_cache(maxsize=128)
def resident_input_region(m, k, n):
    """Empirical two-N-panel guard, followed by exact selected-block slot fit.

    Called only after the incumbent full-RHS regional predicate. The original
    checked retained plan supplies its actual block; the ready owner separately
    verifies the complete transformed source and both advancing storage slots.
    """
    kr, edge, _, operand_rows, _, _, acc_rows, _ = _capacities()
    balanced = isqrt(acc_rows // edge)
    if (n + edge - 1) // edge > 2 * balanced:
        return False
    p, retained = plan_dense(m, k, n, alternative="retained", panel_shape="balanced")
    retained.revalidate()
    schedule = retained.realization.get("schedule_ir", {})
    if schedule.get("pipeline_kind") != "retained_full_k":
        return False
    from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan
    module = parse_dense_schedule_plan(schedule["text"])
    block, = (op for op in module.walk() if op.name == "schedule.block_contraction")
    mb, kb, nb = tuple(block.block_tiles.iter_values())
    lhs_panel_rows = mb * kb * p.tile_m
    return (kb == p.tiles[1] and (p.tiles[2] + nb - 1) // nb <= 2
            # Only complete tile panels form the counted advancing driver. A tail
            # is emitted separately, so ceil(panel count) would admit one driver
            # plus a tail although the input-pipeline pass needs extent >= 2.
            and p.tiles[0] // mb >= 2
            and 2 * lhs_panel_rows <= operand_rows // 2)


def emit_dense(enc, lhs, rhs, out, m, k, n, *, out_dtype="i32", scale=1.0,
               relu=False, bias=None, alternative=None, lhs_pitch=None, rhs_pitch=None,
               out_pitch=None, lhs_offset=0, rhs_offset=0, out_offset=0, bias_offset=0, panel_shape=None):
    """Emit the checked actual loop IR through Encoder.dense_effect.

    Encoder owns row-major pointer arithmetic, load block stride, accumulator
    address bits and numerical store configuration. No tensor loops are rebuilt.
    """
    if alternative is None:
        settings = Path(__file__).parent / "DENSE_SCHEDULE.json"
        alternative = os.environ.get("DENSE_SCHEDULE") or (
            json.loads(settings.read_text()).get("schedule", "compact_mn") if settings.exists() else "compact_mn")
    if panel_shape is None:
        from merlin.compile.tuning_knobs import knob
        panel_shape = os.environ.get("DENSE_PANEL_SHAPE") or knob("dense_panel_shape", default="balanced",
                           values=("balanced", "rectangular"), root=Path(__file__).parent)
    from merlin.compile.tuning_knobs import knob
    resident_readout = os.environ.get("DENSE_RESIDENT_READOUT") or knob(
        "dense_resident_readout", default="serial", values=("serial", "ready", "input_pipeline", "regional_input_pipeline"),
        root=Path(__file__).parent)
    requested_alternative = alternative
    selection_receipt = None
    if alternative in ("regional_epochs", "regional_resident_rhs", "regional_transfers", "regional_ready_transfers"):
        alternative, selection_receipt = select_dense_alternative(m,k,n,
            resident_rhs=alternative != "regional_epochs", bias=bias is not None,
            wide_transfers=alternative in ("regional_transfers", "regional_ready_transfers"),
            ready_readout=alternative == "regional_ready_transfers")
    if alternative == "shared_rhs_epochs" and resident_readout != "serial":
        if resident_readout == "ready":
            alternative = "resident_ready"
        elif resident_readout == "input_pipeline" or resident_input_region(m, k, n):
            alternative = "resident_input_pipeline"
    lhs_pitch = k if lhs_pitch is None else lhs_pitch
    rhs_pitch = n if rhs_pitch is None else rhs_pitch
    out_pitch = n if out_pitch is None else out_pitch
    if any(type(v) is not int or v <= 0 for v in (lhs_pitch, rhs_pitch, out_pitch)):
        raise ValueError("dense physical row pitches must be positive element counts")
    if lhs_pitch < k or rhs_pitch < n or out_pitch < n:
        raise ValueError("dense physical row pitch is narrower than its logical view")
    if any(type(v) is not int or v < 0 for v in (lhs_offset, rhs_offset, out_offset, bias_offset)):
        raise ValueError("dense physical base offsets must be nonnegative element counts")
    if isinstance(scale, (tuple, list)):
        if out_dtype != "i8":
            raise ValueError("column-dependent scale requires the narrowing i8 readout")
        from merlin.llvmlower.integer_readout import partition_column_scales
        partition = partition_column_scales(scale, n, relu=relu)
        receipts = []
        for region in partition["regions"]:
            first, columns = region["first_column"], region["columns"]
            receipts.append(emit_dense(enc, lhs, rhs, out, m, k, columns,
                out_dtype=out_dtype, scale=region["scale"], relu=relu, bias=bias,
                alternative=alternative, panel_shape=panel_shape, lhs_pitch=lhs_pitch, rhs_pitch=rhs_pitch,
                out_pitch=out_pitch, lhs_offset=lhs_offset, rhs_offset=rhs_offset+first,
                out_offset=out_offset+first, bias_offset=bias_offset+first))
        return {"requested_schedule": requested_alternative, "selected_schedule": alternative,
                "schedule_selection": selection_receipt, "column_readout": partition["schema"],
                "partition_count": partition["partition_count"],
                "regions": [{"first_column": r["first_column"], "columns": r["columns"],
                             "scale": r["scale"]} for r in partition["regions"]],
                "emissions": receipts}
    p, result = plan_dense(m, k, n, bias=bias is not None, alternative=alternative, panel_shape=panel_shape)
    output_bytes = 4 if out_dtype == "i32" else 1
    if lhs_offset:
        lhs = enc.offset(lhs, lhs_offset)
    if rhs_offset:
        rhs = enc.offset(rhs, rhs_offset)
    if out_offset:
        out = enc.offset(out, out_offset * output_bytes)
    if bias is not None and bias_offset:
        bias = enc.offset(bias, bias_offset * 4)
    pointers = dict(lhs=lhs, rhs=rhs, out=out, out_dtype=out_dtype,
                    scale=scale, relu=relu, bias=bias)

    def effect(instruction, fields):
        if alternative == "shared_rhs_epochs" and "member" in fields:
            fields = dict(fields)
            # This is the source-view binding of the checked grouped product.
            # The second member starts at a primitive-row-aligned split.
            split = (p.tiles[0] // 2) * p.tile_m
            rowbase = enc.mul(fields.pop("member"), split)
            if instruction == "lhs_load":
                fields["src_row"] = enc.add(fields["src_row"], rowbase)
            elif instruction == "store":
                fields["row"] = enc.add(fields["row"], rowbase)
        # The schedule describes logical rectangles, while these explicit pitches
        # bind them to original source subviews. A sliced RHS remains K x full_N
        # in physical memory; neither its row stride nor the output stride shrinks.
        if instruction in ("lhs_load", "weight_load"):
            side = "lhs" if instruction == "lhs_load" else "rhs"
            pitch = lhs_pitch if side == "lhs" else rhs_pitch
            logical_pitch = p.k if side == "lhs" else p.n
            if pitch != logical_pitch:
                byteoffset = enc.add(enc.mul(fields["src_row"], pitch), fields["src_col"])
                enc.load(enc.offset(pointers[side], byteoffset), fields["dst"], fields["rows"],
                         fields["cols"], enc.mul(pitch, fields.get("step", 1)),
                         stream=0 if side == "lhs" else 1)
                return
        elif instruction == "store" and out_pitch != p.n:
            byteoffset = enc.mul(enc.add(enc.mul(fields["row"], out_pitch), fields["col"]), output_bytes)
            enc.store(enc.offset(out, byteoffset), fields["acc"], fields["rows"], fields["cols"],
                      out_pitch * output_bytes, dtype=out_dtype, scale=scale, relu=relu)
            return
        enc.dense_effect(instruction, fields, pointers, p)

    def issue(instruction, fields):
        effect(instruction, fields)
        if instruction != "preload":
            enc.fence()

    if hasattr(result, "emit_index_program"):
        from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
        receipt = result.emit_index_program(FnBuilderControlAdapter(enc.fn),
            effect, complete_epoch=enc.fence)
    else:
        receipt = result.emit_llvm(enc.fn, issue)
    enc.fence()
    realization = result.realization
    schedule = realization.get("schedule_ir", {})
    return {**receipt, "requested_schedule": requested_alternative,
            "selected_schedule": realization.get("verification_fallback", {}).get("selected_schedule", alternative),
            "verification_fallback": realization.get("verification_fallback"),
            "schedule_selection": selection_receipt, "kernel_sha256": result.kernel.digest(),
            "schedule_plan_sha256": schedule.get("sha256", realization.get("schedule_sha256")),
            "dynamic_epoch_drains": realization.get("dynamic_epoch_drains"),
            "tile_load_projection": realization.get("tile_load_projection"),
            "overlapped_compute_pairs": realization.get("overlapped_compute_pairs", 0)}
