"""Counted, checked row-major integer GEMM schedules for Gemmini.

The shared compact scheduler owns blocking, reduction order, retention and tails.
This adapter owns the translation to the selected target's ordinary commands.
"""
from __future__ import annotations

from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store, InitializeAccumulator
from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter, DenseLoopStorage, DenseMemoryView, DenseBiasSource, schedule_compact_dense,
)
from merlin.sched.ir import Call, as_expr
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def _encode(op):
    # These are target adapter instructions: every operand is retained explicitly
    # and the decoder below reconstructs it without a side table.
    if isinstance(op, InitializeAccumulator):
        names = ("tile0", "tile1", "col", "rows", "cols", "acc", "row_stride", "col_stride")
        values = (*op.block, op.dram_col, op.rows, op.cols, op.row, op.source_row_stride, op.source_col_stride)
        name = "gemmini_bias"
    elif isinstance(op, Load):
        names = ("tile0", "tile1", "row", "col", "rows", "cols", "dst", "step")
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
        name = "gemmini_" + op.role + "_load"
    elif isinstance(op, Preload):
        names = ("weight", "tile0", "tile1", "acc", "continue", "rows", "cols", "depth")
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate,
                  op.rows, op.cols, op.weight_rows)
        name = "gemmini_preload"
    elif isinstance(op, Compute):
        names = ("lhs", "tile0", "tile1", "rows", "depth")
        values = (op.input_row, *op.input_block, op.rows, op.cols)
        name = "gemmini_compute"
    elif isinstance(op, Store):
        names = ("row", "col", "rows", "cols", "acc")
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
        name = "gemmini_store"
    else:
        raise ValueError("dense adapter does not implement this operation")
    return Call(name, tuple(zip(names, map(as_expr, values))))


def _project(call):
    a = dict(call.args)
    if call.instr == "gemmini_bias":
        return InitializeAccumulator((a["tile0"], a["tile1"]), a["col"], a["rows"], a["cols"],
                                     a["acc"], a["row_stride"], a["col_stride"])
    if call.instr in ("gemmini_lhs_load", "gemmini_weight_load"):
        role = "lhs" if call.instr == "gemmini_lhs_load" else "weight"
        return Load(role, (a["tile0"], a["tile1"]), a["row"], a["col"],
                    a["rows"], a["cols"], a["dst"], a["step"])
    if call.instr == "gemmini_preload":
        return Preload(a["weight"], (a["tile0"], a["tile1"]), a["acc"],
                       a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "gemmini_compute":
        return Compute(a["lhs"], (a["tile0"], a["tile1"]), a["rows"], a["depth"], True)
    if call.instr == "gemmini_store":
        return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
    raise ValueError("unknown dense target instruction: " + call.instr)


def dense_adapter():
    # Each data effect is completed by a target fence; preload and its consuming
    # compute form one ordered execute transaction. Accumulator bit 30 selects
    # exact i32 continuation, with full-width readout bypassing requantization.
    return DenseLoopAdapter(_encode, _project, "gemmini-ws-i8-fenced-v1",
                            "signed-i8-product-modular-i32-v1", True, True, 64)


@lru_cache(maxsize=1)
def _geometry():
    root = Path(__file__).resolve().parent
    facts = json.loads((root / "rtl_facts/facts.json").read_text())["facts"]
    array = next(a for a in facts["arrays"] if a["name"] == "mesh")
    mem = {a["name"]: a for a in facts["memories"]}
    edge = array["rows"]
    if array["cols"] != edge or mem["scratchpad"]["elem_bits"] != 8 or mem["accumulator"]["elem_bits"] != 32:
        raise ValueError("dense target requires square i8 mesh and i32 accumulators")
    return edge, mem["scratchpad"]["bytes"] // edge, mem["accumulator"]["bytes"] // (4 * edge)


@lru_cache(maxsize=128)
def plan_matmul(m, k, n, max_m_blocks=8, max_k_blocks=8, max_n_blocks=8, bias=False, resident=False, segmented=False):
    """Two counted operand windows; each K epoch writes disjoint outputs.

    No epoch contains overlapping accumulator updates, and every epoch drains.
    The next reduction step therefore observes the completed previous step.
    """
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    from merlin.transforms.dense_planning import dense_blocks

    edge, operand_rows, acc_rows = _geometry()
    split = (operand_rows // (2 * edge)) * edge
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", "scratchpad", edge, 0, split),
        DenseMemoryView("rhs", "scratchpad", edge, split, operand_rows - split),
        DenseMemoryView("acc", "accumulator", edge * 4, 0, acc_rows),
    )
    problem = DenseProblem(m, k, n, edge, edge, edge, 1, 1, 4, bias=bias)
    resources = DenseResources(operand_rows * edge, acc_rows * edge * 4,
                              max_m_blocks=max_m_blocks, max_k_blocks=1,
                              max_n_blocks=max_n_blocks,
                              partial_output_tiles=True, partial_output_blocks=True)
    candidates = dense_blocks(problem, resources,
        k_limit=lambda bm,bn: min(1, storage.k_limit(problem,resources,bm,bn)))
    legal = [b for b in candidates
             if 2*b.m*edge <= storage.lhs.capacity and 2*b.n*edge <= storage.rhs.capacity]
    if not legal:
        raise ValueError("two-window panel has no capacity-legal block")
    block = min(legal, key=lambda b:(b.loads, -b.m*b.n, b.m,b.n))
    adapter = dense_adapter()
    facts_bytes = (Path(__file__).resolve().parent / "rtl_facts/facts.json").read_bytes()
    facts = json.loads(facts_bytes)["facts"]
    classes = facts["timing_structure"]["command_classes"]["classes"]
    queue = lambda role: classes[role]["command_queue"]
    if any(queue(role).get("status") != "derived" for role in ("execute","load","store")):
        raise ValueError("panel issue requires selected derived command queues")
    # Queue bounds are physical facts, never compiler-chosen epoch lengths.
    # All configurations stay fixed throughout one invocation, so accepted
    # commands cannot observe a later configuration mutation.
    pending_pairs = max(1,min(c["entries"] for c in queue("execute")["chain"])//2)
    qualification = "gemmini-selected-queue-panel-epoch-v1:" + hashlib.sha256(facts_bytes).hexdigest()
    admission = DensePanelOverlapAdmission(
        adapter.configuration_identity, adapter.numerical_policy_identity, qualification,
        block.m*block.n, 2*(block.m+block.n), pending_pairs,
        queue("load")["entries_total"],
        max_issued_readouts=block.m*block.n,
        pending_readouts=queue("store")["entries_total"],
        readout_configuration="ordered_snapshot_before_store_issue")
    source = hashlib.sha256(json.dumps({"operation":"row-major-i8-matmul-i32",
        "m":m,"k":k,"n":n,"bias":bias},sort_keys=True).encode()).hexdigest()
    seed = DenseBiasSource(n,"i32") if bias else None
    decision = make_dense_schedule_plan(problem,resources,storage,adapter,
        source_identity=source,dtypes=("i8","i8","i32"),static_budget=128,
        block_tiles=(block.m,1,block.n),pipeline_kind="two_window",pipeline_prefetch=True,
        bias_source=seed)
    result = schedule_dense_pipeline(problem,resources,storage,adapter,
        name="gemmini_panel_matmul",dtypes=("i8","i8","i32"),
        schedule_plan=decision,source_identity=source,external_buffers_disjoint=True,
        pipeline_admission=admission,batch_readouts=True,bias_source=seed,
        static_budget=128,timeout_ms=15000)
    if resident:
        from resident import retain_rhs
        result = retain_rhs(result,problem,block,resources,storage,adapter)
    if segmented:
        from segmented_k import plan_segmented
        result = plan_segmented(m,k,n,adapter,_geometry(),result,block,resources)
    return result


@lru_cache(maxsize=128)
def _serial_plan_matmul(m, k, n, max_m_blocks=8, max_k_blocks=8, max_n_blocks=8, bias=False):
    """Build a checked compact plan from public hardware capacities."""
    edge, operand_rows, acc_rows = _geometry()
    split = (operand_rows // (2 * edge)) * edge
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", "scratchpad", edge, 0, split),
        DenseMemoryView("rhs", "scratchpad", edge, split, operand_rows - split),
        DenseMemoryView("acc", "accumulator", edge * 4, 0, acc_rows),
    )
    problem = DenseProblem(m, k, n, edge, edge, edge, 1, 1, 4, bias=bias)
    resources = DenseResources(operand_rows * edge, acc_rows * edge * 4,
                              max_m_blocks=max_m_blocks, max_k_blocks=max_k_blocks,
                              max_n_blocks=max_n_blocks,
                              partial_output_tiles=True, partial_output_blocks=True)
    return schedule_compact_dense(problem, resources, storage, dense_adapter(),
                                  name="gemmini_matmul", dtypes=("i8", "i8", "i32"),
                                  static_budget=128, external_buffers_disjoint=True,
                                  bias_source=DenseBiasSource(n, "i32") if bias else None)


@lru_cache(maxsize=64)
def prove_shift_readout(shift, relu=False, lo=-(1 << 31), hi=(1 << 31)-1):
    """Prove half-up shift against the hardware's RNE readout over a domain.

    Both outputs are monotone saturated integer functions. The library computes
    every hardware transition over the complete interval; the source transition
    is the exact integer half-up boundary. Equality proves every accumulator,
    including values far above f32's exact-integer range.
    """
    from merlin.llvmlower.integer_readout import derive
    if type(shift) is not int:
        raise ValueError("requant shift must be a static integer")
    effective = max(0, shift)
    scale = 2.0 ** -min(effective, 149)
    target_proof = derive([scale], lo=lo, hi=hi, relu=relu)
    low = target_proof["output_min"]
    for q, target_threshold in zip(range(low+1, 128), target_proof["thresholds"]):
        threshold = (q << effective) - (1 << (effective-1)) if effective else q
        source_threshold = min(hi+1, max(lo, threshold))
        if source_threshold != target_threshold:
            witness = min(source_threshold, target_threshold)
            raise ValueError(f"half-up shift {shift} differs from RNE readout at accumulator {witness}; "
                             f"source transition {source_threshold}, target transition {target_threshold}")
    return {**target_proof, "scale": scale, "source_rounding": "integer half up",
            "source_shift": shift, "proof": "all source half-up and library hardware RNE output transitions equal"}


def output_intervals(epilogue, n, output_dtype="i32"):
    """Return equal-scale column runs from the declared readout semantics."""
    e = {} if epilogue is None else dict(epilogue)
    stages = e.get("epilogue", ())
    if isinstance(stages, str):
        stages = (stages,)
    relu = bool(e.get("relu", False) or "relu" in stages)
    dtype = e.get("out_dtype", e.get("dtype", output_dtype))
    if dtype not in ("i8", "i32"):
        raise ValueError("dense output dtype must be i8 or i32")
    scale = e.get("acc_scale_by_column", e.get("acc_scale", 1.0))
    if "requant" in stages or "requant_shift" in e:
        if "acc_scale" in stages or "acc_scale_by_column" in e or "acc_scale" in e:
            raise ValueError("combined floating scale and half-up shift require an ordered composition proof")
        shift = e.get("requant_shift", e.get("shift"))
        if shift is None:
            raise ValueError("requant stage requires an explicit shift")
        proof = prove_shift_readout(shift, relu=relu)
        scale = proof["scale"]
    scales = list(scale) if isinstance(scale, (list, tuple)) else [scale] * n
    if len(scales) != n:
        raise ValueError("per-column acc_scale length must equal N")
    if dtype == "i32" and (any(s != 1 for s in scales) or relu):
        raise ValueError("full-width readout cannot apply scale or ReLU")
    runs = []
    for col, s in enumerate(scales):
        from merlin.llvmlower.requantization import f32
        s = f32(float(s))
        if not math.isfinite(s):
            raise ValueError("accumulator scale must be a finite f32 value")
        if relu and s < 0:
            raise ValueError("negative scale cannot commute source ReLU through the target's accumulator ReLU")
        if runs and runs[-1][2] == s:
            first, width, _, relu, dt = runs[-1]
            runs[-1] = (first, width + 1, s, relu, dt)
        else:
            runs.append((col, 1, s, relu, dtype))
    return runs


def seed_bias(builder, bias_ptr, column, acc_row, rows, cols):
    try:
        from . import target
    except ImportError:
        import target
    b = builder
    target.cfg_ld(b, 0, stream=2, scale=1., shrunk=False)
    target.mvin(b, b.ptr(bias_ptr, b.mul_i(b.value(column), b.const(4))),
                b.add_i(b.value(acc_row), b.const(0x80000000)), rows=rows, cols=cols, stream=2)
    b.fence()


def lower_matmul(builder, lhs, rhs, out, m, k, n, epilogue=None, options=None):
    """Emit i8 row-major A[M,K] B[K,N] -> i32 C[M,N].

    Pointers are i64 byte addresses. Optional byte strides permit slices of a
    larger row-major allocation. ``epilogue(builder,out,m,n)`` runs once after
    completed output stores. Returns the shared plan for schedule evidence.
    """
    try:
        from . import target
    except ImportError:
        import target
    options = {} if options is None else dict(options)
    semantic_epilogue = epilogue if isinstance(epilogue, dict) else {}
    intervals = output_intervals(semantic_epilogue, n, options.get("output_dtype", "i32"))
    output_bytes = 1 if intervals[0][4] == "i8" else 4
    bias = options.get("bias", semantic_epilogue.get("bias_ptr"))
    if len(intervals) > 1:
        # Distinct readout scales do not change the product. Compute it once in
        # raw i32 storage, then reload equal-scale column regions for the exact
        # device bias/scale/activation path. This avoids duplicating a complete
        # GEMM schedule for every column multiplier.
        try:
            from .readout_helpers import lower_partitioned_readout
        except ImportError:
            from readout_helpers import lower_partitioned_readout
        if any(dtype != "i8" for _, _, _, _, dtype in intervals):
            raise ValueError("partitioned readout requires i8 output")
        temporary = builder.alloca(m * n, "i32")
        raw_options = {key:value for key,value in options.items() if key != "bias"}
        raw_options.update(out_stride=n * 4, _dense_serial=True)
        plan = lower_matmul(builder, lhs, rhs, temporary, m, k, n,
                            {"out_dtype": "i32"}, raw_options)
        scales = [None] * n
        for first, width, scale, relu, dtype in intervals:
            scales[first:first+width] = [scale] * width
        lower_partitioned_readout(builder, temporary, out, m, n, scales,
            relu=intervals[0][3], bias=bias,
            output_stride=options.get("out_stride", n))
        return plan
    if min(m, k, n) <= 0:
        raise ValueError("matmul requires positive static dimensions")
    strides = (options.get("lhs_stride", k), options.get("rhs_stride", n),
               options.get("out_stride", n * output_bytes))
    if any(type(s) is not int or s <= 0 for s in strides):
        raise ValueError("dense byte strides must be positive integers")
    # A scalar readout has one fixed configuration throughout the checked
    # pipeline. It only changes the final store width and address flag; all
    # partial sums remain in i32 accumulator rows until their sole final store.
    # Bias initialization and per-column reconfiguration retain the serial path.
    pipeline = (bias is None and intervals[0][4] in ("i32", "i8")
                and not callable(epilogue) and not options.get("_dense_serial", False))
    planner = plan_matmul if pipeline else _serial_plan_matmul
    resident_state = options.get("resident_state")
    if resident_state is not None and not pipeline:
        raise ValueError("resident handles require scalar bias-free pipeline readout")
    planning_options = {}
    if resident_state is not None:
        planning_options["resident"] = True
    elif pipeline and intervals[0][4] in ("i32", "i8"):
        from merlin.compile.tuning_knobs import knob
        retained = options.get("retain_full_lhs")
        if retained is None:
            retained = knob("retain_full_lhs", default=True, values=(False, True),
                            root=Path(__file__).resolve().parent)
        if type(retained) is not bool:
            raise ValueError("retain_full_lhs must be a Boolean")
        planning_options["segmented"] = retained
    max_m_blocks = options.get("max_m_blocks")
    if max_m_blocks is None:
        from merlin.compile.tuning_knobs import knob
        max_m_blocks = (knob("pipeline_max_m_blocks", default=8, values=(8, 16),
                             root=Path(__file__).resolve().parent) if pipeline else 8)
    plan = planner(m, k, n, max_m_blocks,
                       options.get("max_k_blocks", 8), options.get("max_n_blocks", 8), bias is not None,
                       **planning_options)
    if resident_state is not None and not hasattr(plan, "emit_resident_index_program"):
        raise ValueError("resident handle requires checked resident RHS plan")
    b = builder
    target.cfg_ex(b, dataflow=1)
    target.cfg_ld(b, strides[0], stream=0)
    target.cfg_ld(b, strides[1], stream=1)
    _, _, scale, relu, dtype = intervals[0]
    target.cfg_st(b, strides[2], scale=scale, activation=int(relu))

    def address(base, row, col, stride, itemsize=1):
        offset = b.add_i(b.mul_i(row, b.const(stride)), b.mul_i(col, b.const(itemsize)))
        return b.ptr(base, offset)

    def lower(instruction, a):
        a = {key:b.value(value) for key,value in a.items()}
        if instruction == "gemmini_bias":
            seed_bias(b, bias, a["col"], a["acc"], a["rows"], a["cols"])
        elif instruction in ("gemmini_lhs_load", "gemmini_weight_load"):
            is_lhs = instruction == "gemmini_lhs_load"
            ptr = address(lhs if is_lhs else rhs, a["row"], a["col"], strides[0 if is_lhs else 1])
            target.mvin(b, ptr, a["dst"], rows=a["rows"], cols=a["cols"], stream=0 if is_lhs else 1)
            if not pipeline:
                b.fence()
        elif instruction == "gemmini_preload":
            caddr = b.add_i(b.add_i(a["acc"], b.const(0x80000000)),
                            b.mul_i(a["continue"], b.const(0x40000000)))
            target.preload(b, a["weight"], caddr, brows=a["depth"], bcols=a["cols"],
                           crows=a["rows"], ccols=a["cols"])
        elif instruction == "gemmini_compute":
            target.compute(b, a["lhs"], arows=a["rows"], acols=a["depth"], flip=True)
            if not pipeline:
                b.fence()
        elif instruction == "gemmini_store":
            ptr = address(out, a["row"], a["col"], strides[2], output_bytes)
            target.mvout(b, ptr, b.add_i(a["acc"], b.const(0xa0000000 if dtype == "i32" else 0x80000000)),
                         rows=a["rows"], cols=a["cols"])
            if not pipeline:
                b.fence()
        else:
            raise ValueError("unhandled dense instruction: " + instruction)

    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    if pipeline:
        if resident_state is not None:
            plan.emit_resident_index_program(FnBuilderControlAdapter(b),lower,
                complete_epoch=b.fence,resident_live=resident_state["live"])
            resident_state["live"] = True
        else:
            plan.emit_index_program(FnBuilderControlAdapter(b), lower, complete_epoch=b.fence)
    else:
        plan.emit_llvm(b, lower)
    b.fence()
    if callable(epilogue):
        epilogue(b, out, m, n)
    return plan
