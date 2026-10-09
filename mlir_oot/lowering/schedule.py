"""Tile scheduling: a kernel plan -> an ordered stream of gemmini commands.

The schedule is derived from the RTL facts (mesh `DIM`, scratchpad depth, accumulator depth) and
from the extents of the capsule that was handed in.  Nothing here is keyed on a capsule name, a
particular shape, or a particular epilogue value: the loop nest walks `ceil(extent / DIM)` tiles
in each of M, K and N, and every field it packs is read from the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tables import isa
from ..tables import rtl_facts as F
from .model_lane import HostSegment
from .plan import (
    Buffer,
    row_pitch,
    Contraction,
    DTYPE_BYTES,
    Epilogue,
    HostBiasAdd,
    LoweringDeclined,
    Movement,
    Plan,
    Transpose,
    pool_out_dims,
)

DIM = F.DIM

#: The emitted kernel is straight-line code (the program oracle faults on a branch), so the CPU
#: lane's cost is paid in emitted instructions.  Past this budget the lowering DECLINES rather
#: than producing a multi-hundred-megabyte artifact nobody can assemble.
HOST_LANE_OP_BUDGET = 120_000
#: The same guard for the accelerator stream itself.
COMMAND_BUDGET = 1_000_000


@dataclass
class Instr:
    """One scheduled command (an accelerator instruction or a compiler-generated host fixup)."""

    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    #: names of the DRAM buffers this command touches, in operand order
    bufs: list[str] = field(default_factory=list)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class Readout:
    """How one commit's epilogue is split between the store path and the CPU lane."""

    mode: str                                   # "native" (straight to dst) | "scratch"
    stage_dtype: str                            # element type the MVOUT writes
    acc_act: int                                # activation encoded in CONFIG_ST
    acc_scale: float                            # scale encoded in CONFIG_ST
    native_stages: list[str] = field(default_factory=list)
    host_stages: list[str] = field(default_factory=list)


def readout_plan(e: Epilogue) -> Readout:
    """Split the declared epilogue between the accumulator store path and the CPU lane.

    The store path applies its activation and float scale only on the NARROWING readout
    (`AccumulatorScale` feeds the full-width port the raw accumulator), so a full-width i32
    readout carries no stage.  `maxpool` always lands on the CPU lane because it changes the
    extent, but it commutes with the monotone stages before it, so those still go to the store
    path and the pool then runs over the already-narrowed values.
    """
    stages = [s for s in e.stages if s not in ("bias_add", "bias")]
    pool = "maxpool" in stages
    pre = [s for s in stages if s != "maxpool"]
    # The store path has exactly two readout widths: the accumulator's own dtype (full width, raw)
    # and the mesh operand dtype (narrowed through the activation + scale unit).  Any OTHER output
    # width has to be produced by the CPU lane from the full-width readout.
    narrowing = e.output_dtype == F.OPERAND_DTYPE
    full_width = e.output_dtype == F.ACCUMULATOR_DTYPE
    if narrowing and set(pre) <= {"acc_scale", "relu"} and e.acc_scale > 0.0:
        native_stages, host_stages, stage_dtype = pre, [], e.output_dtype
    else:
        native_stages, host_stages, stage_dtype = [], pre, F.ACCUMULATOR_DTYPE
    mode = ("native"
            if (not pool and not host_stages and stage_dtype == e.output_dtype)
            else "scratch")
    return Readout(
        mode=mode,
        stage_dtype=stage_dtype,
        # CONFIG_ST states the DECLARED activation even when a full-width readout bypasses it,
        # so the decoded trace reflects the epilogue the capsule asked for.
        acc_act=isa.RELU if "relu" in stages else isa.NO_ACTIVATION,
        acc_scale=e.acc_scale if "acc_scale" in native_stages else 1.0,
        native_stages=native_stages,
        host_stages=host_stages)


class Scheduler:
    """Walks a plan and produces the command stream plus any staging buffers it needs."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.instrs: list[Instr] = []
        self.staging: dict[str, Buffer] = {}
        self._ld_stride: int | None = None
        self._ld_shrunk: bool | None = None
        self._st_key: tuple | None = None
        self._ex_key: tuple | None = None
        self._stage_id = 0

    # -- helpers -----------------------------------------------------------------------------
    def emit(self, __kind: str, __bufs: list[str] | None = None, **attrs) -> None:
        self.instrs.append(Instr(__kind, attrs, list(__bufs or [])))

    def stage_buffer(self, shape: list[int], dtype: str, hint: str) -> Buffer:
        name = f"__stage_{self._stage_id}_{hint}"
        self._stage_id += 1
        buf = Buffer(name, list(shape), dtype, "scratch")
        self.staging[name] = buf
        self.plan.buffers[name] = buf
        return buf

    def buf(self, name: str) -> Buffer:
        return self.plan.buffers[name]

    def config_ld(self, stride: int, *, shrunk: bool = False) -> None:
        if (stride, shrunk) == (self._ld_stride, self._ld_shrunk):
            return
        self._ld_stride, self._ld_shrunk = stride, shrunk
        self.emit("config_ld", stride=stride, shrunk=shrunk, load_id=0, scale=1.0)

    def config_st(self, **kw) -> None:
        key = tuple(sorted(kw.items()))
        if key == self._st_key:
            return
        self._st_key = key
        self.emit("config_st", **kw)

    def config_ex(self, **kw) -> None:
        key = tuple(sorted(kw.items()))
        if key == self._ex_key:
            return
        self._ex_key = key
        self.emit("config_ex", **kw)

    # -- entry point -------------------------------------------------------------------------
    def run(self) -> list[Instr]:
        if not self.plan.tasks:
            # Nothing was placed on the mesh.  Emit no accelerator command at all: a capsule
            # whose family this datapath does not admit must leave NOTHING in the instruction
            # stream, not even a flush, or the placement is not actually on the host lane.
            return self.instrs
        # open with a fence so the kernel starts from a quiescent accelerator, then flush the
        # pipeline before configuring it (the trace gate reads the opening fence).
        self.emit("fence")
        self.emit("flush", skip=0)
        for task in self.plan.tasks:
            if isinstance(task, Contraction):
                self.contraction(task)
            elif isinstance(task, Movement):
                self.movement(task)
            elif isinstance(task, Transpose):
                self.emit("host_transpose", [task.src, task.dst],
                          rows=task.rows, cols=task.cols, dtype=task.dtype)
            elif isinstance(task, HostSegment):
                self.emit("host_segment",
                          [n for _, n in task.inputs] + [n for _, n in task.outputs],
                          segment=task, regions_placed=list(task.regions) or ["glue"])
            elif isinstance(task, HostBiasAdd):
                self.bias_add(task)
            else:                                                   # pragma: no cover - guarded
                raise LoweringDeclined(f"unscheduled task {type(task).__name__}")
        self.emit("fence")
        if len(self.instrs) > COMMAND_BUDGET:
            raise LoweringDeclined(
                f"the tile schedule needs {len(self.instrs)} accelerator commands, past this "
                f"backend's {COMMAND_BUDGET} budget for a fully unrolled program")
        return self.instrs

    # -- data movement -----------------------------------------------------------------------
    def movement(self, mv: Movement) -> None:
        src, dst = self.buf(mv.src), self.buf(mv.dst)
        operand, accum = F.OPERAND_DTYPE, F.ACCUMULATOR_DTYPE
        # The two on-chip stores this round trip can pass through are the operand scratchpad
        # (`operand` elements) and the accumulator (`accum` elements).  A container the RTL has
        # neither width for would be moved as whatever the hardware's element width happens to
        # be -- the data would come back reinterpreted, which is a WRONG ANSWER wearing the shape
        # of a correct one -- so it is refused with the widths stated.
        legal = {(operand, operand), (operand, accum), (accum, accum)}
        if (src.dtype, dst.dtype) not in legal:
            raise LoweringDeclined(
                f"movement {src.dtype} -> {dst.dtype} has no on-chip container on this target: "
                f"the scratchpad holds {operand} elements and the accumulator {accum}, so the "
                f"legal round trips are {sorted(legal)}",
                op="movement", shape=[mv.rows, mv.cols])
        in_bytes = DTYPE_BYTES[src.dtype]
        out_bytes = DTYPE_BYTES[dst.dtype]
        # `widen` means the round trip is staged in the ACCUMULATOR rather than the scratchpad:
        # either because the readout is wider than the operand (i8 -> i32) or because the data is
        # already accumulator-width and the operand store cannot hold it (i32 -> i32).  The load
        # is `shrunk` only in the first case -- that bit says the DRAM side is operand-width.
        widen = dst.dtype == accum
        shrunk = widen and src.dtype == operand
        # the mesh is untouched by a pure round trip, but the dataflow/stride state is still
        # part of the accelerator's configuration, so state it explicitly.
        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=1, c_stride=1)
        src_pitch, dst_pitch = src.pitch, dst.pitch
        self.config_ld(src_pitch * in_bytes, shrunk=shrunk)
        self.config_st(stride=dst_pitch * out_bytes, acc_act=isa.NO_ACTIVATION, acc_scale=1.0)
        slots = 8
        slot = 0
        for i in range(_ceil_div(mv.rows, DIM)):
            rows = min(DIM, mv.rows - i * DIM)
            for j in range(_ceil_div(mv.cols, DIM)):
                cols = min(DIM, mv.cols - j * DIM)
                if widen:
                    row = (slot % slots) * DIM
                    if row + DIM > F.ACC_ROWS:
                        row = 0
                    local_in = isa.acc_addr(row)
                    local_out = isa.acc_addr(row, full_row=True)
                else:
                    row = (slot % slots) * DIM
                    local_in = local_out = row
                slot += 1
                self.emit("mvin", [mv.src], local=local_in, rows=rows, cols=cols, load_id=0,
                          offset=(i * DIM * src_pitch + j * DIM) * in_bytes)
                self.emit("mvout", [mv.dst], local=local_out, rows=rows, cols=cols,
                          offset=(i * DIM * dst_pitch + j * DIM) * out_bytes)

    # -- the standalone bias add -------------------------------------------------------------
    def bias_add(self, t: HostBiasAdd) -> None:
        """`dst[i, j] = src[i, j] + bias[j]`, FOLDED INTO THE ACCUMULATOR READ-OUT.

        This datapath has no vector-add class, and the ABI says so: a target without one is
        expected to fold the op into its accumulator read-out.  The shipped ISA header spells the
        idiom out in `sp_tiled_matmul_ws` (the `repeating_bias` move-in with a ZERO DRAM row
        stride into `D_sp_addr_start = 1 << (ADDR_LEN-1)`) and in `sp_tiled_resadd` (a second
        move-in into `3 << (ADDR_LEN-2)`, the same address with the accumulate bit set, then a
        move-out of the summed row).  So per output tile: move the bias row in with a zero row
        stride so it repeats down the tile, move the source in ON TOP with the accumulate bit,
        and read the accumulator row out.

        Both operands are in the ACCUMULATOR's dtype -- the ABI states that outright, because the
        bias lands on the accumulator before any requant.  A declaration in any other width is
        left to the CPU lane rather than fed to an accumulator port that does not carry it.
        """
        src, bias, dst = self.buf(t.src), self.buf(t.bias), self.buf(t.dst)
        acc_dt = F.ACCUMULATOR_DTYPE
        if not (src.dtype == bias.dtype == dst.dtype == acc_dt):
            self.emit("host_bias_add", [t.src, t.bias, t.dst],
                      rows=t.rows, cols=t.cols, dtype=t.dtype)
            return
        wide = DTYPE_BYTES[acc_dt]
        src_pitch, dst_pitch = src.pitch, dst.pitch
        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=1, c_stride=1)
        self.config_st(stride=dst_pitch * wide, acc_act=isa.NO_ACTIVATION, acc_scale=1.0)
        slot = 0
        for i in range(_ceil_div(t.rows, DIM)):
            rows = min(DIM, t.rows - i * DIM)
            for j in range(_ceil_div(t.cols, DIM)):
                cols = min(DIM, t.cols - j * DIM)
                row = (slot % 8) * DIM
                if row + DIM > F.ACC_ROWS:
                    row = 0
                slot += 1
                # the bias vector repeats down the tile: a zero DRAM row stride is what the ISA
                # header's `repeating_bias` path uses, so one length-N row feeds every output row
                self.config_ld(0)
                self.emit("mvin", [t.bias], load_id=0, local=isa.acc_addr(row),
                          rows=rows, cols=cols, offset=j * DIM * wide)
                self.config_ld(src_pitch * wide)
                self.emit("mvin", [t.src], load_id=0,
                          local=isa.acc_addr(row, accumulate=True), rows=rows, cols=cols,
                          offset=(i * DIM * src_pitch + j * DIM) * wide)
                self.emit("mvout", [t.dst], local=isa.acc_addr(row, full_row=True),
                          rows=rows, cols=cols,
                          offset=(i * DIM * dst_pitch + j * DIM) * wide)

    # -- budget ------------------------------------------------------------------------------
    #: The epilogue stages the GENERATED CPU-lane readout implements.  `maxpool` is the windowed
    #: max; the rest are the elementwise stages `codegen.llvm_emit._apply_scalar_stages` builds.
    #: Kept here, next to the split that decides what lands on the CPU lane, so a stage with no
    #: readout is refused while the PLAN is being built -- every entrypoint then answers the same
    #: way about the capsule, instead of the command buffer claiming a program the artifact
    #: cannot emit (which reads as a crashed tool, not as a refusal).
    HOST_READOUT_STAGES = frozenset({"relu", "requant", "acc_scale", "maxpool"})

    def _check_host_stages(self, c: Contraction, stages: list[str]) -> None:
        """Refuse an epilogue stage the generated CPU-lane readout does not implement."""
        unsupported = [s for s in stages if s not in self.HOST_READOUT_STAGES]
        if unsupported:
            raise LoweringDeclined(
                f"epilogue stage {unsupported[0]!r} has no generated CPU-lane readout, and this "
                f"commit's readout does not go through the accelerator store path",
                op=unsupported[0], shape=[c.m, c.n])

    def _check_host_budget(self, c: Contraction, e: Epilogue, ro, stages: list[str]) -> None:
        """Refuse a CPU-lane readout whose straight-line form would not be assemblable."""
        per_element = 6 + 4 * len(ro.host_stages)
        if e.has_pool:
            ih, iw = e.pool_in_dims
            ph, pw = e.pool_size
            ho, wo = pool_out_dims(ih, iw, e.pool_size, e.pool_stride, e.pool_padding)
            out_rows = (c.m // (ih * iw)) * ho * wo
            cost = out_rows * c.n * (ph * pw * (per_element + 4) + 10)
        else:
            cost = c.m * c.n * per_element
        if cost > HOST_LANE_OP_BUDGET:
            raise LoweringDeclined(
                f"the epilogue {stages} over a [{c.m}, {c.n}] readout needs about {cost} "
                f"straight-line CPU-lane instructions, past this backend's {HOST_LANE_OP_BUDGET} "
                f"budget; the emitted kernel must be single-block on this target, so the work "
                f"cannot be rolled into a loop",
                op=stages[-1] if stages else "commit", shape=[c.m, c.n])

    # -- contraction -------------------------------------------------------------------------
    def block_shape(self, mt: int, nt: int, kt: int) -> tuple[int, int, int]:
        """Tile-block extents that respect the RTL accumulator and scratchpad depths."""
        acc_tiles = F.ACC_ROWS // DIM                 # C tiles resident at once
        spad_tiles = F.SPAD_ROWS // DIM               # A + B tiles resident at once
        bn = min(nt, 8)
        bm = min(mt, max(1, acc_tiles // bn))
        bn = min(nt, max(1, acc_tiles // bm))
        bk = min(kt, max(1, spad_tiles // (bm + bn)))
        while bk > 1 and (bm + bn) * bk > spad_tiles:
            bk -= 1
        return bm, bn, bk

    def contraction(self, c: Contraction) -> None:
        lhs, rhs, dst = self.buf(c.lhs), self.buf(c.rhs), self.buf(c.dst)
        if lhs.dtype != "i8" or rhs.dtype != "i8":
            raise LoweringDeclined(
                f"the gemmini mesh contracts i8 operands; got {lhs.dtype} x {rhs.dtype}",
                op="matmul", shape=[c.m, c.k, c.n])
        e = c.epilogue
        ro = readout_plan(e)
        mode = ro.mode
        lhs_pitch = row_pitch(c.lhs_row_elems)
        rhs_pitch = row_pitch(c.rhs_row_elems)
        lhs_base = getattr(c, "lhs_batch_rows", 0) * lhs_pitch
        rhs_base = getattr(c, "rhs_batch_rows", 0) * rhs_pitch

        if mode == "native":
            out_buf = dst
            out_row_elems = row_pitch(c.dst_row_elems or c.n)
            out_row_offset = c.dst_row_offset
        else:
            out_buf = self.stage_buffer([c.m, c.n], ro.stage_dtype, "acc")
            out_row_elems = row_pitch(c.n)
            out_row_offset = 0

        out_dtype = ro.stage_dtype
        full_row = out_dtype == "i32"
        out_bytes = DTYPE_BYTES[out_dtype]
        acc_act, acc_scale = ro.acc_act, ro.acc_scale

        mt, nt, kt = _ceil_div(c.m, DIM), _ceil_div(c.n, DIM), _ceil_div(c.k, DIM)
        bm, bn, bk = self.block_shape(mt, nt, kt)
        a_base_tile = 0
        b_base_tile = bm * bk

        self.config_ex(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                       acc_scale=1.0, a_stride=1, c_stride=1)

        for m0 in range(0, mt, bm):
            for n0 in range(0, nt, bn):
                for k0 in range(0, kt, bk):
                    mi = list(range(m0, min(m0 + bm, mt)))
                    ni = list(range(n0, min(n0 + bn, nt)))
                    ki = list(range(k0, min(k0 + bk, kt)))
                    if e.bias and k0 == 0:
                        bias_buf = self.buf(e.bias)
                        self.config_ld(0)
                        for a, i in enumerate(mi):
                            for d, j in enumerate(ni):
                                self.emit("mvin", [e.bias], load_id=0,
                                          local=isa.acc_addr((a * bn + d) * DIM),
                                          rows=min(DIM, c.m - i * DIM),
                                          cols=min(DIM, c.n - j * DIM),
                                          offset=j * DIM * DTYPE_BYTES[bias_buf.dtype])
                    self.config_ld(lhs_pitch * DTYPE_BYTES[lhs.dtype])
                    for a, i in enumerate(mi):
                        for b, k in enumerate(ki):
                            self.emit("mvin", [c.lhs], load_id=0,
                                      local=(a_base_tile + a * bk + b) * DIM,
                                      rows=min(DIM, c.m - i * DIM),
                                      cols=min(DIM, c.k - k * DIM),
                                      offset=(lhs_base + i * DIM * lhs_pitch + k * DIM)
                                      * DTYPE_BYTES[lhs.dtype])
                    self.config_ld(rhs_pitch * DTYPE_BYTES[rhs.dtype])
                    for b, k in enumerate(ki):
                        for d, j in enumerate(ni):
                            self.emit("mvin", [c.rhs], load_id=0,
                                      local=(b_base_tile + b * bn + d) * DIM,
                                      rows=min(DIM, c.k - k * DIM),
                                      cols=min(DIM, c.n - j * DIM),
                                      offset=(rhs_base + k * DIM * rhs_pitch + j * DIM)
                                      * DTYPE_BYTES[rhs.dtype])
                    for b, k in enumerate(ki):
                        for d, j in enumerate(ni):
                            for a, i in enumerate(mi):
                                first_i = a == 0
                                accumulate = bool(e.bias) or k != 0
                                c_row = (a * bn + d) * DIM
                                self.emit(
                                    "preload",
                                    bd=(b_base_tile + b * bn + d) * DIM if first_i
                                    else isa.GARBAGE_ADDR,
                                    c=isa.acc_addr(c_row, accumulate=accumulate),
                                    bd_cols=min(DIM, c.n - j * DIM),
                                    bd_rows=min(DIM, c.k - k * DIM),
                                    c_cols=min(DIM, c.n - j * DIM),
                                    c_rows=min(DIM, c.m - i * DIM))
                                self.emit(
                                    "compute",
                                    a=(a_base_tile + a * bk + b) * DIM,
                                    bd=isa.GARBAGE_ADDR,
                                    a_cols=min(DIM, c.k - k * DIM),
                                    a_rows=min(DIM, c.m - i * DIM),
                                    accumulate=not first_i)
                # readout of this [bm x bn] accumulator block
                mi = list(range(m0, min(m0 + bm, mt)))
                ni = list(range(n0, min(n0 + bn, nt)))
                self.config_st(stride=out_row_elems * out_bytes, acc_act=acc_act,
                               acc_scale=acc_scale)
                for a, i in enumerate(mi):
                    for d, j in enumerate(ni):
                        self.emit("mvout", [out_buf.name],
                                  local=isa.acc_addr((a * bn + d) * DIM, full_row=full_row),
                                  rows=min(DIM, c.m - i * DIM),
                                  cols=min(DIM, c.n - j * DIM),
                                  offset=((out_row_offset + i * DIM) * out_row_elems + j * DIM)
                                  * out_bytes)

        if mode == "native":
            return
        stages = list(ro.host_stages) + (["maxpool"] if e.has_pool else [])
        self._check_host_stages(c, stages)
        self._check_host_budget(c, e, ro, stages)
        self.emit("fence")
        bufs = [out_buf.name, c.dst]
        self.emit("host_epilogue", bufs, stages=stages, rows=c.m, cols=c.n,
                  stage_dtype=ro.stage_dtype,
                  out_dtype=e.output_dtype, acc_scale=e.acc_scale,
                  requant_shift=e.requant_shift,
                  pool_in_dims=e.pool_in_dims, pool_size=e.pool_size,
                  pool_stride=e.pool_stride, pool_padding=e.pool_padding,
                  pool_pad_value=e.pool_pad_value,
                  src_row_elems=out_row_elems,
                  src_row_offset=out_row_offset,
                  dst_row_elems=row_pitch(dst.shape[-1]),
                  dst_row_offset=c.dst_row_offset)


def schedule(plan: Plan) -> tuple[list[Instr], dict[str, Buffer]]:
    s = Scheduler(plan)
    return s.run(), s.staging


def epilogue_out_rows(rows: int, e: Epilogue) -> int:
    if not e.has_pool:
        return rows
    ih, iw = e.pool_in_dims
    ho, wo = pool_out_dims(ih, iw, e.pool_size, e.pool_stride, e.pool_padding)
    return (rows // (ih * iw)) * ho * wo
