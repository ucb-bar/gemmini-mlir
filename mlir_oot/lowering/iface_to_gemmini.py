"""The interface -> target pass: a verified ``merlin_iface`` workload becomes a ``gemmini`` module.

One rewrite per interface op, all of them funnelling into the single weight-stationary contraction
emitter. The pass also decides the kernel's pointer ABI, following
``mlir_oot_backend_contract.yaml``'s ``arg_order_by_command_shape`` rows TOP-DOWN, and records the
command-shape it matched so the emitted artifact and the command buffer cannot disagree about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr

from ..ir.workload import Epilogue, Op, Workload
from ..target import dialect as gd
from ..target import facts
from ..target import layout
from . import conv as convlib
from .contraction import (
    AOperand,
    BiasSpec,
    KTile,
    LoweringRefusal,
    OutSpec,
    Stats,
    TileLoad,
    dense_a,
    dense_b,
    dense_ktiles,
    emit_contraction,
    transposed_b,
)
from .epilogue import NO_ACTIVATION, RELU, UnsupportedEpilogue, plan_readout
from .kernel import KernelBuilder
from . import hybrid
from .schedule import Scheduler
from .sumsq import PARTIAL_DTYPE

DIM = facts.DIM
WHOLE_OP_OPCODES = ("ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL", "CONV2D")

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "bf16": 2, "f16": 2, "f32": 4, "f64": 8}

#: What the mesh can READ. RTL-derived: ``datapaths`` reports the operand store as ``UInt<8>`` and
#: the accumulator as ``SInt<32>``; there is no floating-point operand port anywhere in this design.
MESH_OPERAND_DTYPES = ("i8",)
#: What the readout can WRITE: the raw accumulator, or the scaled/clamped operand width.
MESH_RESULT_DTYPES = ("i8", "i32")


def dtype_bytes(name: str) -> int:
    try:
        return DTYPE_BYTES[name]
    except KeyError as exc:
        raise LoweringRefusal(f"no on-chip container for dtype {name!r}") from exc


def require_mesh_dtypes(operands: dict[str, str], result: str, op: str) -> None:
    """Refuse a contraction whose operands the mesh has no encoding for.

    This is the one place the datapath is checked, and it is checked by NAME rather than by width:
    an f32 operand is the same four bytes as an i32 accumulator and would otherwise be moved in and
    contracted as if it were integer data, which is a wrong answer that looks like a right one.
    """
    for role, dt in operands.items():
        if dt not in MESH_OPERAND_DTYPES:
            raise LoweringRefusal(
                f"{op}: this mesh reads {'/'.join(MESH_OPERAND_DTYPES)} operands (RTL-derived "
                f"scratchpad UInt<8>); operand '{role}' is {dt}, for which it has no encoding"
            )
    if result not in MESH_RESULT_DTYPES:
        raise LoweringRefusal(
            f"{op}: the readout writes {'/'.join(MESH_RESULT_DTYPES)} (raw i32 accumulator, or the "
            f"scaled operand width); the declared result is {result}"
        )


def require_movement_dtypes(src: str, dst: str) -> None:
    """Refuse a movement whose containers this datapath cannot make the round trip in.

    Movement is the one interface operation that never enters the mesh: the ABI defines it as an
    identity load->store trip in which the values are carried UNCHANGED and only the container
    widens. So the datapath question it asks is not the mesh's -- it is which ON-CHIP container can
    hold the data on the way through. The RTL declares two: the operand scratchpad (`UInt<8>`) and
    the accumulator (`AccumulatorMem SInt<32>`). A source in either one is movable; a source in
    neither is not, and a NARROWING trip is not a movement at all, because carrying values
    unchanged into a smaller container is a clamp the ABI does not define.

    Kept beside :func:`require_mesh_dtypes` and called from both the lane gate and the lowering, so
    this package still has exactly one predicate per operation class rather than two that drift.
    """
    if dst not in MESH_RESULT_DTYPES:
        raise LoweringRefusal(
            f"movement: the readout writes {'/'.join(MESH_RESULT_DTYPES)}; the declared result "
            f"is {dst}"
        )
    if src not in MESH_OPERAND_DTYPES and dtype_bytes(src) != facts.ACC_BYTES_PER_ELEM:
        raise LoweringRefusal(
            f"movement: this target holds data on chip as {'/'.join(MESH_OPERAND_DTYPES)} operands "
            f"(RTL-derived scratchpad UInt<8>) or as i{8 * facts.ACC_BYTES_PER_ELEM} accumulator "
            f"rows (AccumulatorMem SInt<32>); operand 'src' is {src}, which is neither"
        )
    if dtype_bytes(dst) < dtype_bytes(src):
        raise LoweringRefusal(
            f"movement narrows {src} to {dst}: the ABI's movement carries its values UNCHANGED and "
            "only widens the container, so a narrowing trip is not one of its forms and this "
            "package will not invent a clamp for it"
        )


@dataclass
class Lowered:
    module: ModuleOp
    arg_tensors: list[str]
    command_shape: str
    stats: Stats
    kernel_abi: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    #: kernel-internal buffers the kernel allocates in its own frame (route W's digit staging)
    scratch: list[str] = field(default_factory=list)
    #: the (possibly REWRITTEN) workload the off-mesh stages were compiled against
    workload: Workload | None = None
    #: route X: the off-mesh stages, in marker order, that the code generator expands in place
    lane_stages: list[list[Op]] = field(default_factory=list)


# --------------------------------------------------------------------------- ABI
def _matmul_groups(wl: Workload) -> list[tuple[str, list[tuple[Op, Op]]]]:
    """Resident weight -> its (matmul, commit) pairs, in resident-pack then command order."""
    commit_of: dict[str, Op] = {}
    for op in wl.ops:
        if op.kind == "commit":
            commit_of[op.operands["src"]] = op
    order: list[str] = [op.out for op in wl.ops if op.kind == "resident_pack"]
    groups: dict[str, list[tuple[Op, Op]]] = {h: [] for h in order}
    for op in wl.ops:
        if op.kind != "matmul":
            continue
        commit = commit_of.get(op.out)
        if commit is None:
            raise LoweringRefusal("a matmul accumulator is never committed")
        groups.setdefault(op.operands["rhs"], []).append((op, commit))
    return [(h, groups.get(h, [])) for h in order]


def dram_reads(wl: Workload, op: Op) -> set[str]:
    """The DRAM tensors issuing ``op``'s commands will LOAD.

    A ``matmul`` issues nothing of its own -- its operands are loaded by the ``commit`` that reads
    its accumulator out -- so the matmul's reads are reported against that commit instead. Resident
    handles are resolved to the weight tensor they pack, because that is the buffer the DMA names.
    """
    names: set[str] = set()

    def take(ref: str | None) -> None:
        if not ref:
            return
        name = wl.residents.get(ref, ref)
        if name in wl.tensors:
            names.add(name)

    if op.kind in ("matmul", "evict"):
        return names            # the commit carries them / an evict touches no DRAM
    if op.kind == "commit":
        for mm in wl.ops:
            if mm.kind == "matmul" and mm.out == op.operands.get("src"):
                take(mm.operands.get("lhs"))
                take(mm.operands.get("rhs"))
    else:
        for ref in op.operands.values():
            take(ref)
    if op.epilogue is not None:
        take(op.epilogue.bias)
    return names


def dram_writes(wl: Workload, op: Op) -> set[str]:
    """The DRAM tensors issuing ``op``'s commands will STORE."""
    if op.kind in ("matmul", "resident_pack", "evict"):
        return set()
    many = op.attrs.get("outs")
    if many:
        return {n for n in many if n in wl.tensors}
    return {op.out} if op.out in wl.tensors else set()


def plan_abi(wl: Workload) -> tuple[str, list[str], dict[str, Any] | None]:
    """Pick the command shape and the pointer order, trying the contract's rows top-down."""
    kinds = {op.kind for op in wl.ops}
    whole_ops = [op for op in wl.ops if op.kind in ("conv2d", "matmul_batched", "attention_qk", "attention_pv")]

    def _whole_program() -> tuple[str, list[str], dict[str, Any]]:
        args = list(wl.decl_order)
        return (
            "whole_program",
            args,
            {
                "kind": "whole_program",
                "args": [
                    {
                        "tensor": n,
                        "access": (
                            "write"
                            if wl.tensors[n].role in ("output", "intermediate")
                            else "read"
                        ),
                    }
                    for n in args
                ],
                "outputs": [n for n in args if wl.tensors[n].role == "output"],
            },
        )

    if hybrid.is_hybrid(wl):
        # A kernel that runs stages off the array reaches EVERY declared tensor -- the off-mesh
        # ones through ordinary loads and stores, which no narrower row can name. The contract's
        # own first row declares the pointer boundary explicitly, so that is the one it takes.
        return _whole_program()

    if kinds & {"bias_add", "k_chain", "residual_add", "depthwise_conv2d"}:
        return _whole_program()

    if wl.quant is not None:
        # Route Q's weight is a kernel-internal staging buffer, so the resident-matmul row -- whose
        # first pointer block IS the resident weight -- would name a buffer the harness does not
        # have. Declaring the pointer boundary explicitly is the contract's own first row.
        return _whole_program()

    if "resident_pack" not in kinds and "movement" in kinds:
        mv = next(op for op in wl.ops if op.kind == "movement")
        return "movement", [mv.operands["src"], mv.out], None

    if len(whole_ops) == 1 and not any(op.kind == "matmul" for op in wl.ops):
        # The native-whole-op harness renders one pointer per external tensor from its declared
        # RANK, so a rank-1 operand (a per-channel bias) has no shape it can render. Declaring the
        # pointer boundary explicitly is the contract's own answer to that, and it is the row tried
        # FIRST -- not a different program, the same one with its ABI written down.
        if any(len(wl.tensors[n].shape) < 2 for n in wl.decl_order):
            return _whole_program()
        return "native_whole_op", list(wl.decl_order), None

    weights: list[str] = []
    lhs: list[str] = []
    outs: list[str] = []
    biases: list[str] = []
    for handle, pairs in _matmul_groups(wl):
        weights.append(wl.residents[handle])
        for mm, commit in pairs:
            lhs.append(mm.operands["lhs"])
            outs.append(commit.out)
            ep = commit.epilogue or Epilogue()
            if ep.bias and ("bias_add" in ep.has or "bias" in ep.has):
                biases.append(ep.bias)
    return "resident_matmul", weights + lhs + outs + biases, None


# --------------------------------------------------------------------------- the pass
def rewrite_conv_to_residency(wl: Workload) -> None:
    """Lower a convolution into the RESIDENCY command form when the whole-op pointer ABI cannot
    carry one of its operands.

    The native-whole-op harness renders one pointer per external tensor from that tensor's declared
    RANK, and refuses a rank < 2 one -- which is exactly what a per-channel bias is. The contract's
    own alternative is the residency form, "including a CONV2D the package itself lowers to that
    form (its derived im2col lhs then IS a pointer argument)", with the activation gather declared
    through `params.im2col_recipes` so the reference, the simulator and the device are handed
    byte-identical stimulus. This is a narrower capability than the address-stream im2col every
    other convolution takes -- the gather is the harness's, and the buffer SAYS so -- and it is
    applied only where the wider route has no pointer ABI at all.
    """
    convs = [op for op in wl.ops if op.kind == "conv2d"]
    if not convs or any(op.kind == "matmul" for op in wl.ops):
        return
    external = [t for t in wl.tensors.values() if t.role in ("input", "weight", "bias", "output")]
    if all(len(t.shape) >= 2 for t in external):
        return

    rebuilt: list[Op] = []
    for op in wl.ops:
        if op.kind != "conv2d":
            if op.kind in ("resident_pack", "evict"):
                continue
            rebuilt.append(op)
            continue
        ifm = wl.tensors[op.operands["ifm"]]
        handle = op.operands["weight"]
        weight = wl.tensors[wl.residents.get(handle, handle)]
        g = convlib.geometry_from(op.attrs, ifm.shape, weight.shape)
        im2col = f"{ifm.name}_im2col"
        wl.tensors[im2col] = type(ifm)(im2col, (g.m, g.k), ifm.dtype, "input")
        wl.im2col_recipes.append(
            {
                "target": im2col, "source": ifm.name,
                "kh": g.kh, "kw": g.kw, "ci": g.ci,
                "stride": [g.sh, g.sw],
                "padding": [g.pad_t, g.pad_l, g.pad_b, g.pad_r],
                "dilation": [g.dh, g.dw],
                "layout": op.attrs.get("layout") or "nhwc",
            }
        )
        res = f"res_{im2col}"
        acc = f"acc_{op.out}"
        wl.residents[res] = weight.name
        rebuilt.append(Op("resident_pack", res, {"src": weight.name}, {"layout": "packed_conv_rhs"}))
        rebuilt.append(
            Op("matmul", acc, {"lhs": im2col, "rhs": res}, {},
               shape={"m": g.m, "k": g.k, "n": g.co})
        )
        rebuilt.append(Op("commit", op.out, {"src": acc}, {}, epilogue=op.epilogue))
        rebuilt.append(Op("evict", "", {"handle": res}))
    wl.ops = rebuilt


class IfaceToGemmini:
    """The interface -> gemmini rewrite pipeline."""

    def __init__(self, scheduler: Scheduler | None = None) -> None:
        self.scheduler = scheduler or Scheduler()

    def run(self, wl: Workload) -> Lowered:
        if not wl.ops and wl.lane_placement:
            return self._host_lane(wl)
        rewrite_conv_to_residency(wl)
        shape, args, abi = plan_abi(wl)
        # Staging buffers are addressed exactly like a DRAM tensor but are allocated by the kernel
        # itself, so they are kernel operands without being ABI pointers.
        staged = [s.dst for s in wl.quant.staged] if wl.quant is not None else []
        staged += [n for n in wl.scratch if n not in staged]
        b = KernelBuilder(args + staged)
        # The harness fills the operand buffers before it calls in, so the kernel opens by ordering
        # those stores against the DMA it is about to issue; Rocket's fence also drains any
        # accelerator command still in flight from a previous call.
        b.fence()
        b.flush(0)
        total = Stats()
        # A tensor an earlier region STORED and a later region LOADS is a dependency the DMA does
        # not carry on its own: the load may issue while the store is still in flight, which the
        # functional planes never see because they retire each command before the next. So the
        # dependency is ordered explicitly, once, at the region that consumes it.
        in_flight: set[str] = set()
        # Route X: each off-mesh stage becomes ONE marker in the stream, at the position its
        # operations occupy in the program. The stage itself is compiled by the code generator into
        # the same function body; here it only has to hold its place and order its memory.
        lane_stages = [st.ops for st in hybrid.scalar_stages(wl)]
        stage_of = {id(op): i for i, ops in enumerate(lane_stages) for op in ops}
        opened: set[int] = set()
        for op in wl.ops:
            if dram_reads(wl, op) & in_flight:
                b.fence()
                in_flight.clear()
            idx = stage_of.get(id(op))
            if idx is not None:
                if idx not in opened:
                    b.lane_stage(idx)
                    opened.add(idx)
                in_flight |= dram_writes(wl, op)
                continue
            st = self._lower_op(b, wl, op)
            in_flight |= dram_writes(wl, op)
            if st is not None:
                total.mvin += st.mvin
                total.mvout += st.mvout
                total.compute += st.compute
        b.fence()
        b.terminate()
        kernel = gd.KernelOp.build(
            regions=[b.region()],
            attributes={"sym_name": StringAttr("gemmini_kernel"),
                        "command_shape": StringAttr(shape)},
        )
        module = ModuleOp([kernel])
        module.verify()
        return Lowered(module, args, shape, total, abi, lane_stages=lane_stages,
                       scratch=list(wl.scratch), workload=wl)

    def _host_lane(self, wl: Workload) -> Lowered:
        """Route H: every region is declared on the host lane, so the mesh issues NOTHING.

        The kernel still exists and still orders memory, because the harness links against the
        symbol either way -- but it carries no accelerator command, which is the whole claim.
        """
        args = list(wl.decl_order)
        b = KernelBuilder(args)
        b.fence()
        b.terminate()
        kernel = gd.KernelOp.build(
            regions=[b.region()],
            attributes={"sym_name": StringAttr("gemmini_kernel"),
                        "command_shape": StringAttr("host_lane")},
        )
        module = ModuleOp([kernel])
        module.verify()
        abi = {
            "kind": "whole_program",
            "args": [
                {
                    "tensor": n,
                    "access": "write" if wl.tensors[n].role in ("output", "intermediate") else "read",
                }
                for n in args
            ],
            "outputs": [n for n in args if wl.tensors[n].role == "output"],
        }
        return Lowered(module, args, "host_lane", Stats(), abi)

    # -- per-op rewrites ---------------------------------------------------
    def _lower_op(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats | None:
        if op.kind in ("resident_pack", "evict"):
            return None
        if op.kind == "matmul":
            return None  # emitted by its commit, which carries the readout
        if op.kind == "commit":
            return self._matmul_commit(b, wl, op)
        if op.kind == "conv2d":
            return self._conv(b, wl, op)
        if op.kind == "movement":
            return self._movement(b, wl, op)
        if op.kind == "matmul_batched":
            return self._batched(b, wl, op)
        if op.kind in ("attention_qk", "attention_pv"):
            return self._attention(b, wl, op)
        if op.kind == "row_sumsq":
            return self._row_sumsq(b, wl, op)
        if op.kind == "bias_add":
            return self._bias_add(b, wl, op)
        if op.kind == "residual_add":
            return self._residual_add(b, wl, op)
        raise LoweringRefusal(f"no lowering for interface operation {op.kind!r}")

    def _matmul(self, wl: Workload, acc_id: str) -> Op:
        for op in wl.ops:
            if op.kind == "matmul" and op.out == acc_id:
                return op
        raise LoweringRefusal("a commit consumes an accumulator no matmul produced")

    def _matmul_commit(self, b: KernelBuilder, wl: Workload, commit: Op) -> Stats:
        mm = self._matmul(wl, commit.operands["src"])
        lhs = wl.tensors[mm.operands["lhs"]]
        weight = wl.tensors[wl.residents[mm.operands["rhs"]]]
        m, k, n = mm.shape["m"], mm.shape["k"], mm.shape["n"]
        out_decl = wl.tensors[commit.out]
        require_mesh_dtypes(
            {"lhs": lhs.dtype, "weight": weight.dtype}, out_decl.dtype, "matmul"
        )
        ep = commit.epilogue or Epilogue()
        readout = plan_readout(ep, m)
        out = wl.tensors[commit.out]
        bias = BiasSpec(readout.bias, 0) if readout.bias else None
        if bias is not None and bias.tensor not in b.arg_tensors:
            raise LoweringRefusal(f"the bias tensor {bias.tensor!r} is not a kernel argument")
        out_pitch = layout.row_pitch(out.shape)
        return emit_contraction(
            b,
            a=dense_a(lhs.name, m, k, dtype_bytes(lhs.dtype), layout.row_pitch(lhs.shape)),
            bop=dense_b(weight.name, k, n, dtype_bytes(weight.dtype), layout.row_pitch(weight.shape)),
            ktiles=dense_ktiles(k),
            m=m, n=n,
            out=OutSpec(out.name, out_pitch, out_pitch * dtype_bytes(out.dtype)),
            readout=readout,
            bias=bias,
            scheduler=self.scheduler,
        )

    def _conv(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        ifm = wl.tensors[op.operands["ifm"]]
        wname = op.operands["weight"]
        weight = wl.tensors[wl.residents.get(wname, wname)]
        g = convlib.geometry_from(op.attrs, ifm.shape, weight.shape)
        out = wl.tensors[op.out]
        require_mesh_dtypes({"ifm": ifm.dtype, "weight": weight.dtype}, out.dtype, "conv2d")
        ep = op.epilogue or Epilogue()
        readout = plan_readout(ep, g.m)
        expected_rows = g.m
        if readout.pool_enabled:
            # A fused pool commits the POOLED extent, and its declared pool_in_dims must be the
            # conv's own output extent -- a disagreement is rejected, never reconciled.
            if tuple(ep.pool_in_dims or ()) != (g.ho, g.wo):
                raise LoweringRefusal(
                    f"conv2d pool_in_dims {tuple(ep.pool_in_dims or ())} disagrees with the "
                    f"geometry's output extent [{g.ho}, {g.wo}]"
                )
            expected_rows = readout.pool_batches * readout.porows * readout.pocols
        if tuple(out.shape) != (expected_rows, g.co):
            raise LoweringRefusal(
                f"conv2d result {tuple(out.shape)} disagrees with the geometry's "
                f"[{expected_rows}, {g.co}]"
            )
        bias = BiasSpec(readout.bias, 0) if readout.bias else None
        out_pitch = layout.row_pitch(out.shape)
        return emit_contraction(
            b,
            a=convlib.conv_a_operand(ifm.name, g, layout.row_pitch(ifm.shape)),
            bop=dense_b(weight.name, g.k, g.co, dtype_bytes(weight.dtype),
                        layout.row_pitch(weight.shape)),
            ktiles=convlib.conv_ktiles(g),
            m=g.m, n=g.co,
            out=OutSpec(out.name, out_pitch, out_pitch * dtype_bytes(out.dtype)),
            readout=readout,
            bias=bias,
            scheduler=self.scheduler,
        )

    def _movement(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        src = wl.tensors[op.operands["src"]]
        dst = wl.tensors[op.out]
        if len(src.shape) != 2 or tuple(src.shape) != tuple(dst.shape):
            raise LoweringRefusal(
                f"movement expects one rank-2 extent through the accelerator; got "
                f"{tuple(src.shape)} -> {tuple(dst.shape)}"
            )
        rows, cols = src.shape
        require_movement_dtypes(src.dtype, dst.dtype)
        in_b, out_b = dtype_bytes(src.dtype), dtype_bytes(dst.dtype)
        in_pitch, out_pitch = layout.row_pitch(src.shape), layout.row_pitch(dst.shape)
        # An accumulator-width container at either end forces the trip through the accumulator; the
        # load is `shrunk` only when the SOURCE is the narrow one being widened on the way in.
        via_acc = facts.ACC_BYTES_PER_ELEM in (in_b, out_b)
        widen = via_acc
        st = Stats()
        b.config_ex(act=0)
        b.config_ld(0, in_pitch * in_b, shrunk=via_acc and in_b < facts.ACC_BYTES_PER_ELEM)
        b.config_st(out_pitch * out_b)
        pool = max(1, facts.ACC_ROWS // DIM) if widen else max(1, facts.SP_ROWS // DIM)
        idx = 0
        for i in range((rows + DIM - 1) // DIM):
            r = min(DIM, rows - i * DIM)
            for j in range((cols + DIM - 1) // DIM):
                c = min(DIM, cols - j * DIM)
                slot = (idx % pool) * DIM
                idx += 1
                local_in = facts.acc_addr(slot, False, False) if widen else slot
                local_out = facts.acc_addr(slot, True, True) if widen else slot
                b.mvin(b.addr(src.name, (i * DIM * in_pitch + j * DIM) * in_b), local_in, c, r)
                b.mvout(b.addr(dst.name, (i * DIM * out_pitch + j * DIM) * out_b), local_out, c, r)
                st.mvin += 1
                st.mvout += 1
        return st

    def _batched(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        a = wl.tensors[op.operands["a"]]
        w = wl.tensors[op.operands["w"]]
        out = wl.tensors[op.out]
        *prefix, m, k = a.shape
        batch = 1
        for d in prefix:
            batch *= d
        n = w.shape[-1]
        if w.shape[-2] != k or tuple(out.shape[-2:]) != (m, n):
            raise LoweringRefusal("batched matmul extents do not contract")
        require_mesh_dtypes({"a": a.dtype, "w": w.dtype}, out.dtype, "matmul_batched")
        ep = op.epilogue or Epilogue()
        readout = plan_readout(ep, m)
        ab, wb, ob = dtype_bytes(a.dtype), dtype_bytes(w.dtype), dtype_bytes(out.dtype)
        a_pitch = layout.row_pitch(a.shape)
        w_pitch = layout.row_pitch(w.shape)
        o_pitch = layout.row_pitch(out.shape)
        st = Stats()
        for bi in range(batch):
            sub = emit_contraction(
                b,
                a=_offset_a(dense_a(a.name, m, k, ab, a_pitch), bi * m * a_pitch * ab),
                bop=_offset_b(dense_b(w.name, k, n, wb, w_pitch), bi * k * w_pitch * wb),
                ktiles=dense_ktiles(k),
                m=m, n=n,
                out=OutSpec(out.name, o_pitch, o_pitch * ob),
                readout=readout,
                bias=None,
                scheduler=self.scheduler,
                out_base=bi * m * o_pitch * ob,
            )
            st.mvin += sub.mvin
            st.mvout += sub.mvout
            st.compute += sub.compute
        return st

    def _attention(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        qk = op.kind == "attention_qk"
        lhs = wl.tensors[op.operands["q" if qk else "p"]]
        rhs = wl.tensors[op.operands["k" if qk else "v"]]
        out = wl.tensors[op.out]
        m, d = lhs.shape
        require_mesh_dtypes({"lhs": lhs.dtype, "rhs": rhs.dtype}, out.dtype, op.kind)
        ep = op.epilogue or Epilogue()
        readout = plan_readout(ep, m)
        lb, rb, ob = dtype_bytes(lhs.dtype), dtype_bytes(rhs.dtype), dtype_bytes(out.dtype)
        if qk:
            n = rhs.shape[0]
            if rhs.shape[1] != d:
                raise LoweringRefusal("attention_qk contracts the trailing head dim of both operands")
            bop = transposed_b(rhs.name, n, d, rb, layout.row_pitch(rhs.shape))
        else:
            n = rhs.shape[1]
            if rhs.shape[0] != d:
                raise LoweringRefusal("attention_pv: p's key extent must match v's row extent")
            bop = dense_b(rhs.name, d, n, rb, layout.row_pitch(rhs.shape))
        if tuple(out.shape) != (m, n):
            raise LoweringRefusal(f"attention result {tuple(out.shape)} is not [{m}, {n}]")
        out_pitch = layout.row_pitch(out.shape)
        return emit_contraction(
            b,
            a=dense_a(lhs.name, m, d, lb, layout.row_pitch(lhs.shape)),
            bop=bop,
            ktiles=dense_ktiles(d),
            m=m, n=n,
            out=OutSpec(out.name, out_pitch, out_pitch * ob),
            readout=readout,
            bias=None,
            scheduler=self.scheduler,
            b_transpose=qk,
        )

    def _row_sumsq(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        """``ss[i] = sum_k x[i,k]**2``, contracted on the mesh one row BAND at a time.

        Band ``bd`` is the DIM-row slice of ``src`` starting at ``bd*DIM``. It is contracted against
        ITSELF -- the moving operand is the band, the stationary operand is the same band read
        through ``b_transpose`` (which is what :func:`.contraction.transposed_b` expresses: a
        stationary operand stored row-per-output-column, and a band's rows ARE the output columns
        here). The result is a ``rows_b x rows_b`` tile whose diagonal holds the band's reductions,
        committed full width into the band's slot of the staged intermediate.

        Every extent below comes from the tensor the capsule declared; the band edge is the
        RTL-derived array edge, not a constant chosen here.
        """
        src = wl.tensors[op.operands["src"]]
        ss = wl.tensors[op.out]
        require_mesh_dtypes({"lhs": src.dtype, "rhs": src.dtype}, ss.dtype, "row_sumsq")
        rows = layout.row_count(src.shape)
        k = int(src.shape[-1])
        sb = dtype_bytes(src.dtype)
        s_pitch = layout.row_pitch(src.shape)
        ss_pitch = layout.row_pitch(ss.shape)
        # The intermediate is read out RAW: the row scale has not been applied yet, so no
        # activation and no accumulator scale ride this store.
        readout = plan_readout(Epilogue(output_dtype=PARTIAL_DTYPE), DIM)
        st = Stats()
        for band, row0 in enumerate(range(0, rows, DIM)):
            nb = min(DIM, rows - row0)
            base = row0 * s_pitch * sb
            sub = emit_contraction(
                b,
                a=_offset_a(dense_a(src.name, nb, k, sb, s_pitch), base),
                bop=_offset_b(transposed_b(src.name, nb, k, sb, s_pitch), base),
                ktiles=dense_ktiles(k),
                m=nb, n=nb,
                out=OutSpec(ss.name, ss_pitch, ss_pitch * 4),
                readout=readout,
                bias=None,
                scheduler=self.scheduler,
                b_transpose=True,
                out_base=band * DIM * ss_pitch * 4,
            )
            st.mvin += sub.mvin
            st.mvout += sub.mvout
            st.compute += sub.compute
        return st

    def _acc_add(
        self,
        b: KernelBuilder,
        wl: Workload,
        out,
        srcs: list[tuple[Any, bool, float]],
        act: int,
        acc_scale: float,
    ) -> Stats:
        """Sum several equal-extent tensors inside the ACCUMULATOR and read them out once.

        This is the shape both `bias_add` and `residual_add` have on this datapath: the array's
        accumulator is a read-modify-write memory whose load path carries a per-unit f32 scale, so a
        pure elementwise sum needs no pass through the mesh at all -- which is why the derived
        instruction coverage for these ops asks for movement and a store, and for no compute.

        ``srcs`` is ``(tensor, broadcast, scale)`` in accumulate order: the first lands with the
        accumulate bit CLEAR, so it initialises the rows rather than adding to whatever a previous
        command left there; every later one lands with the bit SET. A ``broadcast`` operand is a
        rank-1 vector replayed under every row, which the load unit expresses as a zero row pitch.
        """
        rows, cols = out.shape
        ob = dtype_bytes(out.dtype)
        o_pitch = layout.row_pitch(out.shape)
        full = ob == 4
        if full and (act != NO_ACTIVATION or acc_scale != 1.0):
            raise LoweringRefusal(
                "a full-width (i32) readout reads the raw accumulator, so it cannot carry the "
                "declared activation or accumulator scale"
            )
        if len(srcs) > 2:
            raise LoweringRefusal(
                "this load path has two register sets that decode cleanly, so at most two operands "
                "can be summed in one accumulator pass"
            )
        # One CONFIG_LD per operand, hoisted out of the tile loop: each operand keeps its own load
        # unit, so its row pitch, its narrow/wide container and its scale are configured once.
        units = []
        for unit, (t, broadcast, scale) in enumerate(srcs):
            tb = dtype_bytes(t.dtype)
            pitch = 0 if broadcast else layout.row_pitch(t.shape) * tb
            b.config_ld(unit, pitch, scale=scale, shrunk=tb < 4)
            units.append((unit, t, broadcast, tb))
        b.config_ex(act=0)
        b.config_st(o_pitch * ob, act=act, acc_scale=acc_scale)
        st = Stats()
        pool = max(1, facts.ACC_ROWS // DIM)
        idx = 0
        for i in range((rows + DIM - 1) // DIM):
            r = min(DIM, rows - i * DIM)
            for j in range((cols + DIM - 1) // DIM):
                c = min(DIM, cols - j * DIM)
                slot = (idx % pool) * DIM
                idx += 1
                for n, (unit, t, broadcast, tb) in enumerate(units):
                    off = (j * DIM) * tb if broadcast else (i * DIM * layout.row_pitch(t.shape) + j * DIM) * tb
                    local = facts.acc_addr(slot, n > 0, False)
                    mv = b.mvin if unit == 0 else b.mvin2
                    mv(b.addr(t.name, off), local, c, r)
                    st.mvin += 1
                b.mvout(b.addr(out.name, (i * DIM * o_pitch + j * DIM) * ob),
                        facts.acc_addr(slot, True, full), c, r)
                st.mvout += 1
        return st

    def _bias_add(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        """``dst[i, j] = src[i, j] + bias[j]`` -- the bias COMMIT stage standing on its own.

        Folded into the accumulator read-out, which is what the ABI says a target with no separate
        vector-add class is expected to do. The bias is a length-N vector replayed under every row.
        """
        src = wl.tensors[op.operands["src"]]
        bias = wl.tensors[op.operands["bias"]]
        out = wl.tensors[op.out]
        if len(src.shape) != 2 or tuple(out.shape) != tuple(src.shape):
            raise LoweringRefusal("bias_add expects a rank-2 source committed at the same extent")
        n = out.shape[1]
        if bias.shape[-1] != n or (len(bias.shape) == 2 and bias.shape[0] != 1):
            raise LoweringRefusal(
                f"bias_add's bias must be one length-{n} vector; got {tuple(bias.shape)}"
            )
        ep = op.epilogue or Epilogue()
        stages = ep.has - {"bias_add", "bias"}
        act = RELU if "relu" in stages else NO_ACTIVATION
        if stages - {"relu"}:
            raise LoweringRefusal(
                f"bias_add epilogue stage(s) {sorted(stages - {'relu'})} have no readout path here"
            )
        # The bias initialises the accumulator rows and the source adds onto them; addition
        # commutes, and this way the broadcast operand is the one that needs no row pitch.
        return self._acc_add(b, wl, out, [(bias, True, 1.0), (src, False, 1.0)], act, 1.0)

    def _residual_add(self, b: KernelBuilder, wl: Workload, op: Op) -> Stats:
        """``dst = relu?(sat(roundeven(lhs*lhs_scale + rhs*rhs_scale)))`` -- a residual connection.

        The reference rounds ONCE, in f32, and this datapath has two places it can multiply: the
        load unit's per-operand scale and the store path's accumulator scale. Putting the WHOLE
        multiplier on the load units rounds each operand separately, which is a different function
        and needs the capsule's `bound_lsb` slack to be legal at all.

        So the pair is FACTORED instead. Every finite f32 is a dyadic rational, so for some `t`
        both `lhs_scale * 2**t` and `rhs_scale * 2**t` are integers: load each operand with its
        INTEGER multiplier (exact -- an integer times an integer needs no rounding), let the
        accumulator add them exactly, and hand the single remaining factor `2**-t` to the store's
        accumulator scale, which rounds ONCE. That reproduces the reference's own arithmetic, so
        `bound_lsb = 0` is met for any pair the factorisation's exactness bound admits -- and the
        capsules that do declare slack no longer need it.

        The bound is derived from the DECLARED operand containers, not assumed: the reference's own
        f32 arithmetic is exact only while the scaled sum stays inside the f32 significand, so the
        worst-case magnitude each container can reach is what decides it. A pair that does not
        factor inside the bound keeps the old per-operand scaling, which remains legal from
        `bound_lsb >= 1` and is refused by name below it.
        """
        lhs = wl.tensors[op.operands["lhs"]]
        rhs = wl.tensors[op.operands["rhs"]]
        out = wl.tensors[op.out]
        if len(out.shape) != 2 or tuple(lhs.shape) != tuple(out.shape) or tuple(rhs.shape) != tuple(out.shape):
            raise LoweringRefusal(
                f"residual_add sums two tensors at the output's own rank-2 extent; got "
                f"{tuple(lhs.shape)} + {tuple(rhs.shape)} -> {tuple(out.shape)}"
            )
        ls = float(op.attrs.get("lhs_scale", 1.0))
        rs = float(op.attrs.get("rhs_scale", 1.0))
        bound = int(op.attrs.get("bound_lsb", 0))
        ep = op.epilogue or Epilogue()
        stages = ep.has
        if stages - {"relu"}:
            raise LoweringRefusal(
                f"residual_add epilogue stage(s) {sorted(stages - {'relu'})} have no readout path here"
            )
        act = RELU if "relu" in stages else NO_ACTIVATION
        if (ls, rs) == (1.0, 1.0):
            # No multiplier anywhere: the accumulator's integer sum IS the reference.
            return self._acc_add(b, wl, out, [(lhs, False, 1.0), (rhs, False, 1.0)], act, 1.0)
        split = _factor_scales(ls, rs, lhs.dtype, rhs.dtype, out.dtype)
        if split is not None:
            p, q, store = split
            return self._acc_add(b, wl, out, [(lhs, False, p), (rhs, False, q)], act, store)
        if bound < 1:
            raise LoweringRefusal(
                f"residual_add declares bound_lsb {bound} with multipliers ({ls}, {rs}): the "
                "reference rounds the scaled sum ONCE. The multipliers do not factor into integer "
                "load scales and one store scale within this container's exactness bound, so the "
                "only remaining path rounds each operand into the output's domain before adding -- "
                "legal only from bound_lsb 1"
            )
        return self._acc_add(b, wl, out, [(lhs, False, ls), (rhs, False, rs)], act, 1.0)


#: The f32 significand. A product of integers below this is represented exactly, so the reference's
#: own f32 evaluation of the scaled sum does no rounding of its own and the single store-path
#: rounding is the only one either side performs.
F32_EXACT = 1 << 24

#: How far the factorisation is allowed to push the binary point. Every finite f32 is a dyadic
#: rational, but a multiplier needing more than this many halvings cannot also keep its integer
#: partner inside `F32_EXACT` for any container this target loads.
MAX_BINARY_SHIFT = 24


def _container_max(dtype: str) -> int:
    """The largest magnitude the DECLARED container can hold. Read from the dtype, never assumed."""
    return 1 << (8 * dtype_bytes(dtype) - 1)


def _factor_scales(
    ls: float, rs: float, lhs_dtype: str, rhs_dtype: str, out_dtype: str
) -> tuple[float, float, float] | None:
    """Split ``(ls, rs)`` into integer LOAD multipliers and one STORE multiplier, or ``None``.

    Returns ``(p, q, 2**-t)`` with ``p = ls * 2**t`` and ``q = rs * 2**t`` both integral, so the
    load path scales exactly, the accumulator adds exactly, and the store's accumulator scale is
    the only rounding in the program -- the same single rounding the reference performs.

    ``None`` means no such split is usable here: the readout is full width (the raw accumulator
    carries no store scale), a multiplier is not finite, or the worst case the declared containers
    admit would leave the f32 significand, where the reference's own arithmetic starts rounding and
    the equality this split is FOR no longer holds.
    """
    for v in (ls, rs):
        if v != v or v in (float("inf"), float("-inf")):
            return None
    for t in range(MAX_BINARY_SHIFT + 1):
        p, q = ls * (2.0 ** t), rs * (2.0 ** t)
        if not (p.is_integer() and q.is_integer()):
            continue
        store = 2.0 ** -t
        if store != 1.0 and dtype_bytes(out_dtype) == 4:
            return None         # a full-width readout reads the raw accumulator; no store scale
        worst = abs(p) * _container_max(lhs_dtype) + abs(q) * _container_max(rhs_dtype)
        if worst > F32_EXACT:
            return None
        return p, q, store
    return None


def _offset_a(a: AOperand, base: int) -> AOperand:
    def loads(i: int, kti: int) -> list[TileLoad]:
        return [
            TileLoad(l.row_offset, l.tensor, l.byte_offset + base, l.cols, l.rows)
            for l in a.loads(i, kti)
        ]

    return AOperand(m=a.m, stride_bytes=a.stride_bytes, loads=loads, runs=a.runs,
                    shrunk=a.shrunk, elem_bytes=a.elem_bytes)


def _offset_b(bop, base: int):
    def load(kspec: KTile, jti: int, ncols: int, n: int) -> TileLoad:
        l = bop.load(kspec, jti, ncols, n)
        return TileLoad(l.row_offset, l.tensor, l.byte_offset + base, l.cols, l.rows)

    return type(bop)(tensor=bop.tensor, stride_bytes=bop.stride_bytes, load=load,
                     elem_bytes=bop.elem_bytes)
