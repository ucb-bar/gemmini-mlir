"""Development aid: execute the package's OWN emitted gemmini-dialect program.

This is a self-consistency check, not an oracle: it models the datapath this backend targets (the
scratchpad, the accumulator, the weight-stationary mesh and the readout) from the same RTL-derived
facts the compiler uses, runs the emitted instruction stream on synthetic operands, and compares the
result with a direct evaluation of the interface program. It catches a wrong tile address, a wrong
accumulate bit or a mis-shaped im2col run in SECONDS, without a simulator and without any golden.

    python devtools/simulate.py ../isa/A2_single_tile_matmul
    python devtools/simulate.py --all
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mlir_oot.driver import read_workload  # noqa: E402
from mlir_oot.lowering.iface_to_gemmini import IfaceToGemmini, dtype_bytes  # noqa: E402
from mlir_oot.target import dialect as gd  # noqa: E402
from mlir_oot.target import facts, layout  # noqa: E402
from mlir_oot.target.dialect import _float, _int  # noqa: E402

DIM = facts.DIM
NP_DTYPE = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64}


def deterministic(name: str, shape, dtype: str) -> np.ndarray:
    """Any repeatable operand will do; this check is about the program, not the numbers."""
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    n = int(np.prod(shape))
    if dtype == "i8":
        return rng.integers(-8, 8, size=n, dtype=np.int64).reshape(shape).astype(np.int8)
    return rng.integers(-200, 200, size=n, dtype=np.int64).reshape(shape).astype(np.int32)


class Dram:
    """The padded row-major buffers the harness allocates, addressed in bytes."""

    def __init__(self, wl):
        self.buf: dict[str, np.ndarray] = {}
        self.pitch: dict[str, int] = {}
        self.dt: dict[str, str] = {}
        for name, t in wl.tensors.items():
            p = layout.row_pitch(t.shape)
            rows = layout.row_count(t.shape)
            self.pitch[name] = p
            self.dt[name] = t.dtype
            self.buf[name] = np.zeros(rows * p, dtype=NP_DTYPE[t.dtype])

    def logical(self, name: str, shape) -> np.ndarray:
        p, rows = self.pitch[name], layout.row_count(shape)
        return self.buf[name].reshape(rows, p)[:, : shape[-1]].reshape(shape)

    def store_logical(self, name: str, shape, values: np.ndarray) -> None:
        p, rows = self.pitch[name], layout.row_count(shape)
        v = self.buf[name].reshape(rows, p)
        v[:, : shape[-1]] = values.reshape(rows, shape[-1])

    def read(self, name: str, byte_off: int, count: int, elem_bytes: int) -> np.ndarray:
        idx = byte_off // elem_bytes
        view = self.buf[name].view(np.int8 if elem_bytes == 1 else np.int32)
        return view[idx : idx + count].astype(np.int64)

    def write(self, name: str, byte_off: int, values: np.ndarray, elem_bytes: int) -> None:
        idx = byte_off // elem_bytes
        view = self.buf[name].view(np.int8 if elem_bytes == 1 else np.int32)
        view[idx : idx + len(values)] = values.astype(view.dtype)


class Machine:
    def __init__(self, dram: Dram, args: list[str]):
        self.dram = dram
        self.args = args
        self.spad = np.zeros((facts.SP_ROWS, DIM), dtype=np.int64)
        self.acc = np.zeros((facts.ACC_ROWS, DIM), dtype=np.int64)
        self.ld = {i: {"stride": 0, "shrunk": 0, "scale": 1.0} for i in range(3)}
        self.st = {"stride": 0, "act": 0, "acc_scale": 1.0}
        self.weights = np.zeros((DIM, DIM), dtype=np.int64)
        self.pending = np.zeros((DIM, DIM), dtype=np.int64)
        self.c_target = (0, 0, 0)
        # ExecuteController: in WS, `bd_transpose` routes the PRELOADED operand through the
        # transposer, and the rs1 rows/cols fields are read swapped.
        self.b_transpose = False

    # -- readout -----------------------------------------------------------
    def _scaled(self, x: np.ndarray) -> np.ndarray:
        if self.st["act"] == 1:
            x = np.maximum(x, 0)
        y = np.asarray(x, dtype=np.float64) * float(np.float32(self.st["acc_scale"]))
        y = np.asarray([_round_near_even(v) for v in y.reshape(-1)]).reshape(x.shape)
        return np.clip(y, -128, 127)

    def run(self, kernel) -> None:
        addr: dict = {}
        for op in kernel.body.block.ops:
            if isinstance(op, gd.DramAddrOp):
                buf = op.operands[0]
                name = self.args[list(kernel.body.block.args).index(buf)]
                addr[op.results[0]] = (name, _int(op, "offset", 0))
            elif isinstance(op, gd.ConfigExOp):
                self.b_transpose = bool(_int(op, "b_transpose", 0))
            elif isinstance(op, gd.ConfigLdOp):
                self.ld[_int(op, "id")] = {
                    "stride": _int(op, "stride"), "shrunk": _int(op, "shrunk", 0),
                    "scale": _float(op, "scale", 1.0),
                }
            elif isinstance(op, gd.ConfigStOp):
                self.st = {
                    "stride": _int(op, "stride"), "act": _int(op, "act", 0),
                    "acc_scale": _float(op, "acc_scale", 1.0),
                    "pool_size": _int(op, "pool_size", 0),
                    "pool_stride": _int(op, "pool_stride", 0),
                    "orows": _int(op, "orows", 0), "ocols": _int(op, "ocols", 0),
                    "porows": _int(op, "porows", 0), "pocols": _int(op, "pocols", 0),
                    "pool_out_dim": _int(op, "pool_out_dim", 0),
                    "upad": _int(op, "upad", 0), "lpad": _int(op, "lpad", 0),
                }
            elif isinstance(op, (gd.MvinOp, gd.Mvin2Op, gd.Mvin3Op)):
                unit = {gd.MvinOp: 0, gd.Mvin2Op: 1, gd.Mvin3Op: 2}[type(op)]
                self._mvin(unit, addr[op.operands[0]], op)
            elif isinstance(op, gd.MvoutOp):
                self._mvout(addr[op.operands[0]], op)
            elif isinstance(op, gd.PreloadOp):
                bd = _int(op, "bd")
                if bd != facts.GARBAGE_ADDR:
                    w = np.zeros((DIM, DIM), dtype=np.int64)
                    r, c = _int(op, "bd_rows"), _int(op, "bd_cols")
                    if self.b_transpose:
                        w[:r, :c] = self.spad[bd : bd + c, :r].T
                    else:
                        w[:r, :c] = self.spad[bd : bd + r, :c]
                    self.pending = w
                self.c_target = (_int(op, "c"), _int(op, "c_cols"), _int(op, "c_rows"))
            elif isinstance(op, gd.ComputeOp):
                if _int(op, "preloaded", 1):
                    self.weights = self.pending
                self._compute(op)

    def _mvin(self, unit: int, src, op) -> None:
        """One load transfer, which may carry several DIM-wide column blocks.

        ``LoadController.scala`` places block ``b`` of a transfer at ``spad + block_stride * b``
        and the compiler always declares ``block_stride = DIM``, so a ``cols`` wider than DIM is
        modelled here as ``ceil(cols/DIM)`` consecutive tiles, each reading its own contiguous
        slice of the same DRAM row. A transfer of exactly one block is the ordinary case and is
        bit-identical to what this model did before blocks existed.
        """
        name, base = src
        local, cols, rows = _int(op, "spad"), _int(op, "cols"), _int(op, "rows")
        stride = self.ld[unit]["stride"]
        to_acc = bool(local & facts.BIT_IS_ACC)
        wide = to_acc and not self.ld[unit]["shrunk"]
        eb = 4 if wide else 1
        scale = float(np.float32(self.ld[unit].get("scale", 1.0)))
        blocks = max(1, -(-cols // DIM))
        for blk in range(blocks):
            width = min(DIM, cols - blk * DIM)
            for r in range(rows):
                data = self.dram.read(name, base + r * stride + blk * DIM * eb, width, eb)
                if scale != 1.0:
                    # LoadController applies the CONFIG_LD f32 scale as the row enters the array;
                    # the product is rounded into the destination container, not truncated.
                    y = data.astype(np.float64) * scale
                    data = np.asarray(
                        [_round_near_even(v) for v in y.reshape(-1)]
                    ).reshape(data.shape)
                if to_acc:
                    row = (local & 0x3FFF) + blk * DIM + r
                    if local & facts.BIT_ACCUMULATE:
                        self.acc[row, :width] += data
                    else:
                        self.acc[row, :width] = data
                else:
                    self.spad[local + blk * DIM + r, :width] = data

    def _mvout(self, dst, op) -> None:
        name, base = dst
        local, cols, rows = _int(op, "spad"), _int(op, "cols"), _int(op, "rows")
        full = bool(local & facts.BIT_READ_FULL_ACC)
        from_acc = bool(local & facts.BIT_IS_ACC)
        eb = 4 if (from_acc and full) else 1
        if self.st.get("pool_stride"):
            self._pooled_mvout(name, base, local, cols, eb)
            return
        for r in range(rows):
            row = (local & 0x3FFF) + r
            data = self.acc[row, :cols] if from_acc else self.spad[row, :cols]
            if from_acc and not full:
                data = self._scaled(data)
            self.dram.write(name, base + r * self.st["stride"], data, eb)

    def _pooled_mvout(self, name, base, local, cols, eb) -> None:
        st = self.st
        ph, sh = st["pool_size"], st["pool_stride"]
        for porow in range(st["porows"]):
            for pocol in range(st["pocols"]):
                best = None
                for wr in range(ph):
                    for wc in range(ph):
                        orow = porow * sh + wr - st["upad"]
                        ocol = pocol * sh + wc - st["lpad"]
                        if orow < 0 or ocol < 0 or orow >= st["orows"] or ocol >= st["ocols"]:
                            v = np.zeros(cols, dtype=np.int64)
                        else:
                            row = (local & 0x3FFF) + orow * st["ocols"] + ocol
                            v = self.acc[row, :cols]
                            if not (local & facts.BIT_READ_FULL_ACC):
                                v = self._scaled(v)
                        best = v if best is None else np.maximum(best, v)
                off = base + (porow * st["pool_out_dim"] + pocol) * st["stride"]
                self.dram.write(name, off, best, eb)

    def _compute(self, op) -> None:
        a, a_cols, a_rows = _int(op, "a"), _int(op, "a_cols"), _int(op, "a_rows")
        c_addr, c_cols, c_rows = self.c_target
        lhs = self.spad[a : a + a_rows, :a_cols]
        prod = lhs @ self.weights[:a_cols, :c_cols]
        row0 = c_addr & 0x3FFF
        n = min(a_rows, c_rows)
        if c_addr & facts.BIT_ACCUMULATE:
            self.acc[row0 : row0 + n, :c_cols] += prod[:n]
        else:
            self.acc[row0 : row0 + n, :c_cols] = prod[:n]


def _round_near_even(v: float) -> int:
    import math

    f = math.floor(v)
    rem = v - f
    if rem < 0.5:
        return int(f)
    if rem > 0.5:
        return int(f + 1)
    return int(f if f % 2 == 0 else f + 1)


# --------------------------------------------------------------------------- reference
def reference(wl, dram: Dram) -> dict[str, np.ndarray]:
    """Evaluate the interface program, CHAINED: a stage reads the value an earlier stage produced.

    Only the values this side computes are chained -- nothing is written back to DRAM. Writing them
    back would pre-fill the intermediates the emitted program is supposed to produce, and would mask
    exactly the bug (a stage that never stores its result) this tool exists to catch.
    """
    out: dict[str, np.ndarray] = {}
    accs: dict[str, np.ndarray] = {}

    def read(name: str) -> np.ndarray:
        """An operand: the value an earlier stage of THIS program produced, else what DRAM holds."""
        if name in out:
            return out[name]
        return dram.logical(name, wl.tensors[name].shape)

    for op in wl.ops:
        if op.kind == "matmul":
            lhs = read(op.operands["lhs"])
            wn = wl.residents[op.operands["rhs"]]
            rhs = read(wn)
            accs[op.out] = lhs.astype(np.int64) @ rhs.astype(np.int64)
        elif op.kind == "commit":
            out[op.out] = _epilogue(accs[op.operands["src"]], op.epilogue, wl, dram)
        elif op.kind == "conv2d":
            out[op.out] = _conv(op, wl, dram)   # leaf operands only; no chained input
        elif op.kind == "movement":
            src = read(op.operands["src"])
            out[op.out] = src.astype(np.int64)
        elif op.kind == "matmul_batched":
            a = read(op.operands["a"]).astype(np.int64)
            w = read(op.operands["w"]).astype(np.int64)
            out[op.out] = a @ w
        elif op.kind == "attention_qk":
            q = read(op.operands["q"]).astype(np.int64)
            k = read(op.operands["k"]).astype(np.int64)
            out[op.out] = _epilogue(q @ k.T, op.epilogue, wl, dram)
        elif op.kind == "attention_pv":
            p = read(op.operands["p"]).astype(np.int64)
            v = read(op.operands["v"]).astype(np.int64)
            out[op.out] = _epilogue(p @ v, op.epilogue, wl, dram)
        elif op.kind == "bias_add":
            s = read(op.operands["src"]).astype(np.int64)
            bnm = op.operands["bias"]
            b = read(bnm).astype(np.int64)
            lo, hi = _dtype_range(op.attrs.get("output_dtype") or wl.tensors[op.out].dtype)
            out[op.out] = np.clip(s + b.reshape(1, -1), lo, hi)
        elif op.kind == "residual_add":
            # command_buffer_abi RESIDUAL_ADD: both products and the sum in IEEE single precision,
            # rounded ONCE to nearest-even, saturated into the output dtype, then the activation.
            l = read(op.operands["lhs"])
            r = read(op.operands["rhs"])
            ls = np.float32(float(op.attrs.get("lhs_scale", 1.0)))
            rs = np.float32(float(op.attrs.get("rhs_scale", 1.0)))
            y = (l.astype(np.float32) * ls + r.astype(np.float32) * rs).astype(np.float64)
            x = np.asarray([_round_near_even(v) for v in y.reshape(-1)]).reshape(y.shape)
            lo, hi = _dtype_range(op.attrs.get("output_dtype") or wl.tensors[op.out].dtype)
            x = np.clip(x, lo, hi)
            if "relu" in ((op.epilogue.stages if op.epilogue else ())):
                x = np.maximum(x, 0)
            out[op.out] = x
    return out


def tolerances(wl) -> dict[str, int]:
    """The absolute per-element slack each produced tensor is graded at.

    Only `residual_add` declares one: its reference rounds the scaled sum ONCE while a scaled,
    rounding load rounds each operand, and `bound_lsb` is the number of output steps the capsule
    admits between the two. Everything else is exact.
    """
    return {
        op.out: int(op.attrs.get("bound_lsb", 0))
        for op in wl.ops
        if op.kind == "residual_add"
    }


def _dtype_range(dt: str) -> tuple[int, int]:
    bits = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}[dt]
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _epilogue(acc: np.ndarray, ep, wl, dram) -> np.ndarray:
    if ep is None:
        return acc
    x = acc.astype(np.int64)
    for stage in ep.stages:
        if stage in ("bias_add", "bias"):
            b = dram.logical(ep.bias, wl.tensors[ep.bias].shape).astype(np.int64)
            x = x + b.reshape(1, -1)
        elif stage == "relu":
            x = np.maximum(x, 0)
        elif stage == "acc_scale":
            y = x.astype(np.float64) * float(np.float32(ep.acc_scale))
            x = np.asarray([_round_near_even(v) for v in y.reshape(-1)]).reshape(x.shape)
            x = np.clip(x, -128, 127)
        elif stage == "requant":
            s = int(ep.requant_shift)
            x = np.clip((x + (1 << (s - 1))) >> s, -128, 127)
        elif stage == "maxpool":
            x = _maxpool(x, ep)
    if ep.output_dtype == "i8":
        x = np.clip(x, -128, 127)
    return x


def _maxpool(x: np.ndarray, ep) -> np.ndarray:
    h, w = ep.pool_in_dims
    ph, pw = ep.pool_size
    sh, sw = ep.pool_stride
    pt, pl, pb, pr = (list(ep.pool_padding) + [0, 0, 0, 0])[:4]
    m, n = x.shape
    batch = m // (h * w)
    ho = (h + pt + pb - ph) // sh + 1
    wo = (w + pl + pr - pw) // sw + 1
    o = np.full((batch * ho * wo, n), np.iinfo(np.int64).min, dtype=np.int64)
    for b in range(batch):
        for oh in range(ho):
            for ow in range(wo):
                best = None
                for r in range(ph):
                    for c in range(pw):
                        ih, iw = oh * sh + r - pt, ow * sw + c - pl
                        if 0 <= ih < h and 0 <= iw < w:
                            v = x[b * h * w + ih * w + iw]
                        else:
                            v = np.full(n, ep.pool_pad_value, dtype=np.int64)
                        best = v if best is None else np.maximum(best, v)
                o[b * ho * wo + oh * wo + ow] = best
    return o


def _conv(op, wl, dram) -> np.ndarray:
    from mlir_oot.lowering import conv as convlib

    ifm_t = wl.tensors[op.operands["ifm"]]
    hnd = op.operands["weight"]
    wt = wl.tensors[wl.residents.get(hnd, hnd)]
    g = convlib.geometry_from(op.attrs, ifm_t.shape, wt.shape)
    ifm = dram.logical(ifm_t.name, ifm_t.shape).astype(np.int64)
    weight = dram.logical(wt.name, wt.shape).astype(np.int64)
    cols = np.zeros((g.m, g.k), dtype=np.int64)
    for p in range(g.m):
        nb, rem = divmod(p, g.ho * g.wo)
        oh, ow = divmod(rem, g.wo)
        for kh in range(g.kh):
            for kw in range(g.kw):
                ih = oh * g.sh - g.pad_t + kh * g.dh
                iw = ow * g.sw - g.pad_l + kw * g.dw
                if 0 <= ih < g.h and 0 <= iw < g.w:
                    base = (kh * g.kw + kw) * g.ci
                    cols[p, base : base + g.ci] = ifm[nb, ih, iw, :]
    return _epilogue(cols @ weight, op.epilogue, wl, dram)


# --------------------------------------------------------------------------- driver
def check(capsule_dir: pathlib.Path) -> tuple[bool, str]:
    text = (capsule_dir / "capsule.interface.mlir").read_text()
    wl = read_workload(text)
    if wl.declined is not None:
        return True, "declined"
    if not wl.ops:
        return True, "host lane"
    lowered = IfaceToGemmini().run(wl)
    dram = Dram(wl)
    derived = {r["target"] for r in wl.im2col_recipes}
    for name, t in wl.tensors.items():
        if t.role != "output" and name not in derived:
            dram.store_logical(name, t.shape, deterministic(name, t.shape, t.dtype))
    for r in wl.im2col_recipes:
        # the ABI materialises a declared im2col activation from its source leaf; do the same here
        src_t = wl.tensors[r["source"]]
        src = dram.logical(r["source"], src_t.shape).astype(np.int64)
        tgt = wl.tensors[r["target"]]
        kh, kw, ci = int(r["kh"]), int(r["kw"]), int(r["ci"])
        sh, sw = r["stride"]
        pt, pl, pb, pr = r["padding"]
        dh, dw = r["dilation"]
        n, h, w, _ = src_t.shape
        ho = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
        wo = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
        mat = np.zeros(tgt.shape, dtype=np.int64)
        for p_i in range(tgt.shape[0]):
            nb, rem = divmod(p_i, ho * wo)
            oh, ow = divmod(rem, wo)
            for a in range(kh):
                for b_ in range(kw):
                    ih, iw = oh * sh - pt + a * dh, ow * sw - pl + b_ * dw
                    if 0 <= ih < h and 0 <= iw < w:
                        base = (a * kw + b_) * ci
                        mat[p_i, base : base + ci] = src[nb, ih, iw, :]
        dram.store_logical(r["target"], tgt.shape, mat)
    want = reference(wl, dram)
    tol = tolerances(wl)
    kernel = next(o for o in lowered.module.body.block.ops if isinstance(o, gd.KernelOp))
    Machine(dram, lowered.arg_tensors).run(kernel)
    bad = []
    for name, expect in want.items():
        got = dram.logical(name, wl.tensors[name].shape).astype(np.int64)
        e = np.asarray(expect).reshape(got.shape)
        slack = tol.get(name, 0)
        diff = np.abs(got - e.astype(np.int64))
        if int(diff.max(initial=0)) > slack:
            n = int((diff > slack).sum())
            bad.append(
                f"{name}: {n}/{got.size} elements outside the declared bound, "
                f"max|diff|={int(diff.max())} > {slack}"
            )
    return (not bad), "; ".join(bad) or "ok"


def main(argv: list[str]) -> int:
    root = HERE.parent.parent
    if "--all" in argv:
        dirs = sorted(
            d.parent
            for r in ("isa", "layers", "model_slices")
            for d in (root / r).glob("*/capsule.interface.mlir")
        )
    else:
        dirs = [pathlib.Path(a) for a in argv if not a.startswith("-")]
    fails = 0
    for d in dirs:
        try:
            ok, msg = check(d)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            fails += 1
        if not ok or "-v" in argv:
            print(f"{'PASS' if ok else 'FAIL'}  {d.name:52s} {msg[:110]}")
    print(f"{len(dirs) - fails}/{len(dirs)} self-consistent")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
