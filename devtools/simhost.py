"""Development aid: check a host-lane artifact by EXECUTING it against a reference evaluation.

Two independent evaluations of the same region:
  * the emitted LLVM-dialect module, interpreted op by op (loads, f32 arithmetic, bit casts,
    selects -- including this package's own exp/erf/rsqrt expansions);
  * the linalg program itself, evaluated with numpy and the C library's transcendentals.
Agreement to the capsule's declared tolerance means the host-lane codegen and its function
approximations are both right. No golden is involved.
"""

from __future__ import annotations

import math as pymath
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from xdsl.dialects import arith, func, linalg, llvm, math, tensor  # noqa: E402
from xdsl.dialects.builtin import FloatAttr, IntegerAttr  # noqa: E402

from mlir_oot.codegen.host_lane import HostLaneCompiler, _flat, _iter, _shape_of  # noqa: E402
from mlir_oot.driver import read_workload  # noqa: E402
from mlir_oot.lowering.iface_to_gemmini import IfaceToGemmini  # noqa: E402
from mlir_oot.target import layout  # noqa: E402


def f32(x):
    return np.float32(x)


def bf16_round(x: float) -> float:
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFF0000))[0]


# --------------------------------------------------------------------------- reference
def _scalar_ref(op, env, idx=()):
    from xdsl.dialects.linalg.ops import IndexOp

    if isinstance(op, IndexOp):
        return f32(idx[int(op.dim.value.data)])
    if isinstance(op, (arith.ExtFOp, arith.TruncFOp, arith.SIToFPOp, arith.IndexCastOp)):
        return env[op.operands[0]]
    if isinstance(op, arith.ConstantOp):
        a = op.value
        return f32(a.value.data)
    ins = [env[o] for o in op.operands]
    if isinstance(op, arith.AddfOp):
        return f32(ins[0] + ins[1])
    if isinstance(op, arith.SubfOp):
        return f32(ins[0] - ins[1])
    if isinstance(op, arith.MulfOp):
        return f32(ins[0] * ins[1])
    if isinstance(op, arith.DivfOp):
        return f32(ins[0] / ins[1])
    if isinstance(op, (arith.MaximumfOp, arith.MaxnumfOp)):
        return f32(max(ins[0], ins[1]))
    if isinstance(op, (arith.MinimumfOp, arith.MinnumfOp)):
        return f32(min(ins[0], ins[1]))
    if isinstance(op, arith.NegfOp):
        return f32(-ins[0])
    if isinstance(op, math.ExpOp):
        return f32(pymath.exp(ins[0]))
    if isinstance(op, math.RsqrtOp):
        return f32(1.0 / pymath.sqrt(ins[0]))
    if isinstance(op, math.SqrtOp):
        return f32(pymath.sqrt(ins[0]))
    if isinstance(op, math.ErfOp):
        return f32(pymath.erf(ins[0]))
    if isinstance(op, math.TanhOp):
        return f32(pymath.tanh(ins[0]))
    if isinstance(op, arith.AddiOp):
        return f32(ins[0] + ins[1])
    if isinstance(op, arith.SubiOp):
        return f32(ins[0] - ins[1])
    if isinstance(op, arith.MuliOp):
        return f32(ins[0] * ins[1])
    if isinstance(op, arith.CmpiOp):
        c = int(op.predicate.value.data)
        return {0: ins[0] == ins[1], 1: ins[0] != ins[1], 2: ins[0] < ins[1], 3: ins[0] <= ins[1],
                4: ins[0] > ins[1], 5: ins[0] >= ins[1], 6: ins[0] < ins[1], 7: ins[0] <= ins[1],
                8: ins[0] > ins[1], 9: ins[0] >= ins[1]}[c]
    if isinstance(op, arith.CmpfOp):
        c = int(op.predicate.value.data)
        return {1: ins[0] == ins[1], 2: ins[0] > ins[1], 3: ins[0] >= ins[1], 4: ins[0] < ins[1],
                5: ins[0] <= ins[1], 6: ins[0] != ins[1]}[c]
    if isinstance(op, arith.SelectOp):
        return ins[1] if ins[0] else ins[2]
    raise NotImplementedError(op.name)


def _body_ref(block, args, idx=()):
    env = dict(zip(block.args, args))
    for op in block.ops:
        if isinstance(op, linalg.YieldOp):
            return env[op.operands[0]]
        env[op.results[0]] = _scalar_ref(op, env, idx)
    raise NotImplementedError("body yielded nothing")


def reference(entry, inputs: list[np.ndarray]) -> list[np.ndarray]:
    from mlir_oot.codegen.host_lane import HostLaneCompiler as HC

    env = {}
    block = entry.body.block
    for a, arr in zip(block.args, inputs):
        env[a] = arr.astype(np.float32).reshape(-1).tolist()
    shapes = {a: tuple(arr.shape) for a, arr in zip(block.args, inputs)}
    helper = HC.__new__(HC)

    def val(v):
        return env[v], shapes[v]

    for op in block.ops:
        if isinstance(op, func.ReturnOp):
            return [np.array(env[v], dtype=np.float32).reshape(shapes[v]) for v in op.operands]
        _op_ref(op, env, shapes, helper)
    raise NotImplementedError("no return")


def _statics(attr):
    if hasattr(attr, "get_values"):
        return [int(x) for x in attr.get_values()]
    return [int(x.value.data) for x in attr.data]


def _op_ref(op, env, shapes, helper):
    if isinstance(op, arith.ConstantOp):
        env[op.results[0]] = [f32(op.value.value.data)]
        shapes[op.results[0]] = ()
        return
    if isinstance(op, (tensor.SplatOp, linalg.FillOp)):
        shape = _shape_of(op.results[0])
        n = int(np.prod(shape)) if shape else 1
        env[op.results[0]] = [env[op.operands[0]][0]] * n
        shapes[op.results[0]] = shape
        return
    if isinstance(op, tensor.EmptyOp):
        shape = _shape_of(op.results[0])
        env[op.results[0]] = [f32(0.0)] * int(np.prod(shape))
        shapes[op.results[0]] = shape
        return
    if isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp, tensor.ReshapeOp)):
        env[op.results[0]] = list(env[op.operands[0]])
        shapes[op.results[0]] = _shape_of(op.results[0])
        return
    if isinstance(op, tensor.InsertSliceOp):
        src, dst = env[op.operands[0]], list(env[op.operands[1]])
        ss, ds = shapes[op.operands[0]], shapes[op.operands[1]]
        offs, sizes, strides = (_statics(op.static_offsets), _statics(op.static_sizes),
                                _statics(op.static_strides))
        for idx in _iter(tuple(sizes)):
            d = tuple(offs[k] + idx[k] * strides[k] for k in range(len(idx)))
            dst[_flat(ds, d)] = src[_flat(ss, idx)]
        env[op.results[0]] = dst
        shapes[op.results[0]] = ds
        return
    if isinstance(op, linalg.TransposeOp):
        src, ss = env[op.operands[0]], shapes[op.operands[0]]
        perm = _statics(op.permutation)
        shape = _shape_of(op.results[0])
        env[op.results[0]] = [
            src[_flat(ss, tuple(idx[perm.index(k)] for k in range(len(perm))))]
            for idx in _iter(shape)
        ]
        shapes[op.results[0]] = shape
        return
    if isinstance(op, linalg.MatmulOp):
        a, b, c = (env[o] for o in op.operands)
        (m, k), (k2, n) = shapes[op.operands[0]], shapes[op.operands[1]]
        out = list(c)
        for i in range(m):
            for j in range(n):
                acc = out[i * n + j]
                for t in range(k):
                    acc = f32(acc + f32(a[i * k + t] * b[t * n + j]))
                out[i * n + j] = acc
        env[op.results[0]] = out
        shapes[op.results[0]] = (m, n)
        return
    if isinstance(op, linalg.ReduceOp):
        src, init = env[op.operands[0]], list(env[op.operands[1]])
        ss = shapes[op.operands[0]]
        dims = set(_statics(op.dimensions))
        out_shape = tuple(d for k, d in enumerate(ss) if k not in dims)
        for idx in _iter(ss):
            o = tuple(v for k, v in enumerate(idx) if k not in dims)
            pos = _flat(out_shape, o) if out_shape else 0
            init[pos] = _body_ref(op.region.block, [init[pos], src[_flat(ss, idx)]])
        env[op.results[0]] = init
        shapes[op.results[0]] = out_shape
        return
    if isinstance(op, linalg.GenericOp):
        maps = [m.data for m in op.get_indexing_maps().data]
        ranges = op.get_static_loop_ranges()
        ins = list(op.inputs)
        outs = list(op.outputs)
        result = list(env[outs[0]])
        out_shape = shapes[outs[0]]
        for idx in _iter(tuple(ranges)):
            args = []
            for k, v in enumerate(ins):
                args.append(env[v][_flat(shapes[v], tuple(maps[k].eval(list(idx), [])))])
            for k, v in enumerate(outs):
                args.append(
                    env[v][_flat(shapes[v], tuple(maps[len(ins) + k].eval(list(idx), [])))]
                )
            result[_flat(out_shape, tuple(maps[len(ins)].eval(list(idx), [])))] = _body_ref(
                op.body.block, args, idx
            )
        env[op.results[0]] = result
        shapes[op.results[0]] = out_shape
        return
    raise NotImplementedError(op.name)


# --------------------------------------------------------------------------- LLVM interpreter
def run_llvm(module, buffers: list[bytearray]) -> None:
    fns = {o.sym_name.data: o for o in module.body.block.ops if isinstance(o, llvm.FuncOp)}
    fn = fns["gemmini_kernel"]
    _exec_block(fn.body.block, [("ptr", i, 0) for i in range(len(fn.body.block.args))],
                buffers, fns)


def _exec_block(blk, arg_values, buffers, fns):
    env = dict(zip(blk.args, arg_values))

    def num(v):
        return env[v]

    for op in blk.ops:
        if isinstance(op, llvm.ConstantOp):
            a = op.value
            env[op.results[0]] = (
                f32(a.value.data) if isinstance(a, FloatAttr) else int(a.value.data)
            )
        elif isinstance(op, llvm.GEPOp):
            kind, buf, _ = env[op.operands[0]]
            idx = [int(x.value.data) if hasattr(x, "value") else int(x)
                   for x in _statics(op.rawConstantIndices)]
            width = 4 if op.elem_type == op.elem_type and str(op.elem_type) == "f32" else 2
            env[op.results[0]] = ("ptr", buf, idx[0] * width)
        elif isinstance(op, llvm.LoadOp):
            _, buf, off = env[op.operands[0]]
            if str(op.results[0].type) == "f32":
                env[op.results[0]] = f32(struct.unpack_from("<f", buffers[buf], off)[0])
            else:
                env[op.results[0]] = struct.unpack_from("<H", buffers[buf], off)[0]
        elif isinstance(op, llvm.StoreOp):
            _, buf, off = env[op.operands[1]]
            v = env[op.operands[0]]
            if isinstance(v, np.float32):
                struct.pack_into("<f", buffers[buf], off, float(v))
            else:
                struct.pack_into("<H", buffers[buf], off, int(v) & 0xFFFF)
        elif isinstance(op, llvm.FAddOp):
            env[op.results[0]] = f32(num(op.operands[0]) + num(op.operands[1]))
        elif isinstance(op, llvm.FSubOp):
            env[op.results[0]] = f32(num(op.operands[0]) - num(op.operands[1]))
        elif isinstance(op, llvm.FMulOp):
            env[op.results[0]] = f32(num(op.operands[0]) * num(op.operands[1]))
        elif isinstance(op, llvm.FDivOp):
            env[op.results[0]] = f32(num(op.operands[0]) / num(op.operands[1]))
        elif isinstance(op, llvm.FCmpOp):
            a, b = num(op.operands[0]), num(op.operands[1])
            code = int(op.predicate.value.data)
            env[op.results[0]] = _FCMP[code](a, b)
        elif isinstance(op, llvm.SelectOp):
            env[op.results[0]] = num(op.operands[1]) if num(op.operands[0]) else num(op.operands[2])
        elif isinstance(op, llvm.BitcastOp):
            v = num(op.operands[0])
            if isinstance(v, np.float32):
                env[op.results[0]] = struct.unpack("<i", struct.pack("<f", float(v)))[0]
            else:
                env[op.results[0]] = f32(struct.unpack("<f", struct.pack("<i", _s32(v)))[0])
        elif isinstance(op, llvm.AddOp):
            env[op.results[0]] = _s32(num(op.operands[0]) + num(op.operands[1]))
        elif isinstance(op, llvm.SubOp):
            env[op.results[0]] = _s32(num(op.operands[0]) - num(op.operands[1]))
        elif isinstance(op, llvm.ShlOp):
            env[op.results[0]] = _s32(num(op.operands[0]) << num(op.operands[1]))
        elif isinstance(op, llvm.LShrOp):
            env[op.results[0]] = (num(op.operands[0]) & 0xFFFFFFFF) >> num(op.operands[1])
        elif isinstance(op, llvm.AShrOp):
            env[op.results[0]] = _s32(num(op.operands[0])) >> num(op.operands[1])
        elif isinstance(op, llvm.AndOp):
            env[op.results[0]] = num(op.operands[0]) & num(op.operands[1])
        elif isinstance(op, llvm.ZExtOp):
            env[op.results[0]] = num(op.operands[0]) & 0xFFFF
        elif isinstance(op, llvm.TruncOp):
            env[op.results[0]] = num(op.operands[0]) & 0xFFFF
        elif isinstance(op, llvm.PtrToIntOp):
            env[op.results[0]] = 0
        elif isinstance(op, llvm.CallOp):
            callee = op.callee.string_value() if hasattr(op.callee, "string_value") else str(op.callee)
            callee = callee.lstrip("@")
            target = fns[callee]
            env[op.results[0]] = _exec_block(
                target.body.block, [num(a) for a in op.operands], buffers, fns
            )
        elif isinstance(op, llvm.ReturnOp):
            return num(op.operands[0]) if op.operands else None
        elif isinstance(op, llvm.InlineAsmOp):
            continue
        else:
            raise NotImplementedError(op.name)


#: LLVM fcmp predicate encodings (LangRef order), the ones this backend emits.
_FCMP = {
    1: lambda a, b: a == b,
    6: lambda a, b: a != b,
    2: lambda a, b: a > b,
    3: lambda a, b: a >= b,
    4: lambda a, b: a < b,
    5: lambda a, b: a <= b,
}


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


# --------------------------------------------------------------------------- driver
def check(capsule: pathlib.Path) -> tuple[bool, str]:
    wl = read_workload((capsule / "capsule.interface.mlir").read_text())
    if not wl.lane_placement or wl.host_entry is None:
        return True, "not a host-lane program"
    lowered = IfaceToGemmini().run(wl)
    args = list(lowered.arg_tensors)
    module = HostLaneCompiler(wl.host_entry, args, wl.tensors).compile()

    rng = np.random.default_rng(7)
    buffers: list[bytearray] = []
    inputs: list[np.ndarray] = []
    for name in args:
        t = wl.tensors[name]
        pitch = layout.row_pitch(t.shape)
        rows = layout.row_count(t.shape)
        width = 4 if t.dtype == "f32" else 2
        buf = bytearray(rows * pitch * width)
        if t.role != "output":
            arr = rng.standard_normal(t.shape).astype(np.float32)
            if t.dtype == "bf16":
                arr = np.vectorize(bf16_round)(arr).astype(np.float32)
            inputs.append(arr)
            flat = arr.reshape(rows, t.shape[-1])
            for r in range(rows):
                for c in range(t.shape[-1]):
                    off = (r * pitch + c) * width
                    if t.dtype == "f32":
                        struct.pack_into("<f", buf, off, float(flat[r, c]))
                    else:
                        b = struct.unpack("<I", struct.pack("<f", float(flat[r, c])))[0]
                        struct.pack_into("<H", buf, off, b >> 16)
        buffers.append(buf)

    run_llvm(module, buffers)
    want = reference(wl.host_entry, inputs)
    msgs = []
    outs = [n for n in args if wl.tensors[n].role == "output"]
    for j, name in enumerate(outs):
        t = wl.tensors[name]
        pitch, rows, width = layout.row_pitch(t.shape), layout.row_count(t.shape), (4 if t.dtype == "f32" else 2)
        got = np.zeros(t.shape, dtype=np.float32).reshape(rows, t.shape[-1])
        for r in range(rows):
            for c in range(t.shape[-1]):
                off = (r * pitch + c) * width
                if t.dtype == "f32":
                    got[r, c] = struct.unpack_from("<f", buffers[args.index(name)], off)[0]
                else:
                    h = struct.unpack_from("<H", buffers[args.index(name)], off)[0]
                    got[r, c] = struct.unpack("<f", struct.pack("<I", h << 16))[0]
        exp = np.asarray(want[j], dtype=np.float32).reshape(rows, t.shape[-1])
        err = np.abs(got - exp)
        rel = err / np.maximum(np.abs(exp), 1e-6)
        if not (err.max() <= 0.03125 or rel.max() <= 0.02):
            msgs.append(f"{name}: max_abs={err.max():.5g} max_rel={rel.max():.5g}")
    return (not msgs), ("; ".join(msgs) or "ok")


def main(argv):
    root = HERE.parent.parent
    dirs = [pathlib.Path(a) for a in argv if not a.startswith("-")] or sorted(
        p.parent for p in (root / "model_slices").glob("*/capsule.interface.mlir")
    )
    bad = 0
    for d in dirs:
        try:
            ok, msg = check(d)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            bad += 1
        if not ok or "-v" in argv:
            print(f"{'PASS' if ok else 'FAIL'}  {d.name:44s} {msg[:120]}")
    print(f"{len(dirs) - bad}/{len(dirs)} host-lane regions agree")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
