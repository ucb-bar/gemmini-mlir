"""Development aid: interpret the scalar-lane kernel this package emits, and check its arithmetic.

:mod:`mlir_oot.codegen.scalar_lane` compiles a region the mesh has no operand encoding for into an
LLVM-dialect LOOP NEST. A loop nest is exactly where an address expression goes wrong silently, so
this tool EXECUTES the emitted module over DRAM buffers laid out the way the harness lays them out,
and compares the result with a direct numpy evaluation of the same interface program.

It is a self-consistency check, not an oracle: both sides are derived from the capsule that was
handed in, and no golden is involved.

    python devtools/simlane.py            # the built-in region matrix
"""

from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from xdsl.dialects import llvm  # noqa: E402
from xdsl.dialects.builtin import FloatAttr  # noqa: E402

from mlir_oot.driver import _lower, read_workload  # noqa: E402
from mlir_oot.target import layout  # noqa: E402

NP = {"f32": np.float32, "bf16": np.float32}
WIDTH = {"f32": 4, "bf16": 2}

#: LLVM icmp predicate encodings (LangRef order).
_ICMP = {
    0: lambda a, b: a == b, 1: lambda a, b: a != b,
    2: lambda a, b: a < b, 3: lambda a, b: a <= b,
    4: lambda a, b: a > b, 5: lambda a, b: a >= b,
    6: lambda a, b: a < b, 7: lambda a, b: a <= b,
    8: lambda a, b: a > b, 9: lambda a, b: a >= b,
}
_FCMP = {1: lambda a, b: a == b, 2: lambda a, b: a > b, 3: lambda a, b: a >= b,
         4: lambda a, b: a < b, 5: lambda a, b: a <= b, 6: lambda a, b: a != b}


def _width(elem_type) -> int:
    return 4 if str(elem_type) == "f32" else 2


def run(module, buffers: list[bytearray]) -> None:
    fns = {o.sym_name.data: o for o in module.body.block.ops if isinstance(o, llvm.FuncOp)}
    entry = list(fns["gemmini_kernel"].body.blocks)[0]
    _call(entry, [("ptr", i, 0) for i in range(len(entry.args))], buffers, fns)


def _call(block, args, buffers, fns):
    env: dict = {}
    cur, incoming = block, args
    for _ in range(1 << 30):
        env.update(zip(cur.args, incoming))
        nxt = None
        for op in cur.ops:
            r = _step(op, env, buffers, fns)
            if r is not None:
                kind = r[0]
                if kind == "ret":
                    return r[1]
                nxt, incoming = r[1], r[2]
                break
        if nxt is None:
            raise RuntimeError("a block fell off its end without a terminator")
        cur = nxt
    raise RuntimeError("interpreter did not converge")


def _step(op, env, buffers, fns):
    def v(x):
        return env[x]

    if isinstance(op, llvm.ConstantOp):
        a = op.value
        env[op.results[0]] = (np.float32(a.value.data) if isinstance(a, FloatAttr)
                              else int(a.value.data))
    elif isinstance(op, llvm.GEPOp):
        _, buf, base = v(op.operands[0])
        idx = v(op.operands[1]) if len(op.operands) > 1 else 0
        env[op.results[0]] = ("ptr", buf, base + int(idx) * _width(op.elem_type))
    elif isinstance(op, llvm.LoadOp):
        _, buf, off = v(op.operands[0])
        if str(op.results[0].type) == "f32":
            env[op.results[0]] = np.float32(struct.unpack_from("<f", buffers[buf], off)[0])
        else:
            env[op.results[0]] = struct.unpack_from("<H", buffers[buf], off)[0]
    elif isinstance(op, llvm.StoreOp):
        _, buf, off = v(op.operands[1])
        val = v(op.operands[0])
        if isinstance(val, np.float32):
            struct.pack_into("<f", buffers[buf], off, float(val))
        else:
            struct.pack_into("<H", buffers[buf], off, int(val) & 0xFFFF)
    elif isinstance(op, llvm.FAddOp):
        env[op.results[0]] = np.float32(v(op.operands[0]) + v(op.operands[1]))
    elif isinstance(op, llvm.FMulOp):
        env[op.results[0]] = np.float32(v(op.operands[0]) * v(op.operands[1]))
    elif isinstance(op, llvm.FSubOp):
        env[op.results[0]] = np.float32(v(op.operands[0]) - v(op.operands[1]))
    elif isinstance(op, llvm.FCmpOp):
        env[op.results[0]] = _FCMP[int(op.predicate.value.data)](
            v(op.operands[0]), v(op.operands[1]))
    elif isinstance(op, llvm.SelectOp):
        env[op.results[0]] = v(op.operands[1]) if v(op.operands[0]) else v(op.operands[2])
    elif isinstance(op, llvm.AddOp):
        env[op.results[0]] = v(op.operands[0]) + v(op.operands[1])
    elif isinstance(op, llvm.SubOp):
        env[op.results[0]] = v(op.operands[0]) - v(op.operands[1])
    elif isinstance(op, llvm.MulOp):
        env[op.results[0]] = v(op.operands[0]) * v(op.operands[1])
    elif isinstance(op, llvm.SDivOp):
        a, b = v(op.operands[0]), v(op.operands[1])
        env[op.results[0]] = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    elif isinstance(op, llvm.SRemOp):
        a, b = v(op.operands[0]), v(op.operands[1])
        env[op.results[0]] = a - (abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)) * b
    elif isinstance(op, llvm.ICmpOp):
        env[op.results[0]] = _ICMP[int(op.predicate.value.data)](
            v(op.operands[0]), v(op.operands[1]))
    elif isinstance(op, llvm.AndOp):
        a, b = v(op.operands[0]), v(op.operands[1])
        env[op.results[0]] = (a and b) if isinstance(a, bool) else (a & b)
    elif isinstance(op, llvm.LShrOp):
        env[op.results[0]] = (v(op.operands[0]) & 0xFFFFFFFF) >> v(op.operands[1])
    elif isinstance(op, llvm.ShlOp):
        env[op.results[0]] = (v(op.operands[0]) << v(op.operands[1])) & 0xFFFFFFFF
    elif isinstance(op, llvm.ZExtOp):
        env[op.results[0]] = int(v(op.operands[0]))
    elif isinstance(op, llvm.TruncOp):
        env[op.results[0]] = int(v(op.operands[0])) & 0xFFFF
    elif isinstance(op, llvm.BitcastOp):
        x = v(op.operands[0])
        if str(op.results[0].type) == "f32":
            env[op.results[0]] = np.float32(struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0])
        else:
            env[op.results[0]] = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    elif isinstance(op, llvm.CallOp):
        name = op.callee.string_value() if hasattr(op.callee, "string_value") else str(op.callee)
        env[op.results[0]] = _call(list(fns[name.lstrip("@")].body.blocks)[0],
                                   [v(a) for a in op.operands], buffers, fns)
    elif isinstance(op, llvm.BrOp):
        return ("br", op.successors[0], [v(a) for a in op.operands])
    elif isinstance(op, llvm.CondBrOp):
        cond = v(op.operands[0])
        n_then = len(op.then_arguments)
        rest = list(op.operands)[1:]
        then_args = [v(a) for a in rest[:n_then]]
        else_args = [v(a) for a in rest[n_then:]]
        return ("br", op.successors[0 if cond else 1], then_args if cond else else_args)
    elif isinstance(op, llvm.ReturnOp):
        return ("ret", v(op.operands[0]) if op.operands else None)
    else:
        raise NotImplementedError(op.name)
    return None


# --------------------------------------------------------------------------- harness
def _alloc(shape, dtype, data=None) -> bytearray:
    """A DRAM buffer laid out the way the contract says the harness lays one out."""
    pitch, rows = layout.row_pitch(shape), layout.row_count(shape)
    trailing = int(shape[-1]) if shape else 1
    buf = bytearray(rows * pitch * WIDTH[dtype])
    if data is None:
        return buf
    flat = np.asarray(data, dtype=np.float32).reshape(rows, trailing)
    for r in range(rows):
        for c in range(trailing):
            off = (r * pitch + c) * WIDTH[dtype]
            if dtype == "f32":
                struct.pack_into("<f", buf, off, float(flat[r, c]))
            else:
                bits = struct.unpack("<I", struct.pack("<f", float(flat[r, c])))[0]
                struct.pack_into("<H", buf, off, (bits >> 16) & 0xFFFF)
    return buf


def _read(buf, shape, dtype) -> np.ndarray:
    pitch, rows = layout.row_pitch(shape), layout.row_count(shape)
    trailing = int(shape[-1]) if shape else 1
    out = np.zeros((rows, trailing), dtype=np.float32)
    for r in range(rows):
        for c in range(trailing):
            off = (r * pitch + c) * WIDTH[dtype]
            if dtype == "f32":
                out[r, c] = struct.unpack_from("<f", buf, off)[0]
            else:
                half = struct.unpack_from("<H", buf, off)[0]
                out[r, c] = struct.unpack("<f", struct.pack("<I", half << 16))[0]
    return out.reshape(shape)


def _bf16(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    r = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return r.view(np.float32)


def check(text: str, inputs: dict[str, np.ndarray], expect: dict[str, np.ndarray],
          tol: float = 1e-4) -> tuple[bool, str]:
    wl = read_workload(text)
    declined, lowered, module = _lower(wl)
    if declined is not None:
        return False, f"declined: {declined.get('reason', '')[:120]}"
    if module is None:
        return False, "no scalar-lane module was emitted (the region went to the mesh)"
    args = lowered.arg_tensors
    buffers = []
    for name in args:
        t = wl.tensors[name]
        buffers.append(_alloc(t.shape, t.dtype, inputs.get(name)))
    run(module, buffers)
    for name, want in expect.items():
        got = _read(buffers[args.index(name)], wl.tensors[name].shape, wl.tensors[name].dtype)
        if not np.allclose(got, want, atol=tol, rtol=tol):
            bad = int(np.argmax(np.abs(got - want)))
            return False, (f"{name}: {got.size} elements, worst at flat {bad}: "
                           f"got {got.flat[bad]} want {want.flat[bad]}")
    return True, ""


def _iface(tensors: dict, ops: list[str]) -> str:
    lines = ['module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
             'merlin_iface.abi_version = "0.1"} {']
    for name, (shape, dt, role) in tensors.items():
        ext = "x".join(str(d) for d in shape)
        lines.append(f'  %{name} = merlin_iface.tensor {{name = "{name}", role = "{role}"}} '
                     f': tensor<{ext}x{dt}>')
    lines += [f"  {o}" for o in ops]
    lines.append("}")
    return "\n".join(lines)


def _cases():
    rng = np.random.default_rng(7)
    out = []

    def f(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    # 1. resident matmul, no epilogue
    X, W = f(32, 32), f(32, 32)
    out.append(("matmul f32 32x32x32", _iface(
        {"X": ((32, 32), "f32", "input"), "W": ((32, 32), "f32", "weight")},
        ['%h = merlin_iface.resident_pack %W {layout = "packed_rhs"} : '
         '(tensor<32x32xf32>) -> !merlin_iface.resident',
         '%a = merlin_iface.matmul %X, %h : (tensor<32x32xf32>, !merlin_iface.resident) '
         '-> !merlin_iface.acc<f32>',
         '%Y0 = merlin_iface.commit %a {name = "Y0", epilogue = [], output_dtype = "f32"} : '
         '(!merlin_iface.acc<f32>) -> tensor<32x32xf32>',
         "merlin_iface.evict %h : (!merlin_iface.resident) -> ()"]),
        {"X": X, "W": W}, {"Y0": X @ W}))

    # 2. a NON-tile-aligned matmul: the row pitch is padded, the extents are not
    X, W = f(5, 7), f(7, 3)
    out.append(("matmul f32 5x7x3 (unaligned)", _iface(
        {"X": ((5, 7), "f32", "input"), "W": ((7, 3), "f32", "weight")},
        ['%h = merlin_iface.resident_pack %W {layout = "packed_rhs"} : '
         '(tensor<7x3xf32>) -> !merlin_iface.resident',
         '%a = merlin_iface.matmul %X, %h : (tensor<5x7xf32>, !merlin_iface.resident) '
         '-> !merlin_iface.acc<f32>',
         '%Y0 = merlin_iface.commit %a {name = "Y0", epilogue = [], output_dtype = "f32"} : '
         '(!merlin_iface.acc<f32>) -> tensor<5x3xf32>',
         "merlin_iface.evict %h : (!merlin_iface.resident) -> ()"]),
        {"X": X, "W": W}, {"Y0": X @ W}))

    # 3. the fused readout: bias -> scale -> relu
    X, W, B = f(4, 6), f(6, 8), f(8)
    ref = np.maximum((X @ W + B) * np.float32(0.25), 0)
    out.append(("matmul f32 + bias/scale/relu", _iface(
        {"X": ((4, 6), "f32", "input"), "W": ((6, 8), "f32", "weight"),
         "B": ((8,), "f32", "bias")},
        ['%h = merlin_iface.resident_pack %W {layout = "packed_rhs"} : '
         '(tensor<6x8xf32>) -> !merlin_iface.resident',
         '%a = merlin_iface.matmul %X, %h : (tensor<4x6xf32>, !merlin_iface.resident) '
         '-> !merlin_iface.acc<f32>',
         '%Y0 = merlin_iface.commit %a {name = "Y0", epilogue = ["bias_add", "acc_scale", "relu"],'
         ' output_dtype = "f32", acc_scale = 0.25 : f32, bias = "B"} : '
         '(!merlin_iface.acc<f32>) -> tensor<4x8xf32>',
         "merlin_iface.evict %h : (!merlin_iface.resident) -> ()"]),
        {"X": X, "W": W, "B": B}, {"Y0": ref}))

    # 4. batched contraction -- the shape M2's depthwise conv tile arrives as
    A, W = f(48, 1, 9), f(48, 9, 16)
    out.append(("batched matmul f32 48x1x9 @ 48x9x16", _iface(
        {"A0": ((48, 1, 9), "f32", "input"), "W": ((48, 9, 16), "f32", "weight")},
        ['%Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", '
         'output_dtype = "f32"} : '
         '(tensor<48x1x9xf32>, tensor<48x9x16xf32>) -> tensor<48x1x16xf32>']),
        {"A0": A, "W": W}, {"Y0": A @ W}))

    # 5. attention, both halves (qk contracts the trailing head dim of BOTH operands)
    Q, K = f(6, 4), f(5, 4)
    out.append(("attention_qk f32", _iface(
        {"Q": ((6, 4), "f32", "input"), "K": ((5, 4), "f32", "input")},
        ['%Y0 = merlin_iface.attention_qk %Q, %K {name = "Y0", epilogue = [], '
         'output_dtype = "f32"} : (tensor<6x4xf32>, tensor<5x4xf32>) -> tensor<6x5xf32>']),
        {"Q": Q, "K": K}, {"Y0": Q @ K.T}))
    P, V = f(6, 5), f(5, 4)
    out.append(("attention_pv f32", _iface(
        {"P": ((6, 5), "f32", "input"), "V": ((5, 4), "f32", "input")},
        ['%Y0 = merlin_iface.attention_pv %P, %V {name = "Y0", epilogue = [], '
         'output_dtype = "f32"} : (tensor<6x5xf32>, tensor<5x4xf32>) -> tensor<6x4xf32>']),
        {"P": P, "V": V}, {"Y0": P @ V}))

    # 6. bf16 containers: loaded widened, stored rounded to nearest even
    X, W = _bf16(f(3, 5)), _bf16(f(5, 2))
    out.append(("matmul bf16", _iface(
        {"X": ((3, 5), "bf16", "input"), "W": ((5, 2), "bf16", "weight")},
        ['%h = merlin_iface.resident_pack %W {layout = "packed_rhs"} : '
         '(tensor<5x2xbf16>) -> !merlin_iface.resident',
         '%a = merlin_iface.matmul %X, %h : (tensor<3x5xbf16>, !merlin_iface.resident) '
         '-> !merlin_iface.acc<bf16>',
         '%Y0 = merlin_iface.commit %a {name = "Y0", epilogue = [], output_dtype = "bf16"} : '
         '(!merlin_iface.acc<bf16>) -> tensor<3x2xbf16>',
         "merlin_iface.evict %h : (!merlin_iface.resident) -> ()"]),
        {"X": X, "W": W}, {"Y0": _bf16(X @ W)}, 1e-2))

    # 7. movement: an identity round trip keeps the values
    X = f(9, 11)
    out.append(("movement f32", _iface(
        {"X": ((9, 11), "f32", "input")},
        ['%Y0 = merlin_iface.movement %X {name = "Y0", output_dtype = "f32", '
         'semantic = "mvin_mvout"} : (tensor<9x11xf32>) -> tensor<9x11xf32>']),
        {"X": X}, {"Y0": X}))

    # 8. convolution, with padding, stride and dilation exercised together
    for tag, kh, kw, sh, sw, pad, dh, dw in (
        ("3x3 s1 pad0", 3, 3, 1, 1, (0, 0, 0, 0), 1, 1),
        ("3x3 s1 pad1", 3, 3, 1, 1, (1, 1, 1, 1), 1, 1),
        ("3x3 s2 pad1", 3, 3, 2, 2, (1, 1, 1, 1), 1, 1),
        ("2x2 s1 dil2", 2, 2, 1, 1, (0, 0, 0, 0), 2, 2),
    ):
        n, h, w, ci, co = 1, 7, 8, 3, 5
        pt, pl, pb, pr = pad
        ho = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
        wo = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
        ifm, wt = f(n, h, w, ci), f(kh * kw * ci, co)
        padded = np.zeros((n, h + pt + pb, w + pl + pr, ci), dtype=np.float32)
        padded[:, pt:pt + h, pl:pl + w, :] = ifm
        ref = np.zeros((n * ho * wo, co), dtype=np.float32)
        for b in range(n):
            for oh in range(ho):
                for ow in range(wo):
                    acc = np.zeros(co, dtype=np.float32)
                    for a in range(kh):
                        for c_ in range(kw):
                            px = padded[b, oh * sh + a * dh, ow * sw + c_ * dw, :]
                            acc = acc + px @ wt[(a * kw + c_) * ci:(a * kw + c_) * ci + ci, :]
                    ref[b * ho * wo + oh * wo + ow] = acc
        out.append((f"conv2d f32 {tag}", _iface(
            {"IFM": ((n, h, w, ci), "f32", "input"), "W": ((kh * kw * ci, co), "f32", "weight")},
            [f'%h = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : '
             f'(tensor<{kh * kw * ci}x{co}xf32>) -> !merlin_iface.resident',
             f'%Y0 = merlin_iface.conv2d %IFM, %h {{kernel = [{kh}, {kw}, {ci}, {co}], '
             f'stride = [{sh}, {sw}], padding = [{pt}, {pl}, {pb}, {pr}], '
             f'dilation = [{dh}, {dw}], name = "Y0", epilogue = [], output_dtype = "f32", '
             f'layout = "nhwc"}} : (tensor<{n}x{h}x{w}x{ci}xf32>, !merlin_iface.resident) '
             f'-> tensor<{n * ho * wo}x{co}xf32>',
             "merlin_iface.evict %h : (!merlin_iface.resident) -> ()"]),
            {"IFM": ifm, "W": wt}, {"Y0": ref}))
    return out


def main() -> int:
    bad = 0
    for case in _cases():
        name, text, ins, exp = case[0], case[1], case[2], case[3]
        tol = case[4] if len(case) > 4 else 1e-4
        try:
            ok, note = check(text, ins, exp, tol)
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"{type(exc).__name__}: {exc}"
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"   {note}" if not ok else ""))
        bad += 0 if ok else 1
    print(f"{'all scalar-lane regions agree' if not bad else f'{bad} region(s) disagree'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
