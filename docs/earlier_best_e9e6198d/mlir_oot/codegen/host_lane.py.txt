"""Host-lane codegen: a ``linalg-on-tensors`` region becomes scalar LLVM-dialect arithmetic.

A region this mesh has no operand encoding for is placed on the host lane (route H) -- but a
DECLARED placement is only half an answer: the program still has to produce the tensor it commits.
So this pass COMPILES the region rather than describing it. Each tensor becomes a compile-time
vector of SSA values, each structural op (reshape, slice, broadcast) is index algebra over that
vector, and each arithmetic op is emitted as single-precision LLVM arithmetic. Nothing is
interpreted at run time and no library is called: the emitted function is the region.

Operands are widened to f32 on load and narrowed back to their declared container on store, which
is what the readback protocol asks for ("a float-declared output is carried as its stored bit
pattern and decoded from that dtype").
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import arith, func, linalg, math, tensor
from xdsl.dialects.linalg.ops import IndexOp as _LinalgIndexOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    DenseArrayBase,
    FloatAttr,
    Float32Type,
    IntegerAttr,
    ModuleOp,
    TensorType,
)
from xdsl.dialects import llvm
from xdsl.dialects.builtin import f32, i16, i32, i64
from xdsl.ir import Block, Region, SSAValue

from ..frontend.extract import dtype_name
from ..target import layout
from .fmath import FloatBuilder

KERNEL_SYMBOL = "gemmini_kernel"


class HostLaneError(Exception):
    """A construct this host-lane compiler does not implement; the caller states it as a decline."""


@dataclass
class Val:
    """A tensor held as one SSA f32 value per element, row-major."""

    shape: tuple[int, ...]
    vals: list[SSAValue]

    @property
    def n(self) -> int:
        k = 1
        for d in self.shape:
            k *= d
        return k


def _strides(shape: tuple[int, ...]) -> list[int]:
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


def _flat(shape: tuple[int, ...], idx: tuple[int, ...]) -> int:
    s = _strides(shape)
    return sum(i * st for i, st in zip(idx, s))


#: arith.cmpi mnemonic order, and the comparison each one becomes over the f32 coordinate.
_ICMP = {0: "oeq", 1: "one", 2: "olt", 3: "ole", 4: "ogt", 5: "oge",
         6: "olt", 7: "ole", 8: "ogt", 9: "oge"}
#: arith.cmpf mnemonic order (LLVM's own predicate numbering).
_CMPF = {1: "oeq", 2: "ogt", 3: "oge", 4: "olt", 5: "ole", 6: "one"}


def _icmp(f, op, ins):
    pred = _ICMP.get(int(op.predicate.value.data))
    if pred is None:
        raise HostLaneError(f"host lane: unsupported integer predicate on {op.name!r}")
    if pred == "one":
        return f._add(llvm.FCmpOp(ins[0], ins[1], "one"))
    return f.cmp(pred, ins[0], ins[1])


def _fcmp(f, op, ins):
    pred = _CMPF.get(int(op.predicate.value.data))
    if pred is None:
        raise HostLaneError(f"host lane: unsupported float predicate on {op.name!r}")
    return f.cmp(pred, ins[0], ins[1])


def _shape_of(v: SSAValue) -> tuple[int, ...]:
    t = v.type
    if not isinstance(t, TensorType):
        raise HostLaneError(f"expected a ranked tensor value, got {t}")
    return tuple(int(d) for d in t.get_shape())


def _iter(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    idx = [0] * len(shape)
    total = 1
    for d in shape:
        total *= d
    for _ in range(total):
        yield tuple(idx)
        for k in range(len(shape) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < shape[k]:
                break
            idx[k] = 0


class HostLaneCompiler:
    def __init__(self, entry: func.FuncOp, arg_names: list[str], tensors: dict) -> None:
        self.entry = entry
        self.arg_names = arg_names
        self.tensors = tensors
        self.ptr = llvm.LLVMPointerType()
        self.block = Block(arg_types=[self.ptr] * len(arg_names))
        self.f = FloatBuilder(self.block)
        self.env: dict[SSAValue, Val] = {}
        #: transcendental expansions are emitted ONCE as internal functions and called per element;
        #: inlining them per element is what makes an elementwise region's code size explode.
        self.helpers: dict[str, llvm.FuncOp] = {}

    def _helper(self, name: str, build) -> str:
        if name not in self.helpers:
            blk = Block(arg_types=[f32])
            fb = FloatBuilder(blk)
            res = build(fb, blk.args[0])
            blk.add_op(llvm.ReturnOp(res))
            self.helpers[name] = llvm.FuncOp(
                name, llvm.LLVMFunctionType([f32], f32),
                linkage=llvm.LinkageAttr("internal"), body=Region([blk]),
            )
        return name

    def _call1(self, name: str, build, x: SSAValue) -> SSAValue:
        self._helper(name, build)
        op = llvm.CallOp(name, x, return_type=f32)
        self.block.add_op(op)
        return op.results[0]

    def _typed_helper(self, name: str, arg_t, res_t, build) -> str:
        if name not in self.helpers:
            blk = Block(arg_types=[arg_t])
            fb = FloatBuilder(blk)
            res = build(fb, blk.args[0])
            blk.add_op(llvm.ReturnOp(res))
            self.helpers[name] = llvm.FuncOp(
                name, llvm.LLVMFunctionType([arg_t], res_t),
                linkage=llvm.LinkageAttr("internal"), body=Region([blk]),
            )
        return name

    def _widen_bf16(self, half: SSAValue) -> SSAValue:
        """bf16 -> f32: the 16 stored bits are the high half of the single-precision pattern."""

        def build(b, x):
            ext = llvm.ZExtOp(x, i32)
            b.b.add_op(ext)
            return b.unbits(b.ishl(ext.results[0], b.ic(16)))

        self._typed_helper("gemmini_host_bf16_to_f32", i16, f32, build)
        op = llvm.CallOp("gemmini_host_bf16_to_f32", half, return_type=f32)
        self.block.add_op(op)
        return op.results[0]

    def _narrow_bf16(self, v: SSAValue) -> SSAValue:
        """f32 -> bf16, round to nearest even, which is what the declared container holds."""

        def build(b, x):
            bits = b.bits(x)
            lsb = b._add(llvm.LShrOp(bits, b.ic(16)))
            lsb = b._add(llvm.AndOp(lsb, b.ic(1)))
            r = b.iadd(b.iadd(bits, b.ic(0x7FFF)), lsb)
            hi = b._add(llvm.LShrOp(r, b.ic(16)))
            tr = llvm.TruncOp(hi, i16)
            b.b.add_op(tr)
            return tr.results[0]

        self._typed_helper("gemmini_host_f32_to_bf16", f32, i16, build)
        op = llvm.CallOp("gemmini_host_f32_to_bf16", v, return_type=i16)
        self.block.add_op(op)
        return op.results[0]

    # -- memory ------------------------------------------------------------
    def _load(self, arg_index: int, name: str, shape: tuple[int, ...], dtype: str) -> Val:
        pitch = layout.row_pitch(shape)
        rows = layout.row_count(shape)
        trailing = shape[-1] if shape else 1
        ptr = self.block.args[arg_index]
        vals: list[SSAValue] = []
        for r in range(rows):
            for c in range(trailing):
                off = r * pitch + c
                if dtype == "f32":
                    gep = llvm.GEPOp(ptr, [off], f32, result_type=self.ptr)
                    self.block.add_op(gep)
                    ld = llvm.LoadOp(gep.result, f32)
                    self.block.add_op(ld)
                    vals.append(ld.dereferenced_value)
                elif dtype == "bf16":
                    gep = llvm.GEPOp(ptr, [off], i16, result_type=self.ptr)
                    self.block.add_op(gep)
                    ld = llvm.LoadOp(gep.result, i16)
                    self.block.add_op(ld)
                    vals.append(self._widen_bf16(ld.dereferenced_value))
                else:
                    raise HostLaneError(f"host lane: no load for dtype {dtype!r}")
        return Val(shape, vals)

    def _store(self, arg_index: int, val: Val, dtype: str) -> None:
        shape = val.shape
        pitch = layout.row_pitch(shape)
        rows = layout.row_count(shape)
        trailing = shape[-1] if shape else 1
        ptr = self.block.args[arg_index]
        for r in range(rows):
            for c in range(trailing):
                off = r * pitch + c
                v = val.vals[r * trailing + c]
                if dtype == "f32":
                    gep = llvm.GEPOp(ptr, [off], f32, result_type=self.ptr)
                    self.block.add_op(gep)
                    self.block.add_op(llvm.StoreOp(v, gep.result))
                elif dtype == "bf16":
                    narrow = self._narrow_bf16(v)
                    gep = llvm.GEPOp(ptr, [off], i16, result_type=self.ptr)
                    self.block.add_op(gep)
                    self.block.add_op(llvm.StoreOp(narrow, gep.result))
                else:
                    raise HostLaneError(f"host lane: no store for dtype {dtype!r}")

    # -- scalar body -------------------------------------------------------
    def _scalar(self, op, env: dict[SSAValue, SSAValue], idx: tuple[int, ...] = ()) -> SSAValue:
        f = self.f
        if isinstance(op, _LinalgIndexOp):
            dim = int(op.dim.value.data)
            if dim >= len(idx):
                raise HostLaneError("host lane: linalg.index outside the iteration space")
            return f.fc(float(idx[dim]))
        if isinstance(op, (arith.ExtFOp, arith.TruncFOp, arith.IndexCastOp)):
            # every value in this lane is already held in single precision
            return env[op.operands[0]]
        if isinstance(op, arith.SIToFPOp):
            return env[op.operands[0]]
        if isinstance(op, arith.ConstantOp):
            a = op.value
            if isinstance(a, FloatAttr):
                return f.fc(float(a.value.data))
            if isinstance(a, IntegerAttr):
                return f.fc(float(a.value.data))
            raise HostLaneError(f"host lane: unsupported constant {a}")
        ins = [env[o] for o in op.operands]
        if isinstance(op, arith.AddfOp):
            return f.add(ins[0], ins[1])
        if isinstance(op, arith.SubfOp):
            return f.sub(ins[0], ins[1])
        if isinstance(op, arith.MulfOp):
            return f.mul(ins[0], ins[1])
        if isinstance(op, arith.DivfOp):
            return f.div(ins[0], ins[1])
        if isinstance(op, arith.MaximumfOp) or isinstance(op, arith.MaxnumfOp):
            return f.maximum(ins[0], ins[1])
        if isinstance(op, arith.MinimumfOp) or isinstance(op, arith.MinnumfOp):
            return f.select(f.cmp("olt", ins[0], ins[1]), ins[0], ins[1])
        if isinstance(op, arith.NegfOp):
            return f.neg(ins[0])
        if isinstance(op, math.ExpOp):
            return self._call1("gemmini_host_expf", lambda b, x: b.exp(x), ins[0])
        if isinstance(op, math.RsqrtOp):
            return self._call1("gemmini_host_rsqrtf", lambda b, x: b.rsqrt(x), ins[0])
        if isinstance(op, math.SqrtOp):
            return f.div(
                f.fc(1.0),
                self._call1("gemmini_host_rsqrtf", lambda b, x: b.rsqrt(x), ins[0]),
            )
        if isinstance(op, math.ErfOp):
            return self._call1("gemmini_host_erff", lambda b, x: b.erf(x), ins[0])
        if isinstance(op, arith.AddiOp):
            return f.add(ins[0], ins[1])
        if isinstance(op, arith.SubiOp):
            return f.sub(ins[0], ins[1])
        if isinstance(op, arith.MuliOp):
            # index arithmetic: the values are loop coordinates, exact in single precision
            return f.mul(ins[0], ins[1])
        if isinstance(op, arith.CmpiOp):
            return _icmp(f, op, ins)
        if isinstance(op, arith.CmpfOp):
            return _fcmp(f, op, ins)
        if isinstance(op, arith.SelectOp):
            return f.select(ins[0], ins[1], ins[2])
        if isinstance(op, math.TanhOp):
            return self._call1(
                "gemmini_host_tanhf",
                lambda b, x: b.div(
                    b.sub(b.exp(b.mul(x, b.fc(2.0))), b.fc(1.0)),
                    b.add(b.exp(b.mul(x, b.fc(2.0))), b.fc(1.0)),
                ),
                ins[0],
            )
        raise HostLaneError(f"host lane: no scalar lowering for {op.name!r}")

    def _run_body(
        self, block: Block, args: list[SSAValue], idx: tuple[int, ...] = ()
    ) -> SSAValue:
        env = dict(zip(block.args, args))
        result: SSAValue | None = None
        for op in block.ops:
            if isinstance(op, linalg.YieldOp):
                result = env[op.operands[0]]
                break
            env[op.results[0]] = self._scalar(op, env, idx)
        if result is None:
            raise HostLaneError("host lane: a linalg body yielded nothing")
        return result

    # -- structural ops ----------------------------------------------------
    def compile(self) -> ModuleOp:
        body = self.entry.body.block
        out_names = [n for n in self.arg_names if self.tensors[n].role == "output"]
        in_args = [n for n in self.arg_names if self.tensors[n].role != "output"]
        for i, a in enumerate(body.args):
            name = in_args[i]
            spec = self.tensors[name]
            self.env[a] = self._load(self.arg_names.index(name), name, spec.shape, spec.dtype)

        ret = None
        for op in body.ops:
            if isinstance(op, func.ReturnOp):
                ret = op
                break
            self._op(op)
        if ret is None:
            raise HostLaneError("host lane: the region has no return")
        for j, v in enumerate(ret.operands):
            name = out_names[j]
            self._store(self.arg_names.index(name), self.env[v], self.tensors[name].dtype)

        self.block.add_op(llvm.ReturnOp())
        fn = llvm.FuncOp(
            KERNEL_SYMBOL,
            llvm.LLVMFunctionType([self.ptr] * len(self.arg_names)),
            linkage=llvm.LinkageAttr("external"),
            body=Region([self.block]),
        )
        module = ModuleOp([*self.helpers.values(), fn])
        module.verify()
        return module

    def _op(self, op) -> None:
        f = self.f
        if isinstance(op, arith.ConstantOp):
            self.env[op.results[0]] = Val((), [self._scalar(op, {})])
            return
        if isinstance(op, tensor.SplatOp):
            shape = _shape_of(op.results[0])
            v = self.env[op.operands[0]].vals[0]
            self.env[op.results[0]] = Val(shape, [v] * Val(shape, []).n)
            return
        if isinstance(op, tensor.EmptyOp):
            shape = _shape_of(op.results[0])
            zero = f.fc(0.0)
            self.env[op.results[0]] = Val(shape, [zero] * Val(shape, []).n)
            return
        if isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp, tensor.ReshapeOp)):
            src = self.env[op.operands[0]]
            self.env[op.results[0]] = Val(_shape_of(op.results[0]), list(src.vals))
            return
        if isinstance(op, tensor.InsertSliceOp):
            self._insert_slice(op)
            return
        if isinstance(op, tensor.ExtractSliceOp):
            self._extract_slice(op)
            return
        if isinstance(op, linalg.FillOp):
            shape = _shape_of(op.results[0])
            v = self.env[op.operands[0]].vals[0]
            self.env[op.results[0]] = Val(shape, [v] * Val(shape, []).n)
            return
        if isinstance(op, linalg.TransposeOp):
            self._transpose(op)
            return
        if isinstance(op, linalg.MatmulOp):
            self._matmul(op)
            return
        if isinstance(op, linalg.ReduceOp):
            self._reduce(op)
            return
        if isinstance(op, linalg.GenericOp):
            self._generic(op)
            return
        raise HostLaneError(f"host lane: no lowering for {op.name!r}")

    def _static_list(self, attr) -> list[int]:
        if isinstance(attr, DenseArrayBase):
            return [int(x) for x in attr.get_values()]
        if hasattr(attr, "as_tuple"):
            return [int(x) for x in attr.as_tuple()]
        if isinstance(attr, ArrayAttr):
            return [int(x.value.data) for x in attr.data]
        raise HostLaneError(f"host lane: cannot read a static index list from {attr}")

    def _insert_slice(self, op) -> None:
        src = self.env[op.operands[0]]
        dst = self.env[op.operands[1]]
        offsets = self._static_list(op.static_offsets)
        sizes = self._static_list(op.static_sizes)
        strides = self._static_list(op.static_strides)
        out = list(dst.vals)
        for idx in _iter(tuple(sizes)):
            d = tuple(offsets[k] + idx[k] * strides[k] for k in range(len(idx)))
            out[_flat(dst.shape, d)] = src.vals[_flat(src.shape, idx)]
        self.env[op.results[0]] = Val(dst.shape, out)

    def _extract_slice(self, op) -> None:
        src = self.env[op.operands[0]]
        offsets = self._static_list(op.static_offsets)
        sizes = self._static_list(op.static_sizes)
        strides = self._static_list(op.static_strides)
        shape = _shape_of(op.results[0])
        out: list[SSAValue] = []
        for idx in _iter(tuple(sizes)):
            s = tuple(offsets[k] + idx[k] * strides[k] for k in range(len(idx)))
            out.append(src.vals[_flat(src.shape, s)])
        self.env[op.results[0]] = Val(shape, out)

    def _transpose(self, op) -> None:
        src = self.env[op.operands[0]]
        perm = self._static_list(op.permutation)
        shape = _shape_of(op.results[0])
        out: list[SSAValue] = []
        for idx in _iter(shape):
            s = tuple(idx[perm.index(k)] for k in range(len(perm)))
            out.append(src.vals[_flat(src.shape, s)])
        self.env[op.results[0]] = Val(shape, out)

    def _matmul(self, op) -> None:
        a = self.env[op.operands[0]]
        b = self.env[op.operands[1]]
        init = self.env[op.operands[2]]
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise HostLaneError("host lane: matmul extents do not contract")
        out: list[SSAValue] = []
        for i in range(m):
            for j in range(n):
                acc = init.vals[i * n + j]
                for t in range(k):
                    acc = self.f.add(
                        acc, self.f.mul(a.vals[i * k + t], b.vals[t * n + j])
                    )
                out.append(acc)
        self.env[op.results[0]] = Val((m, n), out)

    def _reduce(self, op) -> None:
        src = self.env[list(op.operands)[0]]
        init = self.env[list(op.operands)[1]]
        dims = set(self._static_list(op.dimensions))
        in_shape = src.shape
        out_shape = tuple(d for k, d in enumerate(in_shape) if k not in dims)
        acc = list(init.vals)
        for idx in _iter(in_shape):
            o = tuple(v for k, v in enumerate(idx) if k not in dims)
            pos = _flat(out_shape, o) if out_shape else 0
            acc[pos] = self._run_body(op.region.block, [acc[pos], src.vals[_flat(in_shape, idx)]])
        self.env[op.results[0]] = Val(out_shape, acc)

    def _generic(self, op) -> None:
        maps = [m.data for m in op.get_indexing_maps().data]
        ranges = op.get_static_loop_ranges()
        ins = [self.env[v] for v in op.inputs]
        outs = [self.env[v] for v in op.outputs]
        n_in = len(ins)
        result = list(outs[0].vals)
        out_shape = outs[0].shape
        for idx in _iter(tuple(ranges)):
            args: list[SSAValue] = []
            for k, v in enumerate(ins):
                pos = maps[k].eval(list(idx), [])
                args.append(v.vals[_flat(v.shape, tuple(pos))])
            for k, v in enumerate(outs):
                pos = maps[n_in + k].eval(list(idx), [])
                args.append(v.vals[_flat(v.shape, tuple(pos))])
            value = self._run_body(op.body.block, args, idx)
            pos = maps[n_in].eval(list(idx), [])
            result[_flat(out_shape, tuple(pos))] = value
        self.env[op.results[0]] = Val(out_shape, result)
