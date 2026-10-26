"""Static primitive Gemmini instruction streams and their bit-level reference semantics.

A recipe issues instructions into a ``Stream`` exactly as the validated bare-metal recipes
issue ``gemmini.h`` macros: every operand is a compile-time constant except the DRAM base
pointers, which are kernel arguments.  A DRAM operand is ``(argument index, byte offset)``.
``None`` as a MVIN source is the DRAM-address-zero zero-fill primitive.

``simulate`` executes a stream on byte buffers with the stock datapath semantics the
recipes rely on (weight- or output-stationary mesh with zero-padded tiles, int32 accumulator,
fp32 MVIN/readout scale with round-half-even and saturation, ReLU before scale as in
``AccumulatorScale.scala``).  It is the bit-level model the device must match exactly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import NamedTuple
from collections.abc import Sequence
from itertools import islice

import numpy as np

DIM = 16
SPAD_ROWS = 16384
ACC_ROWS = 1024
GARBAGE = 0xFFFFFFFF
ACC_BIT = 1 << 31
ACCUM_BIT = 1 << 30
FULL_BIT = 1 << 29
NO_ACT, RELU = 0, 1
# Dataflow select codes follow the provider's ISA transcription (CONFIG_EX rs1 dataflow).
from ..tables.isa import OUTPUT_STATIONARY as OS, WEIGHT_STATIONARY as WS  # noqa: E402


def SP(row: int) -> int:
    if not 0 <= row < SPAD_ROWS:
        raise ValueError(f"scratchpad row {row} out of range")
    return row


def ACC(row: int, *, accumulate: bool = False, full: bool = False) -> int:
    if not 0 <= row < ACC_ROWS:
        raise ValueError(f"accumulator row {row} out of range")
    return ACC_BIT | (ACCUM_BIT if accumulate else 0) | (FULL_BIT if full else 0) | row


def f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


class Ins(NamedTuple):
    kind: str
    attrs: tuple          # sorted ((key, value), ...) of compile-time fields
    arg: int | None       # DRAM argument index, None when the command has no DRAM operand
    off: int              # byte offset within the argument

    def a(self, key, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class InstructionLoop:
    count: int
    body: tuple
    strides: tuple  # (argument index, byte increment) for each CPU loop iteration


class CompactInstructions(Sequence):
    """Affine CPU loop tree; expansion is streaming for simulation and hazard lint.

    Only DRAM offsets vary. Local addresses and command fields are invariant,
    so a completed MVOUT separates successive overwrites of the same tile.
    """
    def __init__(self, nodes=()):
        self.nodes = list(nodes)

    def append(self, instruction):
        self.nodes.append(instruction)

    def __repr__(self):
        return f'CompactInstructions({self.nodes!r})'

    def counts(self):
        result = {}
        def walk(nodes, weight):
            for node in nodes:
                if isinstance(node,InstructionLoop):
                    walk(node.body,weight*node.count)
                else:
                    result[node.kind]=result.get(node.kind,0)+weight
        walk(self.nodes,1)
        return result

    def __len__(self):
        def size(nodes):
            return sum(n.count * size(n.body) if isinstance(n, InstructionLoop) else 1 for n in nodes)
        return size(self.nodes)

    def __iter__(self):
        def walk(nodes, offsets):
            for node in nodes:
                if isinstance(node, InstructionLoop):
                    for i in range(node.count):
                        shifted = dict(offsets)
                        for arg, stride in node.strides:
                            shifted[arg] = shifted.get(arg, 0) + i * stride
                        yield from walk(node.body, shifted)
                else:
                    yield node._replace(off=node.off + offsets.get(node.arg, 0)) if node.arg is not None else node
        return walk(self.nodes, {})

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if start == 0 and stop == len(self) and step == 1:
                return self
            if start == stop:
                return []
            if step == 1:
                pos, selected = 0, []
                for node in self.nodes:
                    end = pos + len(CompactInstructions([node]))
                    if pos >= start and end <= stop:
                        selected.append(node)
                    elif pos < stop and end > start:
                        break
                    pos = end
                else:
                    return CompactInstructions(selected)
            return list(islice(self, start, stop, step))
        if key < 0:
            key += len(self)
        return next(islice(self, key, key + 1))

    def templates(self):
        def walk(nodes):
            for node in nodes:
                if isinstance(node, InstructionLoop):
                    yield from walk(node.body)
                else:
                    yield node
        return walk(self.nodes)


@dataclass(frozen=True)
class Arg:
    name: str
    dtype: str            # MLIR element type; raw movement keeps the logical type
    shape: tuple[int, ...]
    role: str             # "input" | "output" | "inout" | "const"

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        from .storage import itemsize
        return n * itemsize(self.dtype)


@dataclass
class Stream:
    """Recorder for one kernel's primitive command stream."""

    args: list[Arg]
    ins: list[Ins] | CompactInstructions = field(default_factory=list)
    consts: dict[str, np.ndarray] = field(default_factory=dict)
    readouts: list = field(default_factory=list)
    packs: list = field(default_factory=list)
    bitwise: bool = False

    def __post_init__(self):
        self._index = {a.name: i for i, a in enumerate(self.args)}
        if len(self._index) != len(self.args):
            raise ValueError("duplicate kernel argument name")

    # -- operands --------------------------------------------------------------------------
    def at(self, name: str, byte_offset: int = 0) -> tuple[int, int]:
        idx = self._index[name]
        if byte_offset < 0 or byte_offset >= max(1, self.args[idx].nbytes):
            raise ValueError(f"DRAM offset {byte_offset} outside {name}")
        return (idx, int(byte_offset))

    def signed_scratch(self, source: str, shape, packed: str | None = None) -> str:
        """An i8 scratch argument the host fills with ``source``'s uint8 codes XOR 0x80 before the call."""
        packed = packed or source + "__signed"
        self._index[packed] = len(self.args)
        self.args.append(Arg(packed, "i8", shape, "scratch"))
        self.packs.append((source, packed))
        return packed

    def retarget_signed(self, source: str, shape) -> str:
        """``signed_scratch`` for ``source``, with the instructions already emitted on it moved there."""
        old = self._index[source]
        packed = self.signed_scratch(source, shape)
        self.ins = [Ins(i.kind, i.attrs, self._index[packed] if i.arg == old else i.arg, i.off) for i in self.ins]
        return packed

    def const(self, name: str, data: np.ndarray) -> None:
        """A compile-time constant operand (ones rows, identities, int32 seeds).

        Constants become read-only kernel-private data objects; they are listed as
        ``const`` arguments so the stream stays free of literal DRAM addresses.
        """
        data = np.ascontiguousarray(data)
        dtype = {np.dtype(np.int8): "i8", np.dtype(np.int32): "i32"}[data.dtype]
        if name in self._index:
            raise ValueError(f"constant {name} redefined")
        self.args.append(Arg(name, dtype, tuple(data.shape), "const"))
        self._index[name] = len(self.args) - 1
        self.consts[name] = data

    def bind_constant(self, name: str, data: np.ndarray):
        """Specialize a declared target operand to immutable literal bytes."""
        from dataclasses import replace
        arg = self.args[self._index[name]]
        from .storage import numpy_storage
        data = np.array(data,copy=True,order='C')
        data.flags.writeable = False
        if data.dtype != numpy_storage(arg.dtype) or tuple(data.shape) != arg.shape:
            raise ValueError('constant storage differs from declared target operand')
        self.args[self._index[name]] = replace(arg, role='const')
        self.consts[name] = data

    def _emit(self, kind, dram=None, **attrs):
        arg, off = (None, 0) if dram is None else dram
        self.ins.append(Ins(kind, tuple(sorted(attrs.items())), arg, off))

    # -- commands (names follow gemmini.h) -------------------------------------------------
    def flush(self):
        self._emit("flush", skip=0)

    def fence(self):
        self._emit("fence")

    def config_ex(self, *, dataflow=WS, act=NO_ACT, a_stride=1, c_stride=1, a_transpose=0, b_transpose=0):
        self._emit("config_ex", dataflow=dataflow, act=act, acc_scale=1.0, a_stride=a_stride,
                   c_stride=c_stride, a_transpose=int(a_transpose), b_transpose=int(b_transpose))

    def config_ld(self, state: int, stride: int, scale: float = 1.0, shrink: bool = False,
                  block_stride: int = DIM, pixel_repeats: int = 1):
        if state not in (0, 1, 2):
            raise ValueError("load state must be 0, 1 or 2")
        self._emit("config_ld", load_id=state, stride=int(stride), scale=f32(scale), shrunk=int(bool(shrink)),
                   block_stride=int(block_stride), pixel_repeats=int(pixel_repeats))

    def config_st(self, stride: int, act: int = NO_ACT, scale: float = 1.0, *, multiplier_bits=None, pool=None):
        """``pool`` holds the store-path window fields (``pool_plan.POOL_FIELDS``).

        Pooled fields are checked against the derived CONFIG_ST register widths and
        are always encoded with explicit multiplier bits through the derived layout.
        """
        fields = {}
        if pool is not None:
            from .pool_plan import POOL_FIELDS, register_limits
            limits = register_limits()
            if set(pool) != set(POOL_FIELDS) or any(
                    type(pool[k]) is not int or not 0 <= pool[k] <= limits[k] for k in POOL_FIELDS):
                raise ValueError('pooled store fields do not fit the derived CONFIG_ST register')
            if multiplier_bits is None:
                raise ValueError('a pooled store configuration states its multiplier bits')
            fields.update(pool)
        if multiplier_bits is not None:
            scale = struct.unpack('<f', struct.pack('<I', multiplier_bits))[0]
            if not np.isfinite(scale):
                raise ValueError('readout multiplier bits must denote finite f32')
            fields['multiplier_bits'] = multiplier_bits
        self._emit("config_st", stride=int(stride), acc_act=int(act), acc_scale=f32(scale), **fields)

    def narrow_readout(self, multiplier_bits, *, relu, columns_per_scale):
        """Drain an existing exact MAC stream under source-proved register bits."""
        from dataclasses import replace
        from .readout_encoding import activation, flags

        bits = tuple(int(v) for v in multiplier_bits)
        output = self.args[self._index['out']]
        columns = output.shape[-1]
        if len(bits) != columns or columns_per_scale < DIM or columns_per_scale % DIM:
            raise ValueError('readout gain blocks must cover whole output column tiles')
        for start in range(0, columns, columns_per_scale):
            if len(set(bits[start:start + columns_per_scale])) != 1:
                raise ValueError('unproved column partition within a gain block')
        self.args[self._index['out']] = replace(output, dtype='i8')
        original, self.ins = self.ins, []
        current = None
        for ins in original:
            if ins.kind == 'config_st':
                continue
            if ins.kind == 'mvout':
                column = (ins.off // 4) % columns
                selected = bits[column]
                if selected != current:
                    self.config_st(columns, activation(relu), multiplier_bits=selected)
                    current = selected
                attrs = dict(ins.attrs)
                attrs['local'] &= ~flags()['full_c_bit']
                ins = ins._replace(attrs=tuple(sorted(attrs.items())), off=ins.off // 4)
            self.ins.append(ins)

    def mvin(self, src, local: int, cols: int, rows: int, state: int = 0):
        """``src`` is ``(arg, off)`` or ``None`` for the zero-fill primitive."""
        if not (1 <= rows <= DIM and 1 <= cols <= (DIM if local & ACC_BIT else 4 * DIM)):
            raise ValueError(f"mvin extent {rows}x{cols} out of range")
        self._emit("mvin", src, local=int(local), cols=int(cols), rows=int(rows), load_id=state,
                   zero=int(src is None))

    def mvout(self, dst, local: int, cols: int, rows: int):
        wide = bool(local & ACC_BIT) and not local & FULL_BIT
        if not (1 <= rows <= DIM and 1 <= cols <= (4 * DIM if wide else DIM)):
            raise ValueError(f"mvout extent {rows}x{cols} out of range")
        self._emit("mvout", dst, local=int(local), cols=int(cols), rows=int(rows))

    def preload(self, bd: int, c: int, bd_cols: int, bd_rows: int, c_cols: int, c_rows: int):
        self._emit("preload", bd=int(bd), c=int(c), bd_cols=int(bd_cols), bd_rows=int(bd_rows),
                   c_cols=int(c_cols), c_rows=int(c_rows))

    def compute(self, a: int, a_cols: int, a_rows: int, *, accumulate: bool = False,
                bd: int = GARBAGE, bd_cols: int = DIM, bd_rows: int = DIM):
        self._emit("compute", a=int(a), bd=int(bd), a_cols=int(a_cols), a_rows=int(a_rows),
                   bd_cols=int(bd_cols), bd_rows=int(bd_rows), accumulate=int(bool(accumulate)))

    # -- composite helpers -----------------------------------------------------------------
    def matmul_tile(self, a: int, b: int, c: int, rows: int, k: int, cols: int, *, reuse: bool = False):
        """One WS 16x16x16 step: PRELOAD B (or keep it), COMPUTE A into C."""
        self.preload(GARBAGE if reuse else b, c, cols, k, cols, rows)
        self.compute(a, k, rows, accumulate=reuse)

    def counts(self) -> dict[str, int]:
        if isinstance(self.ins,CompactInstructions):
            return self.ins.counts()
        out: dict[str, int] = {}
        for i in self.ins:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out


# ---------------------------------------------------------------------------------------------
# Bit-level reference semantics
# ---------------------------------------------------------------------------------------------

def _rne_sat(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi).astype(np.int64)


def scale_f32(values: np.ndarray, scale: float) -> np.ndarray:
    """fp32(value) * fp32(scale), as the scale units compute it."""
    return values.astype(np.float32) * np.float32(scale)


class Machine:
    """Functional model of the subset of stock Gemmini the recipes use."""

    def __init__(self, buffers: dict[int, np.ndarray]):
        self.dram = {k: v.reshape(-1).view(np.uint8) for k, v in buffers.items()}
        self.spad = np.zeros((SPAD_ROWS, DIM), np.int64)
        self.acc = np.zeros((ACC_ROWS, DIM), np.int64)
        self.ld = {s: dict(stride=0, scale=1.0, shrunk=0, block_stride=DIM, pixel_repeats=1) for s in range(3)}
        self.st = dict(stride=0, acc_act=0, acc_scale=1.0, pool_stride=0)
        self.ex = dict(dataflow=WS, act=0, a_transpose=0, b_transpose=0, c_stride=1, a_stride=1)
        self.weights = np.zeros((DIM, DIM), np.int64)

    def _read(self, arg, off, rows, cols, stride, width):
        out = np.zeros((rows, cols), np.int64)
        buf = self.dram[arg]
        dt = np.int32 if width == 4 else np.int8
        for r in range(rows):
            start = off + r * stride
            out[r] = np.frombuffer(buf[start:start + cols * width].tobytes(), dt).astype(np.int64)
        return out

    def _write(self, arg, off, block, stride, width):
        buf = self.dram[arg]
        dt = np.int32 if width == 4 else np.int8
        for r in range(block.shape[0]):
            start = off + r * stride
            buf[start:start + block.shape[1] * width] = np.frombuffer(block[r].astype(dt).tobytes(), np.uint8)

    def run(self, stream: Stream):
        for ins in stream.ins:
            getattr(self, "_" + ins.kind)(ins)

    def _flush(self, ins):
        pass

    def _fence(self, ins):
        pass

    def _config_ex(self, ins):
        self.ex.update({k: ins.a(k) for k in ("dataflow", "act", "a_transpose", "b_transpose", "c_stride", "a_stride")})
        if self.ex["dataflow"] not in (WS, OS) or self.ex["act"] not in (NO_ACT, RELU) or self.ex["a_stride"] != 1:
            raise NotImplementedError("model covers WS/OS with no activation or ReLU and unit a_stride")
        if self.ex["dataflow"] == OS and (self.ex["a_transpose"] or self.ex["b_transpose"]):
            raise NotImplementedError("output-stationary transposes are not modelled")

    def _config_ld(self, ins):
        self.ld[ins.a("load_id")] = {k: ins.a(k) for k in ("stride", "scale", "shrunk", "block_stride", "pixel_repeats")}
        if ins.a("pixel_repeats") != 1:
            raise NotImplementedError("pixel repeats are not modelled")

    def _config_st(self, ins):
        from .pool_plan import POOL_FIELDS
        self.st = {k: ins.a(k) for k in ("stride", "acc_act", "acc_scale")}
        self.st.update({k: ins.a(k, 0) for k in POOL_FIELDS})

    def _mvin(self, ins):
        cfg = self.ld[ins.a("load_id")]
        local, rows, cols = ins.a("local"), ins.a("rows"), ins.a("cols")
        is_acc = bool(local & ACC_BIT)
        width = 1 if (not is_acc or cfg["shrunk"]) else 4
        if ins.a("zero"):
            data = np.zeros((rows, cols), np.int64)
        else:
            data = self._read(ins.arg, ins.off, rows, cols, cfg["stride"], width)
        scale = cfg["scale"]
        if is_acc and not cfg["shrunk"]:
            if scale != 1.0:
                # the stock configuration has no int32 MVIN scale unit
                raise NotImplementedError("scaled full-width accumulator load")
        elif scale != 1.0:
            # int8 MVIN scale unit, also for shrunk accumulator loads: the result is int8
            data = _rne_sat(scale_f32(data, scale), -128, 127)
        row = local & 0x3FFF
        for b0 in range(0, cols, DIM):
            blk = data[:, b0:b0 + DIM]
            base = row + (b0 // DIM) * cfg["block_stride"]
            if is_acc:
                if local & ACCUM_BIT:
                    self.acc[base:base + rows, :blk.shape[1]] += blk
                    self.acc[base:base + rows] = ((self.acc[base:base + rows] + (1 << 31)) % (1 << 32)) - (1 << 31)
                else:
                    self.acc[base:base + rows, :blk.shape[1]] = blk
            else:
                self.spad[base:base + rows, :blk.shape[1]] = blk

    def _pooled_mvout(self, ins):
        """Store-path max pooling as StoreController/DMA implement it.

        Window taps outside ``orows x ocols`` (including upad/lpad) are garbage
        addresses, whose write data is all zeros (Scratchpad.scala: is_all_zeros);
        the first tap is stored and later taps are maxed into it.
        """
        local, cols, st = ins.a("local"), ins.a("cols"), self.st
        if cols > DIM or local & FULL_BIT:
            raise NotImplementedError("pooled stores move one narrow column block")
        acc = bool(local & ACC_BIT)
        base = local & 0x3FFF if acc else local
        size, step = st["pool_size"], st["pool_stride"]
        for porow in range(st["porows"]):
            for pocol in range(st["pocols"]):
                value = None
                for wrow in range(size):
                    for wcol in range(size):
                        orow = porow * step + wrow - st["upad"]
                        ocol = pocol * step + wcol - st["lpad"]
                        if not (0 <= orow < st["orows"] and 0 <= ocol < st["ocols"]):
                            elem = np.zeros(cols, np.int64)
                        elif acc:
                            elem = self.acc[base + orow * st["ocols"] + ocol, :cols]
                            if st["acc_act"] == RELU:
                                elem = np.maximum(elem, 0)
                            elem = _rne_sat(scale_f32(elem, st["acc_scale"]), -128, 127)
                        else:
                            elem = self.spad[base + orow * st["ocols"] + ocol, :cols]
                        value = elem if value is None else np.maximum(value, elem)
                offset = ins.off + (porow * st["pool_out_dim"] + pocol) * st["stride"]
                self._write(ins.arg, offset, value.reshape(1, -1), 0, 1)

    def _mvout(self, ins):
        if self.st.get("pool_stride"):
            return self._pooled_mvout(ins)
        local, rows, cols = ins.a("local"), ins.a("rows"), ins.a("cols")
        row = local & 0x3FFF
        stride = self.st["stride"]
        if local & ACC_BIT:
            blocks = [self.acc[row + (b0 // DIM) * DIM: row + (b0 // DIM) * DIM + rows, :min(DIM, cols - b0)]
                      for b0 in range(0, cols, DIM)]
            data = np.concatenate(blocks, axis=1)
            if local & FULL_BIT:
                self._write(ins.arg, ins.off, data, stride, 4)
                return
            if self.st["acc_act"] == RELU:
                data = np.maximum(data, 0)
            data = _rne_sat(scale_f32(data, self.st["acc_scale"]), -128, 127)
            self._write(ins.arg, ins.off, data, stride, 1)
        else:
            self._write(ins.arg, ins.off, self.spad[row:row + rows, :cols], stride, 1)

    def _local_tile(self, addr, rows, cols):
        if addr & ACC_BIT:
            raise ValueError("execute operand from the accumulator (HW1)")
        t = np.zeros((DIM, DIM), np.int64)
        t[:rows, :cols] = self.spad[addr:addr + rows, :cols]
        return t

    def _preload(self, ins):
        self._c = (ins.a("c"), ins.a("c_rows"), ins.a("c_cols"))
        if self.ex["dataflow"] == OS:
            # The PEs hold the partial sums; D seeds them on the next preloaded compute.
            self._d = (np.zeros((DIM, DIM), np.int64) if ins.a("bd") == GARBAGE else
                       self._local_tile(ins.a("bd"), ins.a("bd_rows"), ins.a("bd_cols")))
            return
        if ins.a("bd") != GARBAGE:
            rows, cols = ins.a("bd_rows"), ins.a("bd_cols")
            if self.ex["b_transpose"]:
                # PRELOAD takes the logical post-transpose extents.
                t = self._local_tile(ins.a("bd"), cols, rows).T
            else:
                t = self._local_tile(ins.a("bd"), rows, cols)
            self.weights = t

    def _compute(self, ins):
        a_rows, a_cols = ins.a("a_rows"), ins.a("a_cols")
        if self.ex["a_transpose"]:
            a = self._local_tile(ins.a("a"), a_cols, a_rows).T
        else:
            a = self._local_tile(ins.a("a"), a_rows, a_cols)
        if self.ex["dataflow"] == OS:
            # Exact partial sums: admission proves the mesh never wraps them.
            if not ins.a("accumulate"):
                self._pe = self._d.copy()
            b = (np.zeros((DIM, DIM), np.int64) if ins.a("bd") == GARBAGE else
                 self._local_tile(ins.a("bd"), ins.a("bd_rows"), ins.a("bd_cols")))
            self._pe = self._pe + a @ b
            out = self._pe
        elif ins.a("bd") != GARBAGE:
            raise NotImplementedError("WS D operand is not modelled")
        else:
            out = a @ self.weights
        c, c_rows, c_cols = self._c
        if c == GARBAGE:
            return
        row = c & 0x3FFF
        cs = self.ex["c_stride"]
        if c & ACC_BIT:
            for r in range(c_rows):
                dst = row + r * cs
                if c & ACCUM_BIT:
                    self.acc[dst, :c_cols] += out[r, :c_cols]
                else:
                    self.acc[dst, :c_cols] = out[r, :c_cols]
        else:
            for r in range(c_rows):
                clipped = np.clip(out[r, :c_cols], -128, 127)
                self.spad[row + r * cs, :c_cols] = np.maximum(clipped, 0) if self.ex['act'] == RELU else clipped


def simulate(stream: Stream, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ``stream`` on copies of ``inputs``; returns every non-const argument buffer."""
    buffers = {}
    for i, arg in enumerate(stream.args):
        if arg.role != 'const' and (stream.bitwise or arg.dtype not in ('i8', 'i32')):
            # Opaque elements have no numerical interpretation in byte DMA.
            if arg.name in inputs:
                data = np.ascontiguousarray(inputs[arg.name]).reshape(-1).view(np.uint8)
                if data.nbytes != arg.nbytes:
                    raise ValueError('raw movement input byte extent differs: ' + arg.name)
                buffers[i] = data.copy()
            else:
                buffers[i] = np.full(arg.nbytes, 0x5a, dtype=np.uint8)
            continue
        dt = np.int32 if arg.dtype == "i32" else np.int8
        if arg.role == "const":
            data = stream.consts[arg.name]
        elif arg.name in inputs:
            data = np.asarray(inputs[arg.name])
        else:
            data = np.full(arg.shape, 0x5A if dt is np.int8 else 0x5A5A5A5A, dt)
        buffers[i] = np.ascontiguousarray(data.astype(dt)).reshape(arg.shape).copy()
    for source,dest in stream.packs:
        buffers[stream._index[dest]].view(np.uint8).flat[:] = buffers[stream._index[source]].view(np.uint8).reshape(-1) ^ 128
    from dataclasses import replace
    from .requant import apply
    machine=Machine(buffers)
    groups={}
    for r in stream.readouts: groups.setdefault(len(stream.ins) if r.after is None else r.after,[]).append(r)
    start=0
    for boundary,readouts in sorted(groups.items()):
        machine.run(replace(stream,ins=stream.ins[start:boundary],readouts=[]))
        apply(stream,buffers,readouts)
        start=boundary
    machine.run(replace(stream,ins=stream.ins[start:],readouts=[]))
    if stream.bitwise:
        from .storage import numpy_storage
        return {arg.name: buffers[i].view(numpy_storage(arg.dtype)).reshape(arg.shape)
                for i, arg in enumerate(stream.args) if arg.role != 'const'}
    return {arg.name: buffers[i] for i, arg in enumerate(stream.args) if arg.role != "const"}
