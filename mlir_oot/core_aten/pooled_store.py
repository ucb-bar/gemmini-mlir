"""Window-reduction compositions: store-path max pooling and ones-matmul averages.

* Fused pooled readout (contraction + readout + max pool): Merlin's source-bound
  ``conv_pool`` plan proves the f32 readout; the provider bands the contraction
  rows so each accumulator image holds whole pool windows and drains it with one
  pooled store per channel block.
* Standalone pooled store: identical-qparam int8 max pooling moves codes into the
  scratchpad (an explicit border of the storage minimum realizes ``-inf``
  padding), fences, and drains through a pooled store.
* Window mean: Merlin's ``avgpool`` plan supplies the multiplier for a
  ones-contraction sum read out on the device.

Register padding reads zero and is used only for source-proved nonnegative
fused readouts with zero point zero. Admission replays the emitted stream and checks every
output element, every window tap and every effective configuration against the
source. Anything unproved keeps the unfused route.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .source import module_of
from . import pool_plan
from .stream_proof import ProofError, Replay, contraction_row, execute, is_zero, require, written_once

_RESHAPES = ("tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape")


@dataclass(frozen=True)
class Geometry:
    image: pool_plan.Image
    window: pool_plan.Window


def _float_constant(value):
    """A scalar float constant, including non-finite values, else None."""
    from xdsl.dialects.builtin import FloatAttr

    from .match import _attr_of, producer

    owner = producer(value)
    if owner is not None and owner.name == "arith.constant":
        attr = _attr_of(owner, "value")
        if isinstance(attr, FloatAttr):
            return float(attr.value.data)
    return None


def _scalar_source(value):
    """The scalar of a fill or splat, including non-finite values, else None."""
    from .match import producer

    owner = producer(value)
    if owner is not None and owner.name in ("linalg.fill", "tensor.splat"):
        return _float_constant(owner.operands[0])
    return None


def _pool_window(op):
    """``(size, stride)`` of a square NHWC max-pool generic, from its indexing maps."""
    from .match import generic_maps, iterators, map_coeffs, shape

    if op.name != "linalg.generic" or len(op.results) != 1 or len(op.operands) != 3:
        return None
    body = list(op.regions[0].blocks[0].ops)
    block = op.regions[0].blocks[0]
    if [o.name for o in body] != ["arith.maximumf", "linalg.yield"]:
        return None
    if set(body[0].operands) != {block.args[0], block.args[2]} or list(body[1].operands) != [body[0].results[0]]:
        return None
    if _scalar_source(op.operands[2]) != float("-inf"):
        return None  # the init must be the identity of max
    maps = generic_maps(op)
    if iterators(op) != ["parallel"] * 4 + ["reduction"] * 2 or len(maps) != 3:
        return None
    data, window, out = (map_coeffs(m) for m in maps)
    unit = [[int(i == j) for j in range(6)] for i in range(6)]
    if out != unit[:4] or window != unit[4:]:
        return None
    if data is None or data[0] != unit[0] or data[3] != unit[3]:
        return None
    stride = data[1][1]
    if data[1] != [0, stride, 0, 0, 1, 0] or data[2] != [0, 0, stride, 0, 0, 1] or stride <= 0:
        return None
    extents = shape(op.operands[1])
    if extents is None or len(extents) != 2 or extents[0] != extents[1]:
        return None
    return extents[0], stride


# -- fused contraction readout + pool ------------------------------------------------------

def plan_fused(module, op, options, chain):
    """A Merlin ``conv_pool`` plan the store path can realize, or None (unfused route)."""
    from merlin.compile.composition_fusion import plan_composition

    from .match import shape
    from .readout_encoding import supports_narrow_path
    from .stream import DIM

    if options is None or len(shape(op.operands[0])) != 2 or not supports_narrow_path(options["path"]):
        return None
    operations = list(module.walk())
    members = (op, *chain.ops)
    declaration = dict(id="store_path_window_max", producer="contraction", reduction_kinds=["max"],
                       readout_stages="from_output_path", constraints=[])
    candidate = dict(route="conv_pool", operation_ordinals=[operations.index(o) for o in members],
                     detail=dict(composition=declaration, root_ordinal=operations.index(op)))
    facts = {**(options.get("facts") or {}), **pool_plan.composition_facts()}
    # StoreController's out-of-image addresses select Scratchpad's all-zero
    # data. Supply that hardware fact to Merlin's source-bound composition plan.
    path = {**options["path"], "readout": [
        {**stage, "padding": "zero_tap"} if stage["kind"] == "pool" else dict(stage)
        for stage in options["path"]["readout"]]}
    try:
        plan = plan_composition(module, candidate, path=path, scheme=options["scheme"], facts=facts)
        geometry = _fused_geometry(op, chain, plan.decision["proof"])
    except ValueError:
        return None
    proof = plan.decision["proof"]
    if proof.get("definitional") != "f32_product_rne" or proof.get("output_column_partition"):
        return None
    bits = proof["multiplier_bits"]
    if any(len(set(bits[c0:c0 + DIM])) != 1 for c0 in range(0, len(bits), DIM)):
        return None  # each pooled store drains one column block under one multiplier
    return plan, chain, geometry


def _fused_geometry(op, chain, proof):
    from .match import shape

    m, _ = shape(op.operands[0])
    channels = shape(op.results[0])[1]
    pool = next(stage.op for stage in chain.stages if stage.kind == "pool")
    unpadded = _unpad(pool.operands[0])
    if unpadded is None:
        raise ValueError("pool padding is not a source -inf border")
    value, pad, insert = unpadded
    for member in chain.ops[: chain.ops.index(pool)]:
        if member is insert:
            continue
        dims = shape(member.results[0])
        if (member.name not in ("linalg.generic", *_RESHAPES) or dims[-1] != channels
                or int(np.prod(dims)) != m * channels):
            raise ValueError("pool input is not a channels-last view of the contraction rows")
    found = _pool_window(pool)
    geometry = proof.get("pool_geometry") or {}
    if found is None or geometry.get("padded") != (insert is not None):
        raise ValueError("square channels-last window maximum not established")
    if any(pad):
        kinds = [stage.kind for stage in chain.stages]
        before = kinds[:kinds.index("pool")]
        if ("relu" not in before or any(kind != "pad" for kind in before[before.index("relu") + 1:])
                or chain.stages[-1].detail.get("zero_point") != 0):
            raise ValueError("zero pool taps require a source nonnegative readout and zero point zero")
    size, stride = found
    if geometry.get("window") != [size, size] or geometry.get("stride") != [stride, stride]:
        raise ValueError("pool geometry differs from the composition proof")
    batch, rows, cols, last = shape(value)
    if last != channels or batch * rows * cols != m:
        raise ValueError("pool image does not cover the contraction rows")
    image, window = pool_plan.Image(batch, rows, cols, channels), pool_plan.Window(size, stride, pad)
    out = window.output(image)
    if shape(pool.results[0]) != (batch, *out, channels) or shape(chain.ops[-1].results[0]) != shape(pool.results[0]):
        raise ValueError("pooled result extent differs from floor-mode windows")
    return Geometry(image, window)


def _units(geometry, capacity):
    from .stream import DIM

    bands = pool_plan.plan(geometry.image, geometry.window, capacity_rows=capacity, register_padding=True)
    channels = geometry.image.channels
    return [(band, c0, min(DIM, channels - c0)) for band in bands for c0 in range(0, channels, DIM)]


def emit_fused(kernel, planned):
    """Replace a full-width contraction stream by banded pooled readout stores."""
    from ..tables.rtl_facts import ACC_BANK_ROWS, ACC_BANKS
    from .readout_encoding import activation
    from .stream import ACC, DIM, GARBAGE, SPAD_ROWS, SP, Arg, Stream

    plan, _, geometry = planned
    image, window = geometry.image, geometry.window
    proof = plan.decision["proof"]
    relu = "relu" in plan.decision["stages"]
    old = kernel.stream
    m = image.batch * image.rows * image.cols
    depth, channels = old.args[old._index["a"]].shape[1], image.channels
    out_rows, out_cols = window.output(image)
    s = Stream([Arg("a", "i8", (m, depth), "input"), Arg("b", "i8", (depth, channels), "input"),
                Arg("out", "i8", (image.batch, out_rows, out_cols, channels), "output")])
    seed = old.consts.get("bias_c")
    if seed is not None:
        s.const("bias_c", seed)
    row_seeds = seed is not None and seed.ndim == 2
    s.fence()
    s.flush()
    s.config_ex()
    s.config_ld(0, depth)
    s.config_ld(1, channels)
    s.config_ld(2, channels * 4 if row_seeds else 0)
    k_tiles = -(-depth // DIM)
    half = SPAD_ROWS // 2 // DIM
    blocks = 0
    for unit, (band, c0, width) in enumerate(_units(geometry, ACC_BANK_ROWS)):
        base = (unit % ACC_BANKS) * ACC_BANK_ROWS
        first = (band.batch * image.rows + band.in_rows[0]) * image.cols
        rows = band.orows * band.ocols
        tiles = -(-rows // DIM)
        span = max(1, half // (tiles + 1))
        if seed is not None:
            for t in range(tiles):
                offset = ((first + t * DIM) * channels if row_seeds else 0) + c0
                s.mvin(s.at("bias_c", offset * 4), ACC(base + t * DIM), width, min(DIM, rows - t * DIM), 2)
        for k0 in range(0, k_tiles, span):
            tk = min(span, k_tiles - k0)
            a_base = (blocks & 1) * half * DIM
            b_base = a_base + tiles * tk * DIM
            blocks += 1
            for t in range(tiles):
                mr = min(DIM, rows - t * DIM)
                for k in range(0, tk, 4):
                    cols = min(min(4, tk - k) * DIM, depth - (k0 + k) * DIM)
                    s.mvin(s.at("a", (first + t * DIM) * depth + (k0 + k) * DIM), SP(a_base + (t * tk + k) * DIM),
                           cols, mr, 0)
            for k in range(tk):
                kr = min(DIM, depth - (k0 + k) * DIM)
                s.mvin(s.at("b", (k0 + k) * DIM * channels + c0), SP(b_base + k * DIM), width, kr, 1)
            for k in range(tk):
                kr = min(DIM, depth - (k0 + k) * DIM)
                accumulate = seed is not None or k0 + k > 0
                for t in range(tiles):
                    mr = min(DIM, rows - t * DIM)
                    s.preload(SP(b_base + k * DIM) if t == 0 else GARBAGE,
                              ACC(base + t * DIM, accumulate=accumulate), width, kr, width, mr)
                    s.compute(SP(a_base + (t * tk + k) * DIM), kr, mr, accumulate=t > 0)
        s.config_st(channels, activation(relu), multiplier_bits=proof["multiplier_bits"][c0],
                    pool=band.fields(window, out_cols))
        s.mvout(s.at("out", (band.batch * out_rows + band.out_rows[0]) * out_cols * channels + c0),
                ACC(base), width, 1)
    _reset_store(s)
    s.fence()
    kernel.stream = s
    kernel.meta.update(output_dtype="i8", relu=relu, readout="f32_product_rne", device_readout="f32_product_rne",
                       requantization="device", multiplier_bits=proof["multiplier_bits"],
                       pooled_store=dict(kind="fused", image=[image.batch, image.rows, image.cols, image.channels],
                                         window=window.size, stride=window.stride, pad=list(window.pad)))


def _reset_store(s):
    """Leave pooling disabled for later kernels' stores."""
    from .readout_encoding import flags

    s.config_st(0, multiplier_bits=flags()["f1"])


def _final_store_unpooled(stream):
    from .stream_proof import decoded_store_config

    configs = [i for i in stream.ins if i.kind == "config_st"]
    require(configs and decoded_store_config(configs[-1])["pool_stride"] == 0,
            "kernel leaves store-path pooling enabled")


def _unravel(flat, dims):
    return tuple(int(v) for v in np.unravel_index(flat, dims))


@dataclass(frozen=True)
class FusedExpectation:
    """A fused pooled readout derived from the source contraction and its plan alone."""
    image: pool_plan.Image
    window: pool_plan.Window
    depth: int
    bits: list
    relu: bool
    seed: object  # int64 [N] or [M, N]; None when the source accumulator starts at zero


def _bound(match, operands, result):
    """The match binds exactly these source values (by identity), not emitter-named look-alikes."""
    bound = list(match.operands)
    require(len(bound) == len(operands) and all(
        name == want and value is source for (name, value), (want, source) in zip(bound, operands)),
        "kernel operands are not bound to the source values")
    require(match.result[0] == result[0] and match.result[1] is result[1], "kernel result is not the source result")


def fused_expectation(match, readout_options):
    from .integer_regions import contraction, planned_readout, source_seed

    op = match.root
    dims = contraction(op)
    require(dims is not None and readout_options is not None, "pooled readout root is not a source contraction")
    planned, _ = planned_readout(module_of(op), op, readout_options)
    seeded = source_seed(op, dims, planned)
    planned = seeded.planned
    require(planned is not None and planned[0].route == "conv_pool" and seeded.runtime_seed is None,
            "the source does not plan a fused pooled readout")
    plan, chain, geometry = planned
    proof = plan.decision["proof"]
    require(plan.decision.get("cut") == "pool" and proof.get("definitional") == "f32_product_rne",
            "the source pool plan is not a closed f32 readout")
    _bound(match, [("a", op.operands[0]), ("b", op.operands[1])], ("out", chain.ops[-1].results[0]))
    require({id(o) for o in (op, *chain.ops)} <= {id(o) for o in match.consumed},
            "the kernel does not replace the whole source chain")
    seed = None if seeded.values is None else np.asarray(seeded.values, np.int64)
    return FusedExpectation(geometry.image, geometry.window, dims.k, list(proof["multiplier_bits"]),
                            "relu" in plan.decision["stages"], seed)


def validate_fused(stream, expected):
    """Authenticate every pooled output of a fused readout against the source expectation."""
    from .readout_encoding import activation, flags

    batch, rows, cols, channels = (expected.image.batch, expected.image.rows, expected.image.cols,
                                   expected.image.channels)
    size, step, depth, bits = expected.window.size, expected.window.stride, expected.depth, expected.bits
    top, left, _, _ = expected.window.pad
    require(not any(expected.window.pad) or expected.relu,
            "zero garbage taps require the source nonnegative readout proof")
    require(len(bits) == channels, "readout multiplier bits do not cover the source channels")
    require(not stream.readouts and not stream.packs, "pooled readout retains a host epilogue")
    args = {a.name: a for a in stream.args}
    out = (batch, *expected.window.output(expected.image), channels)
    require(set(args) == {"a", "b", "out", *stream.consts} and
            (args["a"].shape, args["a"].dtype, args["a"].role) == ((batch * rows * cols, depth), "i8", "input") and
            (args["b"].shape, args["b"].dtype, args["b"].role) == ((depth, channels), "i8", "input") and
            (args["out"].shape, args["out"].dtype, args["out"].role) == (out, "i8", "output"),
            "kernel arguments differ from the source operands")
    expected_seed = expected.seed
    for name, data in stream.consts.items():
        require(name == "bias_c" and expected_seed is not None and np.array_equal(data, expected_seed),
                "accumulator seed constant differs from the source-proved bias")
    act = activation(expected.relu)
    counts = np.zeros(int(np.prod(out)), np.int64)
    acc_bit, full = flags()["acc_i8"], flags()["full_c_bit"]

    def seed_ok(tag, row, c0, width):
        if tag is None or is_zero(tag):
            values = np.zeros(width, np.int64)
        else:
            require(tag.name in stream.consts and tag.width >= width, "unproved accumulator seed load")
            values = stream.consts[tag.name].reshape(-1)[tag.flat:tag.flat + width].astype(np.int64)
            index = (row * channels + c0) if stream.consts[tag.name].ndim == 2 else c0
            require(tag.flat == index, "accumulator seed load reads another row or channel")
        if expected_seed is None:
            return not values.any()
        wanted = expected_seed[row, c0:c0 + width] if expected_seed.ndim == 2 else expected_seed[c0:c0 + width]
        return np.array_equal(values, wanted)

    def store(replay, ins, st):
        from ..tables.rtl_facts import ACC_BANK_ROWS

        local, width = ins.a("local"), ins.a("cols")
        require(st["pool_stride"] != 0 and local & acc_bit and not local & full,
                "fused readout store is not a narrow pooled accumulator store")
        require(st["activation"] == act, "decoded store activation differs from the source")
        require(st["pool_size"] == size and st["pool_stride"] == step,
                "decoded pool window or stride differs from the source")
        base = local & 0x3FFF
        checked = {}
        for porow in range(st["porows"]):
            for pocol in range(st["pocols"]):
                byte = ins.off + (porow * st["pool_out_dim"] + pocol) * st["stride"]
                n, pr, pc, c0 = _unravel(byte, out)
                for ch in range(width):
                    require(_unravel(byte + ch, out) == (n, pr, pc, c0 + ch), "pooled store leaves its output row")
                    counts[byte + ch] += 1
                require(all(st["acc_scale"] == bits[c] for c in range(c0, c0 + width)),
                        "decoded multiplier bits differ from the proved column block")
                source = {(n * rows + r) * cols + c
                          for r in range(pr * step - top, pr * step - top + size)
                          for c in range(pc * step - left, pc * step - left + size)
                          if 0 <= r < rows and 0 <= c < cols}
                device = set()
                for wrow in range(st["pool_size"]):
                    for wcol in range(st["pool_size"]):
                        orow = porow * st["pool_stride"] + wrow - st["upad"]
                        ocol = pocol * st["pool_stride"] + wcol - st["lpad"]
                        r, c = pr * step + wrow - top, pc * step + wcol - left
                        padded = not (0 <= orow < st["orows"] and 0 <= ocol < st["ocols"])
                        require(padded == (not (0 <= r < rows and 0 <= c < cols)),
                                "pool window garbage taps differ from the source padding")
                        if padded:
                            require(expected.relu, "pool window reads an unproved zero garbage tap")
                            continue  # decoded out-of-image taps select the hardware all-zero data
                        row = base + orow * st["ocols"] + ocol
                        require(row // ACC_BANK_ROWS == base // ACC_BANK_ROWS,
                                "pooled image leaves the accumulator bank its store is ordered against")
                        state = replay.acc.get(row)
                        require(state is not None and state.terms, "pool window reads an unwritten row")
                        source_row = state.terms[0][0].flat // depth
                        if (row, c0) not in checked:
                            contraction_row(state, lhs="a", row=source_row, depth=depth, cols=width,
                                            weight_ok=lambda tag, k: (tag.name == "b" and tag.flat == k * channels + c0
                                                                      and tag.width >= width),
                                            seed_ok=lambda tag: seed_ok(tag, source_row, c0, width))
                            checked[(row, c0)] = source_row
                        device.add(checked[(row, c0)])
                        require(checked[(row, c0)] == (n * rows + r) * cols + c,
                                "pool window tap differs from the source pixel")
                require(device == source, "pool window taps differ from the source window")
        for row in range(base, base + st["orows"] * st["ocols"]):
            if row in replay.acc:
                replay.acc[row].stored = True  # the pooled store drains its whole image

    _replay(stream, store)
    written_once(counts)
    _final_store_unpooled(stream)
    _backstop(stream, "out", lambda inputs: _fused_reference(inputs, expected))
    return "f32_product_rne"


# -- standalone pooled store ---------------------------------------------------------------

def _identity_codes(qx, q):
    """Q(DQ(c)) == c for every input code, under exact and f32 dequantization."""
    from .exactness import quantized

    if qx.unsigned or q.unsigned or qx.dtype != "i8":
        return False
    from .match import _prop_int

    low, high = _prop_int(qx.dq, "quant_min"), _prop_int(qx.dq, "quant_max")
    if low is None or high is None or (low, high) != (q.qmin, q.qmax):
        return False
    scale, out = Fraction(qx.scale), Fraction(q.scale)
    for code in range(low, high + 1):
        rounded = np.float32(code - qx.zero_point) * np.float32(qx.scale)
        for value in (Fraction(code - qx.zero_point) * scale, Fraction(float(rounded))):
            if quantized(value / out, q.qmin, q.qmax, q.zero_point) != code:
                return False
        if int(np.rint(np.float32(rounded) / np.float32(q.scale))) + q.zero_point != code:
            return False
    return True


def _unpad(value):
    """``(source, (top, left, bottom, right), insert op)`` for a ``-inf`` NHWC border."""
    from .match import producer, shape

    owner = producer(value)
    if owner is None or owner.name != "tensor.insert_slice":
        return value, (0, 0, 0, 0), None
    source, dest = owner.operands[0], owner.operands[1]
    if _scalar_source(dest) != float("-inf") or len(owner.operands) != 2:
        return None
    offsets = list(owner.properties["static_offsets"].iter_values())
    sizes = list(owner.properties["static_sizes"].iter_values())
    strides = list(owner.properties["static_strides"].iter_values())
    inner, outer = shape(source), shape(dest)
    if sizes != list(inner) or set(strides) != {1} or offsets[0] or offsets[3] or inner[::3] != outer[::3]:
        return None
    top, left = offsets[1], offsets[2]
    return source, (top, left, outer[1] - inner[1] - top, outer[2] - inner[2] - left), owner


@dataclass(frozen=True)
class MovementSource:
    qx: object
    q: object
    insert: object
    image: pool_plan.Image
    window: pool_plan.Window


def movement_source(op):
    """The identical-qparam int8 max pool rooted at pool ``op``, read from the source alone."""
    from .match import dequant, quant_sink, shape

    found = _pool_window(op)
    unpadded = None if found is None else _unpad(op.operands[0])
    if unpadded is None:
        return None
    value, pad, insert = unpadded
    qx, q = dequant(value), quant_sink(op.results[0])
    if qx is None or q is None or not _identity_codes(qx, q):
        return None
    batch, rows, cols, channels = shape(qx.value)
    image, window = pool_plan.Image(batch, rows, cols, channels), pool_plan.Window(*found, pad)
    try:
        if shape(op.results[0]) != (batch, *window.output(image), channels):
            return None
    except pool_plan.PoolPlanRefused:
        return None
    return MovementSource(qx, q, insert, image, window)


def match_pooled_movement(op):
    """Identical-qparam int8 max pooling as a fenced scratchpad pooled store."""
    from .match import Match, users

    found = movement_source(op)
    if found is None:
        return None
    qx, q, insert, window = found.qx, found.q, found.insert, found.window
    pad = window.pad
    try:
        kernel = _pooled_movement_kernel(found.image, window)
    except pool_plan.PoolPlanRefused:
        return None
    consumed = [op, q.op, *([insert] if insert is not None else [])]
    if len(users(qx.dq.results[0])) == 1:
        consumed.append(qx.dq)
    return Match("aten.max_pool2d.default", q.op, consumed, [("x", qx.value)], ("out", q.value), kernel,
                 numeric=dict(readout="storage_codes", identical_qparam_proof="exhaustive_code_identity",
                              pad=list(pad), window=window.size, stride=window.stride))


def _pooled_movement_kernel(image, window):
    from .lint import SPAD_STORE_ROWS
    from .ops import Kernel
    from .readout_encoding import flags
    from .stream import DIM, NO_ACT, SP, Arg, Stream

    out_rows, out_cols = window.output(image)
    top, left, _, _ = window.pad
    padded_rows, padded_cols = window.padded(image)
    channels = image.channels
    s = Stream([Arg("x", "i8", (image.batch, image.rows, image.cols, channels), "input"),
                Arg("out", "i8", (image.batch, out_rows, out_cols, channels), "output")])
    border = any(window.pad)
    if border:
        s.const("pool_border", np.full(DIM, np.iinfo(np.int8).min, np.int8))
    s.fence()
    s.flush()
    s.config_ld(0, channels)
    s.config_ld(1, 0)
    for band in pool_plan.plan(image, window, capacity_rows=SPAD_STORE_ROWS):
        for c0 in range(0, channels, DIM):
            width = min(DIM, channels - c0)
            for r in range(*band.in_rows):
                local = (r - band.in_rows[0]) * padded_cols
                h = r - top
                runs = [(0, padded_cols, None)] if not 0 <= h < image.rows else [
                    (0, left, None), (left, left + image.cols, h), (left + image.cols, padded_cols, None)]
                for start, stop, source_row in runs:
                    for p0 in range(start, stop, DIM):
                        count = min(DIM, stop - p0)
                        if source_row is None:
                            s.mvin(s.at("pool_border"), SP(local + p0), width, count, 1)
                        else:
                            pixel = (band.batch * image.rows + source_row) * image.cols + p0 - left
                            s.mvin(s.at("x", pixel * channels + c0), SP(local + p0), width, count, 0)
            s.fence()
            s.config_st(channels, NO_ACT, multiplier_bits=flags()["f1"], pool=band.fields(window, out_cols))
            s.mvout(s.at("out", (band.batch * out_rows + band.out_rows[0]) * out_cols * channels + c0), SP(0), width, 1)
            s.fence()
    _reset_store(s)
    s.fence()
    return Kernel("aten.max_pool2d.default", s, dict(
        composition_kernel=True, composition="pooled_movement", readout="storage_codes",
        device_readout="storage_codes", output_dtype="i8",
        pooled_store=dict(kind="movement", image=[image.batch, image.rows, image.cols, channels],
                          window=window.size, stride=window.stride, pad=list(window.pad))))


def movement_expectation(match):
    from .match import producer

    q = match.root
    pool = producer(q.operands[0]) if q.operands else None
    found = None if pool is None else movement_source(pool)
    require(found is not None and found.q.op is q, "the source is not an identical-qparam max pool")
    _bound(match, [("x", found.qx.value)], ("out", found.q.value))
    require(id(pool) in {id(o) for o in match.consumed}, "the kernel does not replace the source pool")
    return found


def validate_movement(stream, expected):
    """Authenticate every pooled output of a standalone pooled store against the source pool."""
    from .readout_encoding import flags

    image, window = expected.image, expected.window
    batch, rows, cols, channels = image.batch, image.rows, image.cols, image.channels
    size, step = window.size, window.stride
    top, left, _, _ = window.pad
    out = (batch, *window.output(image), channels)
    args = {a.name: a for a in stream.args}
    require(set(args) == {"x", "out", *stream.consts} and
            (args["x"].shape, args["x"].dtype, args["x"].role) == ((batch, rows, cols, channels), "i8", "input") and
            (args["out"].shape, args["out"].dtype, args["out"].role) == (out, "i8", "output"),
            "pooled movement operands differ from the source")
    require(not stream.readouts and not stream.packs and not any(i.kind in ("preload", "compute") for i in stream.ins),
            "pooled movement contains arithmetic")
    minimum = np.iinfo(np.int8).min
    for name, data in stream.consts.items():
        require(name == "pool_border" and bool((data == minimum).all()), "border constant is not the code minimum")
    counts = np.zeros(int(np.prod(out)), np.int64)
    acc_bit = flags()["acc_i8"]

    def store(replay, ins, st):
        local, width = ins.a("local"), ins.a("cols")
        require(st["pool_stride"] != 0 and not local & acc_bit, "standalone store is not a scratchpad pooled store")
        image_rows = range(local, local + st["orows"] * st["ocols"])
        require(not replay.loads_since_fence.intersection(image_rows),
                "scratchpad-source pooled store is not fenced after its loads")
        for porow in range(st["porows"]):
            for pocol in range(st["pocols"]):
                byte = ins.off + (porow * st["pool_out_dim"] + pocol) * st["stride"]
                n, pr, pc, c0 = _unravel(byte, out)
                for ch in range(width):
                    require(_unravel(byte + ch, out) == (n, pr, pc, c0 + ch), "pooled store leaves its output row")
                    counts[byte + ch] += 1
                source = {(pr * step + i - top, pc * step + j - left) for i in range(size) for j in range(size)}
                source = {(h, w) for h, w in source if 0 <= h < rows and 0 <= w < cols}
                device = set()
                for wrow in range(st["pool_size"]):
                    for wcol in range(st["pool_size"]):
                        orow = porow * st["pool_stride"] + wrow - st["upad"]
                        ocol = pocol * st["pool_stride"] + wcol - st["lpad"]
                        require(0 <= orow < st["orows"] and 0 <= ocol < st["ocols"],
                                "pool window reads a zero garbage tap")
                        tag = replay.spad.get(local + orow * st["ocols"] + ocol)
                        require(tag is not None and tag.width >= width, "pool window reads an unloaded row")
                        if tag.name == "pool_border":
                            values = stream.consts[tag.name].reshape(-1)[tag.flat:tag.flat + width]
                            require(values.size == width and bool((values == minimum).all()),
                                    "border tap is not the identity of max")
                            continue
                        require(tag.name == "x", "pool window reads another operand")
                        tn, h, w, tc = _unravel(tag.flat, (batch, rows, cols, channels))
                        require(tn == n and tc == c0, "pool window reads another image or channel block")
                        device.add((h, w))
                require(source and device == source, "pool window taps differ from the source window")

    _replay(stream, store)
    written_once(counts)
    _final_store_unpooled(stream)
    _backstop(stream, "out", lambda inputs: _window_max(inputs["x"].astype(np.int64), size, step, window.pad))
    return "storage_codes"


# -- window mean as a ones contraction ---------------------------------------------------------

@dataclass(frozen=True)
class MeanSource:
    qx: object
    reduce: object
    divide: object
    quantize: object
    plan: object
    rows: int
    window: int
    bits: int


def mean_source(op, module, options):
    """A closed DQ -> trailing sum -> divide -> Q chain and its Merlin avgpool plan, from the source."""
    from merlin.compile.composition_fusion import plan_composition

    from .match import dequant, producer, shape, users

    if options is None or op.name != "linalg.reduce" or len(users(op.results[0])) != 1:
        return None
    divide = users(op.results[0])[0]
    if len(divide.results) != 1 or len(users(divide.results[0])) != 1:
        return None
    quantize = users(divide.results[0])[0]
    dequantize = producer(op.operands[0])
    qx = dequant(op.operands[0])
    if qx is None or qx.dq is not dequantize or qx.unsigned:
        return None
    operations = list(module.walk())
    members = (dequantize, op, divide, quantize)
    declaration = dict(id="ones_contraction_mean", producer=["movement", "contraction"], reduction_kinds=["sum"],
                       constraints=[])
    candidate = dict(route="avgpool", operation_ordinals=[operations.index(o) for o in members],
                     detail=dict(composition=declaration, root_ordinal=operations.index(op)))
    try:
        facts = {**(options.get("facts") or {}), "accumulator_bits": _accumulator_bits()}
        plan = plan_composition(module, candidate, path=options["path"], scheme=options["scheme"], facts=facts)
    except ValueError:
        return None
    decision = plan.decision
    rows, window = decision["rows"], decision["window"]
    xshape, yshape = shape(qx.value), shape(quantize.results[0])
    if (int(np.prod(xshape)) != rows * window or int(np.prod(yshape)) != rows
            or decision.get("definitional") != "f32_product_rne" or decision.get("lowering") != "ones_matmul"
            or len(decision["multiplier_bits"]) != 1):
        return None
    return MeanSource(qx, op, divide, quantize, plan, rows, window, decision["multiplier_bits"][0])


def match_window_mean(op, module, options):
    """The window mean as a ones contraction read out under Merlin's avgpool multiplier."""
    from .match import Match, shape, users

    found = mean_source(op, module, options)
    if found is None:
        return None
    qx, divide, quantize = found.qx, found.divide, found.quantize
    dequantize, rows, window = qx.dq, found.rows, found.window
    kernel = _mean_kernel(shape(qx.value), shape(quantize.results[0]), rows, window, found.bits)
    kernel.meta["composition_plan"] = found.plan.record()
    consumed = [op, divide, quantize, *([dequantize] if len(users(dequantize.results[0])) == 1 else [])]
    return Match("aten.mean.dim", quantize, consumed, [("x", qx.value)], ("y", quantize.results[0]), kernel,
                 numeric=dict(readout="f32_product_rne", rows=rows, window=window))


def _accumulator_bits():
    from .readout_encoding import contract

    dtype = contract()["readout_semantics"]["scalar_abi"]["accumulator_dtype"]
    if not dtype.startswith("i") or not dtype[1:].isdigit():
        raise ValueError("UNKNOWN accumulator width")
    return int(dtype[1:])


def _mean_kernel(xshape, yshape, rows, window, bits):
    from .ops import Kernel
    from .recipes import reduce as rd
    from .stream import Arg, Stream

    s = Stream([Arg("x", "i8", tuple(xshape), "input"), Arg("y", "i8", tuple(yshape), "output")])
    s.fence()
    s.flush()
    rd.mean_last(s, "x", "y", rows, window, 1.0, multiplier_bits=bits)
    s.fence()
    return Kernel("aten.mean.dim", s, dict(composition_kernel=True, composition="window_mean",
                                           readout="f32_product_rne", device_readout="f32_product_rne",
                                           output_dtype="i8", multiplier_bits=[bits],
                                           dimensions=dict(rows=rows, k=window)))


def mean_expectation(match, readout_options):
    from .match import producer

    quantize = match.root
    divide = producer(quantize.operands[0]) if quantize.operands else None
    reduce = None if divide is None or not divide.operands else producer(divide.operands[0])
    found = None if reduce is None else mean_source(reduce, module_of(reduce), readout_options)
    require(found is not None and found.quantize is quantize, "the source is not a planned window mean")
    _bound(match, [("x", found.qx.value)], ("y", quantize.results[0]))
    require({id(reduce), id(divide), id(quantize)} <= {id(o) for o in match.consumed},
            "the kernel does not replace the source mean")
    return found


def validate_mean(stream, expected):
    """Authenticate every window-mean output against the source plan."""
    from .match import shape
    from .readout_encoding import flags

    rows, window, bits = expected.rows, expected.window, expected.bits
    args = {a.name: a for a in stream.args}
    require(set(args) == {"x", "y", *stream.consts} and
            (args["x"].shape, args["x"].dtype, args["x"].role) == (shape(expected.qx.value), "i8", "input") and
            (args["y"].shape, args["y"].dtype, args["y"].role) == (shape(expected.quantize.results[0]), "i8", "output"),
            "window mean operands differ from the source")
    require(not stream.readouts and not stream.packs, "window mean retains a host epilogue")
    counts = np.zeros(rows, np.int64)
    acc_bit, full = flags()["acc_i8"], flags()["full_c_bit"]

    def ones(tag, _k):
        data = stream.consts.get(tag.name)
        return data is not None and tag.width >= 1 and int(data.reshape(-1)[tag.flat]) == 1

    def store(replay, ins, st):
        local = ins.a("local")
        require(st["pool_stride"] == 0 and local & acc_bit and not local & full,
                "window mean store is not a narrow accumulator readout")
        require(st["activation"] == 0 and st["acc_scale"] == bits,
                "decoded readout fields differ from the proved multiplier bits")
        require(ins.a("cols") == 1, "window mean stores one column")
        for i in range(ins.a("rows")):
            element = ins.off + i * st["stride"]
            require(0 <= element < rows, "window mean store leaves its output")
            counts[element] += 1
            state = replay.acc.get((local & 0x3FFF) + i)
            require(state is not None, "window mean reads an unwritten row")
            contraction_row(state, lhs="x", row=element, depth=window, cols=1, weight_ok=ones,
                            seed_ok=lambda tag: tag is None or is_zero(tag))

    _replay(stream, store)
    written_once(counts)
    _backstop(stream, "y", lambda inputs: _mean_reference(inputs["x"], rows, bits))
    return "f32_product_rne"


# -- replay and differential backstop -----------------------------------------------------------

def _replay(stream, store):
    replay = Replay(stream, store, store_fields=pool_plan.POOL_FIELDS).run()
    require(not replay.unfenced, "stream ends with unfenced stores")
    return replay


def _window_max(image, size, step, pad=(0, 0, 0, 0)):
    top, left, bottom, right = pad
    framed = np.pad(image.astype(np.float64), ((0, 0), (top, bottom), (left, right), (0, 0)), constant_values=-np.inf)
    rows, cols = (framed.shape[1] - size) // step + 1, (framed.shape[2] - size) // step + 1
    out = np.full((framed.shape[0], rows, cols, framed.shape[3]), -np.inf)
    for i in range(size):
        for j in range(size):
            out = np.maximum(out, framed[:, i:i + step * (rows - 1) + 1:step, j:j + step * (cols - 1) + 1:step])
    return out.astype(np.int64)


def _fused_reference(inputs, expected):
    from merlin.runtime.exact_readout import f32_readout

    acc = inputs["a"].astype(np.int64) @ inputs["b"].astype(np.int64)
    if expected.seed is not None:
        acc = acc + expected.seed
    multiplier = np.array(expected.bits, dtype=np.uint32).view(np.float32)
    readout = f32_readout(acc, multiplier, "i8", relu=expected.relu).astype(np.int64)
    image = expected.image
    return _window_max(readout.reshape(image.batch, image.rows, image.cols, image.channels),
                       expected.window.size, expected.window.stride, expected.window.pad)


def _mean_reference(x, rows, bits):
    from merlin.runtime.exact_readout import f32_readout

    total = x.reshape(rows, -1).astype(np.int64).sum(axis=1)
    return f32_readout(total, np.array([bits], dtype=np.uint32).view(np.float32)[0], "i8", relu=False)


def _backstop(stream, output, reference, trials=3):
    """Execute the decoded commands on extreme and random inputs; compare with the exact reference."""
    rng = np.random.default_rng(0x9001)
    index = {a.name: i for i, a in enumerate(stream.args)}
    for trial in range(trials):
        buffers, inputs = {}, {}
        for i, arg in enumerate(stream.args):
            dt = np.int32 if arg.dtype == "i32" else np.int8
            if arg.role == "const":
                data = stream.consts[arg.name]
            elif arg.role == "input":
                data = (np.full(arg.shape, (-128, 127)[trial % 2], dt) if trial < 2 else
                        rng.integers(-128, 128, arg.shape).astype(dt))
                inputs[arg.name] = data
            else:
                data = np.full(arg.shape, 0x5A, dt)
            buffers[i] = np.ascontiguousarray(np.asarray(data).astype(dt)).reshape(arg.shape).copy()
        execute(stream, buffers)
        expected = np.asarray(reference(inputs)).reshape(-1)
        require(np.array_equal(buffers[index[output]].reshape(-1).astype(np.int64), expected),
                f"instruction model differs from the exact reference on differential trial {trial}")


# -- admission entry points ---------------------------------------------------------------------

def _prove(stream, meta, match, readout_options):
    """``(device readout, reduction depth or None)`` from source-derived expectations only.

    ``meta`` only selects which proof applies; a wrong selection fails that proof.
    """
    require(match is not None, "pooled admission requires the source match of the emitted stream")
    try:
        if meta.get("composition") == "window_mean":
            expected = mean_expectation(match, readout_options)
            return validate_mean(stream, expected), expected.window
        if meta.get("composition") == "pooled_movement":
            require(readout_options is not None, "standalone pooled store without a selected readout scheme")
            return validate_movement(stream, movement_expectation(match)), None
        expected = fused_expectation(match, readout_options)
        return validate_fused(stream, expected), expected.depth
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ProofError(f"malformed fused-store schedule: {error}") from None


def validate(stream, meta, numeric=None, *, match=None, readout_options=None):
    """Device readout check for a pooled or window-mean kernel; raises ValueError."""
    return _prove(stream, meta, match, readout_options)[0]


def admit(match, numeric_policy, *, tile_dim, readout_options):
    """Numeric admission record of a standalone composition kernel."""
    import hashlib

    from merlin.targetgen.operation_numerics import integer_split_k_limit

    meta, stream = match.kernel.meta, match.kernel.stream
    readout, depth = _prove(stream, meta, match, readout_options)
    strategy = (numeric_policy or {}).get("numerical_semantics", {}).get("readout", {}).get("strategy")
    if readout == "f32_product_rne" and strategy != "f32_product_rne":
        raise ProofError("composition boundary differs from selected numerical policy")
    record = dict(status="proven", schedule=meta, exact_arithmetic=True, device_readout=readout,
                  quantization="device" if readout == "f32_product_rne" else "storage_codes",
                  emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest())
    computes = [i for i in stream.ins if i.kind == "compute"]
    if computes:
        require(depth is not None, "pooled movement contains arithmetic")
        bound = integer_split_k_limit(numeric_policy, tile_dim=tile_dim, full_k=depth)
        require(bound is not None and bound["max_padded_k"] >= tile_dim
                and all(i.a("a_cols") <= tile_dim for i in computes),
                "provider padded mesh K exceeds selected internal MAC bound")
        record["bound"] = bound
    return record
