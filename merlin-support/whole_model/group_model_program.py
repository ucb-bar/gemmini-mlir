#!/usr/bin/env python3
"""A captured model as the program its compute groups ARE, run with the vendor's own kernels.

Group formation says which operations a unit takes whole; the prepack says what numbers each group
is issued with; the stream plan says nothing between two device groups has to visit the host. None
of that is a measurement until a program built from exactly those groups runs. This builds one: a
bare-metal program with ONE library call per compute group, in the model's own order, reading the
folded biases, device-layout weights and readout multipliers the compiler derived, and nothing the
compiler did not derive. The vendor library stands in for the per-group kernel, which is the part a
Phase-1 backend is graded on; everything a whole-model route adds on top of a kernel is what this
measures.

It refuses a model that is not closed: any host region other than the quantization of the model's
input is named and the program is not built, because a number for "the groups" that silently runs
the rest on the host is the defect this work exists to end.

The program checks itself against the capture's own golden (the fake-quantized model's output on
the capture's input): it prints the argmax it got and the one it wanted, and the cosine between
the two logit vectors. It is not held to byte equality, and says why: a residual sum on this unit
rounds each operand, within the bound its group declares.

    group_model_program.py --capture <capture dir> --target <target> --vendor-source <snapshot>/source \\
        --compiler <riscv64-unknown-elf-gcc> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "group_model_program_v1"
CFLAGS = (
    "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
    "-Wa,-march=rv64gc", "-lm", "-lgcc", "-DID_STRING=", "-Wno-incompatible-pointer-types",
    "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1",
)  # fmt: skip

#: What THIS PROGRAM'S generated C needs beyond the target's own harness flags. It hands `const`
#: embedded blobs to library calls typed without `const`, which a recent GCC rejects outright; the
#: target's recipe has no reason to know that, so it is stated here, beside the code that needs it.
PROGRAM_CFLAGS = ("-Wno-incompatible-pointer-types",)


class NotClosed(ValueError):
    """The model has host work between its groups, so "the groups" is not the whole program."""


def _product(extents) -> int:
    out = 1
    for extent in extents:
        out *= int(extent)
    return out


def _through_views(value):
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    owner = getattr(value, "owner", None)
    while owner is not None and mq.op_name(owner) in CG._VIEW_OPS and getattr(owner, "operands", None):
        value = owner.operands[0]
        owner = getattr(value, "owner", None)
    return owner


def capture_sources(capture: Path) -> dict[str, Any]:
    """What :func:`extract` reads, from a capture DIRECTORY laid out the way a recapture writes one.

    A model capsule carries the same four things under its own names (``capsule.yaml`` says which),
    so a caller holding one passes them to :func:`extract` directly instead of copying the capsule
    into a capture's layout. The input and golden may be given as arrays or, as here, as the JSON files
    holding them (a batch whose first sample is the one used), read only once the model is closed.
    """
    return {
        "linalg": capture / "linalg.mlir",
        "weights_manifest": capture / "weights.safetensors.manifest.json",
        "weights": capture / "weights.safetensors",
        "input": capture / "inputs.json",
        "golden": capture / "golden.json",
    }


def _first_sample(given: Any):
    """An array, or the first sample of the JSON batch a path names."""
    import numpy as np

    if isinstance(given, (str, Path)):
        given = json.loads(Path(given).read_text(encoding="utf-8"))[0]
    return np.asarray(given, dtype=np.float32)


def extract(
    capture: Path | None, target: str, *, oracle=None, sources: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The model as an ordered list of device steps over named buffers, with every array they read.

    ``sources`` (see :func:`capture_sources`) names the module, its weights, the float input and the
    golden output; without it they are read from the ``capture`` directory.
    """
    import numpy as np

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.whole_program import input_permutation
    from merlin.runtime.commandbuffer import conv_out_dims
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering import group_numerics as GN
    from merlin.xdsl_dialects.lowering import group_prepack as GP
    from merlin.xdsl_dialects.lowering import stream_plan as SP

    given = dict(sources) if sources is not None else capture_sources(Path(capture))
    text = Path(given["linalg"]).read_text(encoding="utf-8")
    manifest_path, weights_path = Path(given["weights_manifest"]), Path(given["weights"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = CG.form_groups(mq.parse(text), target, oracle=oracle)
    weight_args = SP.weight_args_of(manifest)
    packed = GP.prepack(groups, manifest_path, weights_path, device_layout=True)
    rows = {int(row["group"]): row for row in packed["record"]["groups"]}

    # ONE DEFINITION OF "CLOSED", shared with `llvmlower.whole_program`, which states the same groups
    # as one command buffer. Two spellings would drift, and the second one would silently accept a
    # different set of models -- so a measured cycle count and an emitted buffer refuse together.
    from merlin.xdsl_dialects.lowering import model_closure as MC

    MC.require_closed(groups, error=NotClosed)

    buffers: dict[int, dict[str, Any]] = {}  # id(SSA value) -> buffer
    arrays: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    image = None

    def buffer_of(value, *, why: str) -> str:
        found = buffers.get(id(value))
        if found is None:
            raise NotClosed(f"{why} reads a value no group produced")
        return found["name"]

    def produce(value, name: str, elements: int, ctype: str = "elem_t") -> str:
        buffers[id(value)] = {"name": name, "elements": int(elements), "ctype": ctype}
        return name

    for group in groups:
        tag = f"g{group.index}"
        if group.placement == CG.HOST:
            quantize = group.members[-1]
            scale = GN._scale_source(quantize)
            if image is not None or scale.value is None or scale.zero_point:
                raise NotClosed("the model quantizes more than one input, or not under one static symmetric scale")
            image = {"scale": scale.value, "value": quantize.results[0]}
            shape, _ = mq.type_shape_dtype(quantize.results[0].type)
            produce(quantize.results[0], "IMAGE", _product(shape))
            image["shape"] = [int(v) for v in shape]
            # The layout the device reads the argument in, from the one function the command buffer
            # declares its entry tensor with -- never a literal axis tuple.
            image["permutation"] = input_permutation(quantize, len(shape))
            continue
        sink = group.members[-1].results[0]
        # ONE STATEMENT PER GROUP, whatever the group is. Every accelerator group -- contraction,
        # operand sum, window mean -- is stated once, in the entry vocabulary the capsule corpus is
        # built from, and this harness reads its numbers OFF that entry. The elementwise and mean
        # branches used to re-derive theirs from the capture's own types instead, which is a second
        # statement of a stated fact: the same flattening, spelled twice, in two files that are
        # already required to agree about it.
        try:
            stated = GC.program(group, weight_args=weight_args)
        except CG.NoCapsuleForm as refusal:
            raise NotClosed(f"group {group.index} cannot be stated as a device program: {refusal}") from refusal
        entry = stated.entry
        if group.operand_sum is not None:
            sources = [_through_views(operand) for operand in list(group.root.operands)[:2]]
            rows_out, cols_out = GC.device_output_shape(entry)
            # The factor the readout carries when a multiplier exceeds what a saturating load takes.
            # It is the ONE number of this step the entry does not state, so it is read from the
            # group's own derived sum rather than recomputed from the scales beside it.
            factor = float(group.operand_sum["readout_factor"])
            steps.append(
                {
                    "kind": "sum",
                    "group": group.index,
                    "lhs": buffer_of(sources[0].operands[0], why=f"group {group.index}"),
                    "rhs": buffer_of(sources[1].operands[0], why=f"group {group.index}"),
                    "out": produce(sink, f"B_{tag}", rows_out * cols_out),
                    "rows": rows_out,
                    "cols": cols_out,
                    "lhs_load": float(entry["lhs_scale"]) / factor,
                    "rhs_load": float(entry["rhs_scale"]) / factor,
                    "readout": factor,
                    "relu": "relu" in (entry.get("epilogue") or ()),
                    "bound_lsb": int(entry["bound_lsb"]),
                }
            )
            continue
        if group.window_mean is not None:
            dequantize = next(m for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE)
            # A mean reduces a window per row: the entry states it as a contraction of `K` against a
            # constant column, so the rows it commits are its device output shape and the window is
            # its reduction depth. Both were spelled a second time from `group.window_mean`.
            rows_kept, _one = GC.device_output_shape(entry)
            steps.append(
                {
                    "kind": "mean",
                    "group": group.index,
                    "in": buffer_of(dequantize.operands[0], why=f"group {group.index}"),
                    "out": produce(sink, f"B_{tag}", rows_kept),
                    "rows": rows_kept,
                    "window": int(entry["K"]),
                    "multiplier": float(entry["acc_scale"]),
                }
            )
            continue

        if stated.stored_operand is None:
            # A contraction of two activations (attention's scores and context): this closed harness
            # reads a contraction as activation x stored weight and states nothing else.
            raise NotClosed(f"group {group.index} contracts two activations, which this closed harness does not state")
        activation = list(group.root.operands)[1 - int(stated.stored_operand)]
        _adapters, dequantize, _dtype = CG._input_chain(activation)
        step: dict[str, Any] = {
            "kind": entry["op"],
            "group": group.index,
            "in": buffer_of(dequantize.operands[0], why=f"group {group.index}"),
            "relu": "relu" in entry["epilogue"],
            "bias": None,
        }
        row = rows.get(group.index)
        # A ROW IS NO LONGER PROOF THAT THE GROUP CLOSES. The prepack states a device weight for a
        # group that leaves as the accumulator too -- the layout is a property of the unit, not of
        # how the readout ends -- and such a row carries no multiplier, clamp or folded bias. Which
        # branch this takes is read from what the row SAYS it is, never from whether one exists.
        if row is not None and row.get("leaves_as") == "accumulator":
            row = None
        if row is not None:  # a closed group: the prepack already holds its numbers
            step["scale"] = float(row["multiplier"])
            weight = packed["arrays"][row["weight"]["array"]]
            if "bias" in row:
                step["bias"] = f"BIAS_{tag}"
                arrays[step["bias"]] = packed["arrays"][row["bias"]["array"]].astype(np.int32)
            out_ctype = "elem_t"
        else:
            # Not closed: the contraction's result leaves as the accumulator (a model's final
            # classifier). Its bias is folded by the same rule a closed group's is.
            if "acc_scale" in entry["epilogue"] or CG.QUANTIZE in group.stages:
                raise NotClosed(f"group {group.index} is closed and the prepack holds no row for it")
            sources = [GN._scale_source(m) for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE]
            if len(sources) != 2 or any(s.value is None or s.zero_point for s in sources):
                raise NotClosed(f"group {group.index} leaves as an accumulator whose scales are not static")
            divisor = sources[0].value * sources[1].value
            name, stored, _spelled = GP._stored_raw(stated.stored_arg, manifest, weights_path)
            weight = GP.device_weight(group, stated, stored)
            if stated.bias_arg is not None:
                _bias_name, bias = GP._stored_tensor(stated.bias_arg, manifest, weights_path)
                step["bias"] = f"BIAS_{tag}"
                arrays[step["bias"]] = np.rint(bias / divisor).astype(np.int32)
            step.update({"scale": None, "dequantize": divisor})
            out_ctype = "acc_t"
        if entry["op"] == "conv2d":
            if entry["Himg"] != entry["Wimg"] or entry["kh"] != entry["kw"] or len(set(entry["stride"])) != 1:
                raise NotClosed(f"group {group.index} is not a square convolution, which is all the library states")
            pad = entry.get("padding") or [0, 0, 0, 0]
            if len(set(pad)) != 1:
                raise NotClosed(f"group {group.index} pads asymmetrically ({pad})")
            # The ABI's own output-extent arithmetic, not a restatement of it.
            out_dim, _out_w = conv_out_dims(
                int(entry["Himg"]),
                int(entry["Wimg"]),
                int(entry["kh"]),
                int(entry["kw"]),
                entry["stride"],
                pad,
                entry.get("dilation") or [1, 1],
            )
            # ONE ARITHMETIC FOR THE COMMITTED EXTENT, shared with `llvmlower.whole_program`, which
            # declares its buffers with it: `group_command.device_output_shape`. The pooled extent
            # used to be recomputed here, and a second spelling of it would size a buffer nothing
            # would attribute back to the drift.
            pool = {"size": 0, "stride": 0, "padding": 0}
            if "maxpool" in entry["epilogue"]:
                pool = {
                    "size": int(entry["pool_size"][0]),
                    "stride": int(entry["pool_stride"][0]),
                    "padding": int(entry["pool_padding"][0]),
                }
            # The prepack's device layout IS the library's: [tap_h, tap_w, channel] by output, which
            # is how its convolution indexes a weight (`(krow*k*ci + kcol*ci + kch) * n + och`).
            kh, ci, n = int(entry["kh"]), int(entry["ci"]), int(entry["N"])
            weight = np.ascontiguousarray(weight.reshape(kh, kh, ci, n))
            step.update(
                {
                    "in_dim": int(entry["Himg"]),
                    "ci": ci,
                    "n": n,
                    "out_dim": int(out_dim),
                    "stride": int(entry["stride"][0]),
                    "padding": int(pad[0]),
                    "kernel": kh,
                    "pool": pool,
                }
            )
            elements = _product(GC.device_output_shape(entry))
        else:
            step.update({"m": int(entry["M"]), "k": int(entry["K"]), "n": int(entry["N"])})
            elements = _product(GC.device_output_shape(entry))
        step["weight"] = f"W_{tag}"
        arrays[step["weight"]] = np.ascontiguousarray(weight).astype(np.int8)
        step["out"] = produce(sink, f"B_{tag}", elements, out_ctype)
        steps.append(step)

    if image is None or not steps:
        raise NotClosed("the capture has no quantized input or no device group")
    final = steps[-1]
    if final.get("dequantize") is None:
        raise NotClosed("the model's last group does not leave as an accumulator to dequantize; nothing to compare")

    # The model's input, quantized the way its one host region does and laid out as the library
    # reads it (positions by channel).
    pixels = _first_sample(given["input"]).reshape(image["shape"])
    quantized = np.clip(np.rint(pixels / np.float32(image["scale"])), -128, 127).astype(np.int8)
    arrays["IMAGE_DATA"] = np.ascontiguousarray(quantized.transpose(image["permutation"]))
    golden = _first_sample(given["golden"])
    arrays["GOLDEN"] = golden.reshape(-1)
    return {
        "steps": steps,
        "buffers": [b for b in buffers.values() if b["name"] != "IMAGE"],
        "image_elements": int(arrays["IMAGE_DATA"].size),
        "arrays": arrays,
        "groups": len(groups),
        "device_groups": len(steps),
        "classes": int(golden.size),
    }


def emulate(model: dict[str, Any], *, single_rounding_sums: bool = False) -> dict[str, Any]:
    """The same steps in numpy, with the arithmetic each library call is documented to do.

    It answers one question before any simulator runs: is the INTEGER PROGRAM the model? A wrong
    weight layout, bias rule or multiplier shows here, against the capture's golden, in seconds and
    with every intermediate tensor in hand. Returns the named buffers and the final comparison.

    ``single_rounding_sums`` computes every sum the way the capture does (scale both operands, add,
    round once), so the difference between the two emulations is what the unit's per-operand
    rounding costs THIS model on THIS input, separated from everything else.
    """
    import numpy as np

    def readout(acc, scale, relu):
        out = np.rint(acc.astype(np.float32) * np.float32(scale)) if scale is not None else acc
        out = np.clip(out, -128, 127) if scale is not None else out
        return np.maximum(out, 0) if relu else out

    arrays = model["arrays"]
    values: dict[str, Any] = {"IMAGE": arrays["IMAGE_DATA"].astype(np.int64)}
    # A fused region is emulated as its members: the integer program is the same model either way.
    for step in flat_steps(model):
        if step["kind"] == "conv2d":
            x = values[step["in"]].reshape(step["in_dim"], step["in_dim"], step["ci"])
            pad, k, stride = step["padding"], step["kernel"], step["stride"]
            x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(0, 1))[::stride, ::stride]
            w = arrays[step["weight"]].astype(np.int64)  # [kh, kw, ci, n]
            acc = np.einsum("hwcij,ijcn->hwn", windows, w)
            if step["bias"]:
                acc = acc + arrays[step["bias"]].astype(np.int64)
            out = readout(acc, step["scale"], step["relu"])
            pool = step["pool"]
            if pool["size"]:
                out = np.pad(out, ((pool["padding"],) * 2, (pool["padding"],) * 2, (0, 0)), constant_values=-128)
                out = np.lib.stride_tricks.sliding_window_view(out, (pool["size"],) * 2, axis=(0, 1))
                out = out[:: pool["stride"], :: pool["stride"]].max(axis=(-1, -2))
            values[step["out"]] = out.reshape(-1).astype(np.int64)
        elif step["kind"] == "matmul":
            a = values[step["in"]].reshape(step["m"], step["k"])
            acc = a @ arrays[step["weight"]].astype(np.int64).reshape(step["k"], step["n"])
            if step["bias"]:
                acc = acc + arrays[step["bias"]].astype(np.int64)
            values[step["out"]] = readout(acc, step["scale"], step["relu"]).reshape(-1).astype(np.int64)
        elif step["kind"] == "sum" and single_rounding_sums:
            total = sum(
                values[step[side]].astype(np.float32) * np.float32(step[f"{side}_load"] * step["readout"])
                for side in ("lhs", "rhs")
            )
            values[step["out"]] = readout(total, 1.0, step["relu"]).astype(np.int64)
        elif step["kind"] == "sum":
            loads = [
                np.clip(np.rint(values[step[side]].astype(np.float32) * np.float32(step[f"{side}_load"])), -128, 127)
                for side in ("lhs", "rhs")
            ]
            values[step["out"]] = readout(loads[0] + loads[1], step["readout"], step["relu"]).astype(np.int64)
        else:
            a = values[step["in"]].reshape(step["window"], step["rows"])
            values[step["out"]] = readout(a.sum(axis=0), step["multiplier"], False).astype(np.int64)
    final = model["steps"][-1]
    logits = values[final["out"]].astype(np.float64) * float(final["dequantize"])
    golden = arrays["GOLDEN"].astype(np.float64)
    cosine = float(logits @ golden / (np.linalg.norm(logits) * np.linalg.norm(golden) + 1e-30))
    return {"values": values, "argmax": int(logits.argmax()), "want": int(golden.argmax()), "cosine": cosine}


#: The digest constants are the hardened per-layer route's own, imported rather than restated: the
#: gemmini layer bench digests every window's output with ``lb_fnv1a_words`` built from these, and a
#: second copy of an FNV offset here is a second place the two routes could quietly disagree about
#: what "the same digest" means.
def _digest_constants():
    from merlin.perf.layer_bench import reference as _ref

    return _ref.FNV_OFFSET, _ref.FNV_PRIME, _ref.DIGEST_MASK


def group_digest(values) -> int:
    """The ORDER-SENSITIVE digest of one group's output, computed the way the device computes it.

    FNV-1a over the output's elements, each sign-extended to a little-endian 64-bit word, top bit
    cleared so it prints and stores as a signed integer.  Widening to 64 bits is what makes this
    reproducible without knowing how wide ``elem_t`` and ``acc_t`` are on the target: the C helper
    widens each element the same way, so one function checks an int8 output and an accumulator
    output without either side holding a width as a constant.

    It replaces an additive sum, which could not see the failure it most needed to: reorder a
    group's output -- a transpose, a stride, a tiling bug -- and the sum is unchanged, as it is for
    any pair of compensating +k/-k errors.  Every whole-model row measured before this change was
    admitted on that sum.
    """
    import numpy as np

    from merlin.perf.layer_bench import reference as _ref

    _offset, _prime, mask = _digest_constants()
    return _ref.fnv1a64_words(np.ascontiguousarray(values, dtype="<i8").tobytes()) & mask


def group_checksums(model: dict[str, Any], emulation: dict[str, Any]) -> dict[str, dict[str, int]]:
    """``{group: {"sum": ..., "fnv1a": ...}}`` for every device group, from the numpy emulation.

    THE ORACLE A v2 POLICY DECLARES.  Both numbers are produced here so a policy generator and this
    program cannot drift into computing them differently; only ``fnv1a`` is order-sensitive, and
    only ``fnv1a`` should be the ``value_key`` of a policy a cycle claim is sealed against.

    The emulated buffer must hold exactly as many elements as the program declares for it, or the
    two are not digesting the same thing and the result would be an incomparable number rather than
    a wrong one.  That is checked, not assumed.
    """
    sizes = {buffer["name"]: int(buffer["elements"]) for buffer in model["buffers"]}
    values = emulation["values"]
    rows: dict[str, dict[str, int]] = {}
    for step in model["steps"]:
        name = step["out"]
        emulated = values[name]
        if int(emulated.size) != sizes[name]:
            raise ValueError(
                f"group {step['group']} emulates {emulated.size} elements of {name!r} and the "
                f"program declares {sizes[name]}; these are not the same buffer"
            )
        rows[str(step["group"])] = {"sum": int(emulated.sum()), "fnv1a": group_digest(emulated)}
    return rows


#: The library's own dataflow argument for a call it answers. ``WS`` drives the hardware loop unit;
#: when loop-descriptor instructions are prohibited, a library group needs a path that issues none.
LIBRARY_DEFAULT = "WS"
LIBRARY_HOST = "CPU"
LIBRARY_OUTPUT_STATIONARY = "OS"


def library_path_without_loops(header_text: str) -> dict[str, Any]:
    """The library path that issues no loop-descriptor instruction, per op kind, with why.

    The library's convolution has no such accelerator path (its output-stationary convolution is not
    implemented), so a convolution runs on the host. Its matmul has one, output-stationary, but only a
    board built with that dataflow runs it; that capability is read from the parameter header the
    program compiles against, and when the header does not state it the matmul FAILS CLOSED to the
    host as well."""
    stated = None
    for line in header_text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "#define" and fields[1] == "GEMMINI_OS_DATAFLOW":
            stated = fields[2]
    os_ok = stated not in (None, "0", "false")
    return {
        "conv2d": {
            "path": LIBRARY_HOST,
            "host": True,
            "why": "loop-descriptor instructions are prohibited and the library's convolution has no "
            "loop-free accelerator path (its output-stationary convolution is not implemented): the "
            "library's host convolution",
        },
        "matmul": {
            "path": LIBRARY_OUTPUT_STATIONARY if os_ok else LIBRARY_HOST,
            "host": not os_ok,
            "why": "loop-descriptor instructions are prohibited; output-stationary dataflow is stated by the "
            "parameter header"
            if os_ok
            else "loop-descriptor instructions are prohibited and the parameter header states no "
            "output-stationary dataflow, so the library's matmul fails closed to its host code",
        },
        # THE RESIDUAL ADD TOO. The library's accelerated residual add is a loop-descriptor stream (one
        # loop per tile, the adds flagged), so with the role prohibited its default call is trapped --
        # and a trap is `exit`, which the compiler treats as the end of the program: every group after
        # the first declined residual add was dead code, its output buffer was dropped from the linked
        # program, and the build failed naming a missing symbol. Its host code (`resadd_cpu`) is
        # loop-free and computes the same per-operand load scale, sum and readout scale.
        "sum": dict(_SUM_ON_HOST),
        "residual_add": dict(_SUM_ON_HOST),
    }


_SUM_ON_HOST = {
    "path": LIBRARY_HOST,
    "host": True,
    "why": "loop-descriptor instructions are prohibited and the library's residual add has no loop-free "
    "accelerator path (it is a loop-descriptor stream): the library's host residual add",
}


def _identifiers(text: str) -> set[str]:
    """Every C identifier in ``text``, tokenized by character class (no pattern matching)."""
    found, current = set(), []
    for character in text + " ":
        if character.isalnum() or character == "_":
            current.append(character)
        else:
            if current and not current[0].isdigit():
                found.add("".join(current))
            current = []
    return found


def loop_free_header(text: str, prohibited_selectors: Sequence[int]) -> tuple[str, list[str]]:
    """The library header with every macro that would issue a prohibited instruction replaced by a
    trap, and the names of those macros.

    The header names each instruction's selector (``#define k_<NAME> <selector>``); the prohibited
    selectors come from the target's ISA facts, so which constants are prohibited is data. A macro
    whose body uses one keeps its parameter list and becomes a call that stops the program with a
    message: nothing in the linked program can issue the instruction, and a path that would have is
    a failed run, never a silent substitution."""
    prohibited = {int(s) for s in prohibited_selectors}
    constants = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "#define" and "(" not in fields[1]:
            try:
                value = int(fields[2], 0)
            except ValueError:
                continue
            if value in prohibited:
                constants.add(fields[1])
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    trapped: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "#define" and "(" in fields[1]:
            block = [line]
            while block[-1].rstrip("\n").endswith("\\") and index + 1 < len(lines):
                index += 1
                block.append(lines[index])
            body = "".join(block)
            name = fields[1].split("(", 1)[0]
            if name not in constants and _identifiers(body.split(")", 1)[1] if ")" in body else "") & constants:
                head = body[: body.index(")") + 1]
                out.append(
                    f'{head} do {{ printf("merlin: {name} issues a prohibited instruction\\n"); exit(1); }} while (0)\n'
                )
                trapped.append(name)
            else:
                out.extend(block)
        else:
            out.append(line)
        index += 1
    return "".join(out), trapped


def library_host_full_width(step: Mapping[str, Any], library: Mapping[str, Any] | None) -> bool:
    """Whether ``step`` is a full-width (raw accumulator) matmul the library would answer on the HOST.

    The library's host matmul does not implement a full-width result: ``tiled_matmul`` itself says so
    (``Not implemented: CPU matmul, full_C=1``, printed only under its debug checks) and then calls
    ``matmul_cpu``, whose output is ``elem_t *`` -- every int32 result is scaled, saturated to the
    element range and written as bytes into a buffer read back as int32. Measured on ResNet-50's
    classifier (g71) on a full-width machine: 1000 of 1000 elements wrong, argmax 199 against 21. Such a
    group is computed with this driver's own full-width routine (``hr_acc_matmul``, the one a machine
    without the readout already uses), never with the library's."""
    if step.get("kind") != "matmul" or step.get("scale", 0) is not None:
        return False
    return bool(((library or {}).get("matmul") or {}).get("host"))


def _call(step: dict[str, Any], library: dict[str, Any] | None = None) -> str:
    if step["kind"] == "region":
        # The library has no fused call; a region's fallback is its members' own calls, in order.
        return " ".join(_call(member, library) for member in step["members"])
    if step["kind"] == "fused":
        # The library has no merged call; its fallback is the pair the merge replaced, in order.
        return f"{_call(step['producer'], library)} {_call(step['sum'], library)}"
    if library_host_full_width(step, library):
        return _host_call(step)
    kind_path = (library or {}).get(step["kind"] if step["kind"] in ("conv2d", "sum") else "matmul") or {}
    dataflow = kind_path.get("path") or LIBRARY_DEFAULT
    act = "RELU" if step.get("relu") else "NO_ACTIVATION"
    bias = step.get("bias") or "NULL"
    if step["kind"] == "conv2d":
        pool = step["pool"]
        if dataflow == LIBRARY_HOST:
            # THE LIBRARY'S OWN "CPU" PATH IS NOT LOOP-FREE. Measured: `tiled_conv_auto(..., CPU)`
            # still reaches `gemmini_loop_conv_ws` for some shapes, which this repo's no-FSM header
            # traps into a hard abort -- so asking the vendor for its loop-free convolution can abort
            # the run even though one was asked for. `hr_conv2d` is the program's own plain-C answer
            # (see `_HOST_CONV2D`): it never calls into gemmini.h at all, so it cannot reach a
            # prohibited macro regardless of what the vendor's CPU dataflow does internally.
            scaled = 0 if step.get("scale") is None else 1
            scale = step.get("scale") or 1.0
            return (
                f"hr_conv2d({step['in']}, (const elem_t *){step['weight']}, {bias}, {step['out']}, "
                f"{step['in_dim']}, {step['ci']}, {step['n']}, {step['out_dim']}, {step['stride']}, "
                f"{step['padding']}, {step['kernel']}, {scaled}, {scale!r}f, {1 if step.get('relu') else 0}, "
                f"{pool['size']}, {pool['stride']}, {pool['padding']});"
            )
        return (
            f"tiled_conv_auto(1, {step['in_dim']}, {step['in_dim']}, {step['ci']}, {step['n']}, {step['out_dim']}, "
            f"{step['out_dim']}, {step['stride']}, 1, 1, {step['padding']}, {step['kernel']}, false, false, false, "
            f"false, false, {step['in']}, {step['weight']}, {bias}, {step['out']}, {act}, {step['scale']!r}f, "
            f"{pool['size']}, {pool['stride']}, {pool['padding']}, {dataflow});"
        )
    if step["kind"] == "matmul":
        full = step["scale"] is None
        scale = "ACC_SCALE_IDENTITY" if full else f"{step['scale']!r}f"
        return (
            f"tiled_matmul_auto({step['m']}, {step['n']}, {step['k']}, {step['in']}, {step['weight']}, {bias}, "
            f"{step['out']}, {step['k']}, {step['n']}, {step['n']}, {step['n']}, MVIN_SCALE_IDENTITY, "
            f"MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, {act}, {scale}, 0, true, false, false, "
            f"{'true' if full else 'false'}, false, 0, {dataflow});"
        )
    if step["kind"] == "sum":
        return (
            f"tiled_resadd_auto({step['rows']}, {step['cols']}, {step['lhs_load']!r}f, {step['rhs_load']!r}f, "
            f"{step['readout']!r}f, {step['lhs']}, {step['rhs']}, {step['out']}, "
            f"{'true' if step['relu'] else 'false'}, {dataflow});"
        )
    # A mean over a trailing window: the buffer holds [window positions] by [rows channels], so the
    # contraction reads it transposed, against a constant one.
    return (
        f"tiled_matmul_auto({step['rows']}, 1, {step['window']}, {step['in']}, ONES, NULL, {step['out']}, "
        f"{step['rows']}, 1, 1, 1, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, "
        f"NO_ACTIVATION, {step['multiplier']!r}f, 0, false, true, false, false, false, 0, {dataflow});"
    )


#: The label of the one measured window this program publishes.  It is a DEFAULT, not a constant:
#: the label is what ties a cycle number to the policy entry that declares what the window must
#: print, so a build measuring something else passes its own.
DEFAULT_WINDOW_LABEL = "group_model"

#: EVERY LINE THIS PROGRAM PUBLISHES, ONCE.  The C ``printf`` formats below are built from these,
#: and so is :func:`uart_lines`, which renders the UART a run would produce.  One spelling with two
#: renderers is the point: a test that hand-wrote the expected UART would be grading the harness
#: against the same guess that wrote it, which is how this file came to print a metric line nothing
#: could parse and no test noticed for as long as it did.
UART = {
    "invocations": "MERLIN_INVOCATIONS warmup=1 measured=1",
    "window_begin": "MERLIN_WINDOW begin label={label}",
    "warm_begin": "MERLIN_PROFILE warmup begin",
    "warm_end": "MERLIN_PROFILE warmup end rc=0",
    "measured_begin": "MERLIN_PROFILE measured begin",
    # TWO NUMBERS, AND ONLY ONE OF THEM IS A CHECK.  ``sum`` is the additive total this program has
    # always printed; it is kept because it is the only quantity the seven FPGA runs on record also
    # published, so a new build can still be shown to agree with them.  It is NOT what a cycle claim
    # is admitted against: an additive sum is permutation-blind, so a transpose, a stride or a
    # layout bug that REORDERS a group's output leaves it byte-identical, as does any pair of
    # compensating +k/-k errors -- and reordering is a first-class failure mode of a tensor compiler,
    # which is exactly what this program measures.  ``fnv1a`` is the order-sensitive digest a v2
    # policy declares (``value_key: "fnv1a"``); it is the same FNV-1a the hardened per-layer route
    # digests its outputs with (``merlin.perf.layer_bench.reference``), so both routes are checked by
    # one function rather than by two that happen to agree.
    "group": "GM_GROUP {group} {kind} {cycles} sum={sum} fnv1a={checksum}",
    "argmax": "GM_ARGMAX got={got} want={want} agrees={agrees}",
    "cosine": "GM_COSINE_PPM {ppm}",
    "metric": "METRIC cycles {cycles}",
    # THE QUANTITY A TOLERANCE-GRADED OP IS ACTUALLY GRADED ON. `sum` is permutation-blind and
    # `fnv1a` is exact, so NEITHER can answer "is every element within +/- bound_lsb?" -- which is
    # what `numeric_policy.compare: bounded_int` asks of a residual add. The op's own contract says
    # the reference rounds ONCE while a scaled rounding load rounds each operand, so the two are
    # different functions by design and an exact digest must disagree. Measured at g6's scales:
    # 23.8% of elements differ, by exactly one output step, against a declared bound of 2.
    "bounded": "GM_BOUND {group} max_abs={max_abs} over={over} bound={bound}",
    # THE WORST ELEMENT, WITH ITS OWN OPERANDS. A magnitude alone cannot say WHY a group is out of
    # bound: measured on g19, neither single-rounding nor per-operand saturation reproduces a
    # 60-step deviation at its scales, so the remaining hypotheses are all about what the device
    # actually read and wrote. This reports the one element that deviates most, with the two inputs
    # the device had and the output it produced, so the arithmetic can be redone by hand instead of
    # guessed at.
    "witness": "GM_WITNESS {group} i={index} lhs={lhs} rhs={rhs} device={device} reference={reference}",
    # WHERE the out-of-bound elements are, not just how far out. One index cannot distinguish a
    # skipped tile from a DMA edge from a data-dependent fault; a handful of them, with the row and
    # column they fall on, can. Measured on the vendor arm: the two failing groups' worst elements
    # both land in columns 481 and 495 of 512, which is a tile-boundary signature rather than a
    # scattered one -- and 99.98% of each buffer is within bound, so this is a small set of
    # positions, not a dropped operand.
    "where": "GM_WHERE {group} cols={cols} first={first}",
    # THE LOCAL CHECK OF AN EXACT GROUP. The group's reference recomputed ON THE CORE, after the window,
    # from the inputs the device ACTUALLY held -- so a group is judged on its own arithmetic, and a
    # legitimate bounded difference upstream (a residual sum that rounds each operand) does not turn
    # every exact group below it into a digest mismatch. `mismatches` counts output elements that
    # differ from that reference; `first` is the first such element's flat index, -1 when none.
    "local": "GM_LOCAL {group} mismatches={mismatches} of={elements} first={first}",
    # WHERE A GROUP IS WRONG (``local_map`` builds only). Per output axis, the count of wrong elements at
    # each index that has any (``index:count,...``, or ``none``); the first few wrong elements with their
    # coordinates and both values; and how many are above and below the reference. The axes are the
    # program's own output layout: a convolution's row, column and output channel, a contraction's row
    # and output channel, a mean's channel.
    "local_map": "GM_LOCAL_MAP {group} axis={axis} extent={extent} wrong={wrong}",
    "local_sample": "GM_LOCAL_SAMPLE {group} index={index} row={row} col={col} channel={channel} got={got} want={want}",
    "local_sign": "GM_LOCAL_SIGN {group} over={over} under={under} max_abs={max_abs}",
    # ONE CHEAP DIGEST PER GROUP, over the output buffer's bytes read as 64-bit words (one pass, a few
    # cycles per word). It decides nothing on its own: it is how two builds of ONE program -- a timing
    # build on a board and a locally graded build on a functional model -- are shown to have written the
    # same bytes, group by group, without paying for the grade on the board.
    "words": "GM_WORDS {group} bytes={bytes} digest={digest}",
    # THE END RESULT OF A MODEL GRADED ON AN OUTPUT TENSOR rather than a class (the open-model program,
    # group_model_dispatch): elements within the capsule's numeric policy of the oracle's, of how many.
    "output": "GM_OUTPUT within={within} of={of}",
    # THE WHOLE OUTPUT TENSOR'S BYTES, digested after the window: what an end result is judged on when
    # the reference is the model's own host code with exact devices (a bit-identical output).
    "output_digest": "GM_OUTPUT_DIGEST bytes={bytes} digest={digest}",
    # THE THREE THE PUBLISHED BAND IS QUOTED IN. The reference artifact (`resnet50_autocomp.c`,
    # beside the `d6db18f8` `gemmini_params.h` this repo pins) prints a WHOLE-WINDOW `rdcycle`
    # delta, the additive sum of its per-layer brackets, and the difference. The published band is
    # the FIRST of those; this program has only ever published the SECOND, so the two were being
    # compared across different scopes. All three now leave here under their own names, and
    # `uncounted` is what the brackets never saw -- inter-group setup, buffer handling, control
    # flow and the runtime dequantize.
    "full_model": "FM full model cycles: {cycles}",
    # THE SPLIT, NEVER ONE FUSED NUMBER. Two arms differing by a single total invite the reader to
    # attribute all of the difference to the kernels. `im2col` is the operand materialization an
    # authored kernel required and a library call did not; it is part of `authored`, and the loop
    # that performs it is this harness's, so a better one would move this line.
    "split": "FM split authored={authored} groups={authored_groups} | "
    "im2col={im2col} groups={im2col_groups} (inside authored; harness code, a floor not a verdict) | "
    "vendor={vendor} groups={vendor_groups}",
    "bracket_sum": "FM bracketed compute only sum: {cycles}",
    "uncounted": "FM uncounted delta: {cycles}",
    "measured_end": "MERLIN_PROFILE measured end rc=0",
    "window_end": "MERLIN_WINDOW end label={label}",
}


def _printf(key: str, **conversions: str) -> str:
    """One C ``printf`` of a protocol line, with each field replaced by its conversion."""
    return UART[key].format(**conversions) if conversions else UART[key]


def program_steps(model: dict[str, Any]) -> list[tuple[str, str]]:
    """``(group, kind)`` per device group, in the order the program calls them."""
    return [(str(step["group"]), str(step["kind"])) for step in model["steps"]]


def uart_lines(
    steps: Sequence[tuple[str, str]],
    *,
    cycles: int,
    checksums: Mapping[str, int],
    argmax: int,
    want: int,
    cosine_ppm: int,
    digests: Mapping[str, int] | None = None,
    group_cycles: Mapping[str, int] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
) -> list[str]:
    """The UART this program prints, given what the device computed.  No hardware, no C, no guess.

    ``checksums`` is what each group's output SUMMED TO on the device -- the quantity a v2 policy
    declares an oracle for.  Passing an oracle set renders a correct run's UART; changing one entry
    renders the job-730 shape, a run whose argmax marker is byte-identical to a correct one's and
    whose arithmetic was wrong.  Both are UARTs this program could really emit, which is what makes
    the pair a test of the admission rule rather than of the fixture.

    ``steps`` is ``(group, kind)`` pairs -- :func:`program_steps` derives them from a model, and a
    validation policy's declared groups are the other source, so the UART a policy expects can be
    rendered without the capture that produced the policy.

    ``checksums`` is the ADDITIVE sum; ``digests`` the order-sensitive FNV-1a a cycle claim is
    admitted against (:func:`group_checksums` computes both).  A group with no digest offered
    renders ``fnv1a=UNKNOWN`` rather than being left off the line: the runs on record predate the
    digest and can only be re-framed with the numbers they published, and a rendering that quietly
    dropped the field would let a policy declaring ``fnv1a`` read a missing check as a satisfied
    one.  ``UNKNOWN`` is refused by the admission rule as a value that is not an integer, which is
    the fail-closed reading of "this run cannot answer that question".
    """
    order = [group for group, _kind in steps]
    missing = sorted(set(order) - set(checksums))
    if missing:
        raise ValueError(f"no checksum offered for group(s) {missing}; a partial UART is not one this program prints")
    timings = group_cycles or {}
    lines = [
        UART["invocations"],
        UART["window_begin"].format(label=window_label),
        UART["warm_begin"],
        UART["warm_end"],
        UART["measured_begin"],
    ]
    offered = dict(digests or {})
    lines.extend(
        UART["group"].format(
            group=group,
            kind=kind,
            cycles=int(timings.get(group, 1)),
            sum=int(checksums[group]),
            checksum=int(offered[group]) if group in offered else "UNKNOWN",
        )
        for group, kind in steps
    )
    lines.extend(
        (
            UART["argmax"].format(got=int(argmax), want=int(want), agrees=int(argmax == want)),
            UART["cosine"].format(ppm=int(cosine_ppm)),
            UART["metric"].format(cycles=int(cycles)),
            UART["measured_end"],
            UART["window_end"].format(label=window_label),
        )
    )
    return lines


#: VERIFY MODES. ``local`` is ``on_target`` plus one GM_LOCAL line per exact group (conv2d, matmul,
#: mean): the group's reference computed on the core from its actual inputs. It is the gate a run is
#: admitted on; the chained digests stay as information.
VERIFY_MODES = ("on_target", "host_dump", "local", "words", "local_map")

#: ``words`` is the TIMING build's check: no byte-wise digest pass and no local recomputation, only one
#: GM_WORDS line per group plus the elementwise GM_BOUND checks of the tolerance groups (which catch an
#: operand-ordering fault on the board itself). ``local`` prints GM_WORDS too, so a ``words`` build and a
#: ``local`` build of the same program can be held to writing the same bytes.
WORDS_MODES = ("local", "words", "local_map")

#: ``local_map`` is ``local`` that also says WHERE each exact group is wrong (the GM_LOCAL_MAP, _SAMPLE
#: and _SIGN lines): a diagnostic build for one group's program, never a timing build.
LOCAL_MODES = ("local", "local_map")

#: How many wrong elements a ``local_map`` build reports with their coordinates and values.
_LOCAL_MAP_SAMPLES = 8

#: The verification code is fenced by these two markers wherever it appears, so two builds of one
#: program can be shown to differ ONLY inside them (everything outside is what the window runs).
VERIFY_BEGIN = "/* MERLIN_VERIFY_BEGIN */"
VERIFY_END = "/* MERLIN_VERIFY_END */"

_WORDS_HELPER = r"""
static unsigned long long words_digest(const void *p, size_t bytes) {
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 14695981039346656037ULL;
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        unsigned long long w;
        __builtin_memcpy(&w, b + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
    }
    unsigned long long tail = 0;
    for (size_t k = 0; i + k < bytes; k++) tail |= (unsigned long long)b[i + k] << (8 * k);
    h ^= tail ^ (unsigned long long)bytes;
    h *= 1099511628211ULL;
    return h;
}
"""


def program_without_verification(text: str) -> str:
    """The program's C with every fenced verification region removed: what the window compiles from."""
    out, rest = [], text
    while VERIFY_BEGIN in rest:
        head, _, tail = rest.partition(VERIFY_BEGIN)
        out.append(head)
        if VERIFY_END not in tail:
            raise ValueError("a verification region opens and never closes")
        rest = tail.partition(VERIFY_END)[2]
    out.append(rest)
    return "".join(out)


#: Width of the output-channel block the local reference keeps in registers. A tuning constant of this
#: C, not a fact of any target: it only changes how fast the reference runs, never what it computes.
_LOCAL_BLOCK = 8

_LOCAL_HELPERS = r"""
/* LOCAL REFERENCES, OUTSIDE THE WINDOW. Scalar C, independent of the accelerator and of every kernel
   under test, with the arithmetic `emulate()` documents for each step: int32 accumulation, the bias
   added to the accumulator, a float32 product with the readout multiplier rounded half-to-even,
   saturation to the element range, then the activation. */
static long long lc_rne(float p) {
    long long i = (long long)p;
    long long nxt = p < 0 ? i - 1 : i + 1;
    float rem = p - (float)i;
    if (rem < 0) rem = -rem;
    if (rem < 0.5f) return i;
    if (rem > 0.5f) return nxt;
    return (i % 2 == 0) ? i : nxt;
}
static long long lc_readout(long long acc, int scaled, float scale, int relu) {
    if (!scaled) return acc;
    long long q = lc_rne((float)acc * scale);
    if (q > elem_t_max) q = elem_t_max;
    if (q < elem_t_min) q = elem_t_min;
    if (relu && q < 0) q = 0;
    return q;
}
typedef struct { unsigned long long mismatches; long long first; } lc_verdict;
static void lc_note(lc_verdict *v, long long got, long long want, long long index) {
    if (got != want) { if (!v->mismatches) v->first = index; v->mismatches++; }
}
static int32_t lc_acc[LC_MAX_N];
/* One output row (pixel) of a contraction: acc[n] = bias[n] + sum_t x_t * w_t[n], where each tap t is
   a contiguous input vector of `depth` elements against `depth` weight rows of pitch `n`. */
static void lc_row(const elem_t *const *xs, const elem_t *const *ws, int taps, int depth, int n, const acc_t *bias) {
    for (int nb = 0; nb < n; nb += LC_BLOCK) {
        int width = n - nb < LC_BLOCK ? n - nb : LC_BLOCK;
        int32_t a[LC_BLOCK];
        for (int j = 0; j < LC_BLOCK; j++) a[j] = (bias && j < width) ? (int32_t)bias[nb + j] : 0;
        for (int t = 0; t < taps; t++) {
            const elem_t *x = xs[t];
            const elem_t *w = ws[t] + nb;
            for (int c = 0; c < depth; c++, w += n) {
                int32_t xv = x[c];
                if (!xv) continue;
                if (width == LC_BLOCK) {
                    for (int j = 0; j < LC_BLOCK; j++) a[j] += xv * (int32_t)w[j];
                } else {
                    for (int j = 0; j < width; j++) a[j] += xv * (int32_t)w[j];
                }
            }
        }
        for (int j = 0; j < width; j++) lc_acc[nb + j] = a[j];
    }
}
"""


def _bounded_block(step: Mapping[str, Any], sizes: Mapping[str, int]) -> str:
    """The on-core check of one elementwise sum: its reference recomputed from the two operands it read."""
    return f"""    {{
        long long worst = 0, over = 0, wi = -1, wl = 0, wr = 0, wo = 0, wref = 0;
        long long where[4]; int nw = 0;
        for (size_t i = 0; i < {sizes[step["out"]]}; i++) {{
            double ref = ((double){step["lhs"]}[i] * {step["lhs_load"]!r} + (double){step["rhs"]}[i] * {step["rhs_load"]!r}) * {step["readout"]!r};
            long long want = (long long)(ref < 0 ? ref - 0.5 : ref + 0.5);
            {"if (want < 0) want = 0;" if step.get("relu") else ""}
            if (want > 127) want = 127; else if (want < -128) want = -128;
            long long d = (long long){step["out"]}[i] - want;
            if (d < 0) d = -d;
            if (d > worst) {{ worst = d; wi = (long long)i; wl = {step["lhs"]}[i]; wr = {step["rhs"]}[i]; wo = {step["out"]}[i]; wref = want; }}
            if (d > {step["bound_lsb"]}) {{ if (nw < 4) where[nw++] = (long long)i; over++; }}
        }}
        printf("{_printf("bounded", group=step["group"], max_abs="%lld", over="%lld", bound=step["bound_lsb"])}\\n", worst, over);
        printf("{_printf("witness", group=step["group"], index="%lld", lhs="%lld", rhs="%lld", device="%lld", reference="%lld")}\\n", wi, wl, wr, wo, wref);
        for (int q = nw; q < 4; q++) where[q] = -1;
        printf("{_printf("where", group=step["group"], cols=step["cols"], first="%lld %lld %lld %lld")}\\n",
               where[0], where[1], where[2], where[3]);
    }}"""


def _producer_reference(step: Mapping[str, Any]) -> str:
    """C that writes a contraction step's REFERENCE output into the step's own output buffer.

    Used for a merged group, whose producer's committed value never exists on the device: the sum's
    check needs it, so it is recomputed here from the inputs the device actually held, with the same
    scalar arithmetic as the exact groups' local checks (int32 accumulation, the folded bias, the
    readout multiplier rounded half-to-even, saturation, then the activation).
    """
    out, bias, relu = step["out"], step.get("bias") or "NULL", 1 if step.get("relu") else 0
    if step.get("scale") is None:
        raise NotClosed(f"group {step['group']} leaves as an accumulator; a merged sum reads a requantized value")
    scale = step["scale"]
    if step["kind"] == "matmul":
        return f"""
        for (int i = 0; i < {step["m"]}; i++) {{
            lc_xs[0] = {step["in"]} + (size_t)i * {step["k"]};
            lc_ws[0] = (const elem_t *){step["weight"]};
            lc_row(lc_xs, lc_ws, 1, {step["k"]}, {step["n"]}, {bias});
            for (int j = 0; j < {step["n"]}; j++)
                {out}[(size_t)i * {step["n"]} + j] = (elem_t)lc_readout(lc_acc[j], 1, {scale!r}f, {relu});
        }}"""
    if step["kind"] != "conv2d" or step["pool"]["size"]:
        raise NotClosed(f"group {step['group']} is not an unpooled contraction, so no merged reference is stated")
    k, ci, n, dim = int(step["kernel"]), int(step["ci"]), int(step["n"]), int(step["in_dim"])
    od, stride, pad = int(step["out_dim"]), int(step["stride"]), int(step["padding"])
    return f"""
        for (int oh = 0; oh < {od}; oh++)
          for (int ow = 0; ow < {od}; ow++) {{
            int taps = 0;
            for (int r = 0; r < {k}; r++)
              for (int s = 0; s < {k}; s++) {{
                int ih = oh * {stride} - {pad} + r, iw = ow * {stride} - {pad} + s;
                if (ih < 0 || ih >= {dim} || iw < 0 || iw >= {dim}) continue;
                lc_xs[taps] = {step["in"]} + (size_t)(ih * {dim} + iw) * {ci};
                lc_ws[taps] = (const elem_t *){step["weight"]} + (size_t)((r * {k} + s) * {ci}) * {n};
                taps++;
              }}
            lc_row(lc_xs, lc_ws, taps, {ci}, {n}, {bias});
            for (int j = 0; j < {n}; j++)
                {out}[(size_t)(oh * {od} + ow) * {n} + j] = (elem_t)lc_readout(lc_acc[j], 1, {scale!r}f, {relu});
          }}"""


def _fused_check(step: Mapping[str, Any], sizes: Mapping[str, int]) -> str:
    """The on-core check of a MERGED group, graded as the sum it absorbed, on the bound it declared.

    The producer's reference is recomputed from its actual inputs into its own (otherwise unwritten)
    buffer, then the sum's reference from that and the skip tensor the device read -- the same check a
    standalone sum gets, against the merged group's declared bound rather than the sum's own.
    """
    producer, total = step["producer"], step["sum"]
    graded = {
        **total,
        "group": step["graded_as"],
        "out": step["out"],
        "bound_lsb": step["bound_lsb"],
        "lhs": producer["out"] if step["via"] == "lhs" else step["skip"],
        "rhs": step["skip"] if step["via"] == "lhs" else producer["out"],
    }
    head = f"merged group {step['group']} (absorbs {step['graded_as']}): the producer's reference first"
    return f"""    {{ /* {head} */{_producer_reference(producer)}
    }}
{_bounded_block(graded, sizes)}"""


def merge_steps(model: dict[str, Any], merges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The model's steps with each MERGED group stated as one step, at its producer's position.

    ``merges`` is one row per merged group: ``producer`` (a contraction step's group), ``absorbs`` (the
    one sum step it took), ``via`` (which operand of the sum the producer is) and ``bound_lsb`` (the
    bound the offer declared). The merged step writes the sum's output buffer and reads the skip tensor
    the sum's other operand held; the sum's own step is gone. Its library fallback is the two calls it
    replaced, in order, so a group the driver cannot bind still computes what the pair did.

    Every fact is CHECKED against this driver's own steps: the producer must be a requantizing
    contraction whose output is exactly the sum's ``via`` operand, or the merge is refused by name.
    With no merges the model is returned unchanged -- the same object, so the program is too.
    """
    if not merges:
        return model
    steps = {int(step["group"]): step for step in model["steps"]}
    fused: dict[int, dict[str, Any]] = {}
    absorbed: set[int] = set()
    for merge in merges:
        producer_group, (consumer_group,) = int(merge["producer"]), [int(g) for g in merge["absorbs"]]
        producer, total = steps.get(producer_group), steps.get(consumer_group)
        via = str(merge["via"])
        if producer is None or total is None or total["kind"] != "sum" or producer["kind"] not in ("conv2d", "matmul"):
            raise NotClosed(f"merged group {producer_group} does not pair a contraction step with a sum step here")
        if via not in ("lhs", "rhs") or total[via] != producer["out"]:
            raise NotClosed(
                f"merged group {producer_group}: the sum's {via} operand is {total.get(via)!r}, not the producer's "
                f"{producer['out']!r}"
            )
        _producer_reference(producer)  # refuses a producer no merged reference can be stated for
        fused[producer_group] = {
            "kind": "fused",
            "group": producer_group,
            "graded_as": consumer_group,
            "producer": dict(producer),
            "sum": dict(total),
            "via": via,
            "bound_lsb": int(merge["bound_lsb"]),
            "in": producer["in"],
            "weight": producer["weight"],
            "bias": producer.get("bias"),
            "skip": total["rhs" if via == "lhs" else "lhs"],
            "out": total["out"],
            "scale": producer["scale"],
            "relu": total.get("relu"),
        }
        absorbed.add(consumer_group)
    merged = [fused.get(int(step["group"]), step) for step in model["steps"] if int(step["group"]) not in absorbed]
    return {**model, "steps": merged, "device_groups": len(merged)}


#: The step kinds this driver can RESTATE on the core as a fused region's internal member: their
#: reference is written into the member's own buffer before the region's boundary is checked
#: (:func:`_restated`). A requantizing contraction (pooled or not), a requantizing matmul, a window
#: mean. A sum is never an internal member here: its check is a bound, not a value to restate.
REGION_INTERNAL_KINDS = ("conv2d", "matmul", "mean")

#: The step fields that name a buffer a step READS.
_READ_FIELDS = ("in", "lhs", "rhs")


def flat_steps(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every group's own step in program order, a fused region's members in its place."""
    out: list[dict[str, Any]] = []
    for step in model["steps"]:
        out.extend(step["members"] if step["kind"] == "region" else [step])
    return out


def region_steps(
    model: dict[str, Any], regions: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[int, str]]:
    """``(model, refused)``: each FUSED REGION stated as ONE step, at its members' position.

    ``regions`` is one row per region a package answered with ONE kernel: ``members`` (consecutive
    groups, in program order) and ``boundary`` (the last of them, the only member whose output the
    region must commit). The region's step is the BOUNDARY's own step -- its output, its readout, its
    dequantize -- under the boundary's group, with ``kind: region``, every member's own step under
    ``members`` (in order) and the buffers it reads from outside itself under ``reads``. The members'
    own steps are gone: the region's kernel is called once, in their place.

    Its library fallback is the members' own calls, in order, so a region the driver cannot bind
    still computes what they did. It is GRADED AT ITS BOUNDARY: each internal member is restated on
    the core from the inputs the device held (:func:`_restated`), then the boundary is checked as an
    ordinary group of its kind -- a value a region keeps to itself is never graded, and never needs
    to exist.

    Every fact is CHECKED against this driver's own steps rather than taken from the caller: the
    members must be consecutive steps of this model ending at the boundary, every internal member a
    kind this driver can restate (:data:`REGION_INTERNAL_KINDS`) that commits a requantized value, and
    every internal member's output read by NO step but the next member. A region failing any of
    these is REFUSED BY GROUP (``refused[boundary] = why``) and its members stay ordinary steps; with
    no regions the model is returned unchanged -- the same object, so the program is too.
    """
    if not regions:
        return model, {}
    steps = list(model["steps"])
    position = {int(step["group"]): index for index, step in enumerate(steps)}
    readers: dict[str, list[int]] = {}
    for step in flat_steps(model):
        for field in _READ_FIELDS:
            if step.get(field):
                readers.setdefault(str(step[field]), []).append(int(step["group"]))
    refused: dict[int, str] = {}
    replace: dict[int, dict[str, Any]] = {}
    absorbed: set[int] = set()
    for region in regions:
        groups = [int(g) for g in region["members"]]
        boundary = int(region["boundary"])
        why = None
        if len(groups) < 2 or groups[-1] != boundary:
            why = f"a region names {groups} with boundary {boundary}: its boundary is its last member"
        elif any(g not in position for g in groups):
            why = f"region member(s) {[g for g in groups if g not in position]} are not steps of this program"
        elif [position[g] for g in groups] != list(range(position[groups[0]], position[groups[0]] + len(groups))):
            why = f"region members {groups} are not consecutive steps of this program"
        elif absorbed & set(groups):
            why = f"region members {sorted(absorbed & set(groups))} are already in another region"
        else:
            members = [steps[position[g]] for g in groups]
            for member, after in itertools.pairwise(members):
                if member["kind"] not in REGION_INTERNAL_KINDS:
                    why = f"internal member {member['group']} is a {member['kind']!r} step, which this driver cannot restate"
                elif member["kind"] in ("conv2d", "matmul") and member.get("scale") is None:
                    why = f"internal member {member['group']} leaves as the accumulator; a restated value is requantized"
                elif sorted(set(readers.get(str(member["out"]), ()))) != [int(after["group"])]:
                    why = (
                        f"internal member {member['group']}'s output is read by {sorted(set(readers.get(str(member['out']), ())))}, "
                        f"not by its next member {after['group']} alone"
                    )
                if why:
                    break
        if why:
            refused[boundary] = why
            continue
        produced_inside = {str(m["out"]) for m in members[:-1]}
        reads: list[str] = []
        for member in members:
            for field in _READ_FIELDS:
                name = member.get(field)
                if name and str(name) not in produced_inside and str(name) not in reads:
                    reads.append(str(name))
        replace[groups[0]] = {
            **members[-1],
            "kind": "region",
            "group": boundary,
            "boundary_kind": members[-1]["kind"],
            "members": [dict(m) for m in members],
            "reads": reads,
        }
        absorbed.update(groups)
    if not replace:
        return model, refused
    merged = []
    for step in steps:
        group = int(step["group"])
        if group in replace:
            merged.append(replace[group])
        elif group not in absorbed:
            merged.append(step)
    return {**model, "steps": merged, "device_groups": len(merged)}, refused


def _region_bounded(step: Mapping[str, Any], sizes: Mapping[str, int]) -> str:
    """The on-core check of a fused region whose boundary is a SUM: its internal members restated,
    then the sum's own bounded check, under the region's group."""
    return _region_restatements(step) + "\n" + _bounded_block({**step["members"][-1], "group": step["group"]}, sizes)


_LOCAL_NOTE = """static void lc_note(lc_verdict *v, long long got, long long want, long long index) {
    if (got != want) { if (!v->mismatches) v->first = index; v->mismatches++; }
}"""

_LOCAL_MAP_HELPERS = r"""
static unsigned int lc_by_ch[LC_MAP_CH], lc_by_row[LC_MAP_ROWS], lc_by_col[LC_MAP_COLS];
static long long lc_map_pitch = 1, lc_map_width = 1, lc_max_abs;
static unsigned long long lc_over, lc_under;
static long long lc_samples[LC_MAP_SAMPLES][6];
static int lc_nsamples;
static void lc_map_reset(long long pitch, long long width) {
    for (int i = 0; i < LC_MAP_CH; i++) lc_by_ch[i] = 0;
    for (int i = 0; i < LC_MAP_ROWS; i++) lc_by_row[i] = 0;
    for (int i = 0; i < LC_MAP_COLS; i++) lc_by_col[i] = 0;
    lc_map_pitch = pitch; lc_map_width = width; lc_max_abs = 0; lc_over = 0; lc_under = 0; lc_nsamples = 0;
}
static void lc_map_note(long long got, long long want, long long index) {
    long long ch = index % lc_map_pitch, pos = index / lc_map_pitch;
    long long row = pos / lc_map_width, col = pos % lc_map_width;
    if (ch < LC_MAP_CH) lc_by_ch[ch]++;
    if (row < LC_MAP_ROWS) lc_by_row[row]++;
    if (col < LC_MAP_COLS) lc_by_col[col]++;
    long long d = got - want;
    if (d > 0) lc_over++; else lc_under++;
    if (d < 0) d = -d;
    if (d > lc_max_abs) lc_max_abs = d;
    if (lc_nsamples < LC_MAP_SAMPLES) {
        long long *s = lc_samples[lc_nsamples++];
        s[0] = index; s[1] = row; s[2] = col; s[3] = ch; s[4] = got; s[5] = want;
    }
}
static void lc_map_axis(const char *prefix, const unsigned int *counts, long long extent) {
    printf("%s", prefix);
    int none = 1;
    for (long long i = 0; i < extent; i++)
        if (counts[i]) { printf(none ? "%lld:%u" : ",%lld:%u", i, counts[i]); none = 0; }
    printf(none ? "none\n" : "\n");
}
"""

_LOCAL_MAP_NOTE = """static void lc_note(lc_verdict *v, long long got, long long want, long long index) {
    if (got != want) { if (!v->mismatches) v->first = index; v->mismatches++; lc_map_note(got, want, index); }
}"""


def _local_map_geometry(step: Mapping[str, Any]) -> tuple[int, int, int, tuple[str, ...]]:
    """``(pitch, width, rows, axes)`` of a step's output as its local check indexes it."""
    if step["kind"] == "mean":
        return int(step["rows"]), 1, 1, ("channel",)
    if step["kind"] == "matmul":
        return int(step["n"]), 1, int(step["m"]), ("row", "channel")
    od, pool = int(step["out_dim"]), step["pool"]
    if pool["size"]:
        ps, pst, pp = int(pool["size"]), int(pool["stride"]), int(pool["padding"])
        od = (od + 2 * pp - ps) // pst + 1
    return int(step["n"]), od, od, ("row", "col", "channel")


def _local_map_report(group: Any, step: Mapping[str, Any]) -> str:
    """The C that prints where ``step``'s output was wrong, after its GM_LOCAL line."""
    pitch, width, rows, axes = _local_map_geometry(step)
    extents = {"row": rows, "col": width, "channel": pitch}
    counts = {"row": "lc_by_row", "col": "lc_by_col", "channel": "lc_by_ch"}
    lines = []
    for axis in axes:
        prefix = _printf("local_map", group=str(group), axis=axis, extent=str(extents[axis]), wrong="")
        lines.append(f'        lc_map_axis("{prefix}", {counts[axis]}, {extents[axis]});')
    sample = _printf(
        "local_sample", group=str(group), index="%lld", row="%lld", col="%lld", channel="%lld", got="%lld", want="%lld"
    )
    lines.append(
        "        for (int s = 0; s < lc_nsamples; s++)\n"
        f'            printf("{sample}\\n", lc_samples[s][0], lc_samples[s][1], lc_samples[s][2], '
        "lc_samples[s][3], lc_samples[s][4], lc_samples[s][5]);"
    )
    sign = _printf("local_sign", group=str(group), over="%llu", under="%llu", max_abs="%lld")
    lines.append(f'        printf("{sign}\\n", lc_over, lc_under, lc_max_abs);')
    return "\n".join(lines)


def _local_body(step: Mapping[str, Any], note) -> str:
    """The C that recomputes ``step``'s output from its actual inputs, element by element.

    ``note(index, want)`` is the C statement each recomputed element goes to: a comparison with the
    device's value (a local CHECK) or a store into the step's own buffer (a RESTATEMENT, for a fused
    region's internal member, whose value never exists on the device). One body for both, so the
    value a region's boundary is graded against and the value an ordinary group is graded against
    come from the same arithmetic."""
    bias = step.get("bias") or "NULL"
    relu = 1 if step.get("relu") else 0
    if step["kind"] == "mean":
        return f"""
        for (int r = 0; r < {step["rows"]}; r++) {{
            long long acc = 0;
            for (int w = 0; w < {step["window"]}; w++) acc += {step["in"]}[(size_t)w * {step["rows"]} + r];
            {note("r", f"lc_readout(acc, 1, {step['multiplier']!r}f, 0)")}
        }}"""
    if step["kind"] == "matmul":
        scaled = 0 if step.get("scale") is None else 1
        scale = step.get("scale") or 1.0
        return f"""
        for (int i = 0; i < {step["m"]}; i++) {{
            lc_xs[0] = {step["in"]} + (size_t)i * {step["k"]};
            lc_ws[0] = (const elem_t *){step["weight"]};
            lc_row(lc_xs, lc_ws, 1, {step["k"]}, {step["n"]}, {bias});
            for (int j = 0; j < {step["n"]}; j++)
                {note(f"(long long)i * {step['n']} + j", f"lc_readout(lc_acc[j], {scaled}, {scale!r}f, {relu})")}
        }}"""
    k, ci, n, dim = int(step["kernel"]), int(step["ci"]), int(step["n"]), int(step["in_dim"])
    od, stride, pad, pool = int(step["out_dim"]), int(step["stride"]), int(step["padding"]), step["pool"]
    scaled = 0 if step.get("scale") is None else 1
    scale = step.get("scale") or 1.0
    target = (
        "lc_plane[(size_t)(oh * %d + ow) * %d + j] = (elem_t)want;" % (od, n)
        if pool["size"]
        else note(f"(long long)(oh * {od} + ow) * {n} + j", "want")
    )
    body = f"""
        for (int oh = 0; oh < {od}; oh++)
          for (int ow = 0; ow < {od}; ow++) {{
            int taps = 0;
            for (int r = 0; r < {k}; r++)
              for (int s = 0; s < {k}; s++) {{
                int ih = oh * {stride} - {pad} + r, iw = ow * {stride} - {pad} + s;
                if (ih < 0 || ih >= {dim} || iw < 0 || iw >= {dim}) continue;
                lc_xs[taps] = {step["in"]} + (size_t)(ih * {dim} + iw) * {ci};
                lc_ws[taps] = (const elem_t *){step["weight"]} + (size_t)((r * {k} + s) * {ci}) * {n};
                taps++;
              }}
            lc_row(lc_xs, lc_ws, taps, {ci}, {n}, {bias});
            for (int j = 0; j < {n}; j++) {{
                long long want = lc_readout(lc_acc[j], {scaled}, {scale!r}f, {relu});
                {target}
            }}
          }}"""
    if pool["size"]:
        ps, pst, pp = int(pool["size"]), int(pool["stride"]), int(pool["padding"])
        pd = (od + 2 * pp - ps) // pst + 1
        body += f"""
        for (int ph = 0; ph < {pd}; ph++)
          for (int pw = 0; pw < {pd}; pw++)
            for (int j = 0; j < {n}; j++) {{
                long long best = elem_t_min;
                for (int r = 0; r < {ps}; r++)
                  for (int s = 0; s < {ps}; s++) {{
                    int ih = ph * {pst} - {pp} + r, iw = pw * {pst} - {pp} + s;
                    long long value = (ih < 0 || ih >= {od} || iw < 0 || iw >= {od})
                        ? elem_t_min : (long long)lc_plane[(size_t)(ih * {od} + iw) * {n} + j];
                    if (value > best) best = value;
                  }}
                {note(f"(long long)(ph * {pd} + pw) * {n} + j", "best")}
            }}"""
    return body


def _restated(step: Mapping[str, Any]) -> str:
    """C that writes a fused region's internal member's REFERENCE into the member's own buffer.

    The member's committed value never exists on the device (the region's kernel keeps it to itself),
    and its consumer's check needs it: it is recomputed here, outside the window, from the inputs the
    device actually held -- the same arithmetic as a local check, stored instead of compared."""
    out = step["out"]
    return f"""    {{ /* region member {step["group"]}: restated on the core for its region's check */{
        _local_body(step, lambda index, want: f"{out}[{index}] = (elem_t)({want});")
    }
    }}"""


def _region_restatements(step: Mapping[str, Any]) -> str:
    return "\n".join(_restated(member) for member in step["members"][:-1])


def _local_checks(
    model: dict[str, Any],
    sizes: Mapping[str, int],
    *,
    extra: Sequence[Mapping[str, Any]] = (),
    mapped: bool = False,
) -> tuple[str, str]:
    """``(helpers, calls)``: the C that recomputes every exact group from its actual inputs.

    Each exact group -- a convolution, a matmul, a window mean -- gets one GM_LOCAL line. The sum
    groups are already checked locally (GM_BOUND recomputes them from the two operands the device
    read). Nothing here reads a weight in any layout but the one the program itself holds and hands
    to every call, so the reference and the device read the same bytes.

    A FUSED REGION whose boundary is exact gets ONE GM_LOCAL line, under the region's group: its
    internal members are restated on the core first (:func:`_restated`), then its boundary is checked
    from them and from the inputs the device held -- the region graded as the one claim it is.
    """
    exact = [
        s
        for s in model["steps"]
        if s["kind"] in ("conv2d", "matmul", "mean") or (s["kind"] == "region" and s["boundary_kind"] != "sum")
    ]
    regions = [s for s in model["steps"] if s["kind"] == "region"]
    # `extra` are the producers of merged groups: not graded here, but their references use the same
    # helpers, so the helpers are sized for them too -- as are every region's members.
    members = [m for r in regions for m in r["members"]]
    sized = [s for s in exact if s["kind"] != "region"] + [*extra, *members]
    if not sized:
        return "", ""
    max_n = max([int(s["n"]) for s in sized if "n" in s] + [1])
    max_taps = max([int(s["kernel"]) ** 2 for s in sized if s["kind"] == "conv2d"] + [1])
    pooled = [s for s in sized if s["kind"] == "conv2d" and s["pool"]["size"]]
    max_plane = max([int(s["out_dim"]) ** 2 * int(s["n"]) for s in pooled] + [1])
    helpers = (
        # The element range is the vendor header's own (`elem_t_max` / `elem_t_min`), never a literal.
        f"#define LC_MAX_N {max_n}\n#define LC_BLOCK {_LOCAL_BLOCK}\n"
        + _LOCAL_HELPERS
        + f"static const elem_t *lc_xs[{max_taps}], *lc_ws[{max_taps}];\n"
        + f"static elem_t lc_plane[{max_plane if pooled else 1}];\n"
    )

    def graded(step: Mapping[str, Any]) -> Mapping[str, Any]:
        """The step a check reads: a region is checked as its boundary member, under its own group."""
        return {**step["members"][-1], "group": step["group"]} if step["kind"] == "region" else step

    if mapped:
        geometry = [_local_map_geometry(graded(s)) for s in exact]
        assert helpers.count(_LOCAL_NOTE) == 1, "the local helpers' lc_note changed; update the mapped one"
        helpers = helpers.replace(
            _LOCAL_NOTE,
            f"#define LC_MAP_CH {max([g[0] for g in geometry] + [1])}\n#define LC_MAP_COLS {max([g[1] for g in geometry] + [1])}\n"
            f"#define LC_MAP_ROWS {max([g[2] for g in geometry] + [1])}\n#define LC_MAP_SAMPLES {_LOCAL_MAP_SAMPLES}\n"
            + _LOCAL_MAP_HELPERS
            + _LOCAL_MAP_NOTE,
        )
    blocks = []
    for step in exact:
        checked = graded(step)
        g, out = checked["group"], checked["out"]
        body = _local_body(checked, lambda index, want, out=out: f"lc_note(&v, {out}[{index}], {want}, {index});")
        reset = ""
        report = ""
        if mapped:
            pitch, width, _rows, _axes = _local_map_geometry(checked)
            reset = f"\n        lc_map_reset({pitch}, {width});"
            report = "\n" + _local_map_report(g, checked)
        restated = (_region_restatements(step) + "\n") if step["kind"] == "region" else ""
        blocks.append(
            f"""{restated}    {{
        lc_verdict v = {{0, -1}};{reset}{body}
        printf("{_printf("local", group=g, mismatches="%llu", elements=sizes[out], first="%lld")}\\n", v.mismatches, v.first);{report}
    }}"""
        )
    return helpers, "\n".join(blocks)


#: The cause a group carries when its full-width accumulator readout is computed on the core.
ACCUMULATOR_READOUT_UNAVAILABLE = "accumulator_readout_unavailable"


def accumulator_readout_groups(model: dict[str, Any]) -> list[dict[str, Any]]:
    """The steps whose output IS the raw accumulator (no readout multiplier): a full-width mvout.

    A fused region counts by its BOUNDARY (stated as the boundary's own step under the region's
    group): whoever writes it -- the region's kernel or the members' fallback -- reads that
    accumulator out, so a machine without the full-width port routes it like any other."""
    groups = []
    for step in model["steps"]:
        if step["kind"] == "region":
            step = {**step["members"][-1], "group": step["group"]}
        if step["kind"] in ("conv2d", "matmul") and step.get("scale", 0) is None:
            groups.append(step)
    return groups


def full_width_readout(header: Path | None) -> bool | None:
    """Whether the machine built its full-width accumulator read port, from ITS parameter header.

    The generator writes ``#define ACC_READ_FULL_WIDTH`` into the header exactly when the elaborated
    config has ``acc_read_full_width`` (``GemminiConfigs.scala``: ``if (acc_read_full_width) header ++=
    "#define ACC_READ_FULL_WIDTH"``), and that port is the one a full-width mvout reads through. The
    header is the one the build compiles against and asserts by content for the machine, so this is a
    fact about THAT machine, not about the target's default configuration. ``None`` when no header is
    at hand -- which the caller must treat as unknown, never as either answer.
    """
    if header is None or not Path(header).is_file():
        return None
    from merlin.targetgen.capability_discovery import parse_c_header

    return parse_c_header(Path(header)).macro("ACC_READ_FULL_WIDTH") is not None


def _trap_prohibited_in_library(
    includes: Sequence[str], vendor: Path | None, selectors: Sequence[int], build_root: Path
) -> dict:
    """Rewrite the build's own copy of the library header so it cannot issue a prohibited instruction.

    Only a copy inside ``build_root`` (the tree this build owns) is ever rewritten, and it is replaced
    by unlinking rather than written through, so a link back into a shared harness tree is never
    followed: any other header is a refusal, not an edit."""
    owned = Path(build_root).resolve()
    roots = [Path(flag[2:]) for flag in includes if flag.startswith("-I")] + ([Path(vendor)] if vendor else [])
    for root in roots:
        header = root / "include" / "gemmini.h"
        if header.is_file():
            if not header.parent.resolve().is_relative_to(owned):
                raise ValueError(f"the library header {header} is not this build's own copy; refusing to rewrite it")
            before = header.read_text(encoding="utf-8")
            after, trapped = loop_free_header(before, selectors)
            header.unlink()
            header.write_text(after, encoding="utf-8")
            return {
                "path": str(header),
                "sha256_before": hashlib.sha256(before.encode("utf-8")).hexdigest(),
                "sha256_after": hashlib.sha256(after.encode("utf-8")).hexdigest(),
                "trapped_macros": trapped,
                "prohibited_selectors": sorted(int(s) for s in selectors),
            }
    raise ValueError("no library header found in the build's include roots to make loop-free")


def _parameter_header(vendor: Path | None, includes: Sequence[str]) -> Path | None:
    """The parameter header the preprocessor reads: the first include root holding it."""
    roots = [Path(flag[2:]) for flag in includes if flag.startswith("-I")]
    if vendor is not None:
        roots.append(Path(vendor))
    for root in roots:
        header = root / "include" / "gemmini_params.h"
        if header.is_file():
            return header
    return None


def readout_routing(model: dict[str, Any], full_width: bool | None) -> list[dict[str, Any]]:
    """Which groups compute their full-width accumulator readout on the core, and why. Fail-closed.

    A machine without the full-width read port answers a full-width mvout with NARROW data -- the
    measured symptom is a classifier 1000/1000 wrong on every arm, the vendor library's included. The
    smallest correct answer is the group itself on the core: its result IS the int32 accumulator, which
    no narrow (int8, scaled) readout can carry exactly, and a device path that reassembled it from
    narrow readouts would be a new kernel with its own correctness burden. A group read out through a
    multiplier is untouched. When the fact is UNKNOWN and such a group exists, the build is refused.
    """
    groups = accumulator_readout_groups(model)
    if not groups or full_width is True:
        return []
    if full_width is None:
        raise SystemExit(
            f"group(s) {[s['group'] for s in groups]} read out the full-width accumulator, and whether this "
            f"machine built that read port is UNKNOWN (no parameter header to read ACC_READ_FULL_WIDTH from)"
        )
    for step in groups:
        if step["kind"] != "matmul":
            raise SystemExit(
                f"group {step['group']} ({step['kind']}) reads out the full-width accumulator on a machine "
                f"without that read port, and only a matmul has an on-core readout here"
            )
    return [
        {
            "group": s["group"],
            "cause": ACCUMULATOR_READOUT_UNAVAILABLE,
            "why": "the machine's parameter header states no ACC_READ_FULL_WIDTH (the elaborated config has "
            "no full-width accumulator read port), so this group's int32 accumulator result is computed "
            "on the core instead of read out by a full-width mvout",
        }
        for s in groups
    ]


LIBRARY_HOST_NARROW_OUTPUT = "library_host_path_writes_narrow_output"


def library_host_readout_routing(model: dict[str, Any], answered: Collection[int] = ()) -> list[dict[str, Any]]:
    """The full-width accumulator groups a LOOP-FREE library cannot answer, routed to the core.

    With loop-descriptor instructions prohibited and no output-stationary dataflow, the library's
    matmul falls to its host code, ``matmul_cpu``, whose output is ``elem_t *C``: a narrow, scaled
    element. It has no full-width form -- the library's own guard prints "Not implemented: CPU matmul,
    full_C=1" and then runs it anyway, writing int8 values into a buffer the group reads as int32.
    Measured on the GSIM certifier (header 3758ae96, which HAS the full-width port): g71 went there.
    So a group whose result is the int32 accumulator is computed by this driver's own core routine,
    the same one a machine without the full-width port uses, on any machine where the library's
    matmul is its host code.

    ONLY A GROUP THE LIBRARY WOULD ANSWER. This is a fact about the LIBRARY's host code, so it says
    nothing about a group whose kernel is the package's (``answered``): on a machine whose own readout
    facts establish the full-width port, that kernel reads its accumulator out through the port, as
    the package compiled it to, and its result is graded like any other. Routing it to the core anyway
    took the classifier away from every package on every loop-free machine, whatever its kernel did."""
    # A fused region is not the library's call: its kernel writes the accumulator itself, and its
    # fallback's members are routed one by one (`_call` -> `library_host_full_width`).
    regions = {int(s["group"]) for s in model["steps"] if s["kind"] == "region"} | {int(g) for g in answered}
    return [
        dict(
            row,
            cause=LIBRARY_HOST_NARROW_OUTPUT,
            why="loop-descriptor instructions are prohibited, so the library's matmul is its host code, "
            "which writes only narrow elements; this group's int32 accumulator result is computed on the "
            "core by the program's own routine",
        )
        for row in readout_routing(model, False)
        if int(row["group"]) not in regions
    ]


def core_readout_routing(
    model: dict[str, Any],
    full_width: bool | None,
    library: dict[str, Any] | None,
    *,
    answered: Collection[int] = (),
) -> list[dict[str, Any]]:
    """Every full-width accumulator group this program computes on the core, with its cause: the
    machine has no full-width read port (every such group, whoever answers it -- a machine fact), or
    the library path it would take is the library's host code (only the groups the LIBRARY answers:
    ``answered`` are the groups whose kernel is the package's, see
    :func:`library_host_readout_routing`)."""
    routed = readout_routing(model, full_width)
    if not routed and library is not None and (library.get("matmul") or {}).get("host"):
        routed = library_host_readout_routing(model, answered)
    return routed


_HOST_ACC_MATMUL = r"""
/* A FULL-WIDTH ACCUMULATOR RESULT, ON THE CORE: out[i][j] = bias[j] + sum_k a[i][k] * w[k][j] in
   int32 -- what a full-width mvout of the accumulator would have returned on a machine that has the
   port. Emitted only for a machine whose header states no ACC_READ_FULL_WIDTH. */
static void hr_acc_matmul(const elem_t *a, const elem_t *w, const acc_t *bias, acc_t *out,
                          int m, int k, int n, int relu) {
    for (int i = 0; i < m; i++) {
        const elem_t *row = a + (size_t)i * k;
        for (int nb = 0; nb < n; nb += 8) {
            int width = n - nb < 8 ? n - nb : 8;
            int32_t acc[8];
            for (int j = 0; j < 8; j++) acc[j] = (bias && j < width) ? (int32_t)bias[nb + j] : 0;
            for (int kk = 0; kk < k; kk++) {
                int32_t x = row[kk];
                if (!x) continue;
                const elem_t *wr = w + (size_t)kk * n + nb;
                for (int j = 0; j < width; j++) acc[j] += x * (int32_t)wr[j];
            }
            for (int j = 0; j < width; j++) out[(size_t)i * n + nb + j] = (relu && acc[j] < 0) ? 0 : acc[j];
        }
    }
}
"""


def _host_call(step: dict[str, Any]) -> str:
    bias = step.get("bias") or "NULL"
    return (
        f"hr_acc_matmul({step['in']}, (const elem_t *){step['weight']}, {bias}, {step['out']}, "
        f"{step['m']}, {step['k']}, {step['n']}, {1 if step.get('relu') else 0});"
    )


def _library_steps(model: Mapping[str, Any], ours: Mapping[Any, str], on_core: set[int]) -> list[dict[str, Any]]:
    """The steps the LIBRARY answers in this program: each step no kernel of the package answers and the
    machine does not route to the core -- a fused region no kernel answers contributing its members."""
    out: list[dict[str, Any]] = []
    for step in model["steps"]:
        group = int(step["group"])
        if step["group"] in ours:
            continue
        if step["kind"] == "region":
            out.extend(m for m in step["members"] if not (int(m["group"]) == group and group in on_core))
        elif group not in on_core:
            out.append(step)
    return out


def host_conv2d_groups(model: dict[str, Any], library: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every conv2d step ``_call`` routes to ``hr_conv2d`` instead of the vendor library: with
    loop-descriptor instructions prohibited, ``library_path_without_loops`` names ``conv2d`` a host
    path for every convolution uniformly (it has no loop-free accelerator path at all), so this is
    every conv2d step once any package-declined convolution reaches the library under that prohibition.
    """
    if not library or not (library.get("conv2d") or {}).get("host"):
        return []
    return [s for s in flat_steps(model) if s["kind"] == "conv2d"]


#: `hr_conv2d`'s pooling scratch needs one plane the size of the LARGEST pooled convolution's
#: pre-pool feature map -- a static buffer, never a VLA: g1's stem conv alone is 112*112*64 = 802,816
#: elements, and this program's whole runtime stack is a 128 KiB reservation (`gemmini.py`'s harness
#: recipe), so a stack-allocated scratch plane at that size is an instant overflow, not a slow path.
_HOST_CONV2D = r"""
/* A CONVOLUTION, ON THE CORE, WITHOUT A SINGLE LOOP-DESCRIPTOR INSTRUCTION. Measured: the vendor
   library's own CPU dataflow for `tiled_conv` still reaches `gemmini_loop_conv_ws` for some shapes,
   which this repo's no-FSM header traps into a hard abort -- so asking the library for ITS loop-free
   convolution path can abort the run. This routine never calls into gemmini.h at all: int32
   accumulate, bias, round-half-to-even requantize (the same rounding `_local_checks` already
   recomputes every exact group against), clamp to the vendor header's own element range, relu, then
   an optional max-pool -- plain C, so it cannot reach a prohibited macro regardless of what the
   vendor's own CPU path does internally. */
static long long hr_rne(float p) {
    long long i = (long long)p;
    long long nxt = p < 0 ? i - 1 : i + 1;
    float rem = p - (float)i;
    if (rem < 0) rem = -rem;
    if (rem < 0.5f) return i;
    if (rem > 0.5f) return nxt;
    return (i % 2 == 0) ? i : nxt;
}
static long long hr_readout(long long acc, int scaled, float scale, int relu) {
    if (!scaled) return acc;
    long long q = hr_rne((float)acc * scale);
    if (q > elem_t_max) q = elem_t_max;
    if (q < elem_t_min) q = elem_t_min;
    if (relu && q < 0) q = 0;
    return q;
}
static elem_t hr_conv_plane[HR_CONV_PLANE_ELEMENTS];
static void hr_conv2d(const elem_t *in, const elem_t *weight, const acc_t *bias, elem_t *out,
                       int dim, int ci, int n, int od, int stride, int pad, int k,
                       int scaled, float scale, int relu,
                       int pool_size, int pool_stride, int pool_padding) {
    elem_t *plane = pool_size ? hr_conv_plane : out;
    for (int oh = 0; oh < od; oh++)
      for (int ow = 0; ow < od; ow++)
        for (int j = 0; j < n; j++) {
            int32_t acc = bias ? (int32_t)bias[j] : 0;
            for (int r = 0; r < k; r++)
              for (int s = 0; s < k; s++) {
                int ih = oh * stride - pad + r, iw = ow * stride - pad + s;
                if (ih < 0 || ih >= dim || iw < 0 || iw >= dim) continue;
                const elem_t *x = in + (size_t)(ih * dim + iw) * ci;
                const elem_t *w = weight + (size_t)(r * k + s) * ci * n + j;
                for (int c = 0; c < ci; c++) {
                    int32_t xv = x[c];
                    if (xv) acc += xv * (int32_t)w[(size_t)c * n];
                }
              }
            plane[(size_t)(oh * od + ow) * n + j] = (elem_t)hr_readout(acc, scaled, scale, relu);
        }
    if (!pool_size) return;
    int pd = (od + 2 * pool_padding - pool_size) / pool_stride + 1;
    for (int ph = 0; ph < pd; ph++)
      for (int pw = 0; pw < pd; pw++)
        for (int j = 0; j < n; j++) {
            long long best = elem_t_min;
            for (int r = 0; r < pool_size; r++)
              for (int s = 0; s < pool_size; s++) {
                int ih = ph * pool_stride - pool_padding + r, iw = pw * pool_stride - pool_padding + s;
                long long value = (ih < 0 || ih >= od || iw < 0 || iw >= od)
                    ? elem_t_min : (long long)plane[(size_t)(ih * od + iw) * n + j];
                if (value > best) best = value;
              }
            out[(size_t)(ph * pd + pw) * n + j] = (elem_t)best;
        }
}
"""


def render(
    model: dict[str, Any],
    *,
    sched_kernels: dict[str, Any] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
    verify: str = "on_target",
    host_readout: Sequence[int] = (),
    library: dict[str, Any] | None = None,
) -> str:
    """The C program: one call per group, each timed, then the comparison with the golden.

    ``verify`` chooses WHERE each group's output is checked. ``on_target`` (the default) digests every
    output buffer and runs every bounded check on the core after the window closes. ``host_dump``
    emits neither: each GM_GROUP line carries its cycles with ``sum=UNKNOWN fnv1a=UNKNOWN`` (which the
    admission rule refuses as a check, so nothing reads it as one), and the outputs are checked by a
    host-side reader over a memory dump, using the buffer map the build records. Measured on GSIM:
    the on-target digests cost about 12 cycles per byte, 13M for one 800 KB output and over 250M for
    the model, which made an elaborated-RTL run of the whole model unaffordable.

    The call is the vendor library's unless ``sched_kernels`` supplies one of our own for that group
    (see :mod:`group_model_sched_kernels`). Groups we cannot express keep the library call, so a run
    is never quietly a mixture reported as one schedule -- the census says which is which.

    WHY THIS PRINTS ``METRIC cycles N`` AND NOT ITS OWN SPELLING.  Until 2026-09-19 the measured
    total left here as ``GROUP_MODEL_TOTAL cycles: %llu`` and no ``MERLIN_INVOCATIONS`` line was
    printed at all, so ``merlin.perf.firesim_receipt._verify_uart`` rejected this program's UART
    outright -- no run of this shape could ever have been sealed, whatever the queue did.  The fix
    is here rather than in the parser: ``METRIC cycles N`` and the byte-exact
    ``MERLIN_INVOCATIONS warmup=1 measured=1`` are not arbitrary constants, they are the lines a
    real FPGA run printed (pinned in ``merlin/tests/data/firesim_queue/job610_uart_marker_skeleton.log``)
    and the ones :mod:`merlin.perf.warm_profile_harness` generates for every other workload.  A
    parser taught to accept a second spelling is a parser with two places a *different* number can
    be read as the measurement -- and the byte-exactness has already earned its keep, catching the
    trailing ``batch=1`` that ``experiments/voyager_h2h/scripts/emit_gemmini_c.py`` appends.  This
    program was the thing that was wrong: it hand-rolled a ``main`` instead of printing the one
    protocol the repository has.

    The window frame (``MERLIN_WINDOW begin/end label=``) is printed even though this program
    measures exactly one window, so a solo UART is a strict prefix of the batched shape
    :mod:`merlin.perf.firesim_batch` reads and one admission rule serves both.
    """
    if not isinstance(window_label, str) or not window_label.strip() or window_label != window_label.strip():
        raise ValueError("the measured window's label must be a nonempty, unpadded token")
    if any(character.isspace() for character in window_label) or "=" in window_label:
        raise ValueError("the measured window's label is one whitespace-free token without '='")
    blobs = "\n".join(
        f'BLOB({name}, "{name}.bin", {"acc_t" if name.startswith("BIAS_") else "float" if name == "GOLDEN" else "elem_t"})'
        for name in model["arrays"]
    )
    # `__attribute__((used))`: a group's committed buffer must be a property of the PROGRAM (every
    # device group's declared output, whoever answers it), never of what the optimizer can prove is
    # read. Measured: routing one group's own consumer to the vendor's `tiled_conv_auto` -- a large
    # inline library routine specialized on this call's literal shape/flag arguments -- let the
    # compiler determine that THIS specialization never dereferences its input pointer, which made the
    # producing group's write a dead store too and eliminated the buffer from the linked program
    # entirely: `memory_map` (the host-side grade) then found "no sized symbol" for a group the model
    # statement still declares. The buffer's placement is a fact the harness needs regardless of
    # whether a chosen vendor specialization happens to touch it, so retention cannot be left to what
    # any one call site proves.
    buffers = "\n".join(
        f"static {b['ctype']} {b['name']}[{b['elements']}] row_align(1) __attribute__((used));"
        for b in model["buffers"]
    )
    window = max([s["window"] for s in flat_steps(model) if s["kind"] == "mean"] or [1])
    sizes = {b["name"]: b["elements"] for b in model["buffers"]}
    ours = (sched_kernels or {}).get("calls") or {}
    definitions = (sched_kernels or {}).get("definitions") or ""
    # WHICH FAMILY EACH GROUP'S CYCLES BELONG TO. A single fused bracket sum cannot say whether a
    # difference between two arms came from the kernels or from the buffer handling around them, and
    # a reader given one number will attribute all of it to the kernels. `fm_authored` is the groups
    # this arm's own compiler answered, `fm_vendor` the ones that fell back to the library, and
    # `fm_im2col` (accumulated inside the authored bracket, see the gather definitions) the operand
    # materialization an authored kernel required and the library call never did.
    # THE MEASURED PASS COMPUTES AND PRINTS NOTHING OF ITS OWN. Each group's bracket keeps its two
    # register reads -- `dt` is what the bracket sum is made of -- and everything else moves out.
    # MEASURED on hardware: with the checksums and the per-group printf inside the window, 98.9% of
    # the whole-model figure was this instrumentation -- 71 scalar passes over every intermediate
    # tensor on a scalar core, plus 71 UART writes. The reference driver this is scope-matched to
    # says it outright: measure the whole model with NO UART traffic inside the measured window.
    # A group whose full-width readout the machine lacks runs on the core (see `readout_routing`); it is
    # neither the package's nor the library's, and is bracketed with the library's calls.
    on_core = {int(g) for g in host_readout}
    ours = {g: call for g, call in ours.items() if int(g) not in on_core}
    # Every conv2d group `_call` will route to `hr_conv2d` (see there) rather than the vendor's own
    # CPU path -- computed the same way `_call` decides it, so the definition is included exactly
    # when something will actually call it.
    host_conv2d = host_conv2d_groups({"steps": _library_steps(model, ours, on_core)}, library)
    host_conv2d_plane_elements = max(
        [int(s["out_dim"]) ** 2 * int(s["n"]) for s in host_conv2d if s["pool"]["size"]] + [1]
    )
    host_conv2d_block = (
        f"#define HR_CONV_PLANE_ELEMENTS {host_conv2d_plane_elements}\n" + _HOST_CONV2D if host_conv2d else ""
    )
    # A library group whose full-width result the library's host code cannot produce is computed by the
    # same on-core routine (see `library_host_full_width`), so that routine is emitted for it too.
    host_full_width = [
        s
        for s in _library_steps(model, ours, on_core)
        if library_host_full_width(s, library)
    ]

    def call_of(step: dict[str, Any]) -> str:
        if step["kind"] == "region" and step["group"] not in ours:
            # A FUSED REGION NO KERNEL ANSWERS runs its members' own calls; a member whose readout the
            # machine routes to the core (the region's accumulator-leaving boundary) runs there.
            return " ".join(
                _host_call(m) if int(m["group"]) == int(step["group"]) and int(step["group"]) in on_core else _call(m, library)
                for m in step["members"]
            )
        if int(step["group"]) in on_core:
            return _host_call(step)
        return ours.get(step["group"]) or _call(step, library)

    calls = "\n".join(
        f"    fm_g = 0; t0 = read_cycles(); "
        f"{call_of(step)}"
        f" dt = read_cycles() - t0; total += dt;"
        f" fm_dt[{index}] = dt; fm_gather[{index}] = fm_g;"
        f" {'fm_authored' if step['group'] in ours else 'fm_vendor'} += dt;"
        f" {'fm_authored_groups' if step['group'] in ours else 'fm_vendor_groups'}++;"
        for index, step in enumerate(model["steps"])
    )
    # THE SAME DIGESTS, COMPUTED AFTER THE CLOCK STOPS. The output buffers are retained, so walking
    # them here reads exactly the bytes the measured pass produced: every `sum` and `fnv1a` is
    # identical to what an in-window pass would have printed, and only their TIMING moves. `dt` is
    # the cycle count taken inside the bracket and carried out unchanged, so the bracketed sum and
    # the whole-model window stay comparable run to run.
    # THE GATHER AND THE KERNEL, APART. Reported as their own line rather than as extra GM_GROUP
    # fields, so a receipt parser that knows the existing protocol keeps working unchanged.
    gather_split = "\n".join(
        f'    if (fm_gather[{index}]) printf("GM_SPLIT {step["group"]} gather=%llu kernel=%llu\\n",'
        f" (unsigned long long)fm_gather[{index}],"
        f" (unsigned long long)(fm_dt[{index}] - fm_gather[{index}]));"
        for index, step in enumerate(model["steps"])
    )
    # THE DEVICE COMPUTES THE REFERENCE ITSELF, so nothing has to be embedded. A bounded check
    # needs per-element expectations, and carrying 71 output buffers would roughly double a 25 MB
    # image. It is not needed: an elementwise sum's reference is a function of the same two inputs
    # and the same scales the device was handed, so the check recomputes it here -- after the
    # window, where its cost is not measured -- and reports the max absolute deviation and how many
    # elements exceed the declared bound. The exact ops keep `fnv1a`, which is the right instrument
    # for them and the only order-sensitive one available.
    bounded = "\n".join(_bounded_block(step, sizes) for step in model["steps"] if step["kind"] == "sum")
    # A FUSED REGION ending in a sum: its internal members restated on the core, then the sum's bound.
    region_sums = [s for s in model["steps"] if s["kind"] == "region" and s["boundary_kind"] == "sum"]
    if region_sums:
        bounded = "\n".join([b for b in [bounded, *(_region_bounded(s, sizes) for s in region_sums)] if b])
    fused = [step for step in model["steps"] if step["kind"] == "fused"]
    if fused:
        bounded = (
            "\n".join([bounded, *(_fused_check(step, sizes) for step in fused)])
            if bounded
            else "\n".join(_fused_check(step, sizes) for step in fused)
        )
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify is one of {VERIFY_MODES}, not {verify!r}")
    local_helpers, local_calls = (
        _local_checks(model, sizes, extra=[step["producer"] for step in fused], mapped=verify == "local_map")
        if verify in LOCAL_MODES or fused or region_sums
        else ("", "")
    )
    if verify not in LOCAL_MODES:
        # A merged group's producer reference needs the helpers; only a local mode prints the checks.
        local_calls = ""
    words_calls = ""
    if verify in WORDS_MODES:
        local_helpers = _WORDS_HELPER + local_helpers
        words_calls = "\n".join(
            '    printf("'
            + _printf("words", group=step.get("graded_as", step["group"]), bytes="%llu", digest="%llu")
            + f'\\n", (unsigned long long)sizeof({step["out"]}), words_digest({step["out"]}, sizeof({step["out"]})));'
            for step in model["steps"]
        )
    if verify in ("host_dump", "words"):
        bounded = ""
        verification = "\n".join(
            '    printf("'
            + _printf(
                "group",
                group=step.get("graded_as", step["group"]),
                kind=step["kind"],
                cycles="%llu",
                sum="UNKNOWN",
                checksum="UNKNOWN",
            )
            + f'\\n", (unsigned long long)fm_dt[{index}]);'
            for index, step in enumerate(model["steps"])
        )
    else:
        verification = ""
    verification = verification or "\n".join(
        '    printf("'
        + _printf(
            "group",
            group=step.get("graded_as", step["group"]),
            kind=step["kind"],
            cycles="%llu",
            sum="%lld",
            checksum="%lld",
        )
        + f'\\n", (unsigned long long)fm_dt[{index}],\n'
        + f"           sum_{'acc' if step.get('scale', 0) is None else 'elem'}"
        + f"({step['out']}, {sizes[step['out']]}),\n"
        + f"           checksum_{'acc' if step.get('scale', 0) is None else 'elem'}"
        + f"({step['out']}, {sizes[step['out']]}));"
        for index, step in enumerate(model["steps"])
    )
    final = model["steps"][-1]
    _fnv_offset, _fnv_prime, _digest_mask = _digest_constants()
    return f"""/* GENERATED by group_model_program.py -- one library call per compute group of a captured model. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include "include/gemmini_testutils.h"

#define BLOB(sym, file, type) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" file "\\"\\n.previous\\n"); \\
    extern const type sym[];
{blobs}

#define IMAGE ((const elem_t *)IMAGE_DATA)
static elem_t ONES[{window}] row_align(1);
{buffers}

/* Outside the timed window: two numbers per group, which the numpy emulation of the same steps
   reproduces, so the first group that differs is named and not searched for.

   `sum_*` is the additive total this program has always printed. It is kept ONLY so a new build can
   be compared with the FPGA runs already on record, and it is not what a cycle claim is admitted
   against: reorder a group's output and the sum does not move, which makes it blind to exactly the
   transpose / stride / layout failures a tensor compiler produces.

   `checksum_*` is FNV-1a over the output, each element sign-extended to a 64-bit word -- the same
   digest, from the same constants, the hardened per-layer route uses (`lb_fnv1a_words`). Widening
   to 64 bits keeps it independent of how wide elem_t and acc_t are, so one host function
   (`group_digest`) reproduces both. Same linear cost as the sum, and outside the timed window in
   any case. */
static long long sum_elem(const elem_t *v, size_t n) {{ long long s = 0; for (size_t i = 0; i < n; i++) s += v[i]; return s; }}
static long long sum_acc(const acc_t *v, size_t n) {{ long long s = 0; for (size_t i = 0; i < n; i++) s += v[i]; return s; }}
static long long checksum_elem(const elem_t *v, size_t n) {{
    uint64_t h = {_fnv_offset}ULL;
    for (size_t i = 0; i < n; i++) {{ h ^= (uint64_t)(int64_t)v[i]; h *= {_fnv_prime}ULL; }}
    return (long long)(h & {_digest_mask}ULL);
}}
static long long checksum_acc(const acc_t *v, size_t n) {{
    uint64_t h = {_fnv_offset}ULL;
    for (size_t i = 0; i < n; i++) {{ h ^= (uint64_t)(int64_t)v[i]; h *= {_fnv_prime}ULL; }}
    return (long long)(h & {_digest_mask}ULL);
}}

/* Our own schedules for the groups we can express; the rest keep their library call above. */
{definitions}{_HOST_ACC_MATMUL if on_core or host_full_width else ""}
{host_conv2d_block}
{VERIFY_BEGIN}
{local_helpers}
{VERIFY_END}

/* The bracket sum, split by who answered each group. `fm_im2col` is accumulated INSIDE an authored
   group's bracket by the gather that group's kernel required, so it is a part of `fm_authored`,
   never a fourth thing added beside it. The gather is HARNESS code, not the submission's: a better
   gather lowers this line, so it is a floor on that cost and not a verdict on the compiler. */
static uint64_t fm_authored, fm_vendor, fm_im2col;
static unsigned fm_authored_groups, fm_vendor_groups, fm_im2col_groups;
/* Each group's bracketed cycles, carried out of the measured pass so the digests that name them can
   be computed after the clock stops. A register read, not a print: it stays inside the bracket. */
static uint64_t fm_dt[{len(model["steps"])}];
/* The gather half of a group whose kernel reads a caller-materialized operand; zero for every
   other group. Both halves are inside the window -- both must happen for the model to compute its
   answer -- and they are reported apart because one is this harness's code and one is not. */
static uint64_t fm_gather[{len(model["steps"])}];
static uint64_t fm_g;

static uint64_t run(int measured) {{
    uint64_t total = 0, t0, dt;
    fm_authored = fm_vendor = fm_im2col = 0;
    fm_authored_groups = fm_vendor_groups = fm_im2col_groups = 0;
{calls}
    return total;
}}

int main(void) {{
    for (size_t i = 0; i < {window}; i++) ONES[i] = 1;
    gemmini_flush(0);
    printf("{_printf("invocations")}\\n");
    printf("{_printf("window_begin", label=window_label)}\\n");
    printf("{_printf("warm_begin")}\\n");
    run(0);
    printf("{_printf("warm_end")}\\n");
    printf("{_printf("measured_begin")}\\n");
    /* THE WHOLE-MODEL WINDOW OPENS HERE, AND NOT AT THE FIRST STATEMENT OF main().
       The reference artifact opens its window at main's first statement because its main holds
       exactly one model pass. This program runs a WARM-UP pass first (run(0) above), which that
       program has no equivalent of, so a window starting at the top of main would time two passes
       and report roughly twice the work under a name claiming one. The window opens at the
       measured pass instead.
       INSIDE:  every compute group, all inter-group setup and control flow, and the runtime
                dequantize + argmax below -- the steps the reference also counts.
       OUTSIDE: the warm-up pass, the golden/cosine comparison (this program's own check, which
                the reference has no equivalent of), and every printf, which the reference's own
                banner also keeps out.
       ONE SCOPE DIFFERENCE REMAINS AND IS NOT REMOVABLE HERE: this measured pass runs warm
       because a warm-up preceded it, and the reference's runs cold. That makes this number
       optimistic against theirs, and it has to be quoted with the difference, not without it. */
    uint64_t fm_start = read_cycles();
    uint64_t total = run(1);
    /* The model's output against the capture's own golden. Not byte equality: a residual sum on
       this unit rounds each operand, within the bound its group declares. */
    int got = 0, want = 0;
    volatile double sink = 0;
    for (int i = 0; i < {model["classes"]}; i++) {{
        sink = (double){final["out"]}[i] * {final["dequantize"]!r};  /* the runtime dequantize */
        if ({final["out"]}[i] > {final["out"]}[got]) got = i;
        if (GOLDEN[i] > GOLDEN[want]) want = i;
    }}
    uint64_t fm_end = read_cycles();  /* closes after the dequantize and the argmax */
    (void)sink;
    /* VERIFICATION, OUTSIDE THE WINDOW. Every per-group digest and every GM_GROUP line is produced
       here, after the clock has stopped, over the retained output buffers. Everything ABOVE this
       point is work a deployment performs; nothing below it is. */
{VERIFY_BEGIN}
{verification}
{gather_split}
{bounded}
{local_calls}
{words_calls}
{VERIFY_END}
    /* THE GOLDEN COMPARISON IS OUTSIDE THE WINDOW. It is this program's own check and the
       reference has no equivalent of it, so timing it would add work to a number meant to be
       comparable. It re-reads the same two arrays rather than sharing the loop above, which is
       what kept it inside the window until now. */
    double dot = 0, mine = 0, theirs = 0;
    for (int i = 0; i < {model["classes"]}; i++) {{
        double y = (double){final["out"]}[i] * {final["dequantize"]!r};
        dot += y * GOLDEN[i]; mine += y * y; theirs += (double)GOLDEN[i] * GOLDEN[i];
    }}
    double cosine = dot / (sqrt(mine) * sqrt(theirs) + 1e-30);
    printf("{_printf("full_model", cycles="%llu")}\\n", (unsigned long long)(fm_end - fm_start));
    printf("{_printf("bracket_sum", cycles="%llu")}\\n", (unsigned long long)total);
    printf("{
        _printf(
            "split",
            authored="%llu",
            authored_groups="%u",
            im2col="%llu",
            im2col_groups="%u",
            vendor="%llu",
            vendor_groups="%u",
        )
    }\\n",
           (unsigned long long)fm_authored, fm_authored_groups,
           (unsigned long long)fm_im2col, fm_im2col_groups,
           (unsigned long long)fm_vendor, fm_vendor_groups);
    printf("{_printf("uncounted", cycles="%llu")}\\n",
           (unsigned long long)((fm_end - fm_start) - total));
    printf("{_printf("argmax", got="%d", want="%d", agrees="%d")}\\n", got, want, got == want);
    printf("{_printf("cosine", ppm="%d")}\\n", (int)(cosine * 1000000.0));
    printf("{_printf("metric", cycles="%llu")}\\n", (unsigned long long)total);
    printf("{_printf("measured_end")}\\n");
    printf("{_printf("window_end", label=window_label)}\\n");
    return 0;
}}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_header_sha256(vendor: Path | None, includes: Sequence[str]) -> str:
    """The digest of the parameter header the program compiled against, found where it was included.

    The first include root holding ``include/gemmini_params.h`` is the one the preprocessor reads, so it
    is the one recorded; with no such root the vendor tree's own copy, and ``UNKNOWN`` when neither
    exists rather than a digest of nothing.
    """
    roots = [Path(flag[2:]) for flag in includes if flag.startswith("-I")]
    if vendor is not None:
        roots.append(Path(vendor))
    for root in roots:
        header = root / "include" / "gemmini_params.h"
        if header.is_file():
            return _sha256(header)
    return "UNKNOWN: no include root holds include/gemmini_params.h"


def _compiler_provenance() -> dict[str, Any]:
    """Which lowering and schedule sources this build actually IMPORTED, by content.

    A result recorded without this cannot say what compiled it. Measured 2026-09-18: tracing the
    25.4M-cycle ResNet-50 result back to its compiler took an exhaustive scan of every file in the
    repo, and it succeeded only because one emitted recipe name happened to be unique to one
    worktree -- nothing anywhere recorded the ``PYTHONPATH``, the package path or a source digest.
    The same model compiled from two checkouts is two different programs, and the manifest said
    nothing that would tell them apart.

    Recorded as the FILES PYTHON RESOLVED, not as a path we expect it to have resolved, so a
    shadowing checkout or a dirty tree shows up as a different digest rather than as the same one.
    """
    import merlin
    from merlin.common import provenance as PROV

    resolved: dict[str, Path] = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("merlin.") or module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin and Path(origin).is_file():
            resolved[name] = Path(origin).resolve()
    try:
        digest = PROV.source_digest(sorted(resolved.values()))
    except Exception as exc:  # noqa: BLE001 -- an undeterminable digest is recorded, never omitted
        digest = f"UNKNOWN: {type(exc).__name__}: {exc}"
    return {
        "merlin_package": str(Path(merlin.__file__).resolve().parent),
        "source_digest": digest,
        "modules": {name: _sha256(path) for name, path in resolved.items()},
        "python": sys.executable,
    }


def build(
    model: dict[str, Any],
    vendor: Path,
    compiler: Path,
    out: Path,
    *,
    sched_kernels: dict[str, Any] | None = None,
    window_label: str = DEFAULT_WINDOW_LABEL,
    extra_objects: Sequence[Path] | None = None,
    recipe: Any = None,
    verify: str = "on_target",
    library_loops: bool = True,
    prohibited_selectors: Sequence[int] = (),
) -> dict[str, Any]:
    """``extra_objects`` are relocatables LINKED IN rather than compiled here.

    ``library_loops=False`` routes every group the library answers to a path that issues no
    loop-descriptor instruction (see :func:`library_path_without_loops`); the receipt says which.

    ``recipe`` is the target's own harness build recipe (``runtime.backends.base.harness_build_recipe``):
    its compiler, include roots, support sources, link script and flags replace ``vendor``/``compiler``
    and this file's ``CFLAGS``, so the program is built exactly the way the capsule harness it is
    compared against is built. Without one the historical ``vendor`` layout and flags are used.

    An arm measuring a submission's own code cannot compile that code from this file: the kernel is
    the backend's artifact, and a transcription written here would put a number on instructions no
    oracle ever graded (see ``merlin.llvmlower.device_shim``, which refuses the same thing for the
    same reason). The package emits each group's object through its own declared entrypoint, and
    this links them. Their digests go in the receipt, so a result names the bytes that produced it.
    """
    # BEFORE ANY WORK. A named object that is not there is a broken arm, not a slow one, and the
    # cheapest place to say so is before a compile whose failure would be reported as the problem.
    supplied = [Path(o) for o in (extra_objects or [])]
    missing = [o for o in supplied if not o.is_file()]
    if missing:
        raise SystemExit(f"extra objects named but not present: {', '.join(str(o) for o in missing)}")
    out.mkdir(parents=True, exist_ok=True)
    for name, array in model["arrays"].items():
        (out / f"{name}.bin").write_bytes(array.tobytes())
    program, elf = out / "group_model_program.c", out / "group_model_program.elf"
    if recipe is not None:
        header_includes = [f"-I{root}" for root in recipe.include_roots]
    else:
        header_includes = [f"-I{vendor / 'riscv-tests'}", f"-I{vendor}"]
    header_path = _parameter_header(vendor, header_includes)
    loop_free = None
    if not library_loops and prohibited_selectors:
        loop_free = _trap_prohibited_in_library(header_includes, vendor, prohibited_selectors, out.parent)
    library = (
        None
        if library_loops
        else library_path_without_loops(header_path.read_text(encoding="utf-8") if header_path is not None else "")
    )
    # The groups whose kernel is the package's: a library fact never routes one of them (see
    # `library_host_readout_routing`); the machine's own readout facts still do.
    answered = {int(g) for g in ((sched_kernels or {}).get("calls") or {})}
    routed = core_readout_routing(model, full_width_readout(header_path), library, answered=answered)
    program.write_text(
        render(
            model,
            sched_kernels=sched_kernels,
            window_label=window_label,
            verify=verify,
            host_readout=[r["group"] for r in routed],
            library=library,
        ),
        encoding="utf-8",
    )
    if recipe is not None:
        compiler = Path(recipe.compiler)
        includes = [f"-I{root}" for root in recipe.include_roots]
        supports = [Path(source) for source in recipe.support_sources]
        link_script = Path(recipe.link_script)
        compile_flags = (*recipe.cflags, *PROGRAM_CFLAGS)
        link_flags = (*recipe.cflags, *recipe.ldflags)
    else:
        common = vendor / "riscv-tests" / "benchmarks" / "common"
        includes = [f"-I{vendor / 'riscv-tests'}", f"-I{vendor / 'riscv-tests' / 'env'}", f"-I{vendor}", f"-I{common}"]
        supports = [common / "syscalls.c", common / "crt.S"]
        link_script = common / "test.ld"
        compile_flags = link_flags = CFLAGS
    # Two phases with NAMED objects. A one-step compile-and-link records the driver's temporary
    # object names in the executable, so two builds of identical sources differ, and a measurement
    # could not be tied to a rebuild of its own program.
    log, objects = [], []
    for source in (program, *supports):
        unit = out / f"{source.stem}.o"
        done = subprocess.run(
            [str(compiler), *compile_flags, *includes, f"-Wa,-I{out}", "-c", str(source), "-o", str(unit)],
            capture_output=True,
            text=True,
            cwd=out,
        )
        log.append(done.stdout + done.stderr)
        if done.returncode != 0:
            (out / "build.log").write_text("".join(log), encoding="utf-8")
            raise SystemExit(f"compile of {source.name} failed (see {out / 'build.log'}):\n{done.stderr[-2000:]}")
        objects.append(str(unit))
    objects.extend(str(o) for o in supplied)
    done = subprocess.run(
        [str(compiler), *link_flags, "-T", str(link_script), *objects, "-o", str(elf)],
        capture_output=True,
        text=True,
        cwd=out,
    )
    (out / "build.log").write_text("".join(log) + done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise SystemExit(f"link failed (see {out / 'build.log'}):\n{done.stderr[-2000:]}")
    return {
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        # The program this build rendered and its relocatable, NAMED here so a caller reading the
        # receipt (a header check, a batch that links variants) never assumes this driver's file names.
        "program_source": str(program),
        "program_object": str(out / f"{program.stem}.o"),
        "support_objects": [str(out / f"{Path(s).stem}.o") for s in supports],
        "program_sha256": _sha256(program),
        "parameter_header_sha256": _parameter_header_sha256(vendor, includes),
        "compiler": str(compiler),
        "flags": list(compile_flags),
        "link_flags": list(link_flags),
        "link_script": str(link_script),
        "includes": includes,
        # Named by CONTENT, not by path: an arm's claim is about the bytes that were linked, and a
        # path is the one thing a rebuild can keep while the bytes change underneath it.
        "linked_objects": [{"path": str(o), "sha256": _sha256(o)} for o in supplied],
        # Groups whose full-width accumulator readout this machine lacks, computed on the core instead.
        "host_routed": routed,
        "library_paths": library,
        "loop_free_header": loop_free,
    }


#: The two lines a BATCH program prints around each variant it runs, so one console demultiplexes into
#: one console per variant. Each variant's own protocol lines sit between them, unchanged.
BATCH_UART = {
    "variant_begin": "MERLIN_VARIANT begin index={index} package={package}",
    "variant_end": "MERLIN_VARIANT end index={index}",
}


def _tool(compiler: Path, name: str) -> Path:
    """A sibling binutils tool of ``compiler`` (``<prefix>-gcc`` -> ``<prefix>-<name>``)."""
    stem = compiler.name
    if not stem.endswith("gcc"):
        raise ValueError(f"cannot derive {name} from compiler {compiler}")
    return compiler.with_name(stem[: -len("gcc")] + name)


def link_batch(
    variants: Sequence[dict[str, Any]], out: Path, *, compiler: Path, compile_flags, link_flags, link_script, supports
) -> dict[str, Any]:
    """ONE bare-metal ELF that runs several whole-model programs in sequence, each fully isolated.

    Each variant is its program object plus the kernel objects it links, merged into one relocatable
    (``gcc -r``); every symbol that relocatable DEFINES is then made local and its ``main`` renamed,
    so two variants with identically named kernels, weights and buffers link side by side without
    aliasing -- each keeps its own copies of everything it writes. A small driver ``main`` prints a
    ``MERLIN_VARIANT begin`` line, calls the variant, prints ``MERLIN_VARIANT end``, and moves on.
    Every variant's own ``main`` starts the way a fresh program does (it flushes the accelerator and
    re-issues its configuration), so nothing a previous variant configured is relied on.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compiler = Path(compiler)
    objcopy, nm = _tool(compiler, "objcopy"), _tool(compiler, "nm")
    merged, calls, declarations = [], [], []
    for index, variant in enumerate(variants, 1):
        rel = out / f"variant_{index}.o"
        done = subprocess.run(
            [
                str(compiler),
                "-r",
                "-nostdlib",
                str(variant["program_object"]),
                *map(str, variant["objects"]),
                "-o",
                str(rel),
            ],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise SystemExit(f"variant {index} did not merge: {done.stderr[-1500:]}")
        listed = subprocess.run([str(nm), "--defined-only", "-g", str(rel)], capture_output=True, text=True, check=True)
        names = sorted({line.split()[-1] for line in listed.stdout.splitlines() if len(line.split()) >= 3})
        if "main" not in names:
            raise SystemExit(f"variant {index} defines no main")
        entry = f"merlin_variant_{index}_main"
        rewrite = [f"--redefine-sym=main={entry}", *(f"--localize-symbol={n}" for n in names if n != "main")]
        subprocess.run([str(objcopy), *rewrite, str(rel)], check=True, capture_output=True)
        merged.append(str(rel))
        declarations.append(f"int {entry}(void);")
        begin = BATCH_UART["variant_begin"].format(index=index, package=variant["label"])
        end = BATCH_UART["variant_end"].format(index=index)
        calls.append(f'    printf("{begin}\\n");\n    {entry}();\n    printf("{end}\\n");')
    driver = out / "batch_main.c"
    driver.write_text(
        "#include <stdio.h>\n"
        + "\n".join(declarations)
        + "\nint main(void) {\n"
        + "\n".join(calls)
        + "\n    return 0;\n}\n",
        encoding="utf-8",
    )
    unit = out / "batch_main.o"
    subprocess.run([str(compiler), *compile_flags, "-c", str(driver), "-o", str(unit)], check=True, capture_output=True)
    elf = out / "batch.elf"
    done = subprocess.run(
        [
            str(compiler),
            *link_flags,
            *(["-T", str(link_script)] if link_script else []),
            *map(str, supports),
            str(unit),
            *merged,
            "-o",
            str(elf),
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"the batch did not link: {done.stderr[-2000:]}")
    return {
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        "variants": [v["label"] for v in variants],
        "driver": str(driver),
    }


def split_batch(console: str, count: int) -> dict[int, str]:
    """``{index: that variant's console}``; a variant without both markers is ABSENT, never empty-but-present."""
    begin_prefix = BATCH_UART["variant_begin"].split("{index}")[0]
    end_prefix = BATCH_UART["variant_end"].split("{index}")[0]
    blocks: dict[int, list[str]] = {}
    closed: set[int] = set()
    current = None
    for line in console.splitlines():
        stripped = line.strip()
        if stripped.startswith(begin_prefix):
            token = stripped[len(begin_prefix) :].split()[0]
            current = int(token)
            blocks[current] = []
            continue
        if stripped.startswith(end_prefix) and current is not None:
            if int(stripped[len(end_prefix) :].split()[0]) == current:
                closed.add(current)
            current = None
            continue
        if current is not None:
            blocks[current].append(line)
    return {index: "\n".join(blocks[index]) + "\n" for index in range(1, count + 1) if index in closed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", required=True, type=Path, help="a capture directory (linalg.mlir, weights, ...)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--vendor-source", required=True, type=Path, help="a snapshot's source/ directory")
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--schedules",
        choices=["vendor", "ours", "ours-explicit", "ours-overlap"],
        default="vendor",
        help="'ours' renders each group our recipes can express as our own schedule, keeping the "
        "library call for the rest; the census records which group went which way. 'ours-explicit' "
        "is the same, with every group -- convolution, contraction, residual add and mean -- stated "
        "as a sequencer-free loop nest where that recipe can state it. 'ours-overlap' is "
        "'ours-explicit' with a software pipeline asked for wherever the nest can state one: two "
        "operand tiles resident, each tile's staging issued before the other's contraction, over the "
        "same tiles and the same instructions as 'ours-explicit' in a different order",
    )
    parser.add_argument(
        "--window-label",
        default=DEFAULT_WINDOW_LABEL,
        help="the label of the one measured window this program frames; a UART validation policy "
        "declares the same label, which is what binds a cycle number to what it had to print",
    )
    args = parser.parse_args(argv)
    try:
        model = extract(args.capture, args.target)
    except NotClosed as refusal:
        print(f"not built: {refusal}", file=sys.stderr)
        return 2
    sched_kernels = None
    if args.schedules.startswith("ours"):
        # A sibling module of this provider, loaded by its own path: no ambient sys.path change, so it
        # resolves the same whether this file is imported by the builder or run directly.
        import importlib.util

        where = Path(__file__).resolve().with_name("group_model_sched_kernels.py")
        spec = importlib.util.spec_from_file_location("group_model_sched_kernels", where)
        sibling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sibling)
        render_kernels, summarize = sibling.render_kernels, sibling.summarize

        family = {"ours-explicit": "explicit", "ours-overlap": "overlap"}.get(args.schedules, "sequencer")
        sched_kernels = render_kernels(model, target=args.target, recipes=family)
        print(summarize(sched_kernels["census"]))
    receipt = build(
        model,
        args.vendor_source,
        args.compiler,
        args.out,
        sched_kernels=sched_kernels,
        window_label=args.window_label,
    )
    kinds: dict[str, int] = {}
    for step in model["steps"]:
        kinds[step["kind"]] = kinds.get(step["kind"], 0) + 1
    manifest = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "capture_linalg_sha256": _sha256(args.capture / "linalg.mlir"),
        "target": args.target,
        "groups": model["groups"],
        "device_groups": model["device_groups"],
        "steps_by_kind": kinds,
        "weight_bytes": sum(int(a.nbytes) for n, a in model["arrays"].items() if n.startswith("W_")),
        # Which groups ran OUR schedule and which kept the library call. A cycle total for this
        # program is only readable next to this: a mixture reported as one schedule is the way a
        # measurement gets attributed to work it did not do.
        "schedules": args.schedules,
        "schedule_census": (sched_kernels or {}).get("census"),
        # The window this program frames. A cycle claim is sealed against the policy entry carrying
        # this same label, so the two agreeing is a content fact and not a convention.
        "window_label": args.window_label,
        # WHAT COMPILED THIS. Recorded last, after every import this build needed has happened, so
        # the record is of the modules actually resolved rather than of the ones a path suggests.
        "compiler_provenance": _compiler_provenance(),
        "final_dequantize_f32_bits": struct.unpack("<I", struct.pack("<f", model["steps"][-1]["dequantize"]))[0],
        "sums": [
            {k: s[k] for k in ("group", "rows", "cols", "lhs_load", "rhs_load", "readout", "bound_lsb")}
            for s in model["steps"]
            if s["kind"] == "sum"
        ],
        **receipt,
    }
    (args.out / "group_model_program.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"built {receipt['elf']}: {model['device_groups']} device group(s) of {model['groups']}, {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
