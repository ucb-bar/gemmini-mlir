"""This target's schedule instruction set (``merlin.sched.isa``) and its reference matmul schedule.

Each schedule instruction IS one of the curated harness header's macros, and nothing about it is
restated here:

- its operand list is that macro's parameter list, read from the header (a header whose parameters
  differ from the binding below is refused, so an edited header can never be emitted against stale
  operand names);
- every bit-field range the checker enforces is the header's own packing expression evaluated on the
  concrete operands, so a value that would spill into its neighbouring field is refused exactly where the
  hardware would mis-decode it;
- capacities come from the params header, and the per-loop scratchpad/accumulator partition from the
  double-buffer split the header's own tilers state (every statement of it must agree). That split is
  the RTL's: ``LoopMatmul`` runs two concurrent loops, each owning half the scratchpad and half the
  accumulator.

The one thing a C macro cannot say is which parameter is a pointer, a flag or a float; ``_BINDINGS``
says that, once.

Cross-instruction rules the checker enforces (all read from ``LoopMatmul.scala``):

- the LOOP_WS FSM issues its own mvin/mvout with the row strides of the most recent ``config_ld``
  (ids 0/1/2 for A/B/D) and ``config_st``; they must equal the loop's element strides in bytes;
- the FSM moves to the other accumulator half only after a loop whose C pointer is non-NULL. So a loop
  that withholds C (a partial sum over one k-tile) must be followed by a loop on the SAME output tile
  that accumulates (``ex_accumulate``) without reloading D, and the kernel must end with every partial
  stored. A fresh tile with no D must not accumulate: it would add onto whatever the half held.
- the convolution sequencer (``LoopConv.scala``) has the same rule with one difference: it has no
  ``ex_accumulate`` operand because it ALWAYS accumulates, and what initialises the half is the bias
  mvin, which is skipped outright when the bias pointer is NULL (and writes zeros, rather than the bias,
  when ``no_bias`` is set). So a convolution reduction chain is: the first descriptor carries the bias,
  the later ones carry NULL, only the last carries the output, and their depths add up to the layer's
  input channels (``_conv_reduction_chain``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from merlin.sched.ir import NULL, Kernel, Ptr, TensorArg, Var, add, as_expr, call, loop, mul, select
from merlin.sched.isa import InstrDef, InstructionSet, IsaError, Operand
from merlin.targetgen.capability_discovery import _balanced_end, _split_top_level, parse_c_header

from . import gemmini_conv_downsample as _DS
from .gemmini_loop_conv import UnsupportedNativeConv, _expression

#: IR instruction -> (header macro, operand kinds). Keys must equal the macro's parameters, in order.
_BINDINGS: dict[str, tuple[str, dict[str, str]]] = {
    "config_ex": (
        "gemmini_extended_config_ex",
        {
            "dataflow": "int",
            "sys_act": "int",
            "sys_shift": "int",
            "A_stride": "int",
            "A_transpose": "flag",
            "B_transpose": "flag",
        },
    ),
    "config_st": ("gemmini_extended_config_st", {"stride": "int", "acc_act": "int", "acc_scale": "float"}),
    "config_ld": ("gemmini_extended3_config_ld", {"stride": "int", "scale": "float", "shrunk": "flag", "id": "int"}),
    "loop_ws": (
        "gemmini_loop_ws",
        {
            "I": "int",
            "J": "int",
            "K": "int",
            "pad_I": "int",
            "pad_J": "int",
            "pad_K": "int",
            "A": "ptr",
            "B": "ptr",
            "D": "ptr",
            "C": "ptr",
            "A_stride": "int",
            "B_stride": "int",
            "D_stride": "int",
            "C_stride": "int",
            "A_transpose": "flag",
            "B_transpose": "flag",
            "full_C": "flag",
            "low_D": "flag",
            "ex_accumulate": "flag",
            "act": "int",
            "a_spad_id": "int",
            "b_spad_id": "int",
            "is_resadd": "flag",
        },
    ),
    # The device-side convolution sequencer. It is the instruction the whole-model emission never
    # issues: measured on ResNet-50, the program used 8 of the 25 functs the RTL declares, and the 17
    # it never emitted are this family (the loop plus its six config words), so every convolution's
    # im2col patch generation ran as host scalar code. Its operands are the loop's whole problem --
    # the layer's geometry, the tile it walks, and the four pointers -- because the FSM issues its own
    # mvin/mvout rather than being fed them.
    "loop_conv_ws": (
        "gemmini_loop_conv_ws",
        {
            "batch_size": "int",
            "in_row_dim": "int",
            "in_col_dim": "int",
            "in_channels": "int",
            "out_channels": "int",
            "out_row_dim": "int",
            "out_col_dim": "int",
            "pool_out_row_dim": "int",
            "pool_out_col_dim": "int",
            "stride": "int",
            "padding": "int",
            "kernel_dim": "int",
            "kernel_dilation": "int",
            "pool_size": "int",
            "pool_stride": "int",
            "pool_padding": "int",
            "batches": "int",
            "porows": "int",
            "pocols": "int",
            "pochs": "int",
            "krows": "int",
            "kcols": "int",
            "kchs": "int",
            "lpad": "int",
            "rpad": "int",
            "upad": "int",
            "dpad": "int",
            "plpad": "int",
            "prpad": "int",
            "pupad": "int",
            "pdpad": "int",
            "orows": "int",
            "ocols": "int",
            "weights": "ptr",
            "output": "ptr",
            "bias": "ptr",
            "input": "ptr",
            "no_bias": "flag",
            "no_pool": "flag",
            "downsample": "flag",
            "wrot180": "flag",
            "input_dilated": "flag",
            "activation": "int",
            "trans_output_1203": "flag",
            "trans_weight_1203": "flag",
            "trans_weight_0132": "flag",
            "trans_input_3120": "flag",
            "max_pixels_per_row": "int",
            "in_stride": "int",
            "weight_stride": "int",
            "out_stride": "int",
            "dw": "flag",
            "a_spad_id": "int",
            "b_spad_id": "int",
        },
    ),
}

_INT_TYPES = {"int8_t": "i8", "int16_t": "i16", "int32_t": "i32"}
_DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4}


def _int_macro(model, name: str) -> int:
    m = model.macro(name)
    if m is None or m.int_value is None:
        raise IsaError(f"{model.path}: states no integer {name}")
    return m.int_value


def _readout_pooling(target: str | None) -> bool | None:
    """Whether the store path this target ELABORATED can pool, from that target's own RTL facts.

    Not a header fact. The parameters header states the accumulator's read widths, but the store
    controller's pooling gate is a build-time parameter of the generator (``pooling_is_enabled``), so a
    unit whose ISA encodes a pooling descriptor may still have been built without the datapath that
    honours it -- and then it stores the UNPOOLED rows instead of refusing. The fact extraction reads
    that gate structurally out of the elaborated FIRRTL; only a literal derived boolean answers here.
    An absent fact, an UNKNOWN extraction or a human declaration returns ``None``, and both the recipe
    and the checker refuse a pooled readout by that name rather than guessing.
    """
    if target is None:
        return None
    try:
        from merlin.targetgen.rtl.facts import load_facts

        record = load_facts(target)
    except Exception:  # noqa: BLE001 -- an unavailable fact is "not derived", never "assume yes"
        return None
    interfaces = (record.get("facts") or {}).get("interfaces") or []
    entry = next((i for i in interfaces if i.get("name") == "elaborated_rtl_features"), None)
    if entry is None or entry.get("status") != "derived":
        return None
    value = (entry.get("features") or {}).get("max_pool")
    return value if isinstance(value, bool) else None


def _machine_crosscheck(facts: dict[str, Any], target: str | None) -> str:
    """Hold the geometry these headers state to the machine derived from this target's RTL facts.

    The comparison itself is not this target's business and does not live here: it is
    ``merlin.sched.mach.crosscheck``, which names no target and is covered by the no-target-name gate.
    What IS this target's business is the translation -- these headers spell the three axes `dim`,
    `spad_rows` and `acc_rows`, and another target will spell them its own way. That mapping is the only
    target-specific part, so it is the only part that stays.
    """
    from merlin.sched.mach.crosscheck import GeometryDisagreement, crosscheck_declared_geometry

    declared = {
        "block": facts["dim"],
        "operand_rows": facts["spad_rows"],
        "accumulator_rows": facts["acc_rows"],
    }
    try:
        return crosscheck_declared_geometry(declared, target)
    except GeometryDisagreement as exc:
        # The shared comparison speaks the shared vocabulary, so the refusal names `block` where these
        # headers say DIM. Carrying the translation into the message is what lets a reader go straight
        # to the declaration that disagrees instead of working out which field that was.
        raise IsaError(
            f"{exc} These headers spell block / operand_rows / accumulator_rows as DIM / BANK_NUM*BANK_ROWS / ACC_ROWS."
        ) from exc


@lru_cache(maxsize=4)
def schedule_facts(header: str, params: str, *, target: str | None = None) -> dict[str, Any]:
    """Capacities, element types, partition and constants, each read from the two headers.

    When ``target`` is given, the geometry among them is cross-checked against the machine derived from
    that target's RTL facts; see :func:`_machine_crosscheck` for why a disagreement raises.
    """
    hdr_path, prm_path = Path(header), Path(params)
    hdr, prm = parse_c_header(hdr_path), parse_c_header(prm_path)
    ints = {n: _int_macro(prm, n) for n in ("DIM", "ADDR_LEN", "BANK_NUM", "BANK_ROWS", "ACC_ROWS", "MAX_BYTES")}
    try:
        for n in ("MAX_BLOCK_LEN", "MAX_BLOCK_LEN_ACC"):
            m = prm.macro(n)
            if m is None:
                raise IsaError(f"{prm_path}: states no {n}")
            ints[n] = _expression(m.body.strip(), ints)
        text = hdr_path.read_text(encoding="utf-8")
        partition = {}
        for name in ("max_spad_rows", "max_acc_rows"):
            values = {
                _expression(line.split("=", 1)[1].split(";", 1)[0].strip(), ints)
                for line in text.splitlines()
                if line.strip().startswith(f"const int {name} =")
            }
            if len(values) != 1:
                raise IsaError(f"{hdr_path}: {name} partition is stated {len(values)} different ways")
            partition[name] = values.pop()
    except UnsupportedNativeConv as exc:
        raise IsaError(str(exc)) from exc
    types = {alias: under for alias, under, _ in prm.typedefs}
    elem, acc = _INT_TYPES.get(types.get("elem_t", "")), _INT_TYPES.get(types.get("acc_t", ""))
    if elem is None or acc is None:
        raise IsaError(f"{prm_path}: unsupported element types {types.get('elem_t')}/{types.get('acc_t')}")
    facts = {
        "dim": ints["DIM"],
        "addr_len": ints["ADDR_LEN"],
        "spad_rows": ints["BANK_NUM"] * ints["BANK_ROWS"],
        "acc_rows": ints["ACC_ROWS"],
        "max_block_len": ints["MAX_BLOCK_LEN"],
        "max_block_len_acc": ints["MAX_BLOCK_LEN_ACC"],
        "spad_rows_per_loop": partition["max_spad_rows"],
        "acc_rows_per_loop": partition["max_acc_rows"],
        "elem_dtype": elem,
        "acc_dtype": acc,
        "elem_bytes": _DTYPE_BYTES[elem],
        "acc_bytes": _DTYPE_BYTES[acc],
        "constants": {n: _int_macro(hdr, n) for n in ("WEIGHT_STATIONARY", "NO_ACTIVATION", "RELU")},
    }
    # Two READOUT capabilities, each read from the target's own sources rather than assumed present.
    #
    # The full-width accumulator readout is the one `ACC_READ_FULL_WIDTH` states: the generator emits
    # that macro into THIS parameters header exactly when it built the accumulator's full-width read
    # port, which is the port a `full_C` mvout reads through (the narrow port is the one the activation
    # and the accumulator scale are applied on). Absent -> False, and a full-width readout is refused
    # by that name rather than emitted against a unit that would answer it with narrow data.
    facts["acc_read_full_width"] = prm.macro("ACC_READ_FULL_WIDTH") is not None
    # The convolution header's own text, kept so the strided-load predicate can be READ from the
    # library rather than restated here. A predicate transcribed once is a predicate that drifts, and
    # this one decides whether a layer may be loaded at four times less traffic.
    facts["conv_header_text"] = text
    facts["readout_pooling"] = _readout_pooling(target)
    facts["machine_crosscheck"] = _machine_crosscheck(facts, target)
    return facts


def _rocc_packings(body: str) -> list[tuple[str, str, str]]:
    """(funct name, rs1 expression, rs2 expression) of each RoCC instruction a macro body issues."""
    marker, rest, out = "ROCC_INSTRUCTION_RS1_RS2(", body, []
    while marker in rest:
        start = rest.index(marker) + len(marker) - 1
        end = _balanced_end(rest, start)
        args = _split_top_level(rest[start + 1 : end - 1], ",")
        if len(args) != 4:
            raise IsaError("unsupported RoCC invocation in header macro")
        out.append((args[3].strip(), args[1].strip(), args[2].strip()))
        rest = rest[end:]
    if not out:
        raise IsaError("header macro issues no RoCC instruction")
    return out


def _legal_functs(target: str | None) -> tuple[int, ...]:
    """Funct codes admitted by this target's RTL decode table, or none if unavailable."""
    if target is None:
        return ()
    try:
        from merlin.targetgen.rtl.facts import load_facts

        record = load_facts(target)
    except Exception:  # noqa: BLE001 -- unavailable evidence never licenses an instruction
        return ()
    for entry in (record.get("facts") or {}).get("interfaces") or []:
        legal = entry.get("legal_funct")
        if legal:
            return tuple(int(f) for f in legal)
    return ()


# Only these declared semantic classes take a bare register operand as a DRAM pointer.
# Header packing alone cannot distinguish a pointer from a bare stride or flag.
_DRAM_OPERAND_CLASSES = frozenset({"MVIN", "MVIN2", "MVIN3", "MVOUT"})


def _semantic_classes(target: str | None) -> dict[int, str]:
    """Funct-to-class declarations from the selected target capability manifest."""
    if not target:
        return {}
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest

        declared = (load_capability_manifest(str(target)).encoding or {}).get("semantic_class") or {}
    except Exception:  # noqa: BLE001 -- missing declaration contributes no inferred semantics
        return {}
    return {int(k): str(v) for k, v in declared.items()}


def _derived_bindings(
    hdr, legal_functs: Collection[int], classes: Mapping[int, str] | None = None
) -> dict[str, tuple[str, dict[str, str]]]:
    """Derive non-curated single-RoCC macros admitted by the target's RTL.

    The hand bindings carry semantics the header cannot express. Other instructions are admitted
    only when the macro's funct is present in the RTL legal set, and their operand kinds come from
    the packing and declared semantic class. Multi-instruction macros remain sequences, not ISA ops.
    """
    codes = {m.name: m.int_value for m in hdr.macros if not m.is_function and m.int_value is not None}
    legal = {int(f) for f in legal_functs}
    out: dict[str, tuple[str, dict[str, str]]] = {}
    for macro in hdr.macros:
        if not macro.is_function or not macro.params or macro.name in {binding[0] for binding in _BINDINGS.values()}:
            continue
        try:
            packings = _rocc_packings(macro.body)
        except IsaError:
            continue
        if len(packings) != 1:
            continue
        funct, rs1, rs2 = packings[0]
        code = codes.get(funct)
        if code is None or code not in legal:
            continue
        bare = (rs1.strip(), rs2.strip())
        semantic_class = (classes or {}).get(code)
        addresses = semantic_class in _DRAM_OPERAND_CLASSES if semantic_class is not None else True
        kinds = {parameter: ("ptr" if addresses and parameter in bare else "int") for parameter in macro.params}
        out[macro.name] = (macro.name, kinds)
    return out


def _numeric(values: Mapping[str, Any]) -> dict[str, int]:
    """Operand values as the integers a packing expression sees; a pointer fills a whole register, so
    its field check is vacuous and it is evaluated as 0."""
    return {k: (0 if v is None or not isinstance(v, int) else v) for k, v in values.items()}


def _extent(rows: int, cols: int, stride: int, esize: int) -> int:
    return ((rows - 1) * stride + cols) * esize if rows > 0 and cols > 0 else 0


def conv_working_rows(
    *,
    acc: bool,
    stride: int,
    batches: int,
    porows: int,
    pocols: int,
    pochs: int,
    krows: int,
    kcols: int,
    kchs: int,
    pool_size: int,
    pool_stride: int,
    dim: int,
    downsample: int = 0,
) -> int:
    """The scratchpad (or accumulator) rows ONE convolution descriptor occupies.

    This is ``tiled_conv_total_spad_rows`` in the curated harness header, for the modes this instruction
    models (no dilation, no transposed operand) -- transcribed rather than reasoned out, because it is
    the arithmetic the RTL's own address generation uses and a tile that overruns it is a silently wrong
    output, not a refusal. ``dim`` and the two capacities it is compared against are read from the
    target's headers, never written here.

    ``downsample`` is the header's own term. With the strided load the scratchpad holds the HALVED
    window -- the execute stage addresses it as ``irows >> downsample`` / ``icols >> downsample`` -- so
    pricing the unstrided window here would refuse tiles the device would hold, which is the same
    mistake as overrunning it, pointed the other way.
    """
    orows = porows * pool_stride + pool_size - 1
    ocols = pocols * pool_stride + pool_size - 1
    irows = (orows * stride + krows - 1) >> downsample
    icols = (ocols * stride + kcols - 1) >> downsample

    def per_bank(channels: int) -> int:
        return -(-channels // dim)

    if acc:
        return per_bank(pochs) * batches * orows * ocols
    return per_bank(kchs) * batches * irows * icols + per_bank(pochs) * kcols * krows * kchs


#: Everything about a convolution descriptor that names WHICH output tile it accumulates into. Two
#: descriptors of one reduction chain must agree on all of it; what they are allowed to differ in is the
#: reduction depth, the three pointers that move with it, and which of the bias and output they carry.
_CONV_OUTPUT_TILE_FIELDS = (
    "batch_size",
    "in_row_dim",
    "in_col_dim",
    "in_channels",
    "out_channels",
    "out_row_dim",
    "out_col_dim",
    "pool_out_row_dim",
    "pool_out_col_dim",
    "pool_size",
    "pool_stride",
    "pool_padding",
    "no_pool",
    "stride",
    "padding",
    "kernel_dim",
    "batches",
    "porows",
    "pocols",
    "pochs",
    "orows",
    "ocols",
    "lpad",
    "rpad",
    "upad",
    "dpad",
    "plpad",
    "prpad",
    "pupad",
    "pdpad",
    "activation",
    "in_stride",
    "weight_stride",
    "out_stride",
)

#: Everything a descriptor says about WHICH input window the loader stages and WHERE in the scratchpad
#: half it lands. Read off `LoopConvLdInput`'s own address generation and `LoopConvExecute`'s A address:
#: the loader writes `addr_start + (ich/DIM)*input_spad_stride + b*(irows>>ds)*(icols>>ds) +
#: (irow_padded>>ds)*(icols>>ds) + (icol_padded>>ds)` and the execute stage reads the same expression
#: with its own `kch`/`irow`/`icol`, where `input_spad_stride = batches*(irows>>ds)*(icols>>ds)` and
#: `irows`/`icols` come from `orows`/`ocols`/`stride`/`krows`/`kcols`/the four pads.
#:
#: `pochs` IS DELIBERATELY ABSENT, and that absence is the whole of why this mode buys anything: nothing
#: in either address depends on the output-channel extent, so one staged input serves every output-channel
#: tile of the same window. `kchs` IS present (it is `ichs`, the loader's channel bound).
_CONV_INPUT_RESIDENCY_FIELDS = (
    "batch_size",
    "in_row_dim",
    "in_col_dim",
    "in_channels",
    "in_stride",
    "stride",
    "padding",
    "kernel_dim",
    "kernel_dilation",
    "batches",
    "orows",
    "ocols",
    "krows",
    "kcols",
    "kchs",
    "lpad",
    "rpad",
    "upad",
    "dpad",
    "downsample",
    "input_dilated",
    "trans_input_3120",
    "max_pixels_per_row",
)


def _conv_input_residency(v: Mapping[str, Any], state: dict, *, facts: Mapping[str, Any]) -> list[str]:
    """The cross-descriptor rule for an input LEFT IN the scratchpad and read again by a later descriptor.

    Three RTL facts make the mode, and this models all three rather than admitting the fields:

    1. ``a_ex_spad_id``/``b_ex_spad_id`` PIN the operand region to a named scratchpad half instead of
       letting it follow the loop slot. ``LoopConv`` gives slot ``i`` ``a_addr_start = i*(max_addr/
       concurrent_loops)`` and ``b_addr_end = (i+1)*(max_addr/concurrent_loops)`` ONCE, at reset; a
       non-zero id replaces both the loader's ``addr_start`` and the execute stage's ``a_addr_start``
       with ``(id-1)*(max_addr/concurrent_loops)``, the same expression on both sides. So id ``h`` means
       half ``h-1``, and an id past the partition names a half the unit does not have.
    2. A NULL input pointer SKIPS THE LOAD ENTIRELY -- ``LoopConvLdInput`` gates its command queue on
       ``req.dram_addr =/= 0.U`` and drops straight back to ``idle`` -- so the descriptor computes over
       whatever the half already holds. That is the saving and it is also the hazard.
    3. Nothing in the A address depends on ``pochs`` (:data:`_CONV_INPUT_RESIDENCY_FIELDS`), so a
       descriptor on another output-channel tile of the SAME window reads exactly the rows a previous
       descriptor staged.

    What is refused, by name rather than by silence: a NULL input with no id (the execute stage would
    read whichever half its loop SLOT happens to own, and which slot the sequencer gives it is not a
    property of the schedule); a NULL input against a half nothing has staged, or staged for a different
    window, or for a different OUTPUT TILE (everything in :data:`_CONV_OUTPUT_TILE_FIELDS` but ``pochs``,
    which is exactly the axis the mode is allowed to move along); an UNPINNED load while a half is pinned
    (an unpinned load lands at its slot's ``a_addr_start``, which may be the pinned half); a NULL input
    inside a split reduction (the chain rule verifies the reduction by how far the input pointer
    ADVANCED, and a NULL pointer advances nothing); and ``b_spad_id``, which pins the WEIGHTS and is a
    mode this checker still does not model.

    WHAT IT CANNOT CHECK, said rather than implied: a NULL-input descriptor names no DRAM pointer, so
    nothing in it states which spatial tile of the image it wants -- the device's semantics really is
    "read whatever half ``h`` holds". The border pads and the readout extents are part of the output tile
    this compares, so a reader on an edge tile of the same layer is caught; two INTERIOR tiles of one
    layer are not distinguishable by any field a descriptor carries. The obligation that a reader follows
    its own stager is therefore the emitter's, and it discharges it by peeling the output-channel loop
    rather than by reordering anything.

    The resident input and the current descriptor's weights share one half -- A from the bottom,
    ``b_addr_start = b_addr_end - B_rows`` from the top -- and they are already bounded together:
    :func:`conv_working_rows` prices ``A_rows + B_rows`` for one descriptor against
    ``spad_rows_per_loop``, and a resident input is by construction one whose A-determining fields are
    identical to this descriptor's, so its ``A_rows`` is this descriptor's.
    """
    errs: list[str] = []
    slots = conv_loop_slots(facts)
    for field in ("a_spad_id", "b_spad_id"):
        if not 0 <= v[field] <= slots:
            errs.append(
                f"{field}={v[field]} names scratchpad half {v[field] - 1} of a unit whose sequencer "
                f"partitions the scratchpad {slots} ways"
            )
    if v["b_spad_id"]:
        errs.append("b_spad_id pins the WEIGHTS to a scratchpad half, which is not modelled by this checker")
    if errs:
        return errs
    resident = dict(state.get("conv_input_resident") or {})
    half = v["a_spad_id"]
    moveable = tuple(f for f in _CONV_OUTPUT_TILE_FIELDS if f != "pochs")
    window = (tuple(v[f] for f in _CONV_INPUT_RESIDENCY_FIELDS), tuple(v[f] for f in moveable))
    if v["input"] is None:
        if not half:
            errs.append(
                "a descriptor passes a NULL input pointer with a_spad_id=0: the loader is skipped "
                "entirely and the execute stage reads whichever scratchpad half its loop SLOT owns, "
                "which is not a property this schedule states"
            )
        elif half not in resident:
            errs.append(
                f"a descriptor passes a NULL input pointer and reads scratchpad half {half - 1}, "
                f"into which nothing before it staged an input"
            )
        else:
            staged = resident[half]
            if staged != window:
                fields = _CONV_INPUT_RESIDENCY_FIELDS + moveable
                differ = [f for f, a, b in zip(fields, staged[0] + staged[1], window[0] + window[1]) if a != b]
                errs.append(
                    f"a descriptor reads the input resident in scratchpad half {half - 1}, which was "
                    f"staged for a different window (differs in {differ})"
                )
        if v["output"] is None or state.get("pending_conv_partial") is not None:
            errs.append(
                "a descriptor inside a split reduction passes a NULL input pointer: the chain is "
                "verified by how far the input pointer advances into the reduction, and a NULL pointer "
                "advances nothing"
            )
    elif half:
        if half in resident:
            # MEASURED, and the reason this is a refusal and not a comment. The reservation station
            # records an mvin's destination rows and a compute's A rows and makes the reload depend on
            # the read (`deps_ex`, war), so a second staging of a half SHOULD be ordered behind the
            # descriptors still reading the first. The ResNet-50 group model says otherwise: with every
            # output-channel-tiled convolution pinned, the stem -- 40 windows, so 40 stagings of half 0 --
            # returned 5,652,929 on its own checksum against the oracle's 5,663,048 (FireSim job 730,
            # whole-model cosine 997,981 -> 988,504 ppm). The shape that stages ONCE is bit-exact. So a
            # half is staged at most once, which is also the bound the vendor library gates `a_reuse` on:
            # one resident window per half, never a reload.
            errs.append(
                f"a descriptor re-stages scratchpad half {half - 1}, which an earlier descriptor already "
                f"staged and later ones read: a reload of a pinned half is not ordered against the "
                f"descriptors still reading it, and a schedule that does it returns a wrong answer rather "
                f"than a slow one"
            )
        resident[half] = window
    elif resident:
        pinned = ", ".join(str(h - 1) for h in sorted(resident))
        errs.append(
            f"a descriptor stages its input with a_spad_id=0 while scratchpad half/halves {pinned} hold "
            f"a pinned input: an unpinned load lands at the a_addr_start of whichever loop slot the "
            f"sequencer gives it, which may be one of them"
        )
    state["conv_input_resident"] = resident
    return errs


def _conv_strided_load(v: Mapping[str, Any], *, facts: Mapping[str, Any], state: Mapping[str, Any]) -> list[str]:
    """What the sequencer's INPUT LOADER is being asked to do, modelled rather than refused.

    ``LoopConvLdInput`` stages one transfer per input pixel per channel block. A convolution of stride
    s only ever reads every s-th pixel in each dimension, so at stride 2 three of every four staged
    pixels are moved and never read. The ``downsample`` bit removes exactly that: it doubles the mvin's
    DRAM row stride (``config.rs2 := dram_stride << downsample``), steps the row iterator by two
    (``next_irow := sFloorAdd(irow, 1.U << downsample, ...)``) and halves the row count
    (``num_rows := I >> downsample``), so the loader gathers the strided subset through its own stride
    field. Measured on this device, one bit: 1,167,983 -> 502,703 cycles on a 56x56 256->512 layer,
    bit-exact.

    TWO FIELDS, ONE FACT. The execute stage reads the staged window back with ``irows >> downsample``,
    so with the bit set the loader has ALREADY applied the stride and ``config_ex``'s ``A_stride`` must
    come back to one. Setting either alone computes a different convolution, silently -- so the pair is
    checked together and a descriptor that sets one without the other is refused by name.

    The eligibility predicate is the LIBRARY's, read from its own header rather than restated, because
    halving the window is the same computation only when each output pixel maps to one input pixel.
    A header whose predicate has grown a clause this checker does not implement is UNKNOWN, and an
    UNKNOWN refuses the mode rather than permitting it.
    """
    if not v["downsample"]:
        return []
    errs: list[str] = []
    header_text = facts.get("conv_header_text")
    if not header_text:
        return ["downsample is set but this checkout states no convolution header to read its predicate from"]
    try:
        _DS.predicate_is_implemented(header_text)
        entitled = _DS.downsample_flag(
            kernel=v["kernel_dim"],
            stride=v["stride"],
            padding=v["padding"],
            in_rows=v["in_row_dim"],
            in_cols=v["in_col_dim"],
            pooled=not v["no_pool"],
            input_dilation=v["kernel_dilation"],
            trans_input_3120=bool(v["trans_input_3120"]),
        )
    except _DS.DownsampleUnknown as exc:
        return [f"downsample is set but the device's own predicate could not be read: {exc}"]
    if not entitled:
        errs.append("downsample is set on a layer the library's own predicate does not admit")
    ex = state.get("config_ex")
    want = v["stride"] >> v["downsample"]
    if ex is None or ex.get("A_stride") != want:
        errs.append(
            f"downsample is set, so the loader has applied the stride and config_ex A_stride must be "
            f"{want}; it is {None if ex is None else ex.get('A_stride')}"
        )
    return errs


def _conv_readout_window(v: Mapping[str, Any], *, facts: Mapping[str, Any]) -> list[str]:
    """What the sequencer's STORE stage is being asked to do, modelled rather than merely permitted.

    ``LoopConvSt`` has two shapes and the ``no_pool`` bit picks between them. Without pooling it walks
    the tile's output positions and mvouts them; with pooling it issues a ``CONFIG_STORE`` carrying the
    window (``pool_size``, ``pool_stride``, ``porows``/``pocols``, ``orows``/``ocols`` and the two
    leading pool pads), mvouts one pooled block per batch and output-channel block at the pooled row
    pitch ``pool_out_col_dim``, then issues a second ``CONFIG_STORE`` that puts the window back. Both
    shapes are modelled here; what is refused is an operand set that states NEITHER consistently.

    The arithmetic is ``sp_tiled_conv``'s own, not re-derived: a descriptor's readout window is

        orows = porows * pool_stride + pool_size - 1 - pupad - pdpad

    -- the pooled extent walked with the pool stride, plus the window's own reach, less whatever part
    of that reach falls outside the convolution's output. With pooling off the library states the
    window as ``1x1`` stride ``1``, for which this collapses onto ``orows = porows``, and this recipe
    spells that collapse by carrying zeros; either spelling is accepted, a mixture is not.
    """
    errs: list[str] = []
    ps, pst, ppad = v["pool_size"], v["pool_stride"], v["pool_padding"]
    if v["no_pool"]:
        if (ps, pst, ppad) not in ((0, 0, 0), (1, 1, 0)):
            errs.append(
                f"no_pool, with a {ps}x{ps} stride-{pst} pad-{ppad} window stated beside it: the store "
                f"stage ignores the window when no_pool is set, so the two disagree about the readout"
            )
        if v["pool_out_row_dim"] != v["out_row_dim"] or v["pool_out_col_dim"] != v["out_col_dim"]:
            errs.append("no_pool, and the pooled output extents differ from the convolution's own")
        if (v["orows"], v["ocols"]) != (v["porows"], v["pocols"]):
            errs.append(
                f"no_pool, and the readout window {v['orows']}x{v['ocols']} differs from the tile's "
                f"{v['porows']}x{v['pocols']} output extent"
            )
        if any(v[p] for p in ("plpad", "prpad", "pupad", "pdpad")):
            errs.append("no_pool, and a pool border is stated: there is no pooling window to fall outside")
        return errs
    if facts["readout_pooling"] is not True:
        errs.append(
            f"a pooled readout, and this target's elaborated RTL does not state a pooling store path "
            f"(`elaborated_rtl_features.max_pool` is {facts['readout_pooling']!r}): the store "
            f"controller's `pooling_is_enabled` is a build gate, and a unit built without it answers "
            f"this descriptor with the UNPOOLED rows rather than refusing it"
        )
    if min(ps, pst) < 1:
        errs.append(f"a pooled readout needs a positive window and stride, not {ps}x{ps} stride {pst}")
        return errs
    if not 0 <= ppad < ps:
        errs.append(f"pool_padding {ppad} is not inside the {ps}-wide pooling window")
        return errs
    for pooled, whole, label in (
        (v["pool_out_row_dim"], v["out_row_dim"], "row"),
        (v["pool_out_col_dim"], v["out_col_dim"], "col"),
    ):
        want = (whole + 2 * ppad - ps) // pst + 1
        if pooled != want:
            errs.append(f"pool_out_{label}_dim {pooled} is not this layer's pooled extent ({want})")
    for readout, extent, lead, trail, label in (
        (v["orows"], v["porows"], v["pupad"], v["pdpad"], "orows"),
        (v["ocols"], v["pocols"], v["plpad"], v["prpad"], "ocols"),
    ):
        if min(lead, trail) < 0:
            errs.append(f"a negative pool border beside {label}")
        elif readout != extent * pst + ps - 1 - lead - trail:
            errs.append(
                f"{label} {readout} is not the readout window {extent} pooled positions need "
                f"({extent * pst + ps - 1 - lead - trail})"
            )
        elif readout < 1:
            errs.append(f"{label} is empty: the pool border covers the whole window")
    return errs


def _conv_reduction_chain(v: Mapping[str, Any], state: dict, *, elem_bytes: int) -> list[str]:
    """The cross-descriptor rule for a reduction split across input channels.

    The convolution's analogue of ``check_loop_ws``'s ``pending_partial``, and read off the same RTL: the
    accumulator half advances only after a descriptor with a non-NULL OUTPUT pointer, the bias stage
    skips its load when the BIAS pointer is NULL, and the execute stage always accumulates. So a
    descriptor that withholds its output must be followed by one on the SAME output tile that does not
    reload the bias and whose operands start exactly where its predecessor's reduction ended; the chain's
    depths must add up to the layer's input channels; and the kernel must not end holding a partial.

    Anything outside that shape is REFUSED rather than passed. In particular a descriptor covering only
    part of the KERNEL WINDOW is refused whether or not it withholds its output, because its input origin
    and border move with the kernel position and nothing here states that.
    """
    errs: list[str] = []
    channels = v["in_channels"]
    tile = tuple(v[f] for f in _CONV_OUTPUT_TILE_FIELDS)
    pending = state.get("pending_conv_partial")
    if v["krows"] != v["kernel_dim"] or v["kcols"] != v["kernel_dim"]:
        errs.append(
            f"krows/kcols {v['krows']}x{v['kcols']} cover only part of the {v['kernel_dim']}x"
            f"{v['kernel_dim']} kernel window: a reduction split across kernel positions moves each "
            f"descriptor's input origin and border with the position, which this checker does not state"
        )
    if pending is None:
        covered = 0
        if v["bias"] is None:
            errs.append(
                "a descriptor that begins an output tile must carry a bias pointer: the bias mvin is the "
                "only thing that initialises the accumulator half, a NULL pointer skips it, and the "
                "compute accumulates onto whatever the half last held"
            )
    else:
        covered = pending["covered"]
        if pending["tile"] != tile:
            differ = [f for f, a, b in zip(_CONV_OUTPUT_TILE_FIELDS, pending["tile"], tile) if a != b]
            errs.append(f"a partial sum is abandoned for a descriptor on another output tile (differs in {differ})")
        if v["bias"] is not None:
            errs.append(
                "a descriptor continuing a partial sum must not reload the bias: the mvin overwrites the "
                "accumulator half rather than adding to what it holds"
            )
        for which, per_channel in (("input", elem_bytes), ("weights", v["weight_stride"] * elem_bytes)):
            now, before = v[which], pending[which]
            want = pending["kchs"] * per_channel
            if now is None or before is None or now.tensor != before.tensor:
                errs.append(f"the {which} of a continued partial sum is not a pointer into the same tensor")
            elif now.offset - before.offset != want:
                errs.append(
                    f"{which} advances {now.offset - before.offset} bytes into the reduction, where the "
                    f"{pending['kchs']} input channels its predecessor consumed span {want}"
                )
    covered += v["kchs"]
    if v["output"] is None:
        if covered >= channels:
            errs.append(
                f"a descriptor withholds its output having covered {covered} of {channels} input "
                f"channels: the reduction is finished and nothing after it will store the sum"
            )
        state["pending_conv_partial"] = {
            "tile": tile,
            "covered": covered,
            "kchs": v["kchs"],
            "input": v["input"],
            "weights": v["weights"],
        }
    else:
        if covered != channels:
            errs.append(
                f"a descriptor stores its output having covered {covered} of the layer's {channels} input channels"
            )
        state["pending_conv_partial"] = None
    return errs


@lru_cache(maxsize=4)
def instruction_set(header: str, params: str, *, target: str) -> InstructionSet:
    facts = schedule_facts(header, params, target=target)
    hdr = parse_c_header(Path(header))
    dim, eb, ab = facts["dim"], facts["elem_bytes"], facts["acc_bytes"]
    consts = facts["constants"]
    # Preserve curated semantic bindings while deriving every other legal instruction from
    # the selected header, RTL table and capability declaration.
    bindings = {**_derived_bindings(hdr, _legal_functs(target), _semantic_classes(target)), **_BINDINGS}
    macros = {}
    for name, (macro_name, kinds) in bindings.items():
        m = hdr.macro(macro_name)
        if m is None or not m.is_function:
            raise IsaError(f"{header}: no function-like macro {macro_name}")
        if tuple(m.params) != tuple(kinds):
            raise IsaError(f"{macro_name} parameters {m.params} differ from the binding {tuple(kinds)}")
        macros[name] = m
    loop_ws_fields = _rocc_packings(macros["loop_ws"].body)

    def check_config_ex(v, state):
        errs = []
        if v["dataflow"] != consts["WEIGHT_STATIONARY"]:
            errs.append("only the weight-stationary dataflow is modelled")
        state["config_ex"] = dict(v)
        return errs

    def check_config_st(v, state):
        state["config_st"] = dict(v)
        return [] if v["stride"] > 0 else ["store stride must be positive"]

    def check_config_ld(v, state):
        if v["id"] not in (0, 1, 2):
            return [f"load configuration id {v['id']} is not one of the three LOOP_WS operands"]
        state.setdefault("config_ld", {})[v["id"]] = v["stride"]
        return [] if v["stride"] >= 0 else ["negative load stride"]

    def check_loop_ws(v, state):
        errs = []
        I, J, K = v["I"], v["J"], v["K"]
        # The residual add is THIS instruction with one bit set, and the bit changes what the operands
        # mean rather than adding a mode beside them. There is no contraction, so K carries no extent;
        # and A and B never reach the mesh -- the library's own `sp_tiled_resadd` mvins them to
        # `1 << (ADDR_LEN-1)` and `3 << (ADDR_LEN-2)`, the two halves of the ACCUMULATOR, and mvouts the
        # sum from the first. So K must be 0 here and must not be 0 anywhere else.
        resadd = bool(v["is_resadd"])
        if resadd:
            if K != 0:
                return [f"resadd carries no contraction extent but K={K}"]
            if min(I, J) < 1:
                return [f"empty tile I={I} J={J}"]
        elif min(I, J, K) < 1:
            return [f"empty tile I={I} J={J} K={K}"]
        for p in ("pad_I", "pad_J", "pad_K"):
            if not 0 <= v[p] < dim:
                errs.append(f"{p}={v[p]} outside [0, {dim})")
        # Zero for a resadd, and correctly so: it never touches the scratchpad.
        spad = (I * K + K * J) * dim
        if spad > facts["spad_rows_per_loop"]:
            errs.append(f"A+B tiles need {spad} scratchpad rows > {facts['spad_rows_per_loop']} per loop")
        # Both accumulator halves are live in a resadd, which is why the library bounds its tiler at
        # ACC_ROWS/2 -- and `acc_rows_per_loop` IS the header's `max_acc_rows = ACC_ROWS / 2`, so this
        # one bound is already the library's own without a resadd-specific spelling.
        if I * J * dim > facts["acc_rows_per_loop"]:
            errs.append(f"C tile needs {I * J * dim} accumulator rows > {facts['acc_rows_per_loop']} per loop")
        if v["A_transpose"] and v["B_transpose"]:
            errs.append("A and B transposed together (the RTL asserts against it)")
        if v["a_spad_id"] or v["b_spad_id"]:
            errs.append("scratchpad-id operand reuse is not modelled by this checker")
        if resadd:
            # Every mode the resadd path does not model, refused rather than passed -- the loop issues
            # its own movement, so nothing downstream would catch a schedule it never reasoned about.
            if v["D"] is not None:
                errs.append("resadd takes no bias: D must be NULL")
            if v["C"] is None:
                errs.append("resadd must store its sum: C is NULL")
            if v["ex_accumulate"]:
                errs.append("resadd does not accumulate onto what the accumulator half already held")
            for flag in ("A_transpose", "B_transpose", "full_C", "low_D"):
                if v[flag]:
                    errs.append(f"resadd does not model {flag}")
        if v["full_C"]:
            # The full-width readout is the RAW accumulator row, and that is the whole of what this
            # flag means. `LoopMatmulStC` sets the mvout's `read_full` bit from it; the accumulator's
            # scale unit carries the unscaled row through as `full_data` while the activation, the
            # accumulator scale and the clip to the narrow type are applied only to the `data` the
            # NARROW readout takes. So a schedule that asks for a full-width store AND for either of
            # them is asking for arithmetic this readout does not do, and is refused by the property
            # it named -- not silently stored unscaled, which is the same bytes with a wrong meaning.
            if not facts["acc_read_full_width"]:
                errs.append(
                    "full_C moves the accumulator at full width and this target's parameters header "
                    "states no ACC_READ_FULL_WIDTH: no full-width accumulator read port was built"
                )
            if v["act"] != consts["NO_ACTIVATION"]:
                errs.append(f"a full_C readout moves the raw accumulator, so activation {v['act']} is not applied")
            full_st = state.get("config_st")
            if full_st is not None and float(full_st["acc_scale"]) != 1.0:
                errs.append(
                    f"a full_C readout moves the raw accumulator, so config_st's acc_scale "
                    f"{full_st['acc_scale']} is not applied"
                )
        if v["act"] not in (consts["NO_ACTIVATION"], consts["RELU"]):
            errs.append(f"activation {v['act']} is not modelled by this checker")
        for funct, rs1, rs2 in loop_ws_fields:
            for src in (rs1, rs2):
                try:
                    _expression(src, _numeric(v))
                except UnsupportedNativeConv as exc:
                    errs.append(f"{funct}: {exc}")
        ld = state.get("config_ld", {})
        want = {0: v["A_stride"] * eb, 1: v["B_stride"] * eb}
        if v["D"] is not None:
            want[2] = v["D_stride"] * (eb if v["low_D"] else ab)
        for i, stride in want.items():
            if ld.get(i) != stride:
                errs.append(f"config_ld id {i} stride {ld.get(i)} != the loop's {stride} bytes")
        if v["C"] is not None:
            st = state.get("config_st")
            if st is None or st["stride"] != v["C_stride"] * (ab if v["full_C"] else eb):
                errs.append("config_st stride does not match the loop's C stride")
            elif st["acc_act"] != v["act"]:
                errs.append("config_st activation differs from the loop's")
        ex = state.get("config_ex")
        if ex is None or (ex["A_transpose"], ex["B_transpose"]) != (v["A_transpose"], v["B_transpose"]):
            errs.append("config_ex transposes differ from the loop's (or no config_ex)")
        pending = state.get("pending_partial")
        tile = (I, J, v["pad_I"], v["pad_J"])
        if pending is not None:
            if pending != tile:
                errs.append(f"partial sum on tile {pending} is abandoned for tile {tile}")
            if not v["ex_accumulate"] or v["D"] is not None:
                errs.append("a loop continuing a partial sum must accumulate and must not reload D")
        elif v["D"] is None and v["ex_accumulate"]:
            errs.append("a fresh tile with no D accumulates onto whatever the accumulator half held")
        state["pending_partial"] = tile if v["C"] is None else None
        return errs

    def footprint_loop_ws(v):
        ri, cj, ck = I_rows(v), v["J"] * dim - v["pad_J"], v["K"] * dim - v["pad_K"]
        if v["is_resadd"]:
            # A and B are the two ADDENDS, each the shape of the output; the contraction extent is
            # absent, not zero-sized. Reading them through the contraction spelling below would make
            # `ck = 0` and report both operands as touching no bytes at all -- which `_extent` returns
            # for a zero column count -- and a footprint that understates a read is one that lets a
            # tile be admitted whose operands do not fit.
            return [
                ("A", _extent(ri, cj, v["A_stride"], eb), "read"),
                ("B", _extent(ri, cj, v["B_stride"], eb), "read"),
                ("D", 0, "read"),
                ("C", _extent(ri, cj, v["C_stride"], eb), "write"),
            ]
        a = (ck, ri) if v["A_transpose"] else (ri, ck)
        b = (cj, ck) if v["B_transpose"] else (ck, cj)
        d_rows = ri if v["D_stride"] else 1
        return [
            ("A", _extent(*a, v["A_stride"], eb), "read"),
            ("B", _extent(*b, v["B_stride"], eb), "read"),
            ("D", _extent(d_rows, cj, v["D_stride"], eb if v["low_D"] else ab), "read"),
            ("C", _extent(ri, cj, v["C_stride"], ab if v["full_C"] else eb), "write"),
        ]

    def I_rows(v):
        return v["I"] * dim - v["pad_I"]

    loop_conv_fields = _rocc_packings(macros["loop_conv_ws"].body)

    def check_loop_conv_ws(v, state):
        """What the convolution sequencer requires of its own operands.

        Refuses every mode it does not model rather than passing it: a checker that stays silent about
        pooling, dilation, a transposed operand or depthwise would certify a schedule whose numbers it
        never reasoned about, and the loop issues its own movement, so nothing downstream would catch
        it. The field widths are not written here -- they are whatever the header's own packing
        expressions admit, evaluated the same way the matmul loop's are.

        The cross-descriptor rule is the convolution's version of what ``check_loop_ws`` enforces on the
        matmul's k-tiles, and it is read off ``LoopConv.scala`` the same way. The sequencer advances to
        the other accumulator half only after a descriptor whose OUTPUT pointer is non-NULL, its bias
        stage SKIPS its load entirely when the BIAS pointer is NULL, and its execute stage always
        accumulates. So a descriptor that withholds its output leaves a partial sum in a half that the
        next descriptor must be about to finish: the same output tile, no bias reload, and its own slice
        of the reduction stepped off the end of its predecessor's. The whole chain must add up to the
        layer's input channels and the kernel must not end holding one.
        """
        errs = []
        k, st, pad, dil = v["kernel_dim"], v["stride"], v["padding"], v["kernel_dilation"]
        for name, value in (("stride", st), ("kernel_dim", k)):
            if value < 1:
                errs.append(f"{name} must be positive")
        if pad < 0:
            errs.append("padding must not be negative")
        if st >= 1 and k >= 1:
            for rows, dim_in, label in (
                (v["out_row_dim"], v["in_row_dim"], "row"),
                (v["out_col_dim"], v["in_col_dim"], "col"),
            ):
                want = (dim_in + 2 * pad - k) // st + 1
                if rows != want:
                    errs.append(f"out_{label}_dim {rows} is not this layer's output extent ({want})")
        if dil != 1:
            errs.append("kernel dilation is not modelled by this checker")
        errs.extend(_conv_readout_window(v, facts=facts))
        errs.extend(_conv_strided_load(v, facts=facts, state=state))
        for flag in (
            "wrot180",
            "input_dilated",
            "dw",
            "trans_output_1203",
            "trans_weight_1203",
            "trans_weight_0132",
            "trans_input_3120",
        ):
            if v[flag]:
                errs.append(f"{flag} is not modelled by this checker")
        errs.extend(_conv_input_residency(v, state, facts=facts))
        if v["activation"] not in (consts["NO_ACTIVATION"], consts["RELU"]):
            errs.append(f"activation {v['activation']} is not modelled by this checker")
        # The tile the loop walks must FIT, not merely lie inside the problem. The sequencer issues
        # its own mvin/mvout against a scratchpad and an accumulator half, and neither it nor the RTL
        # bounds-checks the addresses it generates: a tile whose working set is larger than the half
        # wraps, and the output is silently wrong rather than refused. That is exactly what happened
        # -- a whole 56x56x64 layer as one descriptor asks for 12544 accumulator rows against 512, and
        # every convolution in a ResNet-50 program came back corrupt while the static check passed.
        if min(v["batches"], v["porows"], v["pocols"], v["pochs"], v["krows"], v["kcols"], v["kchs"]) >= 1:
            footprint = dict(
                batches=v["batches"],
                porows=v["porows"],
                pocols=v["pocols"],
                pochs=v["pochs"],
                krows=v["krows"],
                kcols=v["kcols"],
                kchs=v["kchs"],
                stride=max(st, 1),
                pool_size=1 if v["no_pool"] else v["pool_size"],
                pool_stride=1 if v["no_pool"] else v["pool_stride"],
                dim=dim,
                # The strided load stages the halved window, so pricing the unstrided one here would
                # refuse a tile the device holds -- the mirror of the overrun this check exists for.
                downsample=v["downsample"],
            )
            spad = conv_working_rows(acc=False, **footprint)
            acc = conv_working_rows(acc=True, **footprint)
            if spad > facts["spad_rows_per_loop"]:
                errs.append(f"the tile's input and weights need {spad} scratchpad rows > {facts['spad_rows_per_loop']}")
            if acc > facts["acc_rows_per_loop"]:
                errs.append(f"the tile's output needs {acc} accumulator rows > {facts['acc_rows_per_loop']}")
        # The tile the loop walks must lie inside the problem it was given. Its two spatial extents are
        # named in the POOLED output space -- that is the space the nest walks and the space the store
        # stage addresses -- so they are bounded by the pooled extents, which collapse onto the
        # convolution's own when there is no window.
        for tile, whole in (
            ("batches", "batch_size"),
            ("porows", "pool_out_row_dim"),
            ("pocols", "pool_out_col_dim"),
            ("pochs", "out_channels"),
            ("krows", "kernel_dim"),
            ("kcols", "kernel_dim"),
            ("kchs", "in_channels"),
        ):
            if v[tile] < 1:
                errs.append(f"{tile} must be positive")
            elif v[tile] > v[whole]:
                errs.append(f"{tile} {v[tile]} exceeds {whole} {v[whole]}")
        if v["weights"] is None:
            errs.append("the convolution loop needs a weights pointer")
        if v["input"] is None and not v["a_spad_id"]:
            # A NULL input is a real mode -- the loader skips its load and the descriptor computes over a
            # scratchpad half a previous descriptor staged -- but only when the schedule NAMES that half.
            # `_conv_input_residency` is what says whether the naming holds up.
            errs.append("the convolution loop needs an input pointer unless a_spad_id names a staged scratchpad half")
        if v["no_bias"] and v["bias"] is None:
            errs.append(
                "no_bias asks the sequencer to write zeros over the accumulator half, but a NULL bias "
                "pointer makes it skip that load entirely, so nothing is written"
            )
        # `max_pixels_per_row` packs several kernel columns into one mesh row, so the row it builds is
        # `max_pixels_per_row * min(kchs, DIM)` wide and must fit. Derived from the operands rather than
        # pinned, because the header's own rule reads it off THIS descriptor's reduction depth.
        mppr = v["max_pixels_per_row"]
        if mppr < 1:
            errs.append("max_pixels_per_row must be positive")
        elif v["kcols"] >= 1 and v["kchs"] >= 1:
            if mppr > v["kcols"]:
                errs.append(f"max_pixels_per_row {mppr} exceeds the {v['kcols']} kernel columns of the tile")
            elif mppr * min(v["kchs"], dim) > dim:
                errs.append(
                    f"{mppr} kernel columns of {min(v['kchs'], dim)} channels do not fit one {dim}-wide mesh row"
                )
        errs.extend(_conv_reduction_chain(v, state, elem_bytes=eb))
        st_cfg = state.get("config_st")
        if st_cfg is None:
            errs.append("no config_st before a convolution loop that stores its own output")
        elif st_cfg["acc_act"] != v["activation"]:
            errs.append("config_st activation differs from the loop's")
        for funct, rs1, rs2 in loop_conv_fields:
            for src in (rs1, rs2):
                try:
                    _expression(src, _numeric(v))
                except UnsupportedNativeConv as exc:
                    errs.append(f"{funct}: {exc}")
        return errs

    def check_derived(values, state):
        """Apply RTL-derived array and DMA bounds to newly derived instructions."""
        errors = []
        for key, value in values.items():
            if not isinstance(value, int):
                continue
            low = key.lower()
            if low.endswith("rows") and value > dim:
                errors.append(f"{key}={value} exceeds the array edge {dim}")
            elif low.endswith("cols") and value > facts["max_block_len"] * dim:
                errors.append(
                    f"{key}={value} exceeds the derived block limit "
                    f"{facts['max_block_len'] * dim} ({facts['max_block_len']} tile(s) of {dim})"
                )
        return errors

    checks = {
        "config_ex": check_config_ex,
        "config_st": check_config_st,
        "config_ld": check_config_ld,
        "loop_ws": check_loop_ws,
        "loop_conv_ws": check_loop_conv_ws,
    }
    instrs = {}
    for name, (macro_name, kinds) in bindings.items():
        instrs[name] = InstrDef(
            name=name,
            operands=tuple(Operand(p, kinds[p]) for p in macros[name].params),
            render_c=(lambda mn: lambda args: f"{mn}({', '.join(args)});")(macro_name),
            check=checks.get(name, check_derived),
            footprint=footprint_loop_ws if name == "loop_ws" else None,
            doc=f"header macro {macro_name} (line {macros[name].line})",
        )

    def finish(state):
        out = []
        if state.get("pending_partial") is not None:
            out.append(f"kernel ends with the partial sum on tile {state['pending_partial']} never stored")
        conv = state.get("pending_conv_partial")
        if conv is not None:
            out.append(
                f"kernel ends with a convolution partial sum over {conv['covered']} input channels "
                f"still in the accumulator, never stored"
            )
        return out

    import hashlib

    return InstructionSet(
        target=target,
        instrs=instrs,
        c_types={facts["elem_dtype"]: "elem_t", facts["acc_dtype"]: "acc_t"},
        drain_c="gemmini_fence();",
        facts=facts,
        finish=finish,
        provenance={
            "header_sha256": hashlib.sha256(Path(header).read_bytes()).hexdigest(),
            "params_sha256": hashlib.sha256(Path(params).read_bytes()).hexdigest(),
        },
    )


# --- reference schedule ------------------------------------------------------------------------------


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def choose_tiles(m: int, n: int, k: int, facts: Mapping[str, Any]) -> tuple[int, int, int]:
    """The (I, J, K) tile in DIM blocks this recipe uses by default.

    Every tile the per-loop capacities admit is considered; K is always the largest that fits the
    scratchpad beside the chosen I and J (fewest partial-sum passes). The objective is an explicit
    traffic model, not a measurement: the A matrix is re-read once per column tile and B once per row
    tile, so minimise ``|A|*J0 + |B|*I0``, then the number of loop instructions, then prefer the larger
    output tile. It only picks a starting point; measured alternatives replace it."""
    dim, eb = facts["dim"], facts["elem_bytes"]
    ib, jb, kb = _ceil(m, dim), _ceil(n, dim), _ceil(k, dim)
    spad_cap = facts["spad_rows_per_loop"] // dim
    acc_cap = facts["acc_rows_per_loop"] // dim
    best = None
    for ti in range(1, ib + 1):
        for tj in range(1, jb + 1):
            if ti * tj > acc_cap:
                break
            tk = min(kb, spad_cap // (ti + tj))
            if tk < 1:
                continue
            i0, j0, k0 = _ceil(ib, ti), _ceil(jb, tj), _ceil(kb, tk)
            key = (m * k * eb * j0 + k * n * eb * i0, i0 * j0 * k0, -ti * tj)
            if best is None or key < best[0]:
                best = (key, (ti, tj, tk))
    if best is None:
        raise IsaError("no tile fits the per-loop capacities")
    return best[1]


def matmul_reference(
    *,
    name: str,
    m: int,
    n: int,
    k: int,
    operands: Mapping[str, TensorArg],
    relu: bool,
    scale: float | None,
    facts: Mapping[str, Any],
    tiles: tuple[int, int, int] | None = None,
) -> Kernel:
    """``C[m][n] = readout(A[m][k] @ B[k][n] + D[n])`` as LOOP_WS macro-instructions.

    ``operands`` maps ``a``, ``b``, ``c`` and optionally ``d`` (the bias row, repeated down every output
    row) to tensor arguments; A, B and C are dense row-major with row strides ``k``, ``n``, ``n``. The
    structure: the three configurations once, then row tiles x column tiles, and within one output tile
    the k-tiles in order -- the first loads D (or overwrites), the last stores C, the ones between only
    accumulate (see the module docstring for why k must be innermost).

    ``scale`` is the accumulator scale the NARROW readout requantizes through. Passing ``None`` asks
    instead for the readout a model's final contraction leaves as: the accumulator moved out at its OWN
    width, which is ``full_C``. That readout is the raw accumulator row -- the activation, the scale and
    the clip to the operand type are applied on the narrow port and not on this one -- so ``C`` is an
    ``acc_t`` buffer with an ``acc_t`` row stride, and a ReLU asked for beside it is refused rather than
    silently dropped. The port itself must be one the target states it built (``acc_read_full_width``).
    """
    dim, eb, ab = facts["dim"], facts["elem_bytes"], facts["acc_bytes"]
    consts = facts["constants"]
    a, b, c, d = operands["a"], operands["b"], operands["c"], operands.get("d")
    full_c = scale is None
    if full_c:
        if not facts["acc_read_full_width"]:
            raise IsaError(
                f"{name}: an accumulator-resident readout moves the accumulator at its own width, and "
                f"this target's parameters header states no ACC_READ_FULL_WIDTH -- the generator built "
                f"no full-width accumulator read port for that mvout to read through"
            )
        if relu:
            raise IsaError(
                f"{name}: an accumulator-resident readout moves the RAW accumulator row; the activation "
                f"is applied on the narrow readout port only, so a ReLU cannot ride this store"
            )
    c_bytes = ab if full_c else eb
    ib, jb, kb = _ceil(m, dim), _ceil(n, dim), _ceil(k, dim)
    pad_i, pad_j, pad_k = ib * dim - m, jb * dim - n, kb * dim - k
    ti, tj, tk = tiles or choose_tiles(m, n, k, facts)
    i_tiles, j_tiles, k_tiles = _ceil(ib, ti), _ceil(jb, tj), _ceil(kb, tk)
    last_i, last_j, last_k = ib - (i_tiles - 1) * ti, jb - (j_tiles - 1) * tj, kb - (k_tiles - 1) * tk
    act = consts["RELU"] if relu else consts["NO_ACTIVATION"]
    i0, j0, kv = Var("i0"), Var("j0"), Var("k0")
    I = select(i0, i_tiles - 1, last_i, ti)
    J = select(j0, j_tiles - 1, last_j, tj)
    pI = select(i0, i_tiles - 1, pad_i, 0)
    pJ = select(j0, j_tiles - 1, pad_j, 0)

    def ws(kidx, K, pK, first: bool, last: bool):
        return call(
            "loop_ws",
            I=I,
            J=J,
            K=K,
            pad_I=pI,
            pad_J=pJ,
            pad_K=pK,
            A=Ptr(a.name, (i0 * (ti * dim * k) + kidx * (tk * dim)) * eb),
            B=Ptr(b.name, (kidx * (tk * dim * n) + j0 * (tj * dim)) * eb),
            D=Ptr(d.name, j0 * (tj * dim * ab)) if (first and d is not None) else NULL,
            C=Ptr(c.name, (i0 * (ti * dim * n) + j0 * (tj * dim)) * c_bytes) if last else NULL,
            A_stride=k,
            B_stride=n,
            D_stride=0,
            C_stride=n,
            A_transpose=0,
            B_transpose=0,
            full_C=1 if full_c else 0,
            low_D=0,
            ex_accumulate=(d is not None) if first else 1,
            act=act,
            a_spad_id=0,
            b_spad_id=0,
            is_resadd=0,
        )

    if k_tiles == 1:
        inner = (ws(0, last_k, pad_k, True, True),)
    else:
        middle = (loop("k0", k_tiles - 2, ws(kv + 1, tk, 0, False, False)),) if k_tiles > 2 else ()
        inner = (ws(0, tk, 0, True, False), *middle, ws(k_tiles - 1, last_k, pad_k, False, True))
    body = [
        call(
            "config_ex",
            dataflow=consts["WEIGHT_STATIONARY"],
            sys_act=act,
            sys_shift=0,
            A_stride=1,
            A_transpose=0,
            B_transpose=0,
        ),
        call(
            "config_st",
            stride=n * c_bytes,
            acc_act=act,
            # The library's own `ACC_SCALE_IDENTITY` for the full-width readout: the store carries the
            # accumulator row through unchanged, so any other multiplier would be a scale nobody applies.
            acc_scale=1.0 if full_c else float(np.float32(scale)),
        ),
        call("config_ld", stride=k * eb, scale=1.0, shrunk=0, id=0),
        call("config_ld", stride=n * eb, scale=1.0, shrunk=0, id=1),
    ]
    if d is not None:
        body.append(call("config_ld", stride=0, scale=1.0, shrunk=0, id=2))
    body.append(loop("i0", i_tiles, loop("j0", j_tiles, *inner)))
    args = (a, b) + ((d,) if d is not None else ()) + (c,)
    return Kernel(
        name=name,
        args=args,
        body=tuple(body),
        attrs=(
            ("recipe", "matmul_ws_full_acc_readout_v1" if full_c else "matmul_ws_reference_v1"),
            ("tiles", f"{ti},{tj},{tk}"),
            ("shape", f"{m}x{n}x{k}"),
            ("readout", "the accumulator at its own width" if full_c else "narrow, requantized"),
        )
        + (
            # What was actually measured, not what is expected: the full-width readout was run against
            # the library call for the same group on gsim, over byte-identical operands, and compared
            # element for element (ResNet-50's 1x1000x2048 classifier, 0 of 1000 elements differing).
            (("numerics", "matches the vendor library on gsim, element for element"),) if full_c else ()
        ),
    )


def conv_tile_padding(
    *,
    in_dim: int,
    out_dim: int,
    kernel: int,
    stride: int,
    padding: int,
    start: int = 0,
    extent: int | None = None,
    pool_size: int = 1,
    pool_stride: int = 1,
    pool_padding: int = 0,
    pool_out_dim: int | None = None,
) -> dict[str, int]:
    """The borders and extents ONE tile of a layer sees, by the library tiler's own arithmetic.

    Transcribed from ``tiled_conv`` in the curated harness header rather than reasoned out here, because
    a padding this gets wrong produces a silently wrong output rather than a refusal. A tile is named in
    the axis the loop nest walks, which is the POOLED output axis -- it covers pooled positions
    ``start .. start + extent`` (the whole axis by default), and with no pooling the library states the
    window as ``1x1`` stride ``1``, for which every pooled quantity below collapses onto the
    convolution's own and the defaults here reproduce exactly that.

    Two nested windows, each with its own border:

    - the READOUT window, in the convolution's own output space. Its origin is
      ``start * pool_stride - pool_padding`` and it spans ``extent * pool_stride + pool_size - 1``
      positions; ``plpad``/``pupad`` and ``prpad``/``pdpad`` are how far that falls outside the
      convolution's output, and ``readout`` is what is left -- ``sp_tiled_conv``'s own ``orows``;
    - the INPUT window. Its origin is the readout origin CLAMPED to the output (the sequencer feeds
      zeros for the pool border rather than reading further), walked with the layer's stride:
      ``origin = max(readout origin, 0) * stride - padding``, spanning ``readout * stride + kernel - 1``.
      ``lpad``/``upad`` and ``rpad``/``dpad`` are how far THAT falls outside the image.

    For 3x3 stride 1 padding 1 over the whole of 56x56 with no pooling this is (1, 1, 1, 1) -- the 58x58
    zero-padded form the library's own kernels build by hand. ``origin`` and ``span`` come back too,
    because the tile's input pointer is ``max(origin, 0)`` and its scratchpad footprint is ``span``.
    """
    pool_out_dim = out_dim if pool_out_dim is None else pool_out_dim
    extent = pool_out_dim if extent is None else extent
    readout_origin = start * pool_stride - pool_padding
    readout_span = extent * pool_stride + pool_size - 1
    pool_lead = max(0, -readout_origin)
    pool_trail = max(0, readout_origin + readout_span - out_dim)
    readout = readout_span - pool_lead - pool_trail
    origin = max(readout_origin, 0) * stride - padding
    span = readout * stride + kernel - 1
    leading, trailing = max(0, -origin), max(0, origin + span - in_dim)
    return {
        "lpad": leading,
        "upad": leading,
        "rpad": trailing,
        "dpad": trailing,
        "plpad": pool_lead,
        "pupad": pool_lead,
        "prpad": pool_trail,
        "pdpad": pool_trail,
        "readout": readout,
        "origin": origin,
        "span": span,
    }


def _tiles_of(whole: int, extent: int) -> list[tuple[int, int]]:
    """``(start, extent)`` of each tile covering ``whole``; the last one is whatever is left."""
    return [(start, min(extent, whole - start)) for start in range(0, whole, extent)]


def conv_loop_slots(facts: Mapping[str, Any]) -> int:
    """How many convolution descriptors the sequencer can have in flight at once.

    Derived, not written: it is the ratio between the whole scratchpad/accumulator and the half one loop
    owns, which is exactly the number of concurrent loops the RTL runs. Both ratios must agree, because a
    partition stated two ways is a partition nobody checked.
    """
    by_spad = facts["spad_rows"] // facts["spad_rows_per_loop"]
    by_acc = facts["acc_rows"] // facts["acc_rows_per_loop"]
    if by_spad != by_acc or by_spad < 1:
        raise IsaError(
            f"the scratchpad is split {by_spad} ways per loop and the accumulator {by_acc}: "
            f"the concurrent-loop count cannot be derived from a partition stated two ways"
        )
    return by_spad


def conv_tile_traffic(
    *,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    in_channels: int,
    tile: Mapping[str, int],
    facts: Mapping[str, Any],
    pool_out_dim: int | None = None,
    pool_size: int = 1,
    pool_stride: int = 1,
    downsample: int = 0,
) -> dict[str, int]:
    """The DRAM row transfers a tiling costs, by the four movements the sequencer issues.

    Each of LoopConv's four stages walks its operand in ROWS and moves at most ``MAX_BLOCK_LEN * DIM``
    columns (``MAX_BLOCK_LEN_ACC * DIM`` for the accumulator ones) per transfer, so a row is the unit the
    DMA actually costs and a narrow tile pays the same per row as a wide one. Each stage's iteration
    space is read off its RTL module and summed in closed form over the tiles, which is why this does not
    enumerate descriptors:

    - the input is re-read once per OUTPUT-CHANNEL tile (the channel tile is not an axis of the input),
      ``batches * irows * icols`` rows per reduction step;
    - the weights are re-read once per output POSITION tile, ``krows * kcols * kchs`` rows;
    - the bias is loaded once per output tile over the READOUT window, ``batches * orows * ocols`` rows,
      and the output is stored over the POOLED window, ``batches * porows * pocols``. With no pooling
      the two windows are the same and so are the two counts; a pooled readout makes the store smaller
      than the accumulation it reads, which is the whole reason the pool rides the readout at all.

    The mesh work is deliberately NOT in here: it is the layer's own MAC count up to the waste of a
    partial channel block, and the candidates :func:`choose_conv_tiles` compares all keep the reduction
    depth on a block boundary, so it does not discriminate between them. What the tiling changes is the
    movement.
    """
    dim, mbl, mbla = facts["dim"], facts["max_block_len"], facts["max_block_len_acc"]
    pool_out_dim = out_dim if pool_out_dim is None else pool_out_dim
    b_t = _tiles_of(batch, tile["batches"])
    r_t = _tiles_of(pool_out_dim, tile["porows"])
    c_t = _tiles_of(pool_out_dim, tile["pocols"])
    o_t = _tiles_of(out_channels, tile["pochs"])
    k_t = _tiles_of(in_channels, tile["kchs"])
    sum_b = sum(e for _, e in b_t)
    sum_p_r, sum_p_c = sum(e for _, e in r_t), sum(e for _, e in c_t)
    # A tile's readout window in the convolution's OWN output space -- the pooled extent walked with the
    # pool stride plus the window's reach, which with no pooling is the pooled extent itself. The input
    # window is that readout walked with the layer's stride plus the kernel's own reach.
    out_rows = [e * pool_stride + pool_size - 1 for _, e in r_t]
    out_cols = [e * pool_stride + pool_size - 1 for _, e in c_t]
    sum_o_r, sum_o_c = sum(out_rows), sum(out_cols)
    # With the strided load the loader issues the HALVED window: it steps its row iterator by
    # 1 << downsample and asks for I >> downsample rows. Counting the unstrided window here would make
    # the search blind to the very saving the mode buys, and it would pick tiles for a cost that is
    # four times what the device pays.
    in_rows = sum((o * stride + kernel - 1) >> downsample for o in out_rows)
    in_cols = sum((o * stride + kernel - 1) >> downsample for o in out_cols)
    och_ld = sum(_ceil(e, mbl * dim) for _, e in o_t)
    och_acc = sum(_ceil(e, mbla * dim) for _, e in o_t)
    kch_ld = sum(_ceil(e, mbl * dim) for _, e in k_t)
    return {
        "input": sum_b * in_rows * in_cols * len(o_t) * kch_ld,
        "weights": len(b_t) * len(r_t) * len(c_t) * och_ld * kernel * kernel * sum(e for _, e in k_t),
        "bias": sum_b * sum_o_r * sum_o_c * och_acc,
        "output": sum_b * sum_p_r * sum_p_c * och_acc,
        "descriptors": len(b_t) * len(r_t) * len(c_t) * len(o_t) * len(k_t),
        "reduction_steps": len(k_t),
    }


def _conv_output_tile(
    *,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    depth: int,
    facts: Mapping[str, Any],
    pool_out_dim: int | None = None,
    pool_size: int = 1,
    pool_stride: int = 1,
    downsample: int = 0,
) -> dict[str, int] | None:
    """The output tile ``tiled_conv_stride_auto``'s search settles on with the reduction depth PINNED.

    The library shrinks the largest axis until the tile fits both halves, then grows the column extent,
    then grows whatever still fits until nothing does -- and it avoids shrinking the column extent below
    ``DIM`` while there is more than one row, because that is the extent that keeps the spatial array
    busy. Transcribed from the library's own search rather than re-derived, for the same reason
    :func:`conv_tile_padding` is. ``None`` when no output tile fits at this depth.
    """
    dim = facts["dim"]
    pool_out_dim = out_dim if pool_out_dim is None else pool_out_dim
    spad_cap, acc_cap = facts["spad_rows_per_loop"], facts["acc_rows_per_loop"]
    axes = ["batches", "porows", "pocols", "pochs"]
    # The two spatial axes are named in the POOLED output space, which is the space the loop nest walks.
    whole = {"batches": batch, "porows": pool_out_dim, "pocols": pool_out_dim, "pochs": out_channels}
    tile = dict(whole)

    def fits(candidate: Mapping[str, int]) -> bool:
        rows = dict(
            stride=stride,
            krows=kernel,
            kcols=kernel,
            kchs=depth,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dim=dim,
            downsample=downsample,
            **candidate,
        )
        return conv_working_rows(acc=False, **rows) <= spad_cap and conv_working_rows(acc=True, **rows) <= acc_cap

    while not fits(tile):
        order = [a for a in axes if not (a == "pocols" and tile[a] <= dim and tile["porows"] > 1)]
        axis = max(order, key=lambda a: tile[a])
        if axis == "pochs":  # channels move a whole block at a time; one fewer buys nothing
            tile[axis] = (tile[axis] // dim) * dim if tile[axis] % dim else tile[axis] - dim
            tile[axis] = tile[axis] or 1
        else:
            tile[axis] -= 1
        if min(tile.values()) < 1:
            return None
    while tile["pocols"] < whole["pocols"] and fits(tile | {"pocols": tile["pocols"] + 1}):
        tile["pocols"] += 1
    grew = True
    while grew:
        grew = False
        for axis in axes:
            if tile[axis] < whole[axis] and fits(tile | {axis: tile[axis] + 1}):
                tile[axis] += 1
                grew = True
    return tile | {"kchs": depth}


def conv_tile_candidates(
    *,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    in_channels: int,
    facts: Mapping[str, Any],
    pool_out_dim: int | None = None,
    pool_size: int = 1,
    pool_stride: int = 1,
    downsample: int = 0,
) -> list[dict[str, int]]:
    """One tiling per REDUCTION DEPTH, each with the deepest output tile that fits beside it.

    The depths are the ones that divide the input channels into ``n`` pieces on a ``DIM`` boundary, for
    every ``n`` from one piece (the whole reduction in each descriptor) down to one block per piece, plus
    the single channel the library's own search can reach. A depth off a block boundary is not offered:
    the mesh consumes the reduction ``DIM`` channels at a time, so a piece of 113 channels does the work
    of 128 and the eight pieces it takes do the work of nine.
    """
    dim = facts["dim"]
    depths: list[int] = []
    for pieces in range(1, _ceil(in_channels, dim) + 1):
        depth = min(in_channels, max(1, _ceil(_ceil(in_channels, pieces), dim) * dim))
        if depth not in depths:
            depths.append(depth)
    if 1 not in depths:
        depths.append(1)
    out = []
    for depth in depths:
        tile = _conv_output_tile(
            batch=batch,
            out_dim=out_dim,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            depth=depth,
            facts=facts,
            pool_out_dim=pool_out_dim,
            pool_size=pool_size,
            pool_stride=pool_stride,
            downsample=downsample,
        )
        if tile is not None:
            out.append(tile)
    return out


def choose_conv_tiles(
    *,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    in_channels: int,
    facts: Mapping[str, Any],
    pool_out_dim: int | None = None,
    pool_size: int = 1,
    pool_stride: int = 1,
    downsample: int = 0,
) -> dict[str, int]:
    """The tile this recipe walks a layer with: the cheapest of :func:`conv_tile_candidates` by traffic.

    The trade this search exists to make is REDUCTION DEPTH against OUTPUT TILE. One descriptor's weights
    occupy ``ceil(pochs/DIM) * krows * kcols * kchs`` scratchpad rows, so holding the reduction whole on a
    deep layer buys it by shrinking the output tile instead -- measured, a 512->512 3x3 stride-2 layer
    fell to 16 output channels and 2 output rows, 128 descriptors that each re-move the whole input
    window, and ran 8.4x the vendor library on the same emulator. Splitting the reduction and spending
    the room on the output tile is what the library does and is worth most of that.

    The objective is an explicit traffic model (:func:`conv_tile_traffic`), not a measurement, and it
    prices one thing the row counts do not: a split reduction FORFEITS the second loop slot. The
    sequencer runs :func:`conv_loop_slots` loops at once, and descriptors on distinct output tiles are
    independent, so they occupy both; the descriptors of one reduction chain all accumulate into the same
    accumulator half and must serialise against each other. So a tiling whose reduction is whole is
    charged its traffic divided by the slot count, and a split one its traffic outright. That term is
    what keeps the three ResNet-50 layers whose split saves little movement -- a 256-channel reduction
    cut into 240 and 16 -- from being split for a modelled gain that measurement does not show.

    Ties go to the whole reduction, then to the schedule with fewer descriptors, then to the larger
    output tile. It only picks a starting point; measured alternatives replace it.
    """
    if min(batch, out_dim, out_channels, in_channels) < 1:
        raise IsaError(
            f"a {out_dim}x{out_dim} {out_channels}-channel output over {batch} batch(es) of "
            f"{in_channels} input channel(s) is empty"
        )
    slots = conv_loop_slots(facts)
    pool = {"pool_out_dim": pool_out_dim, "pool_size": pool_size, "pool_stride": pool_stride}
    best = None
    for tile in conv_tile_candidates(
        batch=batch,
        out_dim=out_dim,
        out_channels=out_channels,
        kernel=kernel,
        stride=stride,
        in_channels=in_channels,
        facts=facts,
        downsample=downsample,
        **pool,
    ):
        traffic = conv_tile_traffic(
            batch=batch,
            out_dim=out_dim,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            in_channels=in_channels,
            tile=tile,
            facts=facts,
            downsample=downsample,
            **pool,
        )
        steps = traffic["reduction_steps"]
        rows = traffic["input"] + traffic["weights"] + traffic["bias"] + traffic["output"]
        cost = rows if steps > 1 else _ceil(rows, slots)
        key = (cost, steps, traffic["descriptors"], -tile["porows"] * tile["pocols"] * tile["pochs"])
        if best is None or key < best[0]:
            best = (key, tile)
    if best is None:
        smallest = conv_working_rows(
            acc=False,
            stride=stride,
            batches=1,
            porows=1,
            pocols=1,
            pochs=1,
            krows=kernel,
            kcols=kernel,
            kchs=1,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dim=facts["dim"],
            downsample=downsample,
        )
        raise IsaError(
            f"no tile of a {kernel}x{kernel} stride-{stride} {in_channels}->{out_channels} convolution "
            f"fits with the {kernel}x{kernel} kernel window whole: one output position of one channel "
            f"reducing one input channel already needs {smallest} scratchpad rows of "
            f"{facts['spad_rows_per_loop']}, and this recipe does not split the reduction across kernel "
            f"rows or columns"
        )
    return best[1]


def _affine(var: str, values: Mapping[int, int]) -> tuple[Any, bool]:
    """``values`` (index -> value) as ``t * d + c``, and whether that expression reproduces them all."""
    keys = sorted(values)
    lo = keys[0]
    step = values[keys[1]] - values[lo] if len(keys) > 1 else 0
    if len(keys) > 1:
        step //= keys[1] - lo
    expr = Var(var) * step + (values[lo] - lo * step)
    return expr, all(values[i] == values[lo] + (i - lo) * step for i in keys)


def _by_tile(var: str, values: list[int], *, what: str) -> Any:
    """One per-tile integer sequence as an index expression over that axis's loop variable.

    The IR's operands are affine in a loop variable with a ``select`` for one distinguished iteration,
    which is exactly the shape a tiled convolution's operands take: an extent or a border pad is the
    same on every tile but the first and the last, and a pointer offset steps by a constant except where
    the first tile's origin is clamped to the edge of the image. Anything else is REFUSED rather than
    approximated -- an operand that is off by one on one tile is a silently wrong output.
    """
    n = len(values)
    if n == 1 or len(set(values)) == 1:
        return as_expr(values[0])
    for ends in ((), (n - 1,), (0,), (0, n - 1)):
        core = {i: v for i, v in enumerate(values) if i not in ends}
        if not core:
            continue
        expr, exact = _affine(var, core)
        if not exact:
            continue
        if n - 1 in ends:
            expr = select(Var(var), n - 1, values[-1], expr)
        if 0 in ends:
            expr = select(Var(var), 0, values[0], expr)
        return expr
    raise IsaError(f"{what} takes {len(set(values))} values across {n} tiles, which is not one loop's operand")


def conv_reference(
    *,
    name: str,
    batch: int,
    in_dim: int,
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    padding: int,
    operands: Mapping[str, TensorArg],
    relu: bool,
    scale: float,
    facts: Mapping[str, Any],
    pool_size: int = 0,
    pool_stride: int = 0,
    pool_padding: int = 0,
    pin_input: bool | None = None,
) -> Kernel:
    """One layer as a NEST of device-convolution descriptors: output tiles outside, reduction inside.

    ``operands`` maps ``input``, ``weights``, ``output`` and optionally ``bias``. The layer is NHWC with
    row strides ``in_channels`` / ``out_channels``, and the weights are the library's ``[kh][kw][ci][co]``.

    Why a tile and not the whole layer in one descriptor: the sequencer issues its own mvin/mvout against
    HALF the scratchpad and half the accumulator (the other half belongs to the loop running beside it),
    and nothing checks the addresses it generates. A whole 56x56 64-channel layer asks for 12544
    accumulator rows against 512, so it wraps, and the layer comes back silently wrong -- measured, on
    every convolution of a ResNet-50 program. :func:`choose_conv_tiles` picks a tile that fits by the
    library's own arithmetic and this walks it in the library's own order.

    Why the REDUCTION is split across input channels when the tile search asks for it: a descriptor's
    weights occupy the scratchpad in proportion to its reduction depth, so holding the reduction whole on
    a deep layer pays for it out of the output tile -- and a small output tile re-moves the whole input
    window for every one of them. The accumulator half is what makes the split expressible: the sequencer
    advances to the other half only after a descriptor that STORES, so a chain of descriptors that
    withhold the output all accumulate into the same half. Within one chain,

    - the FIRST descriptor carries the bias pointer, which is the only thing that initialises the half
      (the mvin overwrites it; ``no_bias`` makes the same mvin write zeros, and a NULL pointer skips the
      load altogether and would accumulate onto whatever the half last held);
    - every LATER descriptor passes NULL for the bias, so its contribution adds to the partial;
    - only the LAST descriptor carries the output pointer, which stores the finished sum and releases the
      half.

    ``check_loop_conv_ws`` enforces exactly that shape, including that the depths of a chain add up to the
    layer's input channels, so a chain that drops or repeats a slice is refused rather than emitted.

    The kernel WINDOW is never split: ``krows`` and ``kcols`` are the kernel extent in every descriptor.
    Splitting them moves each descriptor's input origin and border with the kernel position, which
    neither this recipe nor the checker states; a layer that would need it is refused by name.

    The READOUT may POOL. ``pool_stride = 0`` is how the library spells "no pooling", and it then runs
    the same code with a ``1x1`` stride-1 window, for which the pooled output extent collapses onto the
    convolution's own and every pool border is zero. When a window is asked for, the loop nest walks the
    POOLED output axes instead: a descriptor accumulates the ``porows * pool_stride + pool_size - 1``
    convolution outputs its pooled tile reads, and the store stage mvouts the pooled block -- so
    adjacent tiles' readout windows OVERLAP by ``pool_size - pool_stride`` and those rows are computed
    twice, which is what the library does too. The unit must have BUILT the pooling store path; that is
    a build gate of the generator, not an ISA fact, so it is read from the target's elaborated RTL and a
    layer that needs a gate this target does not state is refused by that name.
    """
    eb, ab, dim = facts["elem_bytes"], facts["acc_bytes"], facts["dim"]
    consts = facts["constants"]
    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    if out_dim < 1:
        raise IsaError(f"{name}: a {kernel}x{kernel} stride-{stride} pad-{padding} conv over {in_dim} has no output")
    no_pool = pool_stride == 0
    if no_pool:
        # `tiled_conv`'s own normalisation, verbatim.
        pool_size, pool_stride, pool_padding = 1, 1, 0
    elif facts["readout_pooling"] is not True:
        raise IsaError(
            f"{name}: the readout pools, and this target's elaborated RTL does not state a pooling "
            f"store path (`elaborated_rtl_features.max_pool` is {facts['readout_pooling']!r}) -- the "
            f"store controller's `pooling_is_enabled` is a build gate, and a unit built without it "
            f"answers a pooling descriptor with the UNPOOLED rows rather than refusing it"
        )
    if min(pool_size, pool_stride) < 1:
        raise IsaError(f"{name}: a {pool_size}x{pool_size} stride-{pool_stride} pooling window is empty")
    if not 0 <= pool_padding < pool_size:
        raise IsaError(f"{name}: pool_padding {pool_padding} is not inside the {pool_size}-wide window")
    pool_out_dim = (out_dim + 2 * pool_padding - pool_size) // pool_stride + 1
    if pool_out_dim < 1:
        raise IsaError(
            f"{name}: a {pool_size}x{pool_size} stride-{pool_stride} pad-{pool_padding} pool over "
            f"{out_dim} has no output"
        )
    pool = {"pool_out_dim": pool_out_dim, "pool_size": pool_size, "pool_stride": pool_stride}
    act = consts["RELU"] if relu else consts["NO_ACTIVATION"]
    # THE STRIDED LOAD. The input loader stages one transfer per input pixel; a stride-2 convolution
    # reads every other one, so three of every four staged pixels are moved and never read. Ask for the
    # strided gather where the library's own predicate admits it -- measured at 1,022,272 cycles across
    # this model's three 1x1 stride-2 layers, bit-exact, one bit. An unreadable predicate refuses the
    # mode rather than guessing at it, so the layer merely stays as slow as it was.
    try:
        downsample = _DS.downsample_flag(
            kernel=kernel,
            stride=stride,
            padding=padding,
            in_rows=in_dim,
            in_cols=in_dim,
            pooled=not no_pool,
            header_text=facts.get("conv_header_text"),
        )
    except _DS.DownsampleUnknown:
        downsample = 0
    tile = choose_conv_tiles(
        batch=batch,
        out_dim=out_dim,
        out_channels=out_channels,
        kernel=kernel,
        stride=stride,
        in_channels=in_channels,
        facts=facts,
        # THE SEARCH IS DELIBERATELY NOT TOLD. Telling it frees scratchpad room and it picks larger
        # tiles -- and that was MEASURED, on hardware, and it is worse: FireSim job 729 came back at
        # 24,483,567 cycles against job 728's 24,359,749, +123,818. Per group it helped g60 (-76,390)
        # and cost g18 (+136,697) and g35 (+60,696), outputs bit-identical throughout. The objective
        # prices DRAM row transfers, and with the input term a quarter of its former size it
        # over-weights what remains: it buys a bigger output tile with input traffic that is no longer
        # the binding cost. Pricing the unstrided window here is not an approximation left in by
        # accident, it is the tiling that measured better, and the refutation is kept rather than the
        # intuition. Fixing the objective is the way to reopen this; enabling the flag is not.
        downsample=0,
        **pool,
    )
    inp, wgt, out = operands["input"], operands["weights"], operands["output"]
    bias = operands.get("bias")
    if bias is None:
        raise IsaError(
            f"{name}: this convolution has no bias, and the sequencer initialises the accumulator half "
            f"ONLY from the bias mvin -- it skips that load entirely when the pointer is NULL and its "
            f"compute always accumulates, so the layer would add onto whatever the half last held. "
            f"A zeroing descriptor needs a non-NULL pointer this recipe has no tensor to name"
        )

    b_tiles = _tiles_of(batch, tile["batches"])
    r_tiles = _tiles_of(pool_out_dim, tile["porows"])
    c_tiles = _tiles_of(pool_out_dim, tile["pocols"])
    o_tiles = _tiles_of(out_channels, tile["pochs"])
    k_tiles = _tiles_of(in_channels, tile["kchs"])

    def borders(tiles: list[tuple[int, int]]) -> list[dict[str, int]]:
        return [
            conv_tile_padding(
                in_dim=in_dim,
                out_dim=out_dim,
                kernel=kernel,
                stride=stride,
                padding=padding,
                start=s,
                extent=e,
                pool_padding=pool_padding,
                **pool,
            )
            for s, e in tiles
        ]

    rows, cols = borders(r_tiles), borders(c_tiles)

    def per(var: str, values: list[int], what: str):
        return _by_tile(var, values, what=f"{name}: {what}")

    # The image row and column a tile's input pointer starts at is its origin CLAMPED to the image; the
    # border the descriptor names covers the rest, and the FSM feeds zeros for it.
    in_row = per("r0", [max(p["origin"], 0) * in_dim * in_channels * eb for p in rows], "input row offset")
    in_col = per("c0", [max(p["origin"], 0) * in_channels * eb for p in cols], "input column offset")
    in_batch = per("b0", [s * in_dim * in_dim * in_channels * eb for s, _ in b_tiles], "input batch offset")
    # The output is addressed in the POOLED space -- `LoopConvSt`'s pooled mvout walks it at the
    # `pool_out_col_dim` row pitch, and with no pooling that IS the convolution's output extent.
    out_row = per("r0", [s * pool_out_dim * out_channels * eb for s, _ in r_tiles], "output row offset")
    out_col = per("c0", [s * out_channels * eb for s, _ in c_tiles], "output column offset")
    out_batch = per(
        "b0", [s * pool_out_dim * pool_out_dim * out_channels * eb for s, _ in b_tiles], "output batch offset"
    )
    # THE INPUT STAYS. The input is re-staged once per OUTPUT-CHANNEL tile -- the channel tile is not an
    # axis of the input, so every one of them moves the same window again (:func:`conv_tile_traffic`
    # prices it as `... * len(o_t) * kch_ld`). On this model that is 819,654 of 964,808 input row
    # transfers. The device removes it: `a_ex_spad_id` PINS the input to a named scratchpad half instead
    # of letting it follow the loop slot, and a NULL input pointer makes `LoopConvLdInput` skip its load
    # entirely -- so the first output-channel tile stages the window and the rest read it where it lies.
    # Nothing in the loader's or the execute stage's A address depends on `pochs`, which is what makes
    # the resident rows the right rows (see :func:`_conv_input_residency`).
    #
    # ONLY WHERE THE WHOLE KERNEL STAGES ONE WINDOW, and that bound is MEASURED, not inferred. A
    # descriptor's input window is a function of the batch, row, column and reduction tile, so a nest
    # walking `len(b_tiles)*len(r_tiles)*len(c_tiles)*len(k_tiles)` of them re-stages the pinned half
    # that many times, and a reload lands on the half a descriptor of the previous window may still be
    # reading. The reservation station tracks that overlap as a WAR, so the reload SHOULD be ordered
    # behind the read -- and the ResNet-50 group model run as job 730 says otherwise: with every
    # output-channel-tiled convolution pinned, the stem (40 windows) came back with 5,652,929 against
    # the oracle's 5,663,048 on its own checksum, the whole model's cosine fell 997,981 -> 988,504 ppm,
    # and the run was discarded. The one shape measured BIT-EXACT is the one that stages once
    # (14x14 256->256 3x3, one window, 16 output-channel tiles).
    #
    # So the bound is the vendor library's own, and it is not the spelling limitation it looks like:
    # `tiled_conv` gates its `a_reuse` on `num_kch*num_krow*num_kcol*num_b*num_porow*num_pocol <= 2`,
    # where 2 is `concurrent_loops` -- one resident window per scratchpad half, and no half ever
    # reloaded. This recipe takes the strict case of that, ONE window, because the two-window form needs
    # the halves alternated per window and there is no shape in this model to measure it on. Everything
    # else keeps the re-staging and says so.
    #
    # ``pin_input`` overrides the recipe's own answer: ``False`` re-stages per output-channel tile, which
    # is what a caller passes to hold the mode as the CONTROL of an A/B. ``True`` asks for it and is
    # refused by name where the recipe cannot state it.
    windows = len(b_tiles) * len(r_tiles) * len(c_tiles) * len(k_tiles)
    pinned = windows == 1 and len(o_tiles) > 1
    if pin_input is False:
        pinned = False
    elif pin_input is True and not pinned:
        raise IsaError(
            f"{name}: the input cannot be pinned to a scratchpad half here -- "
            + (
                f"this nest walks {windows} input windows ({len(b_tiles)} batch x {len(r_tiles)} row x "
                f"{len(c_tiles)} column x {len(k_tiles)} reduction tiles), so the pinned half would be "
                f"re-staged {windows - 1} times over descriptors that may still be reading it; measured "
                f"on the whole model, that is a wrong answer, not a slow one"
                if windows > 1
                else "there is only one output-channel tile, so nothing re-stages the input anyway"
            )
        )
    o_resident = 1 if pinned else 0

    def offsets(o_sel: list[tuple[int, int]]) -> dict[str, Any]:
        """The three output-channel-indexed operand offsets and the channel extent, over ``o_sel``."""
        return {
            "pochs": per("o0", [e for _, e in o_sel], "channel extent"),
            "out_ch": per("o0", [s * eb for s, _ in o_sel], "output channel offset"),
            "wgt_ch": per("o0", [s * eb for s, _ in o_sel], "weight output-channel offset"),
            "bias_ch": per("o0", [s * ab for s, _ in o_sel], "bias offset"),
        }

    def descriptor(kch: Any, kchs: int, *, first: bool, last: bool, o_sel: list[tuple[int, int]], stage: bool):
        """One descriptor at reduction offset ``kch`` (an expression in elements) of depth ``kchs``.

        ``tiled_conv``'s own slicing: the weights step a whole ``[ci][co]`` plane's worth of rows per
        input channel (``weight_stride`` of them), the input steps one element, and the border and
        extents are the output tile's, which the reduction does not move.

        ``o_sel`` is the slice of the output-channel axis the ``o0`` loop around this descriptor walks,
        and ``stage`` says whether this descriptor MOVES the input window or reads the one a previous
        descriptor left in the half ``o_resident`` names.
        """
        o = offsets(o_sel)
        out_ch, wgt_ch, bias_ch = o["out_ch"], o["wgt_ch"], o["bias_ch"]
        # `max_pixels_per_row` is the header's own, on THIS descriptor's reduction depth (the header reads
        # it off `ichs`, which is `kchs`): how many kernel columns fit one mesh row, which is one unless a
        # descriptor's channels fit a single block -- and never more than the kernel is wide.
        pixels = 1 if kchs > dim else min(dim // kchs, kernel)
        return call(
            "loop_conv_ws",
            batch_size=batch,
            in_row_dim=in_dim,
            in_col_dim=in_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            out_row_dim=out_dim,
            out_col_dim=out_dim,
            pool_out_row_dim=pool_out_dim,
            pool_out_col_dim=pool_out_dim,
            stride=stride,
            padding=padding,
            kernel_dim=kernel,
            kernel_dilation=1,
            # With `no_pool` set the store stage never reads the window, so this spells its absence as
            # zeros rather than as the library's normalised 1x1; with pooling it is the window itself.
            pool_size=0 if no_pool else pool_size,
            pool_stride=0 if no_pool else pool_stride,
            pool_padding=0 if no_pool else pool_padding,
            batches=per("b0", [e for _, e in b_tiles], "batch extent"),
            porows=per("r0", [e for _, e in r_tiles], "row extent"),
            pocols=per("c0", [e for _, e in c_tiles], "column extent"),
            pochs=o["pochs"],
            krows=kernel,
            kcols=kernel,
            kchs=kchs,
            lpad=per("c0", [p["lpad"] for p in cols], "left border"),
            rpad=per("c0", [p["rpad"] for p in cols], "right border"),
            upad=per("r0", [p["upad"] for p in rows], "top border"),
            dpad=per("r0", [p["dpad"] for p in rows], "bottom border"),
            plpad=per("c0", [p["plpad"] for p in cols], "left pool border"),
            prpad=per("c0", [p["prpad"] for p in cols], "right pool border"),
            pupad=per("r0", [p["pupad"] for p in rows], "top pool border"),
            pdpad=per("r0", [p["pdpad"] for p in rows], "bottom pool border"),
            # The readout window this tile accumulates, in the convolution's own output space. With no
            # pooling it is the tile's output extent; with a window it is that extent walked with the
            # pool stride plus the window's reach, less whatever of it falls outside the layer.
            orows=per("r0", [p["readout"] for p in rows], "row readout extent"),
            ocols=per("c0", [p["readout"] for p in cols], "column readout extent"),
            weights=Ptr(wgt.name, add(wgt_ch, mul(kch, out_channels * eb))),
            output=Ptr(out.name, add(out_batch, out_row, out_col, out_ch)) if last else NULL,
            bias=Ptr(bias.name, bias_ch) if first else NULL,
            input=Ptr(inp.name, add(in_batch, in_row, in_col, mul(kch, eb))) if stage else NULL,
            no_bias=0,
            no_pool=1 if no_pool else 0,
            downsample=downsample,
            wrot180=0,
            input_dilated=0,
            activation=act,
            trans_output_1203=0,
            trans_weight_1203=0,
            trans_weight_0132=0,
            trans_input_3120=0,
            max_pixels_per_row=pixels,
            in_stride=in_channels,
            weight_stride=out_channels,
            out_stride=out_channels,
            dw=0,
            a_spad_id=o_resident,
            b_spad_id=0,
        )

    # The reduction, innermost: first carries the bias, last carries the output, the ones between neither.
    # The middle steps are one loop because they are identical but for an affine offset; the two ends are
    # not, because which pointers they carry is not an affine function of anything.
    depth, last_depth = tile["kchs"], k_tiles[-1][1]

    def chain(o_sel: list[tuple[int, int]], *, stage: bool) -> tuple:
        if len(k_tiles) == 1:
            return (descriptor(as_expr(0), last_depth, first=True, last=True, o_sel=o_sel, stage=stage),)
        middle = ()
        if len(k_tiles) > 2:
            step = mul(add(Var("k0"), 1), depth)
            middle = (
                loop(
                    "k0",
                    len(k_tiles) - 2,
                    descriptor(step, depth, first=False, last=False, o_sel=o_sel, stage=stage),
                ),
            )
        return (
            descriptor(as_expr(0), depth, first=True, last=False, o_sel=o_sel, stage=stage),
            *middle,
            descriptor(as_expr(k_tiles[-1][0]), last_depth, first=False, last=True, o_sel=o_sel, stage=stage),
        )

    def over(var: str, count: int, *body) -> tuple:
        return (loop(var, count, *body),) if count > 1 else tuple(body)

    # Batch, then rows, then columns, then output channels, then the reduction -- the library's own order,
    # so the accumulator halves rotate the way its kernels leave them. With the input pinned, the FIRST
    # output-channel tile is peeled out of that loop: it is the one that moves the window, and the rest
    # read it where it lies. Peeling is what discharges the obligation the checker cannot see -- the
    # stager is lexically the first descriptor of the loop nest that reads it, on every path.
    if pinned:
        nest: tuple = (*chain(o_tiles[:1], stage=True), *over("o0", len(o_tiles) - 1, *chain(o_tiles[1:], stage=False)))
    else:
        nest = tuple(over("o0", len(o_tiles), *chain(o_tiles, stage=True)))
    for var, count in (("c0", len(c_tiles)), ("r0", len(r_tiles)), ("b0", len(b_tiles))):
        if count > 1:
            nest = (loop(var, count, *nest),)
    body = [
        call(
            "config_ex",
            dataflow=consts["WEIGHT_STATIONARY"],
            sys_act=consts["NO_ACTIVATION"],
            sys_shift=0,
            # The convolution's OWN row stride, which is what the sequencer walks the input with. The
            # matmul path leaves this 1; leaving it 1 here reads every strided layer off by a row.
            # With the strided load the loader has ALREADY applied the stride, so the execute unit
            # must read consecutive staged rows. These two are one fact, never separable.
            A_stride=stride >> downsample,
            A_transpose=0,
            B_transpose=0,
        ),
        call("config_st", stride=out_channels * eb, acc_act=act, acc_scale=float(np.float32(scale))),
        *nest,
    ]
    args = (inp, wgt, bias, out)
    return Kernel(
        name=name,
        args=args,
        body=tuple(body),
        attrs=(
            ("recipe", "conv_loop_conv_ws_split_reduction_v3"),
            ("shape", f"{batch}x{in_dim}x{in_dim}x{in_channels}->{out_channels} k{kernel} s{stride} p{padding}"),
            (
                "readout",
                "narrow, unpooled"
                if no_pool
                else f"max pool {pool_size}x{pool_size} s{pool_stride} p{pool_padding} -> {pool_out_dim}",
            ),
            ("tile", ",".join(f"{k}={v}" for k, v in tile.items())),
            ("descriptors", str(len(b_tiles) * len(r_tiles) * len(c_tiles) * len(o_tiles) * len(k_tiles))),
            ("reduction_steps", str(len(k_tiles))),
            (
                "input_staging",
                f"pinned to scratchpad half {o_resident - 1}, staged once for the whole kernel"
                if pinned
                else f"restaged per output-channel tile ({len(o_tiles)} of them): "
                + (
                    "the caller asked for the re-staging"
                    if pin_input is False
                    else f"this nest walks {windows} input windows, so the pinned half would be reloaded"
                    if windows > 1
                    else "there is only one output-channel tile"
                ),
            ),
            # What was measured, per readout. The unpooled path is the whole-model group comparison;
            # the pooled one was run against the library call for the same group on gsim over
            # byte-identical operands and compared element for element (ResNet-50's stem, 0 of 200,704
            # elements differing, on operands chosen so the requantized readout does not saturate).
            (
                "numerics",
                "matches the vendor library on gsim, ResNet-50 group model"
                if no_pool
                else "matches the vendor library on gsim, element for element",
            ),
        ),
    )


def choose_resadd_tiles(rows: int, cols: int, facts: Mapping[str, Any]) -> tuple[int, int]:
    """The element-space tile ``tiled_resadd_stride_auto`` would pick, by its own arithmetic.

    Transcribed rather than re-derived, for the same reason :func:`conv_tile_padding` is: a tile this
    gets wrong is a silently wrong output or an accumulator overrun, not a refusal. The library starts
    from the whole array and shrinks -- halving the row extent when it is the larger side (or when the
    column extent is already one tile wide), otherwise taking one ``DIM`` column off -- until both
    addends fit the half-accumulator budget. ``acc_rows_per_loop`` IS the header's
    ``max_acc_rows = ACC_ROWS / 2``, so the budget here is the library's own constant, derived.
    """
    dim, budget = facts["dim"], facts["acc_rows_per_loop"]
    if min(rows, cols) < 1:
        raise IsaError(f"resadd over an empty array {rows}x{cols}")

    def acc_rows(ti: int, tj: int) -> int:
        return _ceil(ti, dim) * dim * _ceil(tj, dim)

    ti, tj = rows, cols
    while acc_rows(ti, tj) > budget:
        if ti >= tj or tj <= dim:
            ti //= 2
        else:
            tj -= dim
        if ti < 1 or tj < 1:
            # Fail closed. The library's loop has no such guard and would spin or divide to zero; a
            # single DIM x DIM tile not fitting the accumulator is a machine fact worth stating, not
            # a condition to iterate past.
            raise IsaError(
                f"no resadd tile fits {budget} accumulator rows: one {dim}x{dim} tile needs {acc_rows(dim, dim)}"
            )
    return ti, tj


def resadd_reference(
    *,
    name: str,
    rows: int,
    cols: int,
    operands: Mapping[str, TensorArg],
    relu: bool,
    a_scale: float = 1.0,
    b_scale: float = 1.0,
    c_scale: float = 1.0,
    facts: Mapping[str, Any],
    stride: int | None = None,
    tiles: tuple[int, int] | None = None,
) -> Kernel:
    """``C[rows][cols] = readout(a_scale*A + b_scale*B)`` as LOOP_WS macro-instructions.

    The residual add ResNet-50 issues at every block join, expressed the way the device expresses it:
    ONE ``loop_ws`` per tile with ``is_resadd`` set, which is exactly what the library's own
    ``sp_tiled_resadd`` emits. There is no contraction -- ``K`` is 0 and ``D`` is NULL -- and the two
    addends are mvin'd to opposite halves of the accumulator, scaled by the two ``config_ld`` scales,
    with the sum read out through ``config_st``'s activation and accumulator scale.

    ``operands`` maps ``a``, ``b`` and ``c``; all three are dense row-major sharing one row ``stride``
    (defaulting to ``cols``, as ``tiled_resadd_auto`` does). Numerics are not validated here -- see the
    attribute the kernel carries.

    **Why this is not folded into the producing contraction's readout.** The saving would be large:
    measured over ResNet-50's 71 compute groups, the 16 residual adds are 3,782,502 of 25,419,657
    cycles (14.8%) for one addition per element, because the block output round-trips through DRAM --
    stored by the contraction, read back beside the skip, stored again. Having the skip already IN the
    accumulator when the contraction reduces onto it would remove two of those four movements. The
    device does not express it, and the field that says so is the ``accumulate`` argument each unroller
    passes to ``cast_to_acc_addr``, read off the registered revision's own sources by
    :mod:`gemmini_accumulator_ports`:

    - a contraction's only DRAM -> accumulator port is ``LoopMatmulLdD``, the ``D`` operand, and it
      passes ``false.B``: it REPLACES the region. One accumulator region therefore admits exactly ONE
      initialising tensor -- a second ``D`` later in the same reduction chain discards the first -- so
      the skip tensor and the per-channel bias cannot both occupy it. The port is otherwise the right
      shape (it walks both output iterators against a caller's stride and takes its width from
      ``low_d``, so a full ``elem_t`` skip is precisely what it would carry), but all 16 of ResNet-50's
      producing contractions carry a bias, so all 16 have already spent it;
    - the convolution sequencer's port, ``LoopConvLdBias``, is weaker still: its DRAM address is
      ``och * (acc_w/8).U`` -- the output-channel iterator alone, with the spatial iterators absent --
      and the row stride it configures is ``0.U``. It broadcasts one vector over every position and
      cannot carry a tensor at any operand values;
    - the one accumulate-capable load in either unroller is ``LoopMatmulLdB`` under ``is_resadd``,
      which is THIS recipe's own second addend. It is wired to ``resadd_addr_start``, a reset-time
      constant per loop slot, while a contraction's region is ``ld_d_addr_start`` / ``ex_c_addr_start``,
      free-running registers that no ISA field reads or sets. So neither that port nor a hand-issued
      accumulating mvin can be aimed at the region a ``loop_ws`` descriptor is about to write.

    ``merlin/tests/gemmini/test_accumulator_fusion_ceiling.py`` pins those literals, so a revision that
    changes any of them fails there rather than leaving this paragraph to be believed. Note also that
    the fold would not be a pure schedule change even with the port free: the unfused path requantizes
    the contraction's output to ``elem_t`` and saturates it BEFORE the add scales it, so folding
    removes a rounding and a clip that this model's numerics are defined in terms of.

    **Nor can this group's DMA be OVERLAPPED with a neighbour's compute**, which is the weaker lever
    and fails for its own reasons (:mod:`gemmini_loop_concurrency`,
    ``test_residual_add_overlap_ceiling.py``). Run-ahead across descriptors does exist -- each unroller
    holds ``concurrent_loops`` = 2 -- but ``is_resadd`` is declared at MODULE scope in ``LoopMatmul``
    and ``LoopMatmulState.reset()`` does not clear it, so every stage applies it to whichever
    descriptor that stage is serving; with a contraction in the other slot,
    ``ex.io.req.bits.skip := is_resadd`` skips the CONTRACTION's mesh phase. And both unrollers gate
    non-loop commands on ``!loop_configured`` while they are chained, so a convolution group and a
    ``loop_ws`` group are serialized whatever the emitted C does about fences. The
    ``gemmini_fence()`` each emitted kernel ends with is therefore load-bearing for CORRECTNESS at a
    contraction-to-residual-add boundary, not merely a timing bracket -- do not remove it to chase
    overlap.
    """
    dim, eb = facts["dim"], facts["elem_bytes"]
    consts = facts["constants"]
    a, b, c = operands["a"], operands["b"], operands["c"]
    row_stride = cols if stride is None else stride
    ti, tj = tiles or choose_resadd_tiles(rows, cols, facts)
    i_tiles, j_tiles = _ceil(rows, ti), _ceil(cols, tj)
    last_i, last_j = rows - (i_tiles - 1) * ti, cols - (j_tiles - 1) * tj
    act = consts["RELU"] if relu else consts["NO_ACTIVATION"]

    def geom(extent: int) -> tuple[int, int]:
        """A tile's loop count and its pad, the way ``sp_tiled_resadd`` computes them."""
        tiles_ = _ceil(extent, dim)
        return tiles_, tiles_ * dim - extent

    (I_full, pI_full), (I_last, pI_last) = geom(ti), geom(last_i)
    (J_full, pJ_full), (J_last, pJ_last) = geom(tj), geom(last_j)
    i0, j0 = Var("i0"), Var("j0")
    body = [
        # The activation rides the ACCUMULATOR readout, not the array: the library configures the
        # systolic activation off and lets config_st apply it, because in a resadd nothing flows
        # through the mesh at all.
        call("config_st", stride=row_stride * eb, acc_act=act, acc_scale=float(np.float32(c_scale))),
        call(
            "config_ex",
            dataflow=consts["WEIGHT_STATIONARY"],
            sys_act=consts["NO_ACTIVATION"],
            sys_shift=0,
            A_stride=1,
            A_transpose=0,
            B_transpose=0,
        ),
        # `shrunk` says the addends arrive as elem_t while landing in an acc_t-wide half.
        call("config_ld", stride=row_stride * eb, scale=float(np.float32(a_scale)), shrunk=1, id=0),
        call("config_ld", stride=row_stride * eb, scale=float(np.float32(b_scale)), shrunk=1, id=1),
        loop(
            "i0",
            i_tiles,
            loop(
                "j0",
                j_tiles,
                call(
                    "loop_ws",
                    I=select(i0, i_tiles - 1, I_last, I_full),
                    J=select(j0, j_tiles - 1, J_last, J_full),
                    K=0,
                    pad_I=select(i0, i_tiles - 1, pI_last, pI_full),
                    pad_J=select(j0, j_tiles - 1, pJ_last, pJ_full),
                    pad_K=0,
                    A=Ptr(a.name, (i0 * (ti * row_stride) + j0 * tj) * eb),
                    B=Ptr(b.name, (i0 * (ti * row_stride) + j0 * tj) * eb),
                    D=NULL,
                    C=Ptr(c.name, (i0 * (ti * row_stride) + j0 * tj) * eb),
                    A_stride=row_stride,
                    B_stride=row_stride,
                    D_stride=0,
                    C_stride=row_stride,
                    A_transpose=0,
                    B_transpose=0,
                    full_C=0,
                    low_D=0,
                    ex_accumulate=0,
                    act=act,
                    a_spad_id=0,
                    b_spad_id=0,
                    is_resadd=1,
                ),
            ),
        ),
    ]
    return Kernel(
        name=name,
        args=(a, b, c),
        body=tuple(body),
        attrs=(
            ("recipe", "resadd_loop_ws_reference_v1"),
            ("tiles", f"{ti},{tj}"),
            ("shape", f"{rows}x{cols}"),
            ("numerics", "matches the vendor library on gsim, ResNet-50 group model"),
        ),
    )


def window_mean_reference(
    *,
    name: str,
    rows: int,
    channels: int,
    operands: Mapping[str, TensorArg],
    facts: Mapping[str, Any],
    window: int | None = None,
    stride: int | None = None,
    scale: float | None = None,
    tiles: tuple[int, int, int] | None = None,
) -> Kernel:
    """``C[1][channels] = mean over the plane of X[rows][channels]``, as a contraction.

    The mean-pool ResNet-50 ends on, expressed with no new instruction and no host arithmetic. This
    target's headers offer **no average pool** -- the only pooling ``config_st`` and the convolution
    sequencer carry is a MAX pool -- so the reduction is written as what it already is: a contraction
    against a vector of ones, ``ones[1][rows] @ X[rows][channels]``, with the division folded into
    ``config_st``'s accumulator scale as ``1/rows``. The sum is therefore exact in the accumulator's
    integer domain and the only rounding is the readout's, which is where a requantizing pool's
    rounding belongs.

    ``operands`` maps ``a`` (the ones vector, which the CALLER materialises -- it is a real buffer, not
    a constant this recipe can conjure), ``b`` (the plane) and ``c``. ``scale`` defaults to ``1/rows``,
    the plain mean; a caller whose readout already folds the division together with a requantization
    passes that combined multiplier instead, because recomputing ``1/rows`` underneath it would apply
    the division twice.

    Two honest limits. **The mesh is mostly idle**: the contraction has one output row, so it occupies a
    single row of a ``dim``-row array. This is a correctness reference for a shape that is 0.003% of
    ResNet-50's MACs, not a schedule anyone should tune. And **only a whole-plane mean is a
    contraction**: a sliding window is a depthwise convolution against ones weights, which the device
    does express (``loop_conv_ws`` carries ``dw``) but which the checker refuses because it does not
    model it -- so this refuses rather than silently reducing over the wrong extent.
    """
    if window is not None and window != rows:
        raise IsaError(
            f"a sliding window mean (window={window}, plane={rows}) is not a contraction: it is a "
            f"depthwise convolution against ones weights, and `dw` is not modelled by this checker"
        )
    if stride not in (None, rows):
        raise IsaError(f"a strided window mean (stride={stride}) is not a whole-plane reduction")
    if rows < 1:
        raise IsaError(f"window mean over an empty plane {rows}x{channels}")
    kernel = matmul_reference(
        name=name,
        m=1,
        n=channels,
        k=rows,
        operands=operands,
        relu=False,
        scale=(1.0 / rows) if scale is None else scale,
        facts=facts,
        tiles=tiles,
    )
    attrs = dict(kernel.attrs)
    attrs["recipe"] = "window_mean_ones_contraction_v1"
    attrs["reduction"] = f"mean over {rows}"
    if scale is not None:
        attrs["readout_scale"] = "caller's own, the division already folded in"
    attrs["numerics"] = "matches the vendor library on gsim, ResNet-50 group model"
    return Kernel(name=kernel.name, args=kernel.args, body=kernel.body, attrs=tuple(attrs.items()))
