"""Gemmini operand ABI interpretation for the selected host support backend."""
from __future__ import annotations

import struct

from merlin.common.facts_view import interface as _facts_interface

GARBAGE = MASK32 = 0xFFFFFFFF
_MVIN_CLASSES = {"MVIN", "MVIN2", "MVIN3"}

_F32_ONE_BITS = struct.unpack("<I", struct.pack("<f", 1.0))[0]  # 0x3F800000 — IEEE-754 float32(1.0)


def derived_readout_bits(addr_len: int) -> dict[str, int]:
    """The RoCC accumulator-readout bit constants DERIVED from ``addr_len`` + the 3-flag-bit accumulator-
    address convention + float32(1.0). Byte-identical to the former hand-declared ``readout_bits`` hex:

      * ``acc_i8``     = ``1 << (addr_len-1)`` — accumulator-select / scaled-i8 readout base (bit 31)
      * ``acc_accum``  = ``1 << (addr_len-2)`` — accumulate-onto (vs overwrite) (bit 30)
      * ``full_c_bit`` = ``1 << (addr_len-3)`` — full-i32 (vs scaled-i8) readout width (bit 29)
      * ``c_acc``      = ``acc_i8 | full_c_bit`` — full-i32 accumulator readout base (0xA0000000)
      * ``f1``         = float32(1.0) bits (0x3F800000) — the identity acc_scale
    """
    acc_i8 = 1 << (addr_len - 1)
    acc_accum = 1 << (addr_len - 2)
    full_c_bit = 1 << (addr_len - 3)
    return {
        "f1": _F32_ONE_BITS,
        "c_acc": acc_i8 | full_c_bit,
        "acc_i8": acc_i8,
        "acc_accum": acc_accum,
        "full_c_bit": full_c_bit,
    }

def encoding_fields(declared: dict) -> dict:
    """Complete this target's encoding without mutating the declared contract."""
    encoding = dict(declared)
    if "readout_bits" not in encoding and encoding.get("addr_len") is not None:
        encoding["readout_bits"] = derived_readout_bits(int(encoding["addr_len"]))
    return encoding

def isa_constants(target: str) -> dict:
    """Derive the RoCC ISA constants for ``target`` from its RTL facts + capability manifest. No target
    is baked in — the caller passes the target it is grading; the decoder holds no default."""
    from merlin.targetgen.rtl.facts import load_facts
    from merlin.targetgen.target_experiment import load_capability_manifest
    m = load_capability_manifest(target)
    enc = encoding_fields(m.encoding)
    rb = enc["readout_bits"]
    facts = load_facts(target)["facts"]
    # DIM (systolic mesh dimension) is a CIRCT-extracted FACT (arrays[mesh]), not a manifest field —
    # same source the codegen emitter reads, so the decoder's DIM cannot drift from the encoder's.
    mesh = next((a for a in facts.get("arrays", []) if a.get("name") == "mesh"), {})
    dim = mesh.get("rows")   # UNKNOWN (None) if the target declares no mesh — no baked gemmini DIM=16
    # The custom major opcode is a per-target FACT read from facts (funct_decode_table.custom_opcode),
    # NOT a baked literal: it is the RISC-V-standard encoding of the custom SLOT the target's RoCC is
    # wired to (SoC OpcodeSet), resolved by circt_introspect from the target's reviewed
    # encoding.rocc_custom_slot. It may be None (UNKNOWN) for a target that declares no slot — the
    # decoder then does not filter by major opcode (see _parse_insn). funct3 is the RoCC xd/xs1/xs2
    # register-usage field — it VARIES per instruction (e.g. a result-returning op sets xd=1), so it is
    # NOT an identity constraint; instruction identity is func7 (-> FUNCT_CLASS).
    fdt = _facts_interface(facts, "funct_decode_table") or {}
    layouts = next((i.get("bundles", {}) for i in facts.get("interfaces", [])
                    if i.get("name") == "register_bundle_layouts"), {})
    custom_opcode = fdt.get("custom_opcode")
    return {"DIM": dim, "F1": rb["f1"], "C_ACC": rb["c_acc"], "ACC_I8": rb["acc_i8"],
            "ACC_ACCUM": rb["acc_accum"], "FULL_C_BIT": rb["full_c_bit"],
            "CUSTOM_OPCODE": custom_opcode, "FUNCT3": fdt.get("funct3"),
            "FUNCT_CLASS": dict(enc["semantic_class"]), "CONFIG_SUBTYPE": dict(enc["config_subtype"]),
            # Extracted from the target's Scala Bundle by circt_introspect.  Keeping the layout beside
            # the other decoded ISA facts lets every consumer read CONFIG_ST without copying bit offsets.
            "CONFIG_ST_LAYOUT": layouts.get("ConfigMvoutRs1")}


def _bundle_fields(raw: int | None, layout: dict | None) -> dict[str, int]:
    """Unpack fields from a CIRCT/Scala-derived register-bundle layout.

    An absent or incomplete layout yields no fields.  Callers then leave those facts UNKNOWN rather
    than substituting the pinned target's offsets.
    """
    if raw is None or not isinstance(layout, dict):
        return {}
    out: dict[str, int] = {}
    for name, spec in (layout.get("fields") or {}).items():
        if not isinstance(spec, dict):
            continue
        offset, width = spec.get("offset"), spec.get("width")
        if isinstance(offset, int) and isinstance(width, int) and width > 0:
            out[str(name)] = (raw >> offset) & ((1 << width) - 1)
    return out

def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _pack_fields(v: int) -> dict:
    return {"rows": (v >> 48) & 0xFFFF, "cols": (v >> 32) & 0xFFFF, "addr": v & MASK32}


def decode_instruction(funct: int, rs1: dict, rs2: dict, isa: dict) -> tuple[str, dict]:
    """Return (class, decoded-fields) for one .insn given resolved operands and the target ``isa``."""
    full_c_bit, acc_accum = isa["FULL_C_BIT"], isa["ACC_ACCUM"]
    base = isa["FUNCT_CLASS"].get(funct, "UNKNOWN")
    r1 = rs1["raw"] if rs1["kind"] == "const" else None
    r2 = rs2["raw"] if rs2["kind"] == "const" else None
    dec: dict = {}

    if base == "CONFIG":
        sub = isa["CONFIG_SUBTYPE"].get((r1 & 0x3) if r1 is not None else -1, "CONFIG_UNKNOWN")
        if sub == "CONFIG_LD":
            # The mvin scale float is packed in rs1[63:32] (same high-word layout CONFIG_ST uses for the
            # store/acc scale below). Expose it so a degenerate load scale (e.g. 0.0, which multiplies every
            # loaded element to zero) is visible in the trace instead of only manifesting as all-zeros on the
            # hardware oracle. Identity is 1.0; absent/unresolved rs1 -> None (fail-open, never a wrong value).
            ld_scale_bits = ((r1 >> 32) & MASK32) if r1 is not None else None
            dec = {"subtype": "LD", "stride": r2,
                   "scale": _f32_from_bits(ld_scale_bits) if ld_scale_bits is not None else None,
                   "scale_bits": ld_scale_bits}
        elif sub == "CONFIG_ST":
            config_fields = _bundle_fields(r1, isa.get("CONFIG_ST_LAYOUT"))
            # activation's old spelling remains a compatibility path for facts records predating the
            # extracted bundle. Pool geometry has no fallback: without the RTL layout it is UNKNOWN.
            acc_act = config_fields.get("activation")
            if acc_act is None:
                acc_act = ((r1 >> 2) & 0x3) if r1 is not None else None
            scale_bits = ((r2 >> 32) & MASK32) if r2 is not None else None
            dec = {
                "subtype": "ST",
                "acc_act": acc_act,
                "relu": (acc_act == 1) if acc_act is not None else None,
                "acc_scale": _f32_from_bits(scale_bits) if scale_bits is not None else None,
                "acc_scale_bits": scale_bits,
                "out_stride_bytes": (r2 & MASK32) if r2 is not None else None,
            }
            for field in ("pool_stride", "pool_size", "pool_out_dim", "porows", "pocols",
                          "orows", "ocols", "upad", "lpad"):
                if field in config_fields:
                    dec[field] = config_fields[field]
        else:
            dec = {"subtype": "EX"}
        return (sub if sub != "CONFIG_UNKNOWN" else "UNKNOWN"), dec

    if base in _MVIN_CLASSES:
        dec = {"dram": dict(rs1)}
        if r2 is not None:
            dec.update(_pack_fields(r2))
            dec["spad_addr"] = r2 & MASK32
        return base, dec

    if base == "MVOUT":
        dec = {"dram": dict(rs1)}
        if r2 is not None:
            dec.update(_pack_fields(r2))
            acc_addr = r2 & MASK32
            dec["acc_addr"] = acc_addr
            dec["readout"] = "i32" if (acc_addr & full_c_bit) else "i8"
        return base, dec

    if base == "PRELOAD":
        dec = {}
        if r1 is not None:
            dec["weight_spad"] = r1 & MASK32
        if r2 is not None:
            c_addr = r2 & MASK32
            dec["c_addr"] = c_addr
            dec["accumulate"] = bool(c_addr & acc_accum)
            dec["readout"] = "i32" if (c_addr & full_c_bit) else "i8"
        return base, dec

    if base in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
        dec = {}
        if r1 is not None:
            dec["a_spad"] = r1 & MASK32
        if r2 is not None:
            dec["bd"] = r2 & MASK32
            dec["garbage"] = (r2 & MASK32) == GARBAGE
        return base, dec

    if base == "FLUSH":
        return base, {}

    return ("UNKNOWN" if base == "UNKNOWN" else base), dec


def _class_to_funct(isa: dict) -> dict[str, int]:
    """``class-name -> func7`` from the derived FUNCT_CLASS map (plus the CONFIG subtypes, all func7=0)."""
    inv = {v: k for k, v in isa["FUNCT_CLASS"].items()}
    config = inv.get("CONFIG")
    if config is not None:
        for sub in isa["CONFIG_SUBTYPE"].values():  # CONFIG_EX / CONFIG_LD / CONFIG_ST -> same func7
            inv[sub] = config
    return inv


def _config_subtype_bits(isa: dict, name: str) -> int | None:
    """The required ``rs1 & 0x3`` for a CONFIG subtype name, else None (not a CONFIG subtype)."""
    for bits, sub in isa["CONFIG_SUBTYPE"].items():
        if sub == name:
            return bits
    return None


def instruction_funct(name: str, rs1: int, isa: dict) -> int:
    classes = _class_to_funct(isa)
    funct = classes.get(name)
    if funct is None:
        raise ValueError(f"unknown instruction class {name!r}; legal classes: {sorted(classes)}")
    want = _config_subtype_bits(isa, name)
    if want is not None and (rs1 & 0x3) != want:
        raise ValueError(
            f"{name} requires (rs1 & 0x3) == {want}; got rs1={rs1} "
            f"(rs1 & 0x3 == {rs1 & 0x3}). Set the low 2 bits of rs1 to select the subtype."
        )
    return funct


def __getattr__(name):
    if name == "rtl_checks":
        from . import rtl_checks
        return rtl_checks
    raise AttributeError(name)
