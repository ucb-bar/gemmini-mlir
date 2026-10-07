"""Map a commit's declared epilogue onto this target's accumulator-readout path.

Derived from the RTL, not assumed: ``AccumulatorScale`` applies the activation and then the f32
accumulator scale and clips to the operand width, and ``StoreController`` fuses pooling into the
same store. ``LocalAddr.read_full_acc_row`` selects the RAW accumulator instead, which bypasses
both -- so a full-width readout cannot carry an activation, and this module says so by name rather
than emitting a store that silently drops the stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.workload import Epilogue
from ..target import isa

#: Activation encodings from the ISA header (``NO_ACTIVATION``/``RELU``).
NO_ACTIVATION = 0
RELU = 1

#: Stages this target's readout can execute, and the field each one rides on.
READOUT_STAGES = {"relu", "acc_scale", "requant", "maxpool", "bias_add", "bias"}


class UnsupportedEpilogue(Exception):
    """A declared stage this target's readout cannot execute for the requested output width."""


@dataclass(frozen=True)
class Readout:
    """The store-path configuration one commit implies."""

    full_width: bool
    out_bytes: int
    act: int
    acc_scale: float
    bias: str | None
    pool_enabled: bool
    pool_size: int
    pool_stride: int
    pool_out_dim: int
    porows: int
    pocols: int
    orows: int
    ocols: int
    upad: int
    lpad: int
    #: number of committed rows per pooled batch element, and the batch count
    pool_batches: int


def _check_store_field(field: str, value: int) -> None:
    """Refuse a pooling geometry this target's store path has no field wide enough to carry.

    The widths come from the encoder's own layout table, so this check and the instruction it
    guards cannot drift apart.
    """
    limit = isa.config_st_field_max(field)
    if int(value) > limit:
        raise UnsupportedEpilogue(
            f"this store path carries {field} in a "
            f"{isa.CONFIG_ST_FIELD_BITS[field]}-bit field, so the declared geometry's "
            f"{field}={int(value)} cannot be expressed (max {limit})"
        )


def _pool_out(extent: int, pad_a: int, pad_b: int, window: int, stride: int) -> int:
    return (extent + pad_a + pad_b - window) // stride + 1


def plan_readout(ep: Epilogue, acc_rows_committed: int) -> Readout:
    """Derive the readout for one commit, from the attributes the capsule declared."""
    stages = ep.has
    unknown = stages - READOUT_STAGES
    if unknown:
        raise UnsupportedEpilogue(f"epilogue stage(s) {sorted(unknown)} have no readout path here")

    full = ep.output_dtype in ("i32", "i64")
    out_bytes = 4 if full else 1
    act = RELU if "relu" in stages else NO_ACTIVATION

    scale = 1.0
    if "acc_scale" in stages:
        if ep.acc_scale is None:
            raise UnsupportedEpilogue("acc_scale stage declares no multiplier")
        scale = float(ep.acc_scale)
    if "requant" in stages:
        if ep.requant_shift is None:
            raise UnsupportedEpilogue("requant stage declares no requant_shift")
        scale *= 2.0 ** (-int(ep.requant_shift))

    if full and (act != NO_ACTIVATION or scale != 1.0):
        raise UnsupportedEpilogue(
            "a full-width (i32) readout reads the raw accumulator, so it cannot carry the declared "
            f"{'activation' if act else 'accumulator scale'} stage"
        )

    pool = "maxpool" in stages
    ph = pw = sh = sw = 0
    porows = pocols = orows = ocols = upad = lpad = 0
    batches = 1
    if pool:
        if not (ep.pool_in_dims and ep.pool_size and ep.pool_stride):
            raise UnsupportedEpilogue("maxpool without its declared geometry")
        h, w = int(ep.pool_in_dims[0]), int(ep.pool_in_dims[1])
        ph, pw = int(ep.pool_size[0]), int(ep.pool_size[1])
        sh, sw = int(ep.pool_stride[0]), int(ep.pool_stride[1])
        pt, pl, pb, pr = (list(ep.pool_padding) + [0, 0, 0, 0])[:4]
        if ph != pw or sh != sw:
            raise UnsupportedEpilogue(
                "this store path carries ONE pooling window and ONE pooling stride field, so a "
                "non-square window or stride cannot be expressed"
            )
        _check_store_field("pool_size", ph)
        _check_store_field("pool_stride", sh)
        _check_store_field("upad", pt)
        _check_store_field("lpad", pl)
        if pb or pr:
            # the store path drops out-of-range windows by bounds check rather than by a field,
            # which is exactly trailing zero padding -- nothing to configure.
            pass
        orows, ocols = h, w
        porows = _pool_out(h, pt, pb, ph, sh)
        pocols = _pool_out(w, pl, pr, pw, sw)
        # The pooled geometry is derived, so it is checked where it is derived: every one of these
        # rides a fixed-width rs1 field, and a value too wide for its field would otherwise reach
        # the encoder as a hard error rather than as a stated decline.
        _check_store_field("orows", orows)
        _check_store_field("ocols", ocols)
        _check_store_field("porows", porows)
        _check_store_field("pocols", pocols)
        _check_store_field("pool_out_dim", pocols)
        upad, lpad = pt, pl
        rows_per_batch = h * w
        batches = max(1, acc_rows_committed // rows_per_batch)

    return Readout(
        full_width=full,
        out_bytes=out_bytes,
        act=act,
        acc_scale=scale,
        bias=ep.bias if ("bias_add" in stages or "bias" in stages) else None,
        pool_enabled=pool,
        pool_size=ph,
        pool_stride=sh,
        pool_out_dim=pocols,
        porows=porows,
        pocols=pocols,
        orows=orows,
        ocols=ocols,
        upad=upad,
        lpad=lpad,
        pool_batches=batches,
    )
