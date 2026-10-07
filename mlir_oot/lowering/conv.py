"""Convolution -> contraction: the im2col reduction, done in the ADDRESS STREAM.

The interface hands a pre-im2col'd weight ``[Kh*Kw*Ci, Co]`` and an NHWC activation. Rather than
materialising an im2col matrix in DRAM (which would need a scratch buffer the harness never fills)
this lowering contracts ONE KERNEL TAP AT A TIME: contraction tile ``(tap, channel-subtile)`` reads
the activation with a row pitch of ``stride_w * Ci``, so a run of consecutive output columns is a
single strided DMA burst. A tap that falls outside the padded activation for some output pixels
simply contributes no run for those rows -- which is exactly the zero the definition calls for, with
no zero buffer and no gather.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..target import facts
from .contraction import AOperand, KTile, TileLoad

DIM = facts.DIM


@dataclass(frozen=True)
class ConvGeometry:
    batch: int
    h: int
    w: int
    ci: int
    kh: int
    kw: int
    co: int
    sh: int
    sw: int
    pad_t: int
    pad_l: int
    pad_b: int
    pad_r: int
    dh: int
    dw: int

    @property
    def ho(self) -> int:
        return (self.h + self.pad_t + self.pad_b - (self.dh * (self.kh - 1) + 1)) // self.sh + 1

    @property
    def wo(self) -> int:
        return (self.w + self.pad_l + self.pad_r - (self.dw * (self.kw - 1) + 1)) // self.sw + 1

    @property
    def m(self) -> int:
        return self.batch * self.ho * self.wo

    @property
    def k(self) -> int:
        return self.kh * self.kw * self.ci

    @property
    def channel_subtiles(self) -> int:
        return (self.ci + DIM - 1) // DIM


@dataclass(frozen=True)
class PackSpec:
    """Host staging of one convolution band into dense reduction rows."""

    source: str
    target: str
    geometry: ConvGeometry
    source_pitch: int
    target_pitch: int
    row0: int
    rows: int


def geometry_from(op_attrs: dict, ifm_shape, weight_shape) -> ConvGeometry:
    kh, kw, ci, co = (int(x) for x in op_attrs["kernel"])
    stride = list(op_attrs.get("stride") or (1, 1))
    padding = list(op_attrs.get("padding") or (0, 0, 0, 0))
    dilation = list(op_attrs.get("dilation") or (1, 1))
    n, h, w, c = (int(x) for x in ifm_shape)
    if c != ci:
        raise ValueError(f"conv2d: kernel ci={ci} does not match the activation's channel dim {c}")
    if int(weight_shape[0]) != kh * kw * ci or int(weight_shape[1]) != co:
        raise ValueError(
            f"conv2d: weight {tuple(weight_shape)} is not the pre-im2col [{kh * kw * ci}, {co}]"
        )
    return ConvGeometry(
        batch=n, h=h, w=w, ci=ci, kh=kh, kw=kw, co=co,
        sh=int(stride[0]), sw=int(stride[1]),
        pad_t=int(padding[0]), pad_l=int(padding[1]),
        pad_b=int(padding[2]), pad_r=int(padding[3]),
        dh=int(dilation[0]), dw=int(dilation[1]),
    )


def conv_ktiles(g: ConvGeometry) -> list[KTile]:
    """One contraction tile per (kernel tap, channel subtile)."""
    tiles: list[KTile] = []
    for tap in range(g.kh * g.kw):
        for sub in range(g.channel_subtiles):
            width = min(DIM, g.ci - sub * DIM)
            tiles.append(KTile(width, tap * g.ci + sub * DIM))
    return tiles


def _runs_for(g: ConvGeometry, i: int, kti: int, pitch: int,
              row0: int = 0, rows_total: int | None = None,
              col0: int = 0, row_width: int | None = None,
              valid_width: int | None = None) -> list[tuple[int, int, int]]:
    """``(row_offset, n_rows, first_input_byte)`` for the in-bounds output pixels of one tile.

    ``row0`` is the first output pixel this operand covers and ``rows_total`` how many it covers:
    a BAND of the output plane is the same convolution read through a window, so every padding
    decision below is still taken against the WHOLE geometry and a band needs no padding of its
    own.
    """
    subs = g.channel_subtiles
    tap, sub = divmod(kti, subs)
    kh_i, kw_i = divmod(tap, g.kw)
    total = g.m if rows_total is None else rows_total
    width = g.wo if row_width is None else row_width
    valid = width if valid_width is None else valid_width
    rows = min(DIM, total - i * DIM)
    out: list[tuple[int, int, int]] = []
    run_start = None
    run_addr = 0
    prev = None
    for r in range(rows):
        local = i * DIM + r
        column = local % width
        p = row0 + (local // width) * g.wo + col0 + column
        nb, rem = divmod(p, g.ho * g.wo)
        oh, ow = divmod(rem, g.wo)
        ih = oh * g.sh - g.pad_t + kh_i * g.dh
        iw = ow * g.sw - g.pad_l + kw_i * g.dw
        ok = column < valid and 0 <= ih < g.h and 0 <= iw < g.w
        contiguous = (
            ok
            and prev is not None
            and prev[0] == nb
            and prev[1] == oh
            and ow == prev[2] + 1
        )
        if ok and not contiguous:
            if run_start is not None:
                out.append((run_start, r - run_start, run_addr))
            run_start = r
            run_addr = (((nb * g.h + ih) * g.w + iw) * pitch + sub * DIM)
        elif not ok:
            if run_start is not None:
                out.append((run_start, r - run_start, run_addr))
            run_start = None
        prev = (nb, oh, ow) if ok else None
    if run_start is not None:
        out.append((run_start, rows - run_start, run_addr))
    return out


def conv_a_operand(tensor: str, g: ConvGeometry, pitch: int | None = None,
                   row0: int = 0, rows_total: int | None = None,
                   col0: int = 0, row_width: int | None = None,
                   valid_width: int | None = None) -> AOperand:
    """``pitch`` is the activation's DRAM row pitch in elements (its channel extent, tile-rounded).

    ``row0``/``rows_total`` select a contiguous band of the output pixels; the default is the
    whole plane.
    """
    subs = g.channel_subtiles
    p = g.ci if pitch is None else pitch
    total = g.m if rows_total is None else rows_total

    def loads(i: int, kti: int) -> list[TileLoad]:
        _, sub = divmod(kti, subs)
        width = min(DIM, g.ci - sub * DIM)
        return [
            TileLoad(r0, tensor, addr, width, n)
            for (r0, n, addr) in _runs_for(g, i, kti, p, row0, total, col0, row_width,
                                          valid_width)
        ]

    def runs(i: int, kti: int) -> list[tuple[int, int]]:
        return [(r0, n) for (r0, n, _) in _runs_for(g, i, kti, p, row0, total,
                                                   col0, row_width, valid_width)]

    return AOperand(m=total, stride_bytes=g.sw * p, loads=loads, runs=runs,
                    group_loads_by_row=True)
