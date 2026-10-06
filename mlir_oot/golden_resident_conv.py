"""Primitive exact 3x3 convolution with one resident, virtually padded input.

A channel tile owns a complete padded spatial plane. CONFIG_LD block_stride
places wide input loads directly into this layout; shifted resident rows
supply all nine taps without repeated input DMA. The mesh row stride follows
the source spatial stride. Each output row fits one DIM tile; resource
admission and optional remaining-row weight placement are explicit.
"""

import json
from dataclasses import asdict, dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

from .golden_gemm import GoldenGemm, Shape, _ceil_div, _groups
from .tables import isa
from .tables import rtl_facts as F


@dataclass(frozen=True)
class ResidentConvOptions:
    rows_per_tile: int = 1
    loop_channels: bool = False
    prefetch_b: bool = False


def choose_compact_resident(conv, *, prefetch_b=False):
    """Derive an opt-in schedule from spatial span and declared resources.

    This is a target compiler policy, independent of source provenance or model
    identity. The largest legal adjacent-row group reduces mesh padding and
    ordinary channel loops bound code size. Resource refusal leaves selection
    of an alternative schedule to the caller; this is no universal speed claim.
    """
    conv.validate()
    if conv.w + 2 > F.DIM:
        raise ValueError("resident width plus halo exceeds DIM")
    rows = min(conv.h, 1 + (F.DIM - conv.w) // (conv.w + 2))
    if type(prefetch_b) is not bool:
        raise ValueError("resident weight prefetch selection must be boolean")
    options = ResidentConvOptions(
        rows_per_tile=rows, loop_channels=not prefetch_b, prefetch_b=prefetch_b
    )
    return GoldenResidentConv(conv, **asdict(options)), options


def retain_resident_commands(control):
    """Apply one explicit command choice to an already admitted resident layout.

    Preserve all declared constructor facts and prior resource decisions.
    Profitability is deliberately absent: smaller bodies can lose from extra
    CPU issue work, and measured alternatives belong in shared plan selection.
    """
    decision = {
        "applied": False,
        "selection": "explicit opt-in; profitability not inferred",
        "refusal": "selected family is not resident channel planes",
        "timing_claim": False,
    }
    if type(control) is not GoldenResidentConv:
        return control, decision
    selected = GoldenResidentConv(
        control.conv,
        rows_per_tile=control.rows_per_tile,
        loop_channels=control.loop_channels,
        prefetch_b=control.prefetch_b,
        weight_base=control.explicit_weight_base,
        source_stride=control.source_stride,
        row_residue=control.row_residue,
        compact_commands=True,
        weight_issue_tiles=control.weight_issue_tiles,
        flat_spatial_planes=control.flat_spatial_planes,
        store_plan=control.store_plan,
    )
    for name in ("resident_stripe_decision", "source_stride_decision"):
        if hasattr(control, name):
            setattr(selected, name, getattr(control, name))
    decision.update(
        applied=True,
        refusal=None,
        rows_per_tile=selected.rows_per_tile,
        prefetch_b=selected.prefetch_b,
        input_plane_rows=selected.plane,
        weight_base=selected.bbase,
        source_stride=selected.source_stride,
        row_residue=selected.row_residue,
        source_reduction_order="increasing HWIO; exact original command/pointer sequence",
    )
    return selected, decision


def issue_resident_weight_packets(control, *, tiles=2, include_flat_planes=True):
    """Choose explicitly requested packet granularity on a proved resident layout."""
    if type(tiles) is not int or not 1 <= tiles <= 4:
        raise ValueError("weight issue tiles must be an integer in 1..4")
    if type(include_flat_planes) is not bool:
        raise ValueError("flat-plane packet admission must be boolean")
    decision = {
        "applied": False,
        "automatic_policy": False,
        "timing_claim": False,
        "performance": "UNKNOWN",
        "selection": "explicit weight packet lookahead",
    }
    if type(control) is not GoldenResidentConv:
        return control, dict(
            decision, refusal="Current family has no proved complete resident input"
        )
    if control.flat_spatial_planes and not include_flat_planes:
        return control, dict(
            decision,
            refusal="Flat spatial planes excluded by explicit packet layout policy",
        )
    try:
        selected = GoldenResidentConv(
            control.conv,
            rows_per_tile=control.rows_per_tile,
            loop_channels=control.loop_channels,
            prefetch_b=True,
            weight_base=control.explicit_weight_base,
            source_stride=control.source_stride,
            row_residue=control.row_residue,
            compact_commands=True,
            weight_issue_tiles=tiles,
            flat_spatial_planes=control.flat_spatial_planes,
            store_plan=control.store_plan,
        )
    except ValueError as failure:
        return control, dict(decision, refusal=str(failure))
    for name in ("resident_stripe_decision", "source_stride_decision"):
        if hasattr(control, name):
            setattr(selected, name, getattr(control, name))
    decision.update(
        applied=True,
        refusal=None,
        weight_issue_tiles=tiles,
        input_reserved_interval=[0, (control.conv.cin // F.DIM) * control.plane],
        weight_slots=[
            [base, base + min(tiles, control.conv.bn) * F.DIM]
            for base in selected.bases
        ],
        accumulator_rows=len(control.row_tiles) * control.conv.bn * F.DIM,
        output_panel_tiles=control.conv.bn,
        source_reduction_order="Increasing HWIO K for every output; only N packet interleaving changes",
        lifetime="Immutable complete A; current and next B packets occupy disjoint banks; same private ACC destinations and stores",
        CPU_commands="Retained bounded ordinary K and spatial loops",
        whole_timing="UNKNOWN",
    )
    selected.weight_issue_decision = decision
    return selected, decision


def select_flat_resident_planes(control):
    """Explicitly replace a proved virtual-pad flat band with complete A.

    Horizontal-tap planes remove halo lanes from the mesh row sequence. The
    capacity proof covers all three immutable copies, live weight storage and
    the complete output accumulator panel. No timing estimate selects shapes.
    """
    from .golden_flat_conv import GoldenFlatConv

    decision = {
        "applied": False,
        "automatic_policy": False,
        "performance": "UNKNOWN",
        "timing_claim": False,
        "selection": "explicit flat kw-plane residency",
    }
    if type(control) is not GoldenFlatConv or not control.virtual_padding:
        return control, dict(
            decision, refusal="Requires proved virtual-pad flat convolution"
        )
    try:
        selected = GoldenResidentConv(
            control.conv,
            flat_spatial_planes=True,
            compact_commands=True,
            store_plan=control.store_plan,
        )
    except ValueError as failure:
        return control, dict(decision, refusal=str(failure))
    decision.update(
        applied=True,
        refusal=None,
        input_reserved_interval=[0, control.conv.cin // F.DIM * selected.plane],
        kw_plane_rows=(control.conv.h + 2) * control.conv.w,
        channel_plane_rows=selected.plane,
        weight_reserved_interval=[
            selected.bbase,
            selected.bbase + control.conv.bn * F.DIM,
        ],
        accumulator_rows=len(selected.row_tiles) * control.conv.bn * F.DIM,
        flat_output_tiles=len(selected.row_tiles),
        source_reduction_order="Original increasing HWIO K for every output",
        lifetime="Disjoint partitioned immutable A planes remain live through all N panels; B is reloaded per increasing K; completed ACC is stored before reuse",
        layout="source cell(ki,kw,padded_y,x)=NHWC(padded_y-1,x+kw-1,ki*DIM+lane) or static zero",
        requested_DMA="Payload only; physical DRAM and overlap UNKNOWN",
    )
    selected.flat_resident_decision = decision
    return selected, decision


class GoldenResidentConv(GoldenGemm):
    def __init__(
        self,
        s,
        *,
        rows_per_tile=1,
        loop_channels=False,
        prefetch_b=False,
        weight_base=None,
        source_stride=False,
        row_residue=False,
        compact_commands=False,
        weight_issue_tiles=None,
        flat_spatial_planes=False,
        store_plan=None,
    ):
        s.validate()
        if store_plan is not None:
            from .readout_store_plan import PairedReadoutPlan

            if type(store_plan) is not PairedReadoutPlan:
                raise ValueError("typed paired readout plan required")
            if not (flat_spatial_planes or source_stride):
                raise ValueError(
                    "paired resident stores require flat spatial or source-stride planes"
                )
            store_plan.require_conv_producer(s)
        self.store_plan = store_plan
        if weight_issue_tiles is not None and (
            type(weight_issue_tiles) is not int or not 1 <= weight_issue_tiles <= 4
        ):
            raise ValueError("weight issue tiles must be an integer in 1..4")
        if weight_issue_tiles is not None and not prefetch_b:
            raise ValueError("weight packet issue requires bank-prefetched commands")
        self.weight_issue_tiles = weight_issue_tiles
        if type(compact_commands) is not bool:
            raise ValueError("compact resident commands require a boolean selection")
        self.compact_commands = compact_commands
        if type(loop_channels) is not bool:
            raise ValueError("resident channel-loop selection must be boolean")
        self.loop_channels = loop_channels or compact_commands
        if type(prefetch_b) is not bool:
            raise ValueError("resident weight prefetch selection must be boolean")
        if prefetch_b and loop_channels and not compact_commands:
            raise ValueError("resident weight prefetch needs static reduction commands")
        self.prefetch_b = prefetch_b
        if type(source_stride) is not bool:
            raise ValueError("source stride residency selection must be boolean")
        self.source_stride = source_stride
        if type(row_residue) is not bool or (
            row_residue and (not source_stride or s.stride != 2)
        ):
            raise ValueError("resident row residue layout requires source stride2")
        self.row_residue = row_residue
        if type(flat_spatial_planes) is not bool or (
            flat_spatial_planes
            and (source_stride or row_residue or s.stride != 1 or rows_per_tile != 1)
        ):
            raise ValueError(
                "flat resident planes require stride1 and no row grouping/residue"
            )
        self.flat_spatial_planes = flat_spatial_planes
        self.conv = s
        if (
            not source_stride
            and not flat_spatial_planes
            and (s.stride != 1 or s.w + 2 > F.DIM)
        ):
            raise ValueError(
                "compact resident convolution needs stride1 and width+halo<=DIM"
            )
        if (
            s.explicit_halo
            or (s.ow > F.DIM and not flat_spatial_planes)
            or s.cin % F.DIM
        ):
            raise ValueError(
                "resident convolution needs unpadded input, output width<=DIM and aligned Cin"
            )
        self.input_pitch = (
            _ceil_div(s.w + 2, s.stride) * s.stride if row_residue else s.w + 2
        )
        self.residue_rows = _ceil_div(s.h + 2, s.stride) if row_residue else s.h + 2
        self.plane = (
            self.residue_rows * self.input_pitch * (s.stride if row_residue else 1)
        )
        if flat_spatial_planes:
            self.input_pitch = s.w
            self.plane = 3 * (s.h + 2) * s.w
        self.output_pitch = (
            self.input_pitch // s.stride if row_residue else self.input_pitch
        )
        if self.plane >= 1 << 16:
            raise ValueError("resident input plane stride exceeds ISA field")
        if (
            type(rows_per_tile) is not int
            or not 1 <= rows_per_tile <= s.oh
            or (
                not flat_spatial_planes
                and (rows_per_tile - 1) * self.output_pitch + s.ow > F.DIM
            )
        ):
            raise ValueError(
                "resident row group must fit its padded spatial span in DIM"
            )
        self.rows_per_tile = rows_per_tile
        self.row_tiles = tuple(
            (
                y,
                min(rows_per_tile, s.oh - y),
                (min(rows_per_tile, s.oh - y) - 1) * self.output_pitch + s.ow,
            )
            for y in range(0, s.oh, rows_per_tile)
        )
        if flat_spatial_planes:
            self.output_pitch = F.DIM
            self.row_tiles = tuple(
                (m, 1, min(F.DIM, s.oh * s.ow - m))
                for m in range(0, s.oh * s.ow, F.DIM)
            )
        self.spatial_step = (
            rows_per_tile * (1 if row_residue else s.stride) * self.input_pitch
        )
        if flat_spatial_planes:
            self.spatial_step = F.DIM
        if compact_commands and any(
            self.source_offset(y, kh, kw)
            != self.source_offset(0, kh, kw) + tile * self.spatial_step
            for tile, (y, _, _) in enumerate(self.row_tiles)
            for kh in range(3)
            for kw in range(3)
        ):
            raise ValueError(
                "compact resident spatial addresses must form a proved affine sequence"
            )
        if weight_base is not None and (
            type(weight_base) is not int or weight_base < 0 or weight_base % F.DIM
        ):
            raise ValueError(
                "resident weight base must be a nonnegative DIM-aligned row"
            )
        if prefetch_b and weight_base is not None:
            raise ValueError(
                "explicit resident weight placement does not admit bank lookahead"
            )
        self.bbase = 2 * F.SPAD_BANK_ROWS if weight_base is None else weight_base
        self.explicit_weight_base = weight_base
        self.bases = (self.bbase, 3 * F.SPAD_BANK_ROWS) if prefetch_b else (self.bbase,)
        if _ceil_div(s.cin, F.DIM) * self.plane > self.bbase:
            raise ValueError("resident input overlaps weight banks")
        if (
            self.bbase + s.bn * F.DIM > F.SPAD_ROWS
            or len(self.row_tiles) * s.bn * F.DIM > F.ACC_ROWS
        ):
            raise ValueError(
                "resident convolution output/weight panel exceeds resources"
            )
        if prefetch_b and (
            s.bn * F.DIM > F.SPAD_BANK_ROWS
            or self.bases[-1] + s.bn * F.DIM > F.SPAD_ROWS
        ):
            raise ValueError("resident weight slots overlap or exceed scratchpad")
        for ki in range(s.cin // F.DIM):
            for y, _, span in self.row_tiles:
                for kh in range(3):
                    for kw in range(3):
                        start = ki * self.plane + self.source_offset(y, kh, kw)
                        stop = start + (span - 1) * s.stride + 1
                        end = (ki + 1) * self.plane
                        if flat_spatial_planes:
                            end = ki * self.plane + (kw + 1) * (s.h + 2) * s.w
                        if self.row_residue:
                            end = (
                                ki * self.plane
                                + (kh % s.stride + 1)
                                * self.residue_rows
                                * self.input_pitch
                            )
                        if stop > end:
                            raise ValueError(
                                "strided resident compute crosses its input plane"
                            )
        self.conv = s
        super().__init__(
            Shape(
                s.oh * s.ow,
                s.cout,
                s.cin,
                bm=len(self.row_tiles),
                bn=s.bn,
                output_dtype=s.output_dtype,
                scale=s.scale,
                relu=s.relu,
                wide_store=True,
                reuse_b=True,
            )
        )
        from .golden_gemm import PTR

        self.second_output = (
            self.fb.entry.insert_arg(PTR, 3) if store_plan is not None else None
        )

    def input_row(self, padded_y):
        """Bijection from allocated source rows into stride-residue slabs."""
        if self.row_residue:
            return (
                padded_y % self.conv.stride
            ) * self.residue_rows + padded_y // self.conv.stride
        return padded_y

    def source_offset(self, output_y, kh, kw):
        if self.flat_spatial_planes:
            return kw * (self.conv.h + 2) * self.conv.w + kh * self.conv.w + output_y
        return self.input_row(output_y * self.conv.stride + kh) * self.input_pitch + kw

    def output_pixel(self, start, row):
        return start if self.flat_spatial_planes else (start + row) * self.conv.ow

    def _load_flat_input(self, zero):
        """Disjoint kw-shifted planes with no inter-row halo lanes.

        Each row owns exactly W cells, corresponding to x+kw-1. The three
        planes duplicate immutable source pixels; only statically out-of-bounds
        cells are zero. Flattened mesh rows may cross source row boundaries.
        """
        s = self.conv
        for ci in range(0, s.cin, 4 * F.DIM):
            cols = min(4 * F.DIM, s.cin - ci)
            base = (ci // F.DIM) * self.plane
            for kw in range(3):
                for y in range(s.h + 2):
                    row = base + (kw * (s.h + 2) + y) * s.w
                    lo, hi = max(0, 1 - kw), min(s.w, s.w + 1 - kw)
                    segments = (
                        [(0, s.w, True)]
                        if y in (0, s.h + 1)
                        else [(0, lo, True), (lo, hi, False), (hi, s.w, True)]
                    )
                    for start, stop, is_zero in segments:
                        for x in range(start, stop, F.DIM):
                            ptr = (
                                zero
                                if is_zero
                                else self._ptr(
                                    self.a,
                                    self.fb.const((y - 1) * s.w + x + kw - 1),
                                    s.cin,
                                    self.fb.const(ci),
                                )
                            )
                            self._rocc(
                                "mvin",
                                {
                                    "local": row + x,
                                    "rows": min(F.DIM, stop - x),
                                    "cols": cols,
                                    "load_id": 0,
                                },
                                ptr,
                            )

    def build(self):
        s = self.conv
        pw = self.input_pitch
        self._emit_config()
        if s.stride != 1:
            from .ir.gemmini_dialect import ConfigExOp

            configs = [op for op in self.fb.entry.ops if isinstance(op, ConfigExOp)]
            assert len(configs) == 1
            configs[0].attributes["a_stride"] = IntegerAttr(s.stride, i64)
        self._rocc(
            "config_ld", {"stride": s.cin, "block_stride": self.plane, "load_id": 0}
        )
        zero = self.fb.add(llvm.IntToPtrOp(self.fb.const(0))).results[0]
        # These loads partition the scratch input exactly: no overwrite races.
        if self.flat_spatial_planes:
            self._load_flat_input(zero)
        for ci in range(0, 0 if self.flat_spatial_planes else s.cin, 4 * F.DIM):
            cols = min(4 * F.DIM, s.cin - ci)
            base = (ci // F.DIM) * self.plane
            padded_rows = self.residue_rows * s.stride if self.row_residue else s.h + 2
            for y in range(padded_rows):
                row = base + self.input_row(y) * pw
                if not 1 <= y <= s.h:
                    for x in range(0, pw, F.DIM):
                        self._rocc(
                            "mvin",
                            {
                                "local": row + x,
                                "rows": min(F.DIM, pw - x),
                                "cols": cols,
                                "load_id": 0,
                            },
                            zero,
                        )
                else:
                    self._rocc(
                        "mvin",
                        {"local": row, "rows": 1, "cols": cols, "load_id": 0},
                        zero,
                    )
                    for x in range(0, s.w, F.DIM):
                        ptr = self._ptr(
                            self.a,
                            self.fb.const((y - 1) * s.w + x),
                            s.cin,
                            self.fb.const(ci),
                        )
                        self._rocc(
                            "mvin",
                            {
                                "local": row + x + 1,
                                "rows": min(F.DIM, s.w - x),
                                "cols": cols,
                                "load_id": 0,
                            },
                            ptr,
                        )
                    self._rocc(
                        "mvin",
                        {
                            "local": row + s.w + 1,
                            "rows": pw - s.w - 1,
                            "cols": cols,
                            "load_id": 0,
                        },
                        zero,
                    )

        def compact_spatial(ki, kh, kw, d, cols, bbase, first):
            """Same resident commands, retaining regular rows as a CPU loop.

            The first spatial tile loads the real stationary weight. Later
            tiles retain it. A and C addresses use proved affine row bounds;
            a final short row group remains a separate exact static command.
            """
            full = (
                s.oh * s.ow // F.DIM
                if self.flat_spatial_planes
                else s.oh // self.rows_per_tile
            )
            span = self.row_tiles[0][2]
            offset = self.source_offset(0, kh, kw)
            a_extent = (s.cin // F.DIM) * self.plane

            def spatial(tile, rows, real_weight=False, static_tile=None):
                common = {
                    "bd": bbase + d * F.DIM if real_weight else isa.GARBAGE_ADDR,
                    "bd_cols": cols,
                    "bd_rows": F.DIM,
                    "c_cols": cols,
                    "c_rows": rows,
                }
                if static_tile is not None:
                    common["c"] = isa.acc_addr(
                        (static_tile * s.bn + d) * F.DIM, accumulate=not first
                    )
                    self._rocc("preload", common)
                    a_offset = offset + static_tile * self.spatial_step
                    address = self.fb.add_i(
                        self.fb.mul_i(ki, self.fb.const(self.plane)),
                        self.fb.const(a_offset),
                    )
                    a_max = (s.cin // F.DIM - 1) * self.plane + a_offset
                else:
                    c_row = self.fb.add_i(
                        self.fb.mul_i(tile, self.fb.const(s.bn * F.DIM)),
                        self.fb.const(d * F.DIM),
                    )
                    common.update(
                        c_accumulate=int(not first),
                        c_max=(full - 1) * s.bn * F.DIM + d * F.DIM,
                        c_reserved_rows=len(self.row_tiles) * s.bn * F.DIM,
                    )
                    self._rocc("preload", common, c_row)
                    row = self.fb.add_i(
                        self.fb.mul_i(tile, self.fb.const(self.spatial_step)),
                        self.fb.const(offset),
                    )
                    address = self.fb.add_i(
                        self.fb.mul_i(ki, self.fb.const(self.plane)), row
                    )
                    a_max = (
                        (s.cin // F.DIM - 1) * self.plane
                        + offset
                        + (full - 1) * self.spatial_step
                    )
                self._rocc(
                    "compute",
                    {
                        "a_cols": F.DIM,
                        "a_rows": rows,
                        "accumulate": not real_weight,
                        "a_max": a_max,
                        "a_reserved_rows": a_extent,
                    },
                    address,
                )

            spatial(None, span, True, 0)
            if full > 1:
                self.fb.for_loop(
                    1, full, 1, lambda tile: spatial(tile, span), retain_loop=True
                )
            if (
                (s.oh * s.ow % F.DIM and full > 0)
                if self.flat_spatial_planes
                else s.oh % self.rows_per_tile
            ):
                spatial(None, self.row_tiles[-1][2], False, len(self.row_tiles) - 1)

        def store_outputs(n0, nr):
            def store(output, dtype):
                for tile, (y, count, span) in enumerate(self.row_tiles):
                    for row in range(count):
                        step = 4 if dtype == "i8" else 1
                        for d in range(0, len(nr), step):
                            ptr = self._ptr(
                                output,
                                self.fb.const(self.output_pixel(y, row)),
                                s.cout,
                                self._tile(n0, d),
                                4 if dtype == "i32" else 1,
                            )
                            self._rocc(
                                "mvout",
                                {
                                    "local": isa.acc_addr(
                                        (tile * s.bn + d) * F.DIM
                                        + row * self.output_pitch,
                                        full_row=dtype == "i32",
                                    ),
                                    "rows": span if self.flat_spatial_planes else s.ow,
                                    "cols": sum(nr[d : d + step]),
                                },
                                ptr,
                            )

            if self.store_plan is None:
                store(self.c, s.output_dtype)
            else:
                # Both passes consume the same completed ACC block. No load or
                # compute overwrites it between them; final fence completes both
                # disjoint byte outputs before the caller's exact decoder.
                for output, scale in zip(
                    (self.c, self.second_output), self.store_plan.store_scales
                ):
                    self._rocc(
                        "config_st",
                        {
                            "stride": s.cout,
                            "acc_scale": scale,
                            "acc_act": isa.RELU
                            if self.store_plan.relu
                            else isa.NO_ACTIVATION,
                        },
                    )
                    store(output, "i8")

        def channel(n0, nr):
            for kh in range(3):
                for kw in range(3):

                    def reduction(ki, first, static_ci=None, kh=kh, kw=kw):
                        krow = (
                            self.fb.const((kh * 3 + kw) * s.cin + static_ci)
                            if static_ci is not None
                            else self.fb.add_i(
                                self.fb.const((kh * 3 + kw) * s.cin),
                                self.fb.mul_i(ki, self.fb.const(F.DIM)),
                            )
                        )
                        for d in range(0, len(nr), 4):
                            ptr = self._ptr(self.b, krow, s.cout, self._tile(n0, d))
                            self._rocc(
                                "mvin",
                                {
                                    "local": self.bbase + d * F.DIM,
                                    "rows": F.DIM,
                                    "cols": sum(nr[d : d + 4]),
                                    "load_id": 1,
                                },
                                ptr,
                            )
                        for d, cols in enumerate(nr):
                            if self.compact_commands:
                                compact_spatial(ki, kh, kw, d, cols, self.bbase, first)
                                continue
                            for tile, (y, count, span) in enumerate(self.row_tiles):
                                self._rocc(
                                    "preload",
                                    {
                                        "bd": self.bbase + d * F.DIM
                                        if tile == 0
                                        else isa.GARBAGE_ADDR,
                                        "c": isa.acc_addr(
                                            (tile * s.bn + d) * F.DIM,
                                            accumulate=not first,
                                        ),
                                        "bd_cols": cols,
                                        "bd_rows": F.DIM,
                                        "c_cols": cols,
                                        "c_rows": span,
                                    },
                                )
                                offset = self.source_offset(y, kh, kw)
                                attrs = {
                                    "a_cols": F.DIM,
                                    "a_rows": span,
                                    "accumulate": tile != 0,
                                }
                                if static_ci is not None:
                                    attrs["a"] = (
                                        static_ci // F.DIM
                                    ) * self.plane + offset
                                    self._rocc("compute", attrs)
                                else:
                                    address = self.fb.add_i(
                                        self.fb.mul_i(ki, self.fb.const(self.plane)),
                                        self.fb.const(offset),
                                    )
                                    attrs.update(
                                        a_max=(s.cin // F.DIM - 1) * self.plane
                                        + offset,
                                        a_reserved_rows=(s.cin // F.DIM) * self.plane,
                                    )
                                    self._rocc("compute", attrs, address)

                    if self.loop_channels:
                        # The first update initializes C; every later channel/tap
                        # accumulates in the original increasing HWIO order.
                        first = kh == 0 and kw == 0
                        if first:
                            reduction(self.fb.const(0), True)
                        self.fb.for_loop(
                            int(first),
                            s.cin // F.DIM,
                            1,
                            lambda ki: reduction(ki, False),
                            retain_loop=self.compact_commands,
                        )
                    else:
                        for ci in range(0, s.cin, F.DIM):
                            reduction(None, kh == 0 and kw == 0 and ci == 0, ci)
            # Padding lanes inside a grouped tile are never materialized. Each
            # valid row has its original NHWC destination and scratch row pitch.
            store_outputs(n0, nr)

        def prefetched_channel(n0, nr):
            kt = s.cin // F.DIM

            def load(index):
                base = self.bases[index % 2]
                for d in range(0, len(nr), 4):
                    ptr = self._ptr(
                        self.b, self.fb.const(index * F.DIM), s.cout, self._tile(n0, d)
                    )
                    self._rocc(
                        "mvin",
                        {
                            "local": base + d * F.DIM,
                            "rows": F.DIM,
                            "cols": sum(nr[d : d + 4]),
                            "load_id": 1,
                        },
                        ptr,
                    )

            load(0)
            for index in range(9 * kt):
                # The next panel writes a disjoint bank while the current one
                # remains live. Source HWIO reduction order is unchanged.
                if index + 1 < 9 * kt:
                    load(index + 1)
                kh, kw = divmod(index // kt, 3)
                ki = index % kt
                for d, cols in enumerate(nr):
                    for tile, (y, count, span) in enumerate(self.row_tiles):
                        self._rocc(
                            "preload",
                            {
                                "bd": self.bases[index % 2] + d * F.DIM
                                if tile == 0
                                else isa.GARBAGE_ADDR,
                                "c": isa.acc_addr(
                                    (tile * s.bn + d) * F.DIM, accumulate=index != 0
                                ),
                                "bd_cols": cols,
                                "bd_rows": F.DIM,
                                "c_cols": cols,
                                "c_rows": span,
                            },
                        )
                        self._rocc(
                            "compute",
                            {
                                "a": ki * self.plane + self.source_offset(y, kh, kw),
                                "a_cols": F.DIM,
                                "a_rows": span,
                                "accumulate": tile != 0,
                            },
                        )
            store_outputs(n0, nr)

        def packet_prefetched_channel(n0, nr):
            """Look ahead one disjoint weight packet, preserving K per output.

            A packet contains adjacent N tiles at one source K position. Its
            successor can be another N packet or the first packet at next K.
            Current/next slots alternate banks; each packet is consumed across
            all spatial rows before its slot is reused. Output addresses retain
            the original full N panel, so stores and accumulator lifetimes do
            not change when weight DMA granularity changes.
            """
            kt = s.cin // F.DIM
            chunks = tuple(range(0, len(nr), self.weight_issue_tiles))
            packets = tuple((index, d) for index in range(9 * kt) for d in chunks)

            def load(ordinal):
                index, d = packets[ordinal]
                count = min(self.weight_issue_tiles, len(nr) - d)
                ptr = self._ptr(
                    self.b, self.fb.const(index * F.DIM), s.cout, self._tile(n0, d)
                )
                self._rocc(
                    "mvin",
                    {
                        "local": self.bases[ordinal % 2],
                        "rows": F.DIM,
                        "cols": sum(nr[d : d + count]),
                        "load_id": 1,
                    },
                    ptr,
                )

            load(0)
            for ordinal, (index, begin) in enumerate(packets):
                if ordinal + 1 < len(packets):
                    load(ordinal + 1)
                kh, kw = divmod(index // kt, 3)
                ki = index % kt
                stop = min(begin + self.weight_issue_tiles, len(nr))
                for d in range(begin, stop):
                    cols = nr[d]
                    for tile, (y, count, span) in enumerate(self.row_tiles):
                        self._rocc(
                            "preload",
                            {
                                "bd": self.bases[ordinal % 2] + (d - begin) * F.DIM
                                if tile == 0
                                else isa.GARBAGE_ADDR,
                                "c": isa.acc_addr(
                                    (tile * s.bn + d) * F.DIM, accumulate=index != 0
                                ),
                                "bd_cols": cols,
                                "bd_rows": F.DIM,
                                "c_cols": cols,
                                "c_rows": span,
                            },
                        )
                        self._rocc(
                            "compute",
                            {
                                "a": ki * self.plane + self.source_offset(y, kh, kw),
                                "a_cols": F.DIM,
                                "a_rows": span,
                                "accumulate": tile != 0,
                            },
                        )
            store_outputs(n0, nr)

        def compact_packet_prefetched_channel(n0, nr):
            """Retain bounded K/row loops with the same packet dependency order."""
            kt = s.cin // F.DIM
            chunks = tuple(range(0, len(nr), self.weight_issue_tiles))

            def load(tap, ki, begin, bank):
                count = min(self.weight_issue_tiles, len(nr) - begin)
                krow = self.fb.add_i(
                    self.fb.const(tap * s.cin), self.fb.mul_i(ki, self.fb.const(F.DIM))
                )
                ptr = self._ptr(self.b, krow, s.cout, self._tile(n0, begin))
                self._rocc(
                    "mvin",
                    {
                        "local": self.bases[bank],
                        "rows": F.DIM,
                        "cols": sum(nr[begin : begin + count]),
                        "load_id": 1,
                    },
                    ptr,
                )

            def one_k(tap, ki, parity, *, first=False, final=False):
                kh, kw = divmod(tap, 3)
                for chunk, begin in enumerate(chunks):
                    following_bank = (parity + chunk + 1) % 2
                    if chunk + 1 < len(chunks):
                        load(tap, ki, chunks[chunk + 1], following_bank)
                    elif not final:
                        load(
                            tap, self.fb.add_i(ki, self.fb.const(1)), 0, following_bank
                        )
                    elif tap < 8:
                        load(tap + 1, self.fb.const(0), 0, following_bank)
                    for d in range(
                        begin, min(begin + self.weight_issue_tiles, len(nr))
                    ):
                        compact_spatial(
                            ki,
                            kh,
                            kw,
                            d,
                            nr[d],
                            self.bases[(parity + chunk) % 2] - begin * F.DIM,
                            first,
                        )

            load(0, self.fb.const(0), 0, 0)
            for tap in range(9):
                base = tap * kt * len(chunks)
                one_k(tap, self.fb.const(0), base % 2, first=tap == 0, final=kt == 1)
                pairs = max(0, (kt - 2) // 2)

                def pair(ki, tap=tap, base=base):
                    one_k(tap, ki, (base + len(chunks)) % 2)
                    one_k(
                        tap,
                        self.fb.add_i(ki, self.fb.const(1)),
                        (base + 2 * len(chunks)) % 2,
                    )

                if pairs:
                    self.fb.for_loop(1, 1 + 2 * pairs, 2, pair, retain_loop=True)
                if kt > 2 and (kt - 2) % 2:
                    one_k(
                        tap, self.fb.const(kt - 2), (base + (kt - 2) * len(chunks)) % 2
                    )
                if kt > 1:
                    one_k(
                        tap,
                        self.fb.const(kt - 1),
                        (base + (kt - 1) * len(chunks)) % 2,
                        final=True,
                    )
            store_outputs(n0, nr)

        def compact_prefetched_channel(n0, nr):
            """Retain alternating B loads across exact increasing K panels."""
            kt = s.cin // F.DIM

            def load(tap, ki, bank):
                krow = self.fb.add_i(
                    self.fb.const(tap * s.cin), self.fb.mul_i(ki, self.fb.const(F.DIM))
                )
                for d in range(0, len(nr), 4):
                    ptr = self._ptr(self.b, krow, s.cout, self._tile(n0, d))
                    self._rocc(
                        "mvin",
                        {
                            "local": self.bases[bank] + d * F.DIM,
                            "rows": F.DIM,
                            "cols": sum(nr[d : d + 4]),
                            "load_id": 1,
                        },
                        ptr,
                    )

            def compute(tap, ki, bank, first=False):
                kh, kw = divmod(tap, 3)
                for d, cols in enumerate(nr):
                    compact_spatial(ki, kh, kw, d, cols, self.bases[bank], first)

            load(0, self.fb.const(0), 0)
            for tap in range(9):
                base = tap * kt
                if kt > 1:
                    load(tap, self.fb.const(1), (base + 1) % 2)
                elif tap < 8:
                    load(tap + 1, self.fb.const(0), (base + 1) % 2)
                compute(tap, self.fb.const(0), base % 2, tap == 0)
                pairs = max(0, (kt - 2) // 2)

                def pair(ki, tap=tap, base=base):
                    following = self.fb.add_i(ki, self.fb.const(1))
                    load(tap, following, base % 2)
                    compute(tap, ki, (base + 1) % 2)
                    load(tap, self.fb.add_i(ki, self.fb.const(2)), (base + 1) % 2)
                    compute(tap, following, base % 2)

                if pairs:
                    self.fb.for_loop(1, 1 + 2 * pairs, 2, pair, retain_loop=True)
                for ki in range(1 + 2 * pairs, kt):
                    if ki + 1 < kt:
                        load(tap, self.fb.const(ki + 1), (base + ki + 1) % 2)
                    elif tap < 8:
                        load(tap + 1, self.fb.const(0), (base + ki + 1) % 2)
                    compute(tap, self.fb.const(ki), (base + ki) % 2)
            store_outputs(n0, nr)

        selected = (
            compact_packet_prefetched_channel
            if self.weight_issue_tiles is not None and self.compact_commands
            else packet_prefetched_channel
            if self.weight_issue_tiles is not None
            else compact_prefetched_channel
            if self.prefetch_b and self.compact_commands
            else prefetched_channel
            if self.prefetch_b
            else channel
        )
        self._for_groups(
            _groups(s.cout, s.bn), selected, retain_loop=self.compact_commands
        )
        module = self._finish("gemmini_golden_resident_conv")
        module.attributes["gemmini.resident_conv_shape"] = StringAttr(
            json.dumps(asdict(s), sort_keys=True)
        )
        module.attributes["gemmini.resident_conv_layout"] = StringAttr(
            "channel-tile-major padded spatial planes; disjoint input DMA partitions"
        )
        if self.flat_spatial_planes:
            module.attributes["gemmini.resident_conv_flat_spatial_planes"] = StringAttr(
                "three kw-shifted halo-free planes per channel tile; flat output panels"
            )
        if self.rows_per_tile != 1:
            module.attributes["gemmini.resident_conv_rows_per_tile"] = StringAttr(
                str(self.rows_per_tile)
            )
        if self.loop_channels:
            module.attributes["gemmini.resident_conv_loop_channels"] = StringAttr(
                "bounded ordinary CPU channel loop"
            )
        if self.prefetch_b:
            module.attributes["gemmini.resident_conv_prefetch_b"] = StringAttr(
                "one panel lookahead; disjoint bank2/bank3 weight lifetimes"
            )
        if self.weight_issue_tiles is not None:
            module.attributes["gemmini.resident_conv_weight_issue_tiles"] = StringAttr(
                str(self.weight_issue_tiles)
            )
        if self.explicit_weight_base is not None:
            module.attributes["gemmini.resident_conv_weight_base"] = StringAttr(
                str(self.bbase)
            )
        if self.source_stride:
            module.attributes["gemmini.resident_conv_source_stride"] = StringAttr(
                str(s.stride)
            )
        if self.row_residue:
            module.attributes["gemmini.resident_conv_row_residue"] = StringAttr(
                json.dumps(
                    {
                        "stride": s.stride,
                        "input_pitch": self.input_pitch,
                        "residue_rows": self.residue_rows,
                        "output_pitch": self.output_pitch,
                        "allocated_plane_rows": self.plane,
                    },
                    sort_keys=True,
                )
            )
        if self.compact_commands:
            module.attributes["gemmini.resident_conv_compact_commands"] = StringAttr(
                "retained bounded ordinary CPU K/spatial loops; unchanged primitive order and alternating B lifetimes"
            )
        if self.store_plan is not None:
            module.attributes["gemmini.paired_readout"] = StringAttr(
                json.dumps(
                    {
                        "certificate": self.store_plan.certificate(),
                        "abi": "A:i8*,B:i8*,first:i8*,second:i8*",
                        "storage": "Two complete disjoint byte outputs, disjoint from immutable inputs; fresh caller-owned scratch remains live until decoder completes",
                        "lifetime": "Both wide store passes precede ACC reuse; final fence completes both outputs",
                    },
                    sort_keys=True,
                )
            )

        return module


def command_counts(s, *, rows_per_tile=1):
    """Exact primitive counts and transferred activation bytes for this schedule."""
    g = GoldenResidentConv(s, rows_per_tile=rows_per_tile)
    kt, nt = _ceil_div(s.cin, F.DIM), _ceil_div(s.cout, F.DIM)
    bloads = sum(_ceil_div(min(s.bn, nt - d), 4) for d in range(0, nt, s.bn))
    compute = 9 * kt * nt * len(g.row_tiles)
    return {
        "mvin_a": (2 * _ceil_div(s.w + 2, F.DIM) + s.h * (2 + _ceil_div(s.w, F.DIM)))
        * _ceil_div(s.cin, 4 * F.DIM),
        "mvin_b": 9 * kt * bloads,
        "compute": compute,
        "preload": compute,
        "mvout": s.oh * (nt if s.output_dtype == "i32" else bloads),
        "padded_array_issue_cycles": compute * F.DIM,
        "activation_dram_bytes": s.h * s.w * s.cin,
        "resident_input_rows": kt * g.plane,
        "input_plane_stride": g.plane,
        "accumulator_rows": len(g.row_tiles) * s.bn * F.DIM,
        "weight_base_row": g.bbase,
        "host_im2col_bytes": 0,
    }
