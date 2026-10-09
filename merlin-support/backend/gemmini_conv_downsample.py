"""The strided-load mode a 1x1 stride-2 convolution is entitled to, and what leaving it off costs.

WHY THIS EXISTS. The sequencer's input loader walks the input window at full resolution: one transfer
per input pixel per channel block. For a convolution of stride s the compute only ever reads every
s-th pixel in each dimension, so at stride 2 three of every four staged pixels are moved into the
scratchpad and never read. The RTL has a mode that removes exactly that waste -- `LoopConvLdInput`'s
``downsample`` bit doubles the mvin's DRAM row stride, steps its row iterator by two, and halves the
row count it asks for, so the loader gathers the strided subset directly through the descriptor's own
stride field:

    config.rs2 := dram_stride << downsample                 (LoopConv.scala, LoopConvLdInput)
    mvin.num_rows := I >> downsample
    next_irow := sFloorAdd(irow, 1.U << downsample, ...)

and the execute unit reads the halved window back with ``irows >> downsample`` / ``icols >>
downsample``, which is why the two must be set together with ``a_stride = stride >> downsample``.

The mode is not general: halving the window is only the same computation when each output pixel maps
to one input pixel, which is a 1x1 kernel at stride 2 over an even extent with no padding and no
pooling. That predicate is the library's own, and it is READ OUT of the target's header here rather
than restated, because a predicate transcribed once is a predicate that drifts.

What this module computes is the input staging both ways, in the unit the DMA actually costs -- a
transfer per pixel per channel block -- so "the loader moves four times what it uses" is a number
rather than a story.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: The clauses this module implements, normalised the way :func:`header_downsample_clauses` returns
#: them. The library declares the predicate at more than one entry point, and the declarations are NOT
#: identical -- the tile-search entry omits the two clauses it has already pinned by construction. So
#: the contract is stated as containment, checked by :func:`predicate_is_implemented`: every clause any
#: declaration states must be one this module checks (otherwise we would enable the mode where the
#: device would not), and every clause this module checks must be stated somewhere (otherwise we would
#: be refusing layers on a condition the device does not have). A declaration that grows a clause is
#: UNKNOWN, not a guess.
IMPLEMENTED_CLAUSES: frozenset[str] = frozenset(
    {
        "stride == 2",
        "kernel_dim == 1",
        "in_row_dim % 2 == 0",
        "in_col_dim % 2 == 0",
        "padding == 0",
        "no_pool",
        "input_dilation == 1",
        "!trans_input_3120",
    }
)

#: The declaration's leading tokens. Matched as tokens, not as text, so a differently spaced or
#: differently wrapped declaration is still found.
_DECLARATION = ("const", "bool", "downsample", "=")


class DownsampleUnknown(Exception):
    """The device's own predicate could not be read, so no eligibility claim is made.

    Raised rather than defaulted: a convolution wrongly declared eligible stages a quarter of the
    input it needs and produces a wrong output, and one wrongly declared ineligible is merely slow.
    Neither is a default worth picking silently.
    """


def header_downsample_clauses(header_text: str) -> tuple[tuple[str, ...], ...]:
    """Each declaration of the library's ``downsample`` predicate, as its `&&`-separated clauses.

    Parsed structurally -- cut the text into statements at `;`, tokenise each on whitespace, and take
    the ones whose tokens open with the declaration; the clauses are what follows, split on the boolean
    connective and whitespace-squeezed. No pattern matching: a predicate spelled with different spacing
    or wrapped at a different column is the same predicate, and a matcher that missed it would report a
    layer ineligible for a mode the device has.
    """
    found: list[tuple[str, ...]] = []
    for statement in header_text.split(";"):
        tokens = statement.split()
        for start in range(len(tokens) - len(_DECLARATION) + 1):
            if tuple(tokens[start : start + len(_DECLARATION)]) != _DECLARATION:
                continue
            body = " ".join(tokens[start + len(_DECLARATION) :])
            clauses = tuple(" ".join(part.split()) for part in body.split("&&"))
            if any(not clause for clause in clauses):
                raise DownsampleUnknown("UNKNOWN downsample predicate: an empty clause")
            found.append(clauses)
    if not found:
        raise DownsampleUnknown(f"UNKNOWN downsample predicate: no {' '.join(_DECLARATION)} statement in the header")
    return tuple(found)


def predicate_is_implemented(header_text: str) -> None:
    """Raise unless every declaration is covered by, and together they cover, the implemented set."""
    declarations = header_downsample_clauses(header_text)
    union: set[str] = set()
    for clauses in declarations:
        extra = set(clauses) - IMPLEMENTED_CLAUSES
        if extra:
            raise DownsampleUnknown(
                f"UNKNOWN downsample predicate: a declaration states {sorted(extra)}, which this module does not check"
            )
        union |= set(clauses)
    missing = IMPLEMENTED_CLAUSES - union
    if missing:
        raise DownsampleUnknown(
            f"UNKNOWN downsample predicate: this module checks {sorted(missing)}, which no declaration states"
        )


def downsample_flag(
    *,
    kernel: int,
    stride: int,
    padding: int,
    in_rows: int,
    in_cols: int,
    pooled: bool,
    input_dilation: int = 1,
    trans_input_3120: bool = False,
    header_text: str | None = None,
) -> int:
    """1 when this layer may be loaded strided, 0 when it may not.

    ``header_text`` is the target's convolution header; passing it checks the device's own predicate
    against :data:`IMPLEMENTED_CLAUSES` first, which is the only way this answer is bound to the
    library rather than to a memory of it.
    """
    if header_text is not None:
        predicate_is_implemented(header_text)
    return int(
        stride == 2
        and kernel == 1
        and in_rows % 2 == 0
        and in_cols % 2 == 0
        and padding == 0
        and not pooled
        and input_dilation == 1
        and not trans_input_3120
    )


def working_rows(
    *,
    acc: bool,
    downsample: int,
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
) -> int:
    """``tiled_conv_total_spad_rows`` with the strided-load mode honoured.

    The same arithmetic the schedule's capacity check already does, plus the one shift the header
    applies to both input extents. It is here and not a flag on the existing helper because the
    difference is load-bearing: at stride 2 the window is a QUARTER of what the unshifted form
    reports, so a capacity check that ignores the mode refuses output tiles the device would hold --
    and the tile it settles on instead is the thing that re-reads the input.
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


def _tiles(extent: int, size: int) -> list[int]:
    return [min(size, extent - start) for start in range(0, extent, size)]


def input_transfers(
    *,
    downsample: int,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    in_channels: int,
    tile: Mapping[str, int],
    facts: Mapping[str, object],
) -> int:
    """DRAM row transfers the input staging costs for this tiling, by the loader's own iteration.

    The loader walks the padded input window one row (one pixel) at a time, moving at most
    ``max_block_len * dim`` channels per transfer, and the whole window again for every OUTPUT-CHANNEL
    tile, because the channel tile is not an axis of the input. With the strided mode set it walks the
    window at half resolution in each dimension. The count is in transfers, not bytes, because the DMA
    costs per transfer and a narrow one costs what a wide one does.
    """
    dim = int(facts["dim"])
    mbl = int(facts["max_block_len"])
    rows = _tiles(out_dim, int(tile["porows"]))
    cols = _tiles(out_dim, int(tile["pocols"]))
    batches = _tiles(batch, int(tile["batches"]))
    channel_tiles = _tiles(in_channels, int(tile["kchs"]))
    out_tiles = _tiles(out_channels, int(tile["pochs"]))
    in_rows = sum((extent * stride + kernel - 1) >> downsample for extent in rows)
    in_cols = sum((extent * stride + kernel - 1) >> downsample for extent in cols)
    per_channel_tile = sum(-(-extent // (mbl * dim)) for extent in channel_tiles)
    return sum(batches) * in_rows * in_cols * len(out_tiles) * per_channel_tile


def staging_waste(
    *,
    batch: int,
    out_dim: int,
    out_channels: int,
    kernel: int,
    stride: int,
    in_channels: int,
    tile: Mapping[str, int],
    facts: Mapping[str, object],
    downsample: int,
) -> dict[str, int]:
    """What the input staging costs with the strided mode off and on, for one tiling.

    ``unstrided`` is what the sequencer moves today; ``strided`` is what it would move with the mode
    set; ``discarded`` is the difference, which is input the loader brings into the scratchpad and the
    mesh never reads. For an ineligible layer the two are equal and ``discarded`` is zero -- the mode
    is not a knob that trades anything, so a layer either qualifies for it or is unaffected.
    """
    common = dict(
        batch=batch,
        out_dim=out_dim,
        out_channels=out_channels,
        kernel=kernel,
        stride=stride,
        in_channels=in_channels,
        tile=tile,
        facts=facts,
    )
    unstrided = input_transfers(downsample=0, **common)
    strided = input_transfers(downsample=downsample, **common)
    return {
        "unstrided": unstrided,
        "strided": strided,
        "discarded": unstrided - strided,
        "row_bytes": int(facts["dim"]) * int(facts["elem_bytes"]),
    }


def descriptor_overrides(downsample: int, *, stride: int) -> dict[str, int]:
    """The two descriptor fields the mode changes, which are only ever correct together.

    ``downsample`` tells the loader to gather the strided subset; ``A_stride`` tells the execute unit
    how far apart the rows it reads are IN THE SCRATCHPAD, and the loader has already applied the
    stride, so it must come back down to one. Setting the first without the second reads every other
    staged row and computes a different convolution -- which is why they are returned as one thing.
    """
    return {"downsample": downsample, "A_stride": stride >> downsample}


def eligible_layers(layers: Sequence[Mapping[str, object]], *, header_text: str | None = None) -> list[dict]:
    """The subset of a model's convolutions the strided mode applies to, with each one's flag.

    Takes the layer descriptions a schedule already carries rather than a bespoke record, so the
    answer for a whole model is one call and not a transcription.
    """
    out = []
    for layer in layers:
        flag = downsample_flag(
            kernel=int(layer["kernel"]),
            stride=int(layer["stride"]),
            padding=int(layer["padding"]),
            in_rows=int(layer["in_rows"]),
            in_cols=int(layer["in_cols"]),
            pooled=bool(layer.get("pooled", False)),
            input_dilation=int(layer.get("input_dilation", 1)),
            trans_input_3120=bool(layer.get("trans_input_3120", False)),
            header_text=header_text,
        )
        if flag:
            out.append({**dict(layer), "downsample": flag})
    return out
