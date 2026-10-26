"""Shape-derived band planning for store-path max pooling.

A pooled store reads one ``orows x ocols`` image of local rows (one row per pixel,
one column per channel) and writes ``porows x pocols`` window maxima. A band is
the run of pooled output rows one store covers; its input rows overlap the next
band's when the window exceeds the stride, and the producer recomputes them.

Window taps outside the stored image are garbage addresses whose data is zero
(``Scratchpad.scala``), which would beat negative inputs. The planner therefore
never relies on register padding: ``upad``/``lpad`` stay zero, every window lies
inside its band image, and source padding is materialized by the caller as an
explicit border that is an identity of max. Register limits come from the
derived ``ConfigMvoutRs1`` field widths, never from literals.
"""

from __future__ import annotations

from dataclasses import dataclass

POOL_FIELDS = ("pool_stride", "pool_size", "pool_out_dim", "porows", "pocols", "orows", "ocols", "upad", "lpad")


class PoolPlanRefused(ValueError):
    pass


def register_limits() -> dict[str, int]:
    """Largest value of each pooling field in the selected CONFIG_ST layout."""
    from .readout_encoding import field

    return {name: (1 << field("ConfigMvoutRs1", name)[1]) - 1 for name in POOL_FIELDS}


def composition_facts() -> dict:
    """Window/stride maxima for Merlin's pool composition plan, per spatial axis."""
    limits = register_limits()
    return dict(pool_window_max=[limits["pool_size"]] * 2, pool_stride_max=[limits["pool_stride"]] * 2)


@dataclass(frozen=True)
class Image:
    batch: int
    rows: int
    cols: int
    channels: int


@dataclass(frozen=True)
class Window:
    size: int
    stride: int
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, left, bottom, right (identity of max)

    def padded(self, image: Image) -> tuple[int, int]:
        top, left, bottom, right = self.pad
        return image.rows + top + bottom, image.cols + left + right

    def output(self, image: Image) -> tuple[int, int]:
        rows, cols = self.padded(image)
        if rows < self.size or cols < self.size:
            raise PoolPlanRefused("pool window exceeds its padded image")
        return (rows - self.size) // self.stride + 1, (cols - self.size) // self.stride + 1


@dataclass(frozen=True)
class Band:
    batch: int
    out_rows: tuple[int, int]  # pooled rows [p0, p1)
    in_rows: tuple[int, int]  # padded-image rows [r0, r1)
    orows: int
    ocols: int
    porows: int
    pocols: int

    def fields(self, window: Window, pool_out_dim: int) -> dict[str, int]:
        return dict(pool_stride=window.stride, pool_size=window.size, pool_out_dim=pool_out_dim,
                    porows=self.porows, pocols=self.pocols, orows=self.orows, ocols=self.ocols, upad=0, lpad=0)


def _real_taps(window: Window, image: Image) -> None:
    """Every window must hold a real pixel; an all-padding window has no source value."""
    top, left, _, _ = window.pad
    out_rows, out_cols = window.output(image)
    for extent, lead, count in ((image.rows, top, out_rows), (image.cols, left, out_cols)):
        if lead >= window.size or (count - 1) * window.stride >= lead + extent:
            raise PoolPlanRefused("a pool window covers only padding")


def plan(image: Image, window: Window, *, capacity_rows: int, limits=None) -> list[Band]:
    """Bands covering every pooled output once with in-image windows, or a refusal."""
    limits = limits or register_limits()
    if min(image.batch, image.rows, image.cols, image.channels) <= 0:
        raise PoolPlanRefused("empty pool image")
    if not 1 <= window.size <= limits["pool_size"] or not 1 <= window.stride <= limits["pool_stride"]:
        raise PoolPlanRefused("pool window or stride exceeds the derived CONFIG_ST fields")
    if any(p < 0 for p in window.pad):
        raise PoolPlanRefused("negative pool padding")
    _real_taps(window, image)
    _, cols = window.padded(image)
    out_rows, out_cols = window.output(image)
    if cols > limits["ocols"] or out_cols > limits["pocols"] or out_cols > limits["pool_out_dim"]:
        raise PoolPlanRefused("pooled image width exceeds the derived CONFIG_ST fields")
    rows = min(limits["orows"], capacity_rows // cols)
    if rows < window.size:
        raise PoolPlanRefused("one pool window row band exceeds local capacity")
    per_band = min(limits["porows"], (rows - window.size) // window.stride + 1)
    bands = []
    for n in range(image.batch):
        for p0 in range(0, out_rows, per_band):
            p1 = min(out_rows, p0 + per_band)
            r0, r1 = p0 * window.stride, (p1 - 1) * window.stride + window.size
            bands.append(Band(n, (p0, p1), (r0, r1), r1 - r0, cols, p1 - p0, out_cols))
    covers(bands, image, window, capacity_rows=capacity_rows)
    return bands


def covers(bands, image: Image, window: Window, *, capacity_rows: int) -> None:
    """Independent check: each pooled row exactly once, every tap inside its band image."""
    out_rows, out_cols = window.output(image)
    seen = {}
    for band in bands:
        p0, p1 = band.out_rows
        if band.porows != p1 - p0 or band.pocols != out_cols or band.in_rows[0] != p0 * window.stride:
            raise PoolPlanRefused("band geometry differs from its pooled rows")
        if ((band.porows - 1) * window.stride + window.size > band.orows or
                (band.pocols - 1) * window.stride + window.size > band.ocols or
                band.in_rows[1] - band.in_rows[0] != band.orows or band.orows * band.ocols > capacity_rows):
            raise PoolPlanRefused("a pool window leaves its band image")
        for p in range(p0, p1):
            key = (band.batch, p)
            seen[key] = seen.get(key, 0) + 1
    if seen != {(n, p): 1 for n in range(image.batch) for p in range(out_rows)}:
        raise PoolPlanRefused("bands do not cover each pooled row exactly once")
