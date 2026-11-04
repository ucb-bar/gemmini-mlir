"""Checked independent cache-reload epochs over original Q1 source effects.

The old synchronous cache API is untouched. This owner admits only disjoint
ACC reload writes with immutable original Q1 source until their complete drain.
The actual consumer and its complete drain finish before producer/ring reuse.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from .spatial_readout_cache import SpatialReadoutCache, emit_cached_spatial_max


@dataclass(frozen=True)
class SpatialCacheEpochLaw:
    hardware_identity: str
    ordering_identity: str
    numerical_identity: str
    source_identity: str
    private_owner_identity: str
    pending_load_capacity: int
    fixed_load_configuration: bool = False
    finite_load_queue_progress: bool = False
    complete_reload_drain: bool = False
    complete_store_drain: bool = False
    immutable_q1_until_drain: bool = False
    exclusive_producer_cache_windows: bool = False

    def require(self, plan):
        if type(plan) is not SpatialReadoutCache:
            raise ValueError('cache epochs require original checked cache owner')
        plan.__post_init__()
        if (self.source_identity != plan.identity or
            any(type(v) is not str or not v for v in (self.hardware_identity,
                self.ordering_identity,self.numerical_identity,self.private_owner_identity)) or
            type(self.pending_load_capacity) is not int or self.pending_load_capacity <= 0 or
            any(getattr(self,k) is not True for k in ('fixed_load_configuration',
                'finite_load_queue_progress','complete_reload_drain','complete_store_drain',
                'immutable_q1_until_drain','exclusive_producer_cache_windows'))):
            raise ValueError('cache epochs lack selected configuration/progress/completion/source ownership premises')


def emit_spatial_cache_epochs(plan, builder, lower, *, image, pool_y, output_x, output_pixels, law):
    """Issue disjoint reloads, complete them, then consume and complete stores.

    This independent API does not weaken emit_cached_spatial_max's synchronous
    callback contract. Each callback exposes its actual phase and fields.
    """
    law.require(plan)
    plan.__post_init__()
    source = plan.source.packed.source
    p, s = source.producer, source.spatial
    height, _ = s.input_hw
    oh, ow = s.output_hw
    if type(output_pixels) is not int or not 0 < output_pixels <= source.output_tile_width:
        raise ValueError("actual stripe exceeds the checked readout footprint")
    op = builder.op
    width = (output_pixels - 1) * 2 + 3
    epoch_index = op("udiv", pool_y, plan.output_rows)
    epoch_first = op("mul", epoch_index, plan.output_rows)
    epoch_start = op("sub", op("mul", epoch_first, 2), 1)
    first = builder.cmp("eq", pool_y, epoch_first)
    original_first = op("sub", op("mul", pool_y, 2), 1)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda i: body(op("mul", i, tile), tile))
        if tail:
            body(whole * tile, tail)

    def channels(column, count):
        lower("begin_reload_epoch", dict(output_column=column, columns=count))
        plane = op("udiv", column, p.column_tile)
        base = op("add", plan.cache.start // plan.row_bytes, op("mul", plane, plan.span * width))

        def row(ky):
            original_y = op("add", original_first, ky)
            clipped = builder.select(builder.cmp("slt", original_y, 0), 0, original_y)
            clipped = builder.select(builder.cmp("slt", clipped, height), clipped, height - 1)
            needed = builder.cmp(
                "slt", 0, op("add", builder.select(first, 1, 0), builder.select(builder.cmp("slt", 0, ky), 1, 0))
            )

            def stage():
                def chunk(offset, rows):
                    lower(
                        "reload_intermediate",
                        dict(
                            accumulator_row=op(
                                "add", base, op("add", op("mul", op("sub", original_y, epoch_start), width), offset)
                            ),
                            scratch_offset=op(
                                "add",
                                plan.source.packed.workspace.start,
                                op(
                                    "add",
                                    op(
                                        "mul",
                                        op("add", op("mul", op("urem", clipped, 3), width), offset),
                                        p.output_channels,
                                    ),
                                    column,
                                ),
                            ),
                            scratch_row_stride=p.output_channels,
                            rows=rows,
                            columns=count,
                            stripe_rows=3 * width,
                            output_column=column,
                            readout_tile=plane,
                        ),
                    )

                split(width, p.row_tile, chunk)

            builder.branch(needed, stage, lambda: None)

        builder.loop(0, 3, 1, row)
        lower("complete_reload_epoch", dict(output_column=column, columns=count))

        # Keep the old complete 3-row rectangle and clipped vertical halo.
        # The selected native pool declaration need not authorize padding.
        lower(
            "store_spatial_max",
            dict(
                output_row=op("add", op("mul", op("add", op("mul", image, oh), pool_y), ow), output_x),
                output_column=column,
                columns=count,
                output_pixels=output_pixels,
                accumulator_row=op("add", base, op("mul", op("sub", original_first, epoch_start), width)),
                local_height=3,
                local_width=width,
                window_height=3,
                window_width=3,
                stride_width=2,
                upad=0,
                readout_tile=plane,
            ),
        )

        lower("complete_store_epoch", dict(output_column=column, columns=count))

    split(p.output_channels, p.column_tile, channels)
