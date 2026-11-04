"""Source-exact linear raw accumulator halo epochs, with full completion.

Retain overlapping unrounded producer rows across spatial maximum readouts.
No accumulator copy, circular pool layout, DRAM intermediate or new target
ordering law is implied. Every raw producer tile and pooled store completes.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from .spatial_producer import SpatialProducerPlan
from .window_transfer import emit_convolution_load_issue_tile
from .convolution_tile_epochs import emit_convolution_compute_epoch_tile


@dataclass(frozen=True)
class SpatialRawRetention:
    source: SpatialProducerPlan
    output_rows: int

    def __post_init__(self):
        if type(self.source) is not SpatialProducerPlan:
            raise ValueError('raw retention requires the original checked spatial source')
        self.source.__post_init__()
        p, s = self.source.producer, self.source.spatial
        if (self.source.readout_mode != 'single_readout' or s.window != (3, 3)
            or s.stride != (2, 2) or type(self.output_rows) is not int
            or not 2 <= self.output_rows <= s.output_hw[0]
            or p.accumulator_row + self.required_rows > p.accumulator_rows
            or self.required_rows * p.primitive * 4 >= 1 << 63):
            raise ValueError('raw retention requires bounded contiguous 3x3/stride2 accumulator epochs')

    @property
    def width(self):
        return (self.source.output_tile_width - 1) * 2 + 3

    @property
    def required_rows(self):
        return ((self.output_rows - 1) * 2 + 3) * self.width

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def propose_spatial_raw_retention(source):
    if type(source) is not SpatialProducerPlan:
        raise ValueError('raw retention requires original typed spatial source')
    source.__post_init__()
    if source.readout_mode != 'single_readout' or source.spatial.window != (3, 3) or source.spatial.stride != (2, 2):
        return None
    p, s = source.producer, source.spatial
    width = (source.output_tile_width - 1) * 2 + 3
    rows = min(s.output_hw[0], ((p.accumulator_rows - p.accumulator_row) // width - 3) // 2 + 1)
    return None if rows < 2 else SpatialRawRetention(source, rows)


def emit_spatial_raw_retention(plan, builder, issue, complete, *, law, epoch_law=None, compute_pairs=None, packed_consumer=None, tile_consumer=None):
    """Emit actual original convolution products and contiguous raw pool views.

    The adapter binds original compute/readout effects and full device drains.
    Each epoch owns one channel tile and a disjoint linear sequence of logical
    halo rows. Original clipped border copies remain separate logical slots.
    """
    if type(plan) is not SpatialRawRetention:
        raise ValueError('raw retention requires a checked epoch plan')
    plan.__post_init__()
    source = plan.source
    source_identity, identity = source.identity, plan.identity
    law.__post_init__()
    original_law = asdict(law)
    p, s = source.producer, source.spatial
    if law.source_identity != p.identity or not callable(issue) or not callable(complete):
        raise ValueError('raw retention requires original load issue law and actual completion')
    epoch_before = None
    if epoch_law is not None:
        epoch_law.require(compute_pairs)
        if epoch_law.source_identity != p.identity:
            raise ValueError("raw retention compute epoch has a different original source")
        epoch_before = asdict(epoch_law)
    elif compute_pairs is not None:
        raise ValueError("raw retention compute epoch needs its stronger selected law")
    if packed_consumer is not None:
        from .spatial_packed_raw import PackedRawConvolution
        if (type(packed_consumer) is not PackedRawConvolution
                or packed_consumer.original != p or epoch_law is None):
            raise ValueError("raw retention packed consumer needs the exact source and epochs")
        packed_consumer.__post_init__()
    if tile_consumer is not None:
        from .spatial_residue_runs import SpatialResidueRuns
        from .spatial_horizontal_flat import SpatialHorizontalFlat
        if (type(tile_consumer) not in (SpatialResidueRuns, SpatialHorizontalFlat) or tile_consumer.original != p
                or epoch_law is None or packed_consumer is not None):
            raise ValueError('raw retention tile consumer needs the exact source and epochs')
        tile_consumer.__post_init__()
    h, w = s.input_hw
    oh, ow = s.output_hw
    pt, pl, _, _ = s.padding
    op = builder.op

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda i: body(op('mul', i, tile), tile))
        if tail:
            body(whole * tile, tail)

    def clip(v, extent):
        v = builder.select(builder.cmp('slt', v, 0), 0, v)
        return builder.select(builder.cmp('ult', v, extent), v, extent - 1)

    def image(image):
        def stripe(x, count):
            width = (count - 1) * 2 + 3
            base_x = op('sub', op('mul', x, 2), pl)
            def channel(column, columns):
                def epoch(first_y, count_y):
                    def output(dy):
                        y = op('add', first_y, dy)
                        base_y = op('sub', op('mul', y, 2), pt)
                        def row(ky):
                            iy = clip(op('add', base_y, ky), h)
                            local_y = op('add', op('mul', dy, 2), ky)
                            def chunk(local_x, rows):
                                raw_x = op('add', base_x, local_x)
                                last_x = op('add', raw_x, rows - 1)
                                valid = builder.op('and', builder.cmp('ult', raw_x, w), builder.cmp('ult', last_x, w), 'i1')
                                def produce(ix, count, local):
                                    global_row = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                                    acc = op('add', p.accumulator_row, op('add', op('mul', local_y, width), local))
                                    emitter = emit_convolution_load_issue_tile if epoch_law is None else emit_convolution_compute_epoch_tile
                                    extra = {} if epoch_law is None else dict(epoch_law=epoch_law, compute_pairs=compute_pairs)
                                    if packed_consumer is not None:
                                        emitter = lambda ignored, *args, **kwargs: packed_consumer.emit(*args, **kwargs)
                                    if tile_consumer is not None:
                                        emitter = lambda ignored, *args, **kwargs: tile_consumer.emit(*args, **kwargs)
                                    emitter(p, builder,
                                        lambda kind, f: issue('finish_raw_tile' if kind == 'store' else kind, f),
                                        complete, output_row=global_row, output_column=column,
                                        rows=count, columns=columns, accumulator_row=acc, law=law, **extra)
                                builder.branch(valid, lambda: produce(raw_x, rows, local_x),
                                    lambda: builder.loop(0, rows, 1, lambda i: produce(clip(op('add', raw_x, i), w), 1, op('add', local_x, i))))
                            split(width, p.row_tile, chunk)
                        builder.branch(builder.cmp('eq', dy, 0),
                            lambda: builder.loop(0, 3, 1, row),
                            lambda: builder.loop(1, 3, 1, row))
                        f = dict(output_row=op('add', op('mul', op('add', op('mul', image, oh), y), ow), x),
                            output_column=column, columns=columns, output_pixels=count,
                            accumulator_row=op('add', p.accumulator_row, op('mul', op('mul', dy, 2), width)),
                            local_height=3, local_width=width, window_height=3, window_width=3,
                            stride_width=2, readout_tile=op('udiv', column, p.column_tile))
                        issue('store_spatial_max', f)
                        complete('store_spatial_max', f)
                    builder.loop(0, count_y, 1, output)
                split(oh, plan.output_rows, epoch)
            split(p.output_channels, p.column_tile, channel)
        begin = min(ow, max(0, (pl + 1) // 2))
        end = min(ow, max(begin, (w + pl - 3) // 2 + 1))
        builder.loop(0, begin, 1, lambda x: stripe(x, 1))
        split(end - begin, source.output_tile_width, lambda x, count: stripe(op('add', x, begin), count))
        builder.loop(end, ow, 1, lambda x: stripe(x, 1))
    builder.loop(0, p.window.input_shape[0], 1, image)
    plan.__post_init__()
    law.__post_init__()
    if source.identity != source_identity or plan.identity != identity or asdict(law) != original_law:
        raise ValueError('raw retention source or resources changed during emission')
    if epoch_law is not None and epoch_before != asdict(epoch_law):
        raise ValueError("raw retention compute epoch law changed during emission")
    return dict(tile_consumer=None if tile_consumer is None else tile_consumer.identity, packed_consumer=None if packed_consumer is None else packed_consumer.identity, compute_epoch_law=epoch_before, compute_pairs=compute_pairs, schema='merlin.spatial_raw_retention.v1', identity=identity,
        source_identity=source_identity, output_rows_per_epoch=plan.output_rows,
        private_accumulator_rows=plan.required_rows,
        private_accumulator_bytes=plan.required_rows*p.primitive*4,
        raw_accumulator_interval=[p.accumulator_row,p.accumulator_row+plan.required_rows],
        channels_live=min(p.column_tile,p.output_channels), full_intermediate_image_bytes=0,
        original_readout_chains=asdict(source)['readouts'],
        overlap_policy='raw row retained until final original max readout; linear contiguous layout',
        overlap_proof='slot(y,2)=(2*y+2)*width=slot(y+1,0); only rows1,2 written for y+1; no write to shared row',
        completion=('full paired-product/raw-tile/spatial-store completion' if epoch_law is None else
            'selected source-index compute epochs; full raw-tile/spatial-store completion'),
        scope='checked counted emission; no target timing qualification')
