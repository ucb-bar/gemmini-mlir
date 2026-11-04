"""Source-bound two-dimensional halo blocks for private pooled convolution.

Horizontal and vertical neighboring windows share their union of raw producer
rows. Every source product remains in kernel/channel order; the final pool is
completed before any of the assigned ACC rectangle can be reused.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan
from merlin.compile.scheduling.spatial_readout import prove_spatial_readout
from merlin.compile.scheduling.window_transfer import emit_convolution_tile
from merlin.compile.scheduling.window_transfer import (
    ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw,
)


@dataclass(frozen=True)
class SpatialComputeEpochs:
    source_identity: str
    issue_law: ConvolutionLoadIssueLaw
    epoch_law: ConvolutionComputeEpochLaw
    compute_pairs: int
    report: dict

    def require(self, plan):
        if (type(self.issue_law) is not ConvolutionLoadIssueLaw
                or type(self.epoch_law) is not ConvolutionComputeEpochLaw
                or self.source_identity != plan.identity
                or self.issue_law.source_identity != plan.identity
                or self.epoch_law.source_identity != plan.identity):
            raise ValueError('spatial epochs do not bind the actual private block source')
        self.issue_law.__post_init__()
        self.epoch_law.require(self.compute_pairs)


def plan_spatial_epochs(plan, *, complete_reduction=False):
    """Reuse the selected completion facts, bound to this exact spatial source."""
    from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
    from .conv_epochs import epoch_admissions
    plan.__post_init__()
    p = plan.source.producer
    intervals = tuple((n, min(p.column_tile, p.output_channels - n))
                      for n in range(0, p.output_channels, p.column_tile))
    source = ConvolutionExplorationSource(p, intervals, plan.identity)
    issue, epoch, pairs, report = epoch_admissions(source,
        'gemmini-fire-aligned-b8ff28edc096392c',
        'signed-i8-products-i32-exact-spatial-readout',
        'private-spatial-block-complete-before-pool-and-reuse-v1',
        complete_reduction=complete_reduction)
    # A spatial tile has one channel panel, so the whole-convolution heuristic
    # (two N traversals) would drain after only two K fragments. The selected
    # finite execute lane holds eight descriptor/product pairs. Use that bound
    # here; correctness still requires every selected dependency/progress fact,
    # not merely the queue's capacity.
    original_pairs = pairs
    pairs = max(pairs, epoch.pending_compute_pairs)
    report = dict(report, base_source_epoch_pairs=original_pairs,
                  max_issued_compute_pairs=pairs,
                  budget='at least selected finite execute descriptor/product pair capacity')
    # The ordering evidence is independent of tensor coordinates. Bind its
    # permissions to the complete transformed block, including its ACC map and
    # final pool observation, rather than just the underlying convolution.
    result = SpatialComputeEpochs(plan.identity,
        replace(issue, source_identity=plan.identity),
        replace(epoch, source_identity=plan.identity, max_compute_pairs=pairs), pairs, report)
    result.require(plan)
    return result


def _emit_epoch_tile(plan, epochs, builder, issue, complete, *, effect_generator=None, **coordinates):
    """Consume canonical original tile effects with real runtime epoch drains.

    The public synchronous tile wrapper cannot grant asynchronous callbacks.
    This separate consumer uses its canonical effect generator, supplies the
    selected issue/retirement laws and owns every required completion boundary.
    """
    from merlin.compile.scheduling.window_transfer import (
        _emit_convolution_tile_effects, _validate_convolution_load_issue,
    )
    epochs.require(plan)
    p = plan.source.producer
    _validate_convolution_load_issue(p, builder, issue, complete,
        coordinates['rows'], epochs.issue_law, int(p.bias))
    pending = None
    ci = p.window.input_shape[-1]
    chunks = (ci + p.channel_tile - 1) // p.channel_tile

    def consume(kind, fields):
        nonlocal pending
        if kind == 'load_weight':
            if pending is not None:
                raise ValueError('spatial epoch lost descriptor/product pairing')
            pending = fields['reduction_row']
        issue('finish_raw_tile' if kind == 'store' else kind, fields)
        if kind == 'compute':
            if pending is None:
                raise ValueError('spatial epoch product lacks original reduction coordinates')
            kernel = builder.op('udiv', pending, ci)
            channel = builder.op('urem', pending, ci)
            pair = builder.op('add', builder.op('add', builder.op('mul', kernel, chunks),
                builder.op('udiv', channel, p.channel_tile)), 1)
            builder.branch(builder.cmp('eq', builder.op('urem', pair, epochs.compute_pairs), 0),
                lambda: complete('compute_epoch', fields), lambda: None)
            pending = None
        elif kind == 'store':
            if pending is not None:
                raise ValueError('spatial epoch reaches its final drain with an unmatched product')
            # This is the exact last producer use of these input/RHS rows and
            # the result-visibility boundary before the stripe's pool reads.
            complete('raw_tile_complete', fields)

    if effect_generator is None:
        _emit_convolution_tile_effects(p, builder, consume, **coordinates)
    else:
        effect_generator(builder, consume, **coordinates)
    if pending is not None:
        raise ValueError('spatial epoch leaves an unmatched original product')
    epochs.require(plan)


@dataclass(frozen=True)
class SpatialBlockPlan:
    source: SpatialProducerPlan
    output_tile_height: int

    def __post_init__(self):
        self.source.__post_init__()
        p, s = self.source.producer, self.source.spatial
        if (self.source.readout_mode != 'single_readout' or s.stride[0] != s.stride[1]
                or not 1 <= self.output_tile_height <= min(255, s.output_hw[0])
                or s.output_hw[1] > 255 or self.local_height > 255
                or p.accumulator_row + self.local_height * self.local_width > p.accumulator_rows):
            raise ValueError('pooled convolution block exceeds its source or physical geometry')

    @property
    def local_height(self):
        return (self.output_tile_height - 1) * self.source.spatial.stride[0] + self.source.spatial.window[0]

    @property
    def local_width(self):
        return (self.source.output_tile_width - 1) * self.source.spatial.stride[1] + self.source.spatial.window[1]

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def emit_spatial_blocks(plan, builder, lower, *, synchronous_completion,
                        epochs=None, complete=None, resident=None, halo=None):
    """Emit actual union-halo products followed by one rectangular pooled store."""
    plan.__post_init__()
    if epochs is None:
        if synchronous_completion is not True:
            raise ValueError('spatial block ownership requires full synchronous callbacks')
    elif synchronous_completion is not False or not callable(complete):
        raise ValueError('spatial epochs require explicit issue-only callbacks and real completion')
    else:
        epochs.require(plan)
    identity = plan.identity
    if resident is not None:
        if epochs is None:
            raise ValueError("resident spatial blocks require explicit completion ownership")
        resident.require(plan)
    if halo is not None:
        halo.require(plan)
    schedule_identity = hashlib.sha256(json.dumps(dict(
        source=identity, resident=None if resident is None else resident.receipt(),
        halo=None if halo is None else halo.receipt(),
        issue=None if epochs is None else asdict(epochs.issue_law),
        epoch=None if epochs is None else asdict(epochs.epoch_law),
        compute_pairs=None if epochs is None else epochs.compute_pairs),
        sort_keys=True).encode()).hexdigest()
    source = plan.source
    p, s = source.producer, source.spatial
    h, w = s.input_hw
    oh, ow = s.output_hw
    ph, pw = s.window
    sh, sw = s.stride
    top, left, _, _ = s.padding
    op = builder.op

    def clip(value, extent):
        positive = builder.select(builder.cmp('slt', value, 0), 0, value)
        return builder.select(builder.cmp('ult', positive, extent), positive, extent - 1)

    def split(extent, tile, body):
        full, tail = divmod(extent, tile)
        if full:
            builder.loop(0, full, 1, lambda i: body(op('mul', i, tile), tile))
        if tail:
            body(full * tile, tail)

    def image_body(image, column, columns, effects):
        def block_y(y, output_rows):
            local_height = (output_rows - 1) * sh + ph
            base_y = op('sub', op('mul', y, sh), top)
            def block_x(x, output_pixels):
                local_width = (output_pixels - 1) * sw + pw
                base_x = op('sub', op('mul', x, sw), left)
                def produce_y(ky):
                    raw_y = op('add', base_y, ky)
                    iy = clip(raw_y, h)
                    def fill(local, count):
                        acc = op('add', p.accumulator_row, op('add', op('mul', ky, local_width), local))
                        effects('initialize_pool_halo', dict(accumulator_row=acc,
                            rows=count, columns=columns, output_column=column))
                    def produce_x(local_x, rows):
                        raw_x = op('add', base_x, local_x)
                        valid = op('and', builder.cmp('ult', raw_x, w),
                                   builder.cmp('ult', op('add', raw_x, rows - 1), w), 'i1')
                        def produce(ix, count, local):
                            global_row = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                            acc = op('add', p.accumulator_row, op('add', op('mul', ky, local_width), local))
                            coordinates = dict(output_row=global_row, output_column=column,
                                rows=count, columns=columns, accumulator_row=acc)
                            if epochs is None:
                                emit_convolution_tile(p, builder,
                                    lambda kind, fields: effects('finish_raw_tile' if kind == 'store' else kind, fields),
                                    **coordinates, synchronous_completion=True)
                            else:
                                _emit_epoch_tile(plan, epochs, builder, effects, complete,
                                    effect_generator=None if resident is None else resident.emit_tile_effects,
                                    **coordinates)
                        def boundary():
                            # A primitive tile is no wider than the original
                            # convolution row, so a partial tile crosses at most
                            # one edge. Peel only duplicated pool halo cells;
                            # retain the complete interior as one vector tile.
                            def choose(index=1):
                                if index >= rows:
                                    builder.loop(0, rows, 1, lambda i: (
                                        builder.branch(builder.cmp('ult', op('add', raw_x, i), w),
                                            lambda: produce(op('add', raw_x, i), 1, op('add', local_x, i)),
                                            lambda: fill(op('add', local_x, i), 1)) if halo is not None else
                                        produce(clip(op('add', raw_x, i), w), 1, op('add', local_x, i))))
                                    return
                                left_edge = builder.cmp('eq', raw_x, -index)
                                right_edge = builder.cmp('eq', raw_x, w - rows + index)
                                def left_band():
                                    if halo is not None:
                                        fill(local_x, index)
                                    else:
                                        builder.loop(0, index, 1, lambda i: produce(0, 1, op('add', local_x, i)))
                                    produce(0, rows - index, op('add', local_x, index))
                                def right_band():
                                    produce(raw_x, rows - index, local_x)
                                    if halo is not None:
                                        fill(op('add', local_x, rows - index), index)
                                    else:
                                        builder.loop(0, index, 1, lambda i: produce(w - 1, 1,
                                            op('add', local_x, op('add', rows - index, i))))
                                builder.branch(left_edge, left_band,
                                    lambda: builder.branch(right_edge, right_band, lambda: choose(index + 1)))
                            choose()
                        builder.branch(valid, lambda: produce(raw_x, rows, local_x), boundary)
                    if halo is None:
                        split(local_width, p.row_tile, produce_x)
                    else:
                        builder.branch(builder.cmp('ult', raw_y, h),
                            lambda: split(local_width, p.row_tile, produce_x),
                            lambda: split(local_width, p.primitive, fill))
                builder.loop(0, local_height, 1, produce_y)
                if halo is not None and epochs is not None:
                    complete('raw_block_ready', {})
                output_row = op('add', op('mul', op('add', op('mul', image, oh), y), ow), x)
                lower('store_spatial_max', dict(output_row=output_row, output_column=column,
                    columns=columns, output_pixels=output_pixels, output_rows=output_rows,
                    output_row_stride=ow, accumulator_row=p.accumulator_row,
                    local_height=local_height, local_width=local_width,
                    window_height=ph, window_width=pw, stride_width=sw,
                    readout_tile=op('udiv', column, p.column_tile)))
                if epochs is not None:
                    complete('pool_complete', dict(output_row=output_row,
                        output_column=column, rows=output_rows, columns=columns))
            split(ow, source.output_tile_width, block_x)
        split(oh, plan.output_tile_height, block_y)
    def phase(column, columns):
        effects = lower if resident is None else resident.phase(builder, lower, complete, column, columns)
        builder.loop(0, p.window.input_shape[0], 1,
            lambda image: image_body(image, column, columns, effects))
        if resident is not None:
            complete('resident_rhs_last_use', {})
    split(p.output_channels, p.column_tile, phase)
    plan.__post_init__()
    if resident is not None:
        resident.require(plan)
    if identity != plan.identity:
        raise ValueError('pooled convolution source or private block changed during emission')
    return dict(schema='merlin.convolution_pool_blocks.v1', plan_sha256=identity,
        schedule_sha256=schedule_identity,
        source_sha256=source.identity, output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
        readout_proofs=[prove_spatial_readout(chain) for chain in source.readouts],
        readout_mode=source.readout_mode,
        retained_readout_chains=[asdict(chain) for chain in source.readouts],
        compute_epochs=None if epochs is None else dict(epochs.report,
            source_sha256=epochs.source_identity,
            issue_law=asdict(epochs.issue_law), epoch_law=asdict(epochs.epoch_law),
            epoch_index='(ky*kw+kx)*ceil(ci/channel_tile)+channel/channel_tile+1',
            ownership='real raw-tile, pooled-store and final drains'),
        output_tile_height=plan.output_tile_height, output_tile_width=source.output_tile_width,
        private_accumulator_rows=plan.local_height * plan.local_width,
        full_intermediate_image_bytes=0, host_payload_operations=0,
        retained_rhs=None if resident is None else resident.receipt(),
        minimum_halo=None if halo is None else halo.receipt(),
        overlap_policy=('compute each valid union cell once per block; source minimum halo DMA' if halo else
            'compute each union-halo cell once per block; source minimum-padding edge duplication'),
        completion=('fact-backed runtime K epochs; raw tile completes before pool; pool completes before reuse'
                    if epochs else 'every producer callback and pooled block store completes before ownership reuse'))
