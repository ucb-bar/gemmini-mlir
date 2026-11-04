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


def plan_spatial_epochs(plan):
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
        complete_reduction=True)
    # The canonical tile below emits exactly one N panel, with real completion
    # after its complete original KH/KW/CI reduction. The selected dependency
    # and finite-backpressure laws permit issuing that bounded reduction without
    # an interior drain. Pending queue capacity is not an issued-work limit.
    chunks = (p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
    tile_pairs = p.window.kernel[0] * p.window.kernel[1] * chunks
    pairs = max(2, 1 << tile_pairs.bit_length())
    report = dict(report, source_compute_pairs_per_block=tile_pairs,
                  live_columns_from_acc_capacity=1,
                  max_issued_compute_pairs=pairs,
                  no_intermediate_compute_boundary=pairs > tile_pairs,
                  epoch_scope='one original reduction per private spatial tile',
                  ownership_boundary='raw tile before pool; pooled store before ACC reuse')
    # The ordering evidence is independent of tensor coordinates. Bind its
    # permissions to the complete transformed block, including its ACC map and
    # final pool observation, rather than just the underlying convolution.
    result = SpatialComputeEpochs(plan.identity,
        replace(issue, source_identity=plan.identity),
        replace(epoch, source_identity=plan.identity, max_compute_pairs=pairs), pairs, report)
    result.require(plan)
    return result


def _emit_epoch_tile(plan, epochs, builder, issue, complete, **coordinates):
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

    _emit_convolution_tile_effects(p, builder, consume, **coordinates)
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


def emit_clamped_spatial_row(builder, produce, *, first, rows, local, width, halo):
    """Partition one source row tile into clamped cells and a valid row run.

    The original tile width is at most the source width, so one tile cannot
    cross both source edges. Only the at-most-window-minus-one halo cells are
    duplicated individually; every valid original pixel stays in one row DMA.
    """
    if type(rows) is not int or not 1 <= rows <= width or not 0 <= halo <= 2:
        raise ValueError('clamped spatial partition needs a bounded original row tile')
    op = builder.op
    def fallback():
        def clipped(index):
            value = op('add',first,index)
            value = builder.select(builder.cmp('slt',value,0),0,value)
            value = builder.select(builder.cmp('ult',value,width),value,width-1)
            produce(value,1,op('add',local,index))
        builder.loop(0,rows,1,clipped)
    def side(cases, case_index=0):
        if case_index == len(cases):
            fallback()
            return
        left, count = cases[case_index]
        condition = (builder.cmp('eq',first,-count) if left else
                     builder.cmp('eq',op('add',first,rows),width+count))
        def partition():
            if left:
                for i in range(count):
                    produce(0,1,op('add',local,i))
                produce(0,rows-count,op('add',local,count))
            else:
                produce(first,rows-count,local)
                for i in range(count):
                    produce(width-1,1,op('add',local,rows-count+i))
        builder.branch(condition,partition,lambda:side(cases,case_index+1))
    valid = op('and',builder.cmp('ult',first,width),
               builder.cmp('ult',op('add',first,rows-1),width),'i1')
    cases = tuple((left,count) for left in (True,False)
                  for count in range(1,min(halo,rows-1)+1))
    builder.branch(valid,lambda:produce(first,rows,local),lambda:side(cases))


def emit_spatial_blocks(plan, builder, lower, *, synchronous_completion,
                        epochs=None, complete=None, horizontal=None):
    """Emit actual union-halo products followed by one rectangular pooled store."""
    plan.__post_init__()
    if epochs is None:
        if synchronous_completion is not True:
            raise ValueError('spatial block ownership requires full synchronous callbacks')
    elif synchronous_completion is not False or not callable(complete):
        raise ValueError('spatial epochs require explicit issue-only callbacks and real completion')
    else:
        epochs.require(plan)
    if horizontal is not None:
        horizontal.__post_init__()
        p = plan.source.producer
        if (horizontal.source != p or horizontal.row_blocks != 1 or epochs is None
                or horizontal.operand_rows > p.rhs_row):
            raise ValueError('horizontal private producer must preserve source and disjoint fallback RHS')
    identity = plan.identity
    schedule_identity = hashlib.sha256(json.dumps(dict(
        source=identity, issue=None if epochs is None else asdict(epochs.issue_law),
        epoch=None if epochs is None else asdict(epochs.epoch_law),
        compute_pairs=None if epochs is None else epochs.compute_pairs,
        horizontal=None if horizontal is None else horizontal.summary()),
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

    if horizontal is not None:
        from merlin.compile.scheduling.horizontal_convolution import (
            emit_horizontal_convolution_weights, emit_horizontal_convolution_tile,
        )
        emit_horizontal_convolution_weights(horizontal, builder, lower, complete)

    def image_body(image):
        def block_y(y, output_rows):
            local_height = (output_rows - 1) * sh + ph
            base_y = op('sub', op('mul', y, sh), top)
            def block_x(x, output_pixels):
                local_width = (output_pixels - 1) * sw + pw
                base_x = op('sub', op('mul', x, sw), left)
                def channels(column, columns):
                    def produce_y(ky):
                        iy = clip(op('add', base_y, ky), h)
                        def produce_x(local_x, rows):
                            raw_x = op('add', base_x, local_x)
                            def produce(ix, count, local):
                                global_row = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                                acc = op('add', p.accumulator_row, op('add', op('mul', ky, local_width), local))
                                coordinates = dict(output_row=global_row, output_column=column,
                                    rows=count, columns=columns, accumulator_row=acc)
                                def original():
                                    if epochs is None:
                                        emit_convolution_tile(p, builder,
                                            lambda kind, fields: lower('finish_raw_tile' if kind == 'store' else kind, fields),
                                            **coordinates, synchronous_completion=True)
                                    else:
                                        _emit_epoch_tile(plan, epochs, builder, lower, complete, **coordinates)
                                if horizontal is None:
                                    original()
                                else:
                                    win = p.window
                                    first = (win.padding[1]+win.stride[1]-1)//win.stride[1]
                                    end = min(win.output_spatial[1],
                                        (win.input_shape[2]+win.padding[1]-win.kernel[1])//win.stride[1]+1)
                                    interior = op('and',builder.cmp('ule',first,ix),
                                        builder.cmp('ult',op('add',ix,count-1),end),'i1')
                                    builder.branch(interior,
                                        lambda:emit_horizontal_convolution_tile(horizontal,builder,lower,complete,
                                            **coordinates),original)
                            emit_clamped_spatial_row(builder,produce,first=raw_x,rows=rows,
                                local=local_x,width=w,halo=pw-1)
                        split(local_width, p.row_tile, produce_x)
                    builder.loop(0, local_height, 1, produce_y)
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
                split(p.output_channels, p.column_tile, channels)
            split(ow, source.output_tile_width, block_x)
        split(oh, plan.output_tile_height, block_y)
    builder.loop(0, p.window.input_shape[0], 1, image_body)
    plan.__post_init__()
    if identity != plan.identity:
        raise ValueError('pooled convolution source or private block changed during emission')
    return dict(schema='merlin.convolution_pool_blocks.v1', plan_sha256=identity,
        schedule_sha256=schedule_identity,
        source_sha256=source.identity, output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
        readout_proofs=[prove_spatial_readout(chain) for chain in source.readouts],
        readout_mode=source.readout_mode,
        retained_readout_chains=[asdict(chain) for chain in source.readouts],
        horizontal_producer=None if horizontal is None else horizontal.summary(),
        compute_epochs=None if epochs is None else dict(epochs.report,
            source_sha256=epochs.source_identity,
            issue_law=asdict(epochs.issue_law), epoch_law=asdict(epochs.epoch_law),
            epoch_index='(ky*kw+kx)*ceil(ci/channel_tile)+channel/channel_tile+1',
            ownership='real raw-tile, pooled-store and final drains'),
        output_tile_height=plan.output_tile_height, output_tile_width=source.output_tile_width,
        private_accumulator_rows=plan.local_height * plan.local_width,
        full_intermediate_image_bytes=0, host_payload_operations=0,
        overlap_policy='compute each union-halo cell once per block; source minimum-padding edge duplication',
        completion=('fact-backed runtime K epochs; raw tile completes before pool; pool completes before reuse'
                    if epochs else 'every producer callback and pooled block store completes before ownership reuse'))
