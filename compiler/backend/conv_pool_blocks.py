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
        'private-spatial-block-complete-before-pool-and-reuse-v1')
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


def emit_spatial_blocks(plan, builder, lower, *, synchronous_completion,
                        epochs=None, complete=None):
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
    schedule_identity = hashlib.sha256(json.dumps(dict(
        source=identity, issue=None if epochs is None else asdict(epochs.issue_law),
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
                            valid = op('and', builder.cmp('ult', raw_x, w),
                                       builder.cmp('ult', op('add', raw_x, rows - 1), w), 'i1')
                            def produce(ix, count, local):
                                global_row = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                                acc = op('add', p.accumulator_row, op('add', op('mul', ky, local_width), local))
                                coordinates = dict(output_row=global_row, output_column=column,
                                    rows=count, columns=columns, accumulator_row=acc)
                                if epochs is None:
                                    emit_convolution_tile(p, builder,
                                        lambda kind, fields: lower('finish_raw_tile' if kind == 'store' else kind, fields),
                                        **coordinates, synchronous_completion=True)
                                else:
                                    _emit_epoch_tile(plan, epochs, builder, lower, complete, **coordinates)
                            builder.branch(valid, lambda: produce(raw_x, rows, local_x),
                                lambda: builder.loop(0, rows, 1, lambda i: produce(
                                    clip(op('add', raw_x, i), w), 1, op('add', local_x, i))))
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


@dataclass(frozen=True)
class QuantizedSpatialBand:
    """A contiguous source output interval and its exact cropped IFM window.

    Q1 is the original quantized convolution readout. Its rows are private and
    consumed once by the original Q2 identity-scale maxpool before reuse.
    """
    output_y: int
    output_rows: int
    producer_y: int
    producer_rows: int
    input_y: int
    input_rows: int
    convolution_padding: tuple
    pool_padding: tuple


def quantized_band_plan(window, pool, channels, *, budget=262144):
    """Prove an affine source crop, with bounded scratch and static expansion."""
    from merlin.compile.scheduling.window_transfer import Window2D
    h, w = window.input_shape[1:3]
    ch, cw = window.output_spatial
    oh, ow = pool.output_hw
    ph, pw = pool.window
    sh, sw = pool.stride
    pt, pl, pb, pr = pool.padding
    ct, cl, cb, cr = window.padding
    kh, kw = window.kernel
    dh, dw = window.dilation
    sy, sx = window.stride
    # A band holds all columns: the ordinary producer retains its full N reuse.
    capacity = min(33, budget // (cw * channels))
    rows = min(16, (capacity - ph) // sh + 1)
    if rows < 1 or (oh + rows - 1) // rows > 64:
        raise ValueError('private quantized band or static expansion exceeds bound')
    bands = []
    for y in range(0, oh, rows):
        count = min(rows, oh-y)
        first, stop = y*sh-pt, (y+count-1)*sh-pt+ph
        py, pe = max(0, first), min(ch, stop)
        if py >= pe:
            raise ValueError('fully padded band needs a separate source route')
        base = py*sy-ct
        end = (pe-1)*sy-ct+(kh-1)*dh+1
        iy, ie = max(0, base), min(h, end)
        cp = (iy-base, cl, end-ie, cr)
        pp = (py-first, pl, stop-pe, pr)
        cropped = Window2D((1, ie-iy, w, window.input_shape[-1]),
            (kh, kw), window.stride, window.dilation, cp)
        if cropped.output_spatial != (pe-py, cw):
            raise ValueError('cropped producer output geometry changed')
        from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
        consumer = SpatialMaxReadout((pe-py, cw), pool.window, pool.stride, pp, pool.pad_value)
        if consumer.output_hw != (count, ow):
            raise ValueError('cropped consumer output geometry changed')
        # These affine coefficient/constant equalities prove every original
        # KH/KW input and every pool patch, including padding, without payloads.
        if (iy-cp[0] != py*sy-ct or cp[1] != cl
                or py-pp[0] != y*sh-pt or pp[1] != pl):
            raise ValueError('source affine coordinates changed')
        bands.append(QuantizedSpatialBand(y,count,py,pe-py,iy,ie-iy,cp,pp))
    if sum(b.output_rows for b in bands) != oh:
        raise ValueError('quantized bands do not partition source outputs')
    return tuple(bands)


class _BandContext:
    def __init__(self, original, specs, offsets):
        self.original, self.specs, self.offsets = original, specs, offsets

    def __getattr__(self, key):
        return getattr(self.original, key)

    def spec(self, name):
        return self.specs[name] if name in self.specs else self.original.spec(name)

    def ptr(self, name, elem_offset=0):
        base = self.offsets.get(name, 0)
        if base:
            elem_offset = (elem_offset+base if isinstance(elem_offset, int)
                           else self.fb.add_i(elem_offset, self.fb.const(base)))
        return self.original.ptr(name, elem_offset)


def emit_quantized_bands(ctx, ifm, rhs, dst, attrs, window, pool, channels):
    """Run real source-exact Q1/Q2 owners over bounded private row intervals."""
    from .conv import lower_conv
    from .pool_stripes import lower as lower_stripes, lower_band_cache, can_lower
    for name in (ifm, rhs, dst):
        if any(key in ctx.spec(name) for key in ('strides','offset','byte_strides','physical_view')):
            return False
    aliases = getattr(ctx,'cb',{}).get('params',{}).get('tensor_aliases',{})
    if any(name in (ifm,rhs,dst) or value.get('source') in (ifm,rhs,dst)
           for name,value in aliases.items()):
        return False
    try:
        bands = quantized_band_plan(window, pool, channels)
    except ValueError:
        return False
    bcount, h, w, ci = window.input_shape
    if bcount * len(bands) > 64:
        return False
    oh, ow = pool.output_hw
    max_rows = max(b.producer_rows for b in bands)
    scratch = '_conv_pool_quantized_band_' + str(len(ctx.tensors))
    while scratch in ctx.tensors:
        scratch += '_'
    # Admission is pure, including the exact pool owner's physical bounds.
    for band in bands:
        ss = {'shape': [band.producer_rows*window.output_spatial[1], channels], 'dtype': 'i8'}
        ds = dict(ctx.spec(dst), shape=[band.output_rows*ow, channels])
        a = dict(layout='nhwc', output_dtype='i8', pool_in_dims=[band.producer_rows,window.output_spatial[1]],
                 pool_size=list(pool.window), pool_stride=list(pool.stride),
                 pool_padding=list(band.pool_padding), pool_pad_value=pool.pad_value)
        view = _BandContext(ctx, {scratch:ss,dst:ds}, {})
        if not can_lower(view, scratch, dst, a):
            return False
    ctx.alloc(scratch, [max_rows*window.output_spatial[1],channels], 'i8')
    ctx.fence()
    for image in range(bcount):
        for band in bands:
            specs = {
                ifm: dict(ctx.spec(ifm),shape=[1,band.input_rows,w,ci]),
                scratch: dict(ctx.spec(scratch),shape=[band.producer_rows*window.output_spatial[1],channels]),
                dst: dict(ctx.spec(dst),shape=[band.output_rows*ow,channels])}
            view = _BandContext(ctx,specs,{ifm:(image*h+band.input_y)*w*ci,
                dst:(image*oh+band.output_y)*ow*channels})
            producer = {key:value for key,value in attrs.items() if not key.startswith('pool_')}
            producer['epilogue'] = [stage for stage in attrs.get('epilogue',()) if stage != 'maxpool']
            producer['padding'] = list(band.convolution_padding)
            # Preserve the caller's admitted source contraction owners. Large
            # KH/KW sources often use the common flattened reduction adapter;
            # dropping to primitive tiles here would lose its retained panels.
            from pathlib import Path
            from merlin.compile.tuning_knobs import knob
            root = Path(__file__).resolve().parents[1]
            lowered = False
            if knob('CONV_CHANNEL_PHASES',default='off',values=('off','on'),root=root) == 'on':
                from .conv_channel_phases import lower as lower_phases
                lowered = bool(lower_phases(view,ifm,rhs,scratch,producer))
            if not lowered and knob('CONV_HORIZONTAL_REDUCTION',default='off',values=('off','on'),root=root) == 'on':
                from .conv_flattened_rows import can_lower as can_flatten, lower as lower_flatten
                if can_flatten(view,ifm,rhs,scratch,producer):
                    lowered = bool(lower_flatten(view,ifm,rhs,scratch,producer))
            if not lowered:
                lower_conv(view,ifm,rhs,scratch,producer)
            ctx.fence()
            consumer = dict(layout='nhwc',output_dtype='i8',
                pool_in_dims=[band.producer_rows,window.output_spatial[1]],pool_size=list(pool.window),
                pool_stride=list(pool.stride),pool_padding=list(band.pool_padding),pool_pad_value=pool.pad_value)
            if not (lower_band_cache(view,scratch,dst,consumer) or lower_stripes(view,scratch,dst,consumer)):
                raise ValueError('admitted source band consumer changed during emission')
            ctx.fence()
    receipt = dict(schema='backend.quantized_spatial_bands.v1',
        source_window=asdict(window),source_pool=asdict(pool),source_attributes=dict(attrs),
        source_operands=[ifm,rhs,dst],bands=[asdict(b) for b in bands],
        scratch_bytes=max_rows*window.output_spatial[1]*channels,
        source_q1='original common convolution binary32 half-even saturated i8 readout',
        source_q2='source-exact identity-readout i8 cached pool rectangle; common stripes fallback',
        output_elements=bcount*oh*ow*channels,host_payload_operations=0,
        produced_rows=bcount*sum(b.producer_rows for b in bands),
        completion='complete Q1 before Q2; complete Q2 before band storage reuse')
    receipt['source_sha256'] = hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    ctx.metadata.setdefault('convolution_pool_compositions',[]).append(receipt)
    return True
