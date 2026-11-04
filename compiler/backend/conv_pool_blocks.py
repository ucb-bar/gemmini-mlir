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


def _emit_epoch_tile(plan, epochs, builder, issue, complete, *, resident_panel=None, **coordinates):
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
        if resident_panel is not None and kind == 'compute':
            tap=builder.op('udiv',pending,ci)
            chunk=builder.op('udiv',builder.op('urem',pending,ci),p.channel_tile)
            slot=builder.op('add',builder.op('mul',tap,chunks),chunk)
            fields=dict(fields,rhs_row=builder.op('add',resident_panel.rhs_base,
                builder.op('mul',slot,p.primitive)))
        if resident_panel is None or kind != 'load_weight':
            issue('finish_raw_tile' if kind == 'store' else kind, fields)
        if kind == 'compute':
            if pending is None:
                raise ValueError('spatial epoch product lacks original reduction coordinates')
            if epochs.compute_pairs > p.window.kernel[0]*p.window.kernel[1]*chunks:
                pending = None
                return
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
                        epochs=None, complete=None, resident_panel=None, horizontal_k=False):
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
        horizontal_k=bool(horizontal_k), source=identity, issue=None if epochs is None else asdict(epochs.issue_law),
        epoch=None if epochs is None else asdict(epochs.epoch_law),
        compute_pairs=None if epochs is None else epochs.compute_pairs,
        resident_rhs=None if resident_panel is None else asdict(resident_panel)),
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

    if resident_panel is not None:
        from merlin.compile.scheduling.affine_gather_runs import ConvolutionResidentPanel
        if (type(resident_panel) is not ConvolutionResidentPanel or epochs is None
                or resident_panel.column_blocks != 1 or resident_panel.tile != p.primitive
                or resident_panel.channel_tiles != (p.window.input_shape[-1]+p.channel_tile-1)//p.channel_tile
                or resident_panel.reduction_tiles != p.window.kernel[0]*p.window.kernel[1]*resident_panel.channel_tiles
                or resident_panel.rhs_base < p.lhs_row+p.row_tile
                or resident_panel.operand_capacity_rows != p.operand_rows):
            raise ValueError('pool RHS owner must preserve the exact complete-K source and private LHS rows')
        resident_panel.__post_init__()

    horizontal_plans = {}
    if horizontal_k:
        for columns in {p.column_tile, p.output_channels % p.column_tile} - {0}:
            panel = _horizontal_pool_tile(p, columns, resident_panel)
            if panel is not None: horizontal_plans[columns] = panel

    def block(image,y,output_rows,x,output_pixels,column,columns):
        local_height = (output_rows - 1) * sh + ph
        local_width = (output_pixels - 1) * sw + pw
        base_y = op('sub', op('mul', y, sh), top)
        base_x = op('sub', op('mul', x, sw), left)
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
                        def canonical():
                            _emit_epoch_tile(plan, epochs, builder, lower, complete,
                                resident_panel=resident_panel, **coordinates)
                        panels = horizontal_plans.get(columns) if count == p.primitive else None
                        if panels is None:
                            canonical()
                        else:
                            _,_,input_width,_ = p.window.input_shape
                            input_x = op('sub', op('mul', ix, p.window.stride[1]), p.window.padding[1])
                            end_x = op('add', input_x, (count-1)*p.window.stride[1]+p.window.kernel[1]-1)
                            horizontal_valid = op('and', builder.cmp('ult', input_x, input_width),
                                builder.cmp('ult', end_x, input_width), 'i1')
                            builder.branch(horizontal_valid,
                                lambda: _emit_horizontal_pool_tile(panels,p,builder,lower,complete,
                                    output_row=global_row,output_column=column,accumulator_row=acc),
                                canonical)
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

    def prepare_rhs(column,columns):
        ci=p.window.input_shape[-1];kh,kw=p.window.kernel
        chunks=resident_panel.channel_tiles
        def kernel_y(ky):
            def kernel_x(kx):
                tap=op('add',op('mul',ky,kw),kx)
                def channel(c,depth):
                    reduction=op('add',op('mul',tap,ci),op('mul',c,p.channel_tile))
                    slot=op('add',op('mul',tap,chunks),c)
                    lower('load_weight',dict(reduction_row=reduction,output_column=column,
                        local_row=op('add',resident_panel.rhs_base,op('mul',slot,p.primitive)),
                        rows=depth,columns=columns))
                full,tail=divmod(ci,p.channel_tile)
                if full:builder.loop(0,full,1,lambda c:channel(c,p.channel_tile))
                if tail:channel(full,tail)
            builder.loop(0,kw,1,kernel_x)
        builder.loop(0,kh,1,kernel_y)
        complete('resident_pool_rhs_ready',dict(output_column=column,columns=columns))
        if columns in horizontal_plans:
            _emit_horizontal_pool_tile(horizontal_plans[columns],p,builder,lower,complete,
                output_row=0,output_column=column,accumulator_row=0,prepare=True)

    if resident_panel is None:
        def image_body(image):
            def block_y(y,rows):
                def block_x(x,outputs):
                    split(p.output_channels,p.column_tile,
                        lambda col,cols:block(image,y,rows,x,outputs,col,cols))
                split(ow,source.output_tile_width,block_x)
            split(oh,plan.output_tile_height,block_y)
        builder.loop(0,p.window.input_shape[0],1,image_body)
    else:
        def channels(column,columns):
            prepare_rhs(column,columns)
            def image_body(image):
                def block_y(y,rows):
                    split(ow,source.output_tile_width,
                        lambda x,outputs:block(image,y,rows,x,outputs,column,columns))
                split(oh,plan.output_tile_height,block_y)
            builder.loop(0,p.window.input_shape[0],1,image_body)
            complete('resident_pool_rhs_last_use',dict(output_column=column,columns=columns))
        split(p.output_channels,p.column_tile,channels)
    plan.__post_init__()
    if identity != plan.identity:
        raise ValueError('pooled convolution source or private block changed during emission')
    return dict(schema='merlin.convolution_pool_blocks.v1', plan_sha256=identity,
        schedule_sha256=schedule_identity,
        source_sha256=source.identity, output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
        readout_proofs=[prove_spatial_readout(chain) for chain in source.readouts],
        readout_mode=source.readout_mode,
        rhs_residency=None if resident_panel is None else asdict(resident_panel),
        horizontal_k_panels={str(k):v.summary() for k,v in horizontal_plans.items()},
        traversal='N phase, invocation images, pooled Y/X blocks' if resident_panel else 'images, pooled Y/X blocks, N',
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


def _horizontal_pool_tile(p, columns, resident_panel):
    """Plan one full primitive output-row tile in an original-coordinate view.

    The view describes only source axes: its origin is bound by each actual
    producer tile. Full horizontal halos must be valid; vertical halo rows are
    mapped to the device zero source. Short/border tiles retain canonical K.
    """
    from merlin.compile.scheduling.horizontal_convolution import plan_horizontal_convolution_panels
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
    w=p.window;_,_,_,ci=w.input_shape;kh,kw=w.kernel;dh,dw=w.dilation
    if (resident_panel is None or ci>=p.primitive or dw!=1
            or w.output_spatial[1]<p.primitive):
        return None
    view=Window2D((1,(kh-1)*dh+1,(p.primitive-1)*w.stride[1]+kw,ci),
                  w.kernel,w.stride,w.dilation)
    source=CountedConvolutionPlan(view,columns,p.primitive,p.primitive,columns,
        min(ci,p.primitive),p.operand_rows,p.accumulator_rows,0,p.primitive,0,bias=p.bias)
    panels=plan_horizontal_convolution_panels(source,max_row_blocks=1,
        maximum_transfer_columns=4*p.primitive,private_rows_owned=True)
    if panels is None or panels.operand_rows>resident_panel.rhs_base:
        return None
    return panels


def _emit_horizontal_pool_tile(panels, p, builder, lower, complete, *,
                               output_row, output_column, accumulator_row,
                               prepare=False):
    """Consume the shared horizontal-K schedule under private pool ownership."""
    from merlin.compile.scheduling.horizontal_convolution import emit_horizontal_convolution_panels
    op=builder.op;w=p.window;_,height,width,ci=w.input_shape
    oh,ow=w.output_spatial;sh,sw=w.stride;pt,pl,_,_=w.padding
    image=op('udiv',output_row,oh*ow)
    oy=op('urem',op('udiv',output_row,ow),oh)
    ox=op('urem',output_row,ow)
    iy=op('sub',op('mul',oy,sh),pt)
    ix=op('sub',op('mul',ox,sw),pl)
    view_width=panels.source.window.input_shape[2]
    def consume(kind,f):
        if kind=='load_weight':
            if prepare:lower(kind,dict(f,output_column=output_column))
            return
        if prepare:return
        if kind=='load_horizontal_input':
            # The shared view has one output row, so each gathered band has a
            # single original input Y and a contiguous KW*CI byte span.
            vy=op('udiv',f['first_element'],view_width*ci)
            vx=op('urem',f['first_element'],view_width*ci)
            y=op('add',iy,vy)
            valid=builder.cmp('ult',y,height)
            def stage(zero):
                first=0 if zero else op('add',op('mul',op('add',op('mul',image,height),y),width*ci),op('add',op('mul',ix,ci),vx))
                for chunk in range(panels.horizontal_chunks):
                    offset=chunk*p.primitive
                    lower('load_input',dict(zero=zero,
                        first_element=0 if zero else op('add',first,offset),
                        row_stride_elements=sw*ci,
                        local_row=op('add',f['local_row'],chunk*f['block_stride_rows']),
                        rows=f['rows'],columns=min(p.primitive,panels.horizontal_columns-offset)))
            builder.branch(valid,lambda:stage(False),lambda:stage(True))
        else:
            mapped=dict(f,output_row=output_row,output_column=output_column,
                        accumulator_row=op('add',accumulator_row,f['accumulator_row']))
            lower('finish_raw_tile'if kind=='store'else kind,mapped)
    def drain(kind,f):
        if prepare:
            if kind=='horizontal_rhs_ready':complete(kind,f)
        elif kind!='horizontal_rhs_ready':complete(kind,f)
    return emit_horizontal_convolution_panels(panels,builder,consume,drain)
