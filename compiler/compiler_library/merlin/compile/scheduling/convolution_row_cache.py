"""Source-exact rolling convolution with an explicit blocked operand binding.

The checked rolling-window dialect owns which original rows are live. This
consumer maps each logical cache row to disjoint channel-block-major scratchpad
rows. Ordinary transport callbacks complete synchronously; selected typed laws
authorize compute epochs and completed, disjoint readout batches.
"""
from dataclasses import dataclass, replace
from .program_limits import AddressWindow
from .rolling_window import (propose_rolling_window, materialize_rolling_window,
    emit_rolling_window, rolling_window_text)
from .convolution_exploration import (ConvolutionExplorationSource,
    plan_convolution_schedule, plan_convolution_rhs_residency,
    _prepare_convolution_rhs)


@dataclass(frozen=True)
class ConvolutionRowCache:
    schedule: object
    transport: object
    module: object
    rhs: object
    pitch: int
    chunks: int
    cache_rows: int
    rhs_base: int
    max_columns: int
    spatial_live: int = 1
    rhs_groups: tuple = ()

    def revalidate(self):
        self.schedule.verify_current()
        p, t = self.schedule.source.plan, self.transport
        w = p.window
        _, _, width, ci = w.input_shape
        expected_pitch = width+w.padding[1]+w.padding[3]
        if (w.stride != (1,1) or w.dilation != (1,1)
            or t.window != w or t.slots != w.kernel[0]
            or t.source_identity != self.schedule.source.source_identity
            or t.source_columns != (0,width) or t.cache.start != 0
            or self.pitch != expected_pitch
            or self.chunks != (ci+p.primitive-1)//p.primitive
            or self.cache_rows != self.chunks*self.pitch*t.slots
            or self.rhs_base != p.rhs_row
            or self.cache_rows > self.rhs_base
            or self.rhs_base+p.channel_tile > p.operand_rows
            or not 1 <= self.spatial_live <= max(1,
                (p.accumulator_rows-p.accumulator_row)//(max(map(len,self.schedule.column_blocks))*p.row_tile))
            or self.max_columns % p.primitive or not p.primitive <= self.max_columns <= 127):
            raise ValueError('rolling convolution physical binding differs from checked source/storage')
        # For a consumer x..x+rows-1 and tap kx, x+rows+kx
        # never exceeds pitch. Each cblock occupies exactly pitch rows;
        # each circular slot exactly chunks*pitch. Thus every live read
        # belongs to one staged pixel (real or padding) within its own slot.
        oh, ow = w.output_spatial
        if ow+w.kernel[1]-1 > self.pitch:
            raise ValueError('rolling consumer escapes staged horizontal halo')
        if self.rhs is not None:
            self.rhs.verify_current()
            if self.rhs.schedule.identity != self.schedule.identity:
                raise ValueError('resident RHS belongs to a different rolling consumer')

        if self.rhs_groups:
            if self.rhs is not None:
                raise ValueError('full and grouped RHS lifetimes cannot overlap')
            columns = []
            for group in self.rhs_groups:
                group.verify_current()
                if group.schedule.identity != self.schedule.identity or group.assigned_columns is None:
                    raise ValueError('RHS group differs from original column source')
                columns.extend(group.assigned_columns)
            if tuple(columns) != self.schedule.source.column_intervals:
                raise ValueError('RHS groups must partition every original output column exactly once')

    def summary(self):
        self.revalidate()
        p = self.schedule.source.plan
        b, h, w, ci = p.window.input_shape
        oh, ow = p.window.output_spatial
        return dict(schema='merlin.convolution_row_cache.v1',
            schedule=self.schedule.summary(), transport_sha256=self.transport.identity,
            transport_ir=rolling_window_text(self.module), pitch=self.pitch,
            channel_blocks=self.chunks, cache_rows=self.cache_rows, spatial_live=self.spatial_live,
            rhs_base=self.rhs_base, max_transfer_columns=self.max_columns,
            resident_rhs=self.rhs is not None, rhs_groups=[g.summary() for g in self.rhs_groups],
            transport_invocations=max(1,len(self.rhs_groups)),
            staged_source_rows=b*(p.window.kernel[0]+oh-1)*max(1,len(self.rhs_groups)),
            logical_source_input_bytes=b*h*w*ci,
            completion='completed transport, selected ordered compute epochs, readout completion before reuse')


def plan_convolution_row_cache(schedule, rhs_law, *, max_transfer_columns=127, spatial_group=False):
    """Return (checked plan, None) or (None, explicit missing premise).

    Capacity includes all circular slots, channel tails padded to physical
    rows, horizontal halo and a disjoint RHS tile. No shape identity is used.
    """
    schedule.verify_current()
    p = schedule.source.plan
    w = p.window
    if w.stride != (1, 1) or w.dilation != (1, 1):
        return None, 'rolling operand binding currently proves unit stride and dilation only'
    if schedule.choice.load_issue or schedule.choice.compute_pairs != 1:
        return None, 'rolling operand binding requires synchronous effects'
    if p.channel_tile != min(p.window.input_shape[-1], p.primitive) or p.column_tile != p.primitive:
        return None, 'blocked operand binding requires full primitive channel/column tiles'
    batch, height, width, ci = w.input_shape
    pitch = width + w.padding[1] + w.padding[3]
    if not 0 < pitch < 2**16:
        return None, 'row pitch does not fit selected load block_stride'
    chunks = (ci+p.primitive-1)//p.primitive
    cache_rows = w.kernel[0]*pitch*chunks
    rhs_base = (cache_rows+p.channel_tile-1)//p.channel_tile*p.channel_tile
    if rhs_base+p.channel_tile > p.operand_rows:
        return None, 'complete rolling rows plus disjoint RHS tile exceed operand capacity'
    columns = max_transfer_columns//p.primitive*p.primitive
    if columns < p.primitive:
        return None, 'selected transfer count cannot encode one primitive'
    transport = propose_rolling_window(w, AddressWindow('logical_operand_cache', 0,
        w.kernel[0]*width*ci), source_identity=schedule.source.source_identity)
    if transport.source_columns != (0, width):
        return None, 'binding requires the checked complete source-row view'
    # This actual schedule owns a separate RHS window after the entire cache.
    source = ConvolutionExplorationSource(replace(p, rhs_row=rhs_base),
        schedule.source.column_intervals, schedule.source.source_identity)
    selected = plan_convolution_schedule(source, schedule.choice,
        retention_law=schedule.retention_law)
    rhs = plan_convolution_rhs_residency(selected, operand_rows=p.operand_rows,
        law=rhs_law, rhs_base=rhs_base)
    groups = ()
    if rhs is None:
        slots = (p.operand_rows-rhs_base)//(w.kernel[0]*w.kernel[1]*chunks*p.channel_tile)
        if slots:
            intervals = selected.source.column_intervals
            groups = tuple(plan_convolution_rhs_residency(selected, operand_rows=p.operand_rows,
                law=rhs_law, rhs_base=rhs_base, column_intervals=intervals[start:start+slots])
                for start in range(0,len(intervals),slots))
            if any(g is None for g in groups):
                raise ValueError('capacity-derived resident RHS group unexpectedly refused')
        elif plan_convolution_rhs_residency(schedule,
                operand_rows=p.operand_rows, law=rhs_law) is not None:
            return None, 'rolling cache leaves no complete single-column RHS reduction resident'
    module = materialize_rolling_window(transport)
    return ConvolutionRowCache(selected, transport, module, rhs,
        pitch, chunks, cache_rows, rhs_base, columns,
        min((w.output_spatial[1]+p.row_tile-1)//p.row_tile,
            (p.accumulator_rows-p.accumulator_row)//(max(map(len,selected.column_blocks))*p.row_tile))
        if spatial_group else 1, groups), None


def emit_convolution_row_cache(cache, builder, lower, *, complete,
                               synchronous_transport=False, epoch_law=None, coalesced_preparation=True,
                               persistent_zero_bands=True, zero_row_elision=True, readout_batch=False):
    if type(cache) is not ConvolutionRowCache or synchronous_transport is not True:
        raise ValueError('row cache needs its checked plan and synchronous transport binding')
    cache.revalidate()
    if readout_batch and epoch_law is None:
        raise ValueError('readout batching needs selected finite queue progress')
    s, b = cache.schedule, builder
    s.verify_current()
    p, t = s.source.plan, cache.transport
    _, height, width, ci = p.window.input_shape
    kh, kw = p.window.kernel
    oh, ow = p.window.output_spatial
    left, right = p.window.padding[1], p.window.padding[3]
    if epoch_law is not None:
        from .window_transfer import ConvolutionComputeEpochLaw
        if type(epoch_law) is not ConvolutionComputeEpochLaw or (cache.rhs is None and not cache.rhs_groups):
            raise ValueError('compute overlap requires resident immutable operand rows and typed ordering law')
        pairs = kh*kw*((ci+p.channel_tile-1)//p.channel_tile)*max(map(len, s.column_blocks))*cache.spatial_live
        epoch_law.require(pairs)
        if epoch_law.source_identity != s.source.source_identity:
            raise ValueError('compute epoch source differs from rolling consumer')
    slot_rows = cache.pitch*cache.chunks
    def op(n, x, y): return b.op(n, x, y)
    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            b.loop(0, whole, 1, lambda i: body(op('mul', i, tile), tile))
        if tail: body(whole*tile, tail)
    base_lower = lower
    active_blocks = s.column_blocks

    if persistent_zero_bands:
        # The halo bands are disjoint from every real-row transport and all
        # resident RHS rows. Establish them once for every circular slot;
        # no consumer writes operand storage and every invocation reinitializes.
        def initialize_slot(slot):
            def channel_band(channel, depth):
                local = op('add', op('mul', slot, slot_rows),
                    op('mul', op('udiv', channel, p.primitive), cache.pitch))
                for first, extent in ((0,left),(left+width,right)):
                    if extent:
                        split(extent,p.primitive,lambda x,rows: lower('load_input',dict(
                            first_element=0,row_stride_elements=ci,
                            local_row=op('add',local,op('add',first,x)),rows=rows,
                            columns=depth,zero=True,block_stride=cache.pitch)))
            split(ci,p.primitive,channel_band)
        b.loop(0,kh,1,initialize_slot)
        complete('persistent_zero_bands', {})

    def stage(f):
        # Logical byte cache row -> blocked physical operand slot. Source-byte
        # remains the original checked NHWC address, guarded before dereference.
        slot = op('udiv', f['cache_byte'], t.row_stride_bytes)
        base = op('mul', slot, slot_rows)
        def channels(channel, cols):
            block = op('udiv', channel, p.primitive)
            local = op('add', base, op('mul', block, cache.pitch))
            def region(xfirst, extent, zero):
                def run(x, rows):
                    pos = op('add', xfirst, x)
                    def transfer(c, count):
                        lower('load_input', dict(first_element=(0 if zero else
                            op('add', f['source_byte'], op('add', op('mul', x, ci), channel))),
                            row_stride_elements=ci, local_row=op('add', op('add', local, pos),
                                op('mul', op('udiv', c, p.primitive), cache.pitch)),
                            rows=rows, columns=count, zero=zero, block_stride=cache.pitch))
                    # A null-source transfer never needs a multi-block source
                    # packet map: use one complete primitive per zero panel.
                    if zero: split(cols, p.primitive, transfer)
                    else: transfer(0, cols)
                if extent: split(extent, p.primitive, run)
            if not persistent_zero_bands: region(0, left, True)
            b.branch(f['valid'], lambda: region(left, width, False),
                     lambda: region(left, width, True))
            if not persistent_zero_bands: region(left+width, right, True)
        split(ci, cache.max_columns, channels)

    def consume(f):
        image, y = f['image'], f['output_y']
        xtiles = (ow+p.row_tile-1)//p.row_tile
        def group_body(first_tile, tile_count):
            for block in active_blocks:
                outputs = []
                for tile_index in range(tile_count):
                    tile = op('add', first_tile, tile_index)
                    x = op('mul', tile, p.row_tile)
                    rows = (b.select(b.cmp('eq', tile, xtiles-1), ow%p.row_tile, p.row_tile)
                            if ow%p.row_tile else p.row_tile)
                    row = op('add', op('mul', op('add', op('mul', image, oh), y), ow), x)
                    for o in block:
                        outputs.append(dict(output_row=row, rows=rows,
                            output_column=o.output_column, columns=o.columns,
                            accumulator_row=o.accumulator_row+tile_index*len(block)*p.row_tile,
                            cache_x=x))
                if p.bias:
                    for out in outputs: lower('initialize_bias', out)
                def kernel_y(ky):
                    slot = op('urem', op('add', y, ky), kh)
                    def kernel_x(kx):
                        def channel_body(channel, depth):
                            reduction = op('add', op('mul', op('add', op('mul', ky, kw), kx), ci), channel)
                            first = b.cmp('eq', reduction, 0)
                            # Source output-channel intervals, then independent
                            # spatial slots; each output retains original K order.
                            for column_slot in range(len(block)):
                                for tile_slot in range(tile_count):
                                    out = outputs[tile_slot*len(block)+column_slot]
                                    lhs = op('add', op('mul', slot, slot_rows),
                                        op('add', op('mul', op('udiv', channel, p.primitive), cache.pitch),
                                            op('add', out['cache_x'], kx)))
                                    lower('load_weight', dict(reduction_row=reduction,
                                        output_column=out['output_column'], local_row=p.rhs_row,
                                        rows=depth, columns=out['columns']))
                                    lower('compute', dict(out, lhs_row=lhs, rhs_row=p.rhs_row,
                                        depth=depth, first_reduction=first, bias=p.bias,
                                        ordered_compute_epoch=epoch_law is not None))
                        split(ci, p.channel_tile, channel_body)
                    if zero_row_elision and p.bias:
                        # Original signed-i8 zero padding contributes exact
                        # i32 additive identity; original bias seeds each output.
                        valid_y = b.cmp('ult', op('sub',op('add',y,ky),p.window.padding[0]),height)
                        b.branch(valid_y,lambda: b.loop(0,kw,1,kernel_x),lambda:None)
                    else:
                        b.loop(0,kw,1,kernel_x)
                b.loop(0, kh, 1, kernel_y)
                if epoch_law is not None: complete('compute_epoch', {})
                for out in outputs: lower('store', dict(out,disjoint_readout_batch=readout_batch))
                if readout_batch: complete('readout_batch', {})
        split(xtiles, cache.spatial_live, group_body)

    def bind(kind, f):
        if kind == 'stage_row': stage(f)
        elif kind == 'consume': consume(f); return cache
        elif kind in ('await_transport', 'complete'): complete(kind, f); return cache
        elif kind == 'begin_transport': return cache
        else: raise ValueError('unknown rolling convolution effect '+kind)
    transport_evidence = []
    from .window_transfer import ConvolutionOutputTile
    for group in cache.rhs_groups or (cache.rhs,):
        lower = base_lower
        def rhs_check(): pass
        active_blocks = s.column_blocks
        if group is not None:
            lower, rhs_check = _prepare_convolution_rhs(group, b, base_lower, complete,
                max_columns=cache.max_columns if coalesced_preparation else p.primitive,
                issue_preparation=epoch_law is not None and coalesced_preparation)
            if group.assigned_columns is not None:
                intervals = group.assigned_columns
                slots = min(max(map(len,s.column_blocks)),len(intervals))
                active_blocks = tuple(tuple(ConvolutionOutputTile(c,n,p.accumulator_row+i*p.row_tile)
                    for i,(c,n) in enumerate(intervals[start:start+slots]))
                    for start in range(0,len(intervals),slots))
        transport_evidence.append(emit_rolling_window(cache.module,t,b,bind))
        rhs_check()
    from dataclasses import asdict
    return dict(cache.summary(), transport_evidence=transport_evidence,
        compute_epoch_law=None if epoch_law is None else asdict(epoch_law),
        readout_batch=readout_batch, persistent_zero_bands=persistent_zero_bands, zero_row_elision=zero_row_elision and p.bias)
