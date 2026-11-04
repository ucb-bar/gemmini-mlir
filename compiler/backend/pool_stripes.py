"""Optional exact i8 NHWC maxpool through bounded horizontal ACC strips.

Only address/control arithmetic is emitted on the host. Each local cell is
loaded exactly once from its original i8 source or an immutable padding value;
all comparisons and output values belong to the selected store-pool path.
The caller selects this alternative; unsupported requests leave ctx untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import prod

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max

from . import isa


def _integers(values):
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise ValueError('pool geometry requires exact source integer constants')
    return values


@dataclass(frozen=True)
class PoolStripes:
    source: SpatialMaxReadout
    batches: int
    channels: int
    stripe: int

    @property
    def stripes(self):
        ow = self.source.output_hw[1]
        return tuple((x, min(self.stripe, ow - x)) for x in range(0, ow, self.stripe))

    def horizontal(self, x, outputs):
        """Disjoint halo/valid/halo intervals covering one complete local row."""
        w = self.source.input_hw[1]
        pw = self.source.window[1]
        sw = self.source.stride[1]
        base = x * sw - self.source.padding[1]
        width = (outputs - 1) * sw + pw
        left = min(width, max(0, -base))
        stop = max(left, min(width, w - base))
        return width, ((0, left, None), (left, stop, base + left), (stop, width, None))

    def projection(self):
        """Check actual interval staging against every source window coordinate.

        Only X is finite-expanded (at most 255 output positions). Y remains the
        identical affine source index y*SH-top+ky, followed by an in-bounds test.
        This proof includes explicit padding, including fully padded windows.
        It supplies no new physical DMA or completion permissions.
        """
        s = self.source
        s.__post_init__()
        h, w = s.input_hw
        ph, pw = s.window
        sh, sw = s.stride
        oh, ow = s.output_hw
        if (not 1 <= ow <= 255 or not 1 <= self.stripe <= 255
                or ph != pw or not 1 <= ph <= 3 or not 1 <= sw <= 3
                or min(self.batches, self.channels, sh) < 1):
            raise ValueError('pool stripe exceeds selected geometry fields')
        stores, rows, loads = [], 0, 0
        for x, outputs in self.stripes:
            width, intervals = self.horizontal(x, outputs)
            if not 1 <= width <= 255 or ph * width > isa.ACC_ROWS:
                raise ValueError('pool stripe exceeds private ACC capacity')
            actual = {}
            for start, stop, source_x in intervals:
                if not 0 <= start <= stop <= width:
                    raise ValueError('invalid pool stripe row partition')
                loads += ph * ((stop - start + isa.DIM - 1) // isa.DIM)
                for local in range(start, stop):
                    if local in actual:
                        raise ValueError('overlapping pool stripe staging')
                    ix = None if source_x is None else source_x + local - start
                    if ix is not None and not 0 <= ix < w:
                        raise ValueError('source stripe DMA outside its original row')
                    actual[local] = ix
            if len(actual) != width:
                raise ValueError('incomplete pool stripe staging')
            for j in range(outputs):
                stores.append(x + j)
                for kx in range(pw):
                    ix = (x + j) * sw - s.padding[1] + kx
                    if actual[j * sw + kx] != (ix if 0 <= ix < w else None):
                        raise ValueError('pool store source-window projection changed')
            rows = max(rows, ph * width)
        if stores != list(range(ow)):
            raise ValueError('pool stripes do not partition all original outputs')
        # Selected store traversal has porows=1: its only Y coordinate is ky.
        # Source Y uses SH independently; output row offset keeps original OW.
        if (oh - 1) * sh + ph + s.padding[0] >= 1 << 63:
            raise ValueError('pool vertical address arithmetic exceeds signed i64')
        result = dict(schema='backend.pool_stripes.v1', source=asdict(s),
            batches=self.batches, channels=self.channels, stripe=self.stripe,
            stripes=[list(v) for v in self.stripes], accumulator_rows=rows,
            source_partition='each patch cell loaded once from source or exact immutable pad',
            output_partition='all batch/y/x/channel outputs once; original OW row pitch',
            vertical_projection='source y*SH-top+ky; local ky; porows=1',
            horizontal_projection='source x*SW-left+kx; local j*SW+kx',
            completion='drain loads before store; drain store before ACC reuse/config changes',
            maximum_row_dma=isa.DIM, maximum_staging_calls_per_output_row=loads,
            timing='UNKNOWN')
        result['source_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result


def _plan(ctx, src, dst, attributes):
    try:
        a = dict(attributes or {})
        ss, ds = ctx.spec(src), ctx.spec(dst)
        if src == dst or ss['dtype'] != 'i8' or ds['dtype'] != 'i8':
            return None
        if str(a.get('layout', 'nhwc')).lower() != 'nhwc':
            return None
        # A standalone pool consumes already-quantized i8. Additional payload
        # stages require their own proof and cannot be silently ignored here.
        if (a.get('epilogue') or a.get('relu') or a.get('bias')
                or a.get('accumulator_init') or a.get('requant_shift') is not None
                or float(a.get('acc_scale', 1.0)) != 1.0
                or any(float(v) != 1.0 for v in (a.get('acc_scale_by_column') or ()))
                or a.get('output_dtype', 'i8') != 'i8'):
            return None
        for spec in (ss, ds):
            if any(key in spec for key in ('strides', 'offset', 'byte_strides', 'physical_view')):
                return None
            if str(spec.get('layout', 'nhwc')).lower() != 'nhwc':
                return None
        aliases = getattr(ctx, 'cb', {}).get('params', {}).get('tensor_aliases', {})
        if any(name in (src, dst) or alias.get('source') in (src, dst)
               for name, alias in aliases.items()):
            return None
        buffers = getattr(ctx, 'buffers', {})
        if dst in buffers and src in buffers and buffers[dst] is buffers[src]:
            return None
        h, w = _integers(a['pool_in_dims'])
        ph, pw = _integers(a['pool_size'])
        sh, sw = _integers(a['pool_stride'])
        padding = _integers(a.get('pool_padding', (0, 0, 0, 0)))
        pad = _integers((a['pool_pad_value'],))[0] if any(padding) else None
        s = SpatialMaxReadout((h, w), (ph, pw), (sh, sw), padding, pad)
        shape, output = _integers(ss['shape']), _integers(ds['shape'])
        if min(*shape, *output) < 1 or len(shape) not in (2, 4) or len(output) not in (2, 4):
            return None
        n = shape[-1]
        if n > 65535 or prod(shape[:-1]) % (h * w):
            return None
        batch = prod(shape[:-1]) // (h * w)
        oh, ow = s.output_hw
        if len(shape) == 4 and shape != (batch, h, w, n):
            return None
        if output not in ((batch * oh * ow, n), (batch, oh, ow, n)):
            return None
        if max(prod(shape), prod(output)) >= 1 << 63:
            return None
        # ConfigST local dimensions and output row pitch are 8-bit fields.
        max_width = min(255, isa.ACC_ROWS // ph)
        stripe = min(ow, 255, (max_width - pw) // sw + 1)
        p = PoolStripes(s, batch, n, stripe)
        proof = p.projection()
        # The shared canonical strip planner applies verbatim to unpadded full
        # strips. Its minimum-padding clamping program is deliberately not used
        # to justify this adapter's explicit arbitrary-valued halo DMAs.
        if not any(padding) and len(p.stripes) == 1:
            shared = plan_striped_spatial_max(s, batches=batch, channels=n,
                channel_block=isa.DIM, transfer_rows=isa.DIM, capacity_rows=isa.ACC_ROWS)
            proof['shared_spatial_plan'] = shared.revalidate()
        return p, proof
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def can_lower(ctx, src, dst, attrs):
    """Pure eligibility, capacity and complete source projection check."""
    return _plan(ctx, src, dst, attrs) is not None


def lower(ctx, src, dst, attrs):
    """Emit an exact selected store-pool strip, or return False before mutation."""
    planned = _plan(ctx, src, dst, attrs)
    if planned is None:
        return False
    p, receipt = planned
    s = p.source
    h, w = s.input_hw
    oh, ow = s.output_hw
    ph, _ = s.window
    sh, sw = s.stride
    top = s.padding[0]
    n = p.channels
    control = FnBuilderControlAdapter(ctx.fb)
    op = control.op
    ctx.fence()
    pad_name = None
    if any(s.padding):
        pad_name = '_pool_stripe_padding_' + str(len(ctx.tensors))
        while pad_name in ctx.tensors:
            pad_name += '_'
        ctx.constant(pad_name, [isa.DIM], 'i8', [s.pad_value] * isa.DIM)
    ctx.emit(*isa.config_ld(n, channel=0, shrink=True))
    if pad_name:
        ctx.emit(*isa.config_ld(0, channel=1, shrink=True))

    for x, outputs in p.stripes:
        local_width, intervals = p.horizontal(x, outputs)
        # Previous traversal is complete before its state or owned rows change.
        ctx.fence()
        ctx.emit(*isa.config_st(n, scale=1.0, activation=0, pool_stride=sw,
            pool_size=ph, pool_out_dim=ow, porows=1, pocols=outputs,
            orows=ph, ocols=local_width))

        def row_body(index):
            image = op('udiv', index, oh)
            y = op('urem', index, oh)
            base_y = op('sub', op('mul', y, sh), top)

            def panel(channel, columns):
                def chunks(start, stop, body):
                    full, tail = divmod(stop - start, isa.DIM)
                    if full:
                        control.loop(0, full, 1,
                            lambda i: body(op('add', start, op('mul', i, isa.DIM)), isa.DIM))
                    if tail:
                        body(start + full * isa.DIM, tail)

                for ky in range(ph):
                    iy = op('add', base_y, ky)

                    def pad_row(start, stop):
                        if start == stop:
                            return
                        if pad_name is None:
                            raise ValueError('unproved padding path reached emission')
                        chunks(start, stop, lambda local, count: ctx.emit(*isa.mvin(
                            ctx.ptr(pad_name), isa.acc_addr(op('add', ky * local_width, local), fb=ctx.fb),
                            count, columns, channel=1, fb=ctx.fb)))

                    def valid_row():
                        for start, stop, source_x in intervals:
                            if start == stop:
                                continue
                            if source_x is None:
                                pad_row(start, stop)
                            else:
                                def load(local, count, start=start, source_x=source_x):
                                    ix = op('add', source_x, op('sub', local, start))
                                    pixel = op('add', op('mul', op('add', op('mul', image, h), iy), w), ix)
                                    offset = op('add', op('mul', pixel, n), channel)
                                    ctx.emit(*isa.mvin(ctx.ptr(src, offset),
                                        isa.acc_addr(op('add', ky * local_width, local), fb=ctx.fb),
                                        count, columns, fb=ctx.fb))
                                chunks(start, stop, load)
                    if any(s.padding):
                        control.branch(control.cmp('ult', iy, h), valid_row,
                                       lambda: pad_row(0, local_width))
                    else:
                        valid_row()
                # Channel queues write disjoint local rows. No early retirement
                # assumption is used: all source/halo loads complete here.
                ctx.fence()
                pixel = op('add', op('mul', index, ow), x)
                offset = op('add', op('mul', pixel, n), channel)
                ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0), 0, columns))
                ctx.fence()

            full, tail = divmod(n, isa.DIM)
            if full:
                control.loop(0, full, 1, lambda block: panel(op('mul', block, isa.DIM), isa.DIM))
            if tail:
                panel(full * isa.DIM, tail)
        control.loop(0, p.batches * oh, 1, row_body)
    ctx.fence()
    receipt = dict(receipt, input=src, output=dst, payload_arithmetic='accelerator_store_maxpool',
                   target_encoder='backend.isa', hardware_implicit_padding=False)
    metadata = getattr(ctx, 'metadata', None)
    if metadata is not None:
        metadata.setdefault('pool_stripes', []).append(receipt)
    return receipt


def lower_band_cache(ctx, src, dst, attrs):
    """Reuse exact Q1 rows in a bounded ACC rectangle for several pool rows.

    This consumes genuine already-quantized source storage. Rectangular store
    traversal maps (y,x,ky,kx) to (y*SH+ky,x*SW+kx), preserving explicit i8 pad.
    It makes no posted-transfer or configuration-snapshot assumption.
    """
    planned = _plan(ctx, src, dst, attrs)
    if planned is None:
        return False
    p, proof = planned
    s = p.source
    h,w = s.input_hw
    oh,ow = s.output_hw
    ph,pw = s.window
    sh,sw = s.stride
    if sh != sw:
        return False
    height = (oh-1)*sh+ph
    max_width = min(255, isa.ACC_ROWS//height)
    stripe = min(ow,(max_width-pw)//sw+1)
    if height > 255 or stripe < 1:
        return False
    p = PoolStripes(s,p.batches,p.channels,stripe)
    proof = p.projection()
    opcontrol = FnBuilderControlAdapter(ctx.fb)
    op = opcontrol.op
    n = p.channels
    pad = None
    ctx.fence()
    if any(s.padding):
        pad = '_pool_band_cache_pad_'+str(len(ctx.tensors))
        ctx.constant(pad,[isa.DIM],'i8',[s.pad_value]*isa.DIM)
    ctx.emit(*isa.config_ld(n,channel=0,shrink=True))
    if pad:
        ctx.emit(*isa.config_ld(0,channel=1,shrink=True))
    for x,outputs in p.stripes:
        width,intervals = p.horizontal(x,outputs)
        if height*width > isa.ACC_ROWS:
            raise ValueError('pool row cache exceeded physical ACC rows')
        ctx.fence()
        ctx.emit(*isa.config_st(n,scale=1.,pool_stride=sw,pool_size=ph,
            pool_out_dim=ow,porows=oh,pocols=outputs,orows=height,ocols=width))
        def image_body(image):
            def panel(channel,columns):
                def row(local_y):
                    iy = op('sub',local_y,s.padding[0])
                    def chunks(start,stop,body):
                        full,tail = divmod(stop-start,isa.DIM)
                        if full:
                            opcontrol.loop(0,full,1,lambda i:body(op('add',start,op('mul',i,isa.DIM)),isa.DIM))
                        if tail:
                            body(start+full*isa.DIM,tail)
                    def padding(start,stop):
                        if start==stop:
                            return
                        if pad is None:
                            raise ValueError('pool cache reached absent source padding')
                        chunks(start,stop,lambda local,count:ctx.emit(*isa.mvin(ctx.ptr(pad),
                            isa.acc_addr(op('add',op('mul',local_y,width),local),fb=ctx.fb),
                            count,columns,channel=1,fb=ctx.fb)))
                    def valid():
                        for start,stop,source_x in intervals:
                            if source_x is None:
                                padding(start,stop)
                            else:
                                def load(local,count,start=start,source_x=source_x):
                                    ix=op('add',source_x,op('sub',local,start))
                                    pixel=op('add',op('mul',op('add',op('mul',image,h),iy),w),ix)
                                    offset=op('add',op('mul',pixel,n),channel)
                                    ctx.emit(*isa.mvin(ctx.ptr(src,offset),
                                        isa.acc_addr(op('add',op('mul',local_y,width),local),fb=ctx.fb),
                                        count,columns,fb=ctx.fb))
                                chunks(start,stop,load)
                    if any(s.padding):
                        opcontrol.branch(opcontrol.cmp('ult',iy,h),valid,lambda:padding(0,width))
                    else:
                        valid()
                opcontrol.loop(0,height,1,row)
                ctx.fence()
                offset=op('add',op('mul',op('add',op('mul',image,oh*ow),x),n),channel)
                ctx.emit(*isa.mvout(ctx.ptr(dst,offset),isa.acc_addr(0),0,columns))
                ctx.fence()
            full,tail=divmod(n,isa.DIM)
            if full:
                opcontrol.loop(0,full,1,lambda block:panel(op('mul',block,isa.DIM),isa.DIM))
            if tail:
                panel(full*isa.DIM,tail)
        opcontrol.loop(0,p.batches,1,image_body)
    ctx.fence()
    receipt=dict(proof,schema='backend.pool_band_cache.v1',input=src,output=dst,
        accumulator_rows=height*max(p.horizontal(x,count)[0] for x,count in p.stripes),
        vertical_projection='local y*SH+ky minus original top; original i8 or exact pad',
        cached_output_rows=oh,cache_completion='all Q1 DMAs complete before rectangular max; store complete before reuse',
        target_encoder='backend.isa',payload_arithmetic='accelerator_store_maxpool')
    ctx.metadata.setdefault('pool_stripes',[]).append(receipt)
    return receipt
