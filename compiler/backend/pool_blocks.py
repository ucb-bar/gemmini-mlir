"""Exact i8 NHWC maxpool through bounded two-dimensional ACC rectangles.

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

from . import isa


def _integers(values):
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise ValueError('pool geometry requires exact source integer constants')
    return values


@dataclass(frozen=True)
class PoolBlocks:
    source: SpatialMaxReadout
    batches: int
    channels: int
    stripe: int
    tile_height: int = 1
    schedule: str = "baseline"

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
                or min(self.batches, self.channels, sh, self.tile_height) < 1
                or self.tile_height > min(255, oh)
                or (self.tile_height - 1) * sh + ph > 255
                or (self.tile_height > 1 and sh != sw)):
            raise ValueError('pool stripe exceeds selected geometry fields')
        stores, rows, loads = [], 0, 0
        for x, outputs in self.stripes:
            width, intervals = self.horizontal(x, outputs)
            if not 1 <= width <= 255 or ((self.tile_height - 1) * sh + ph) * width > isa.ACC_ROWS:
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
            rows = max(rows, ((self.tile_height - 1) * sh + ph) * width)
        if stores != list(range(ow)):
            raise ValueError('pool stripes do not partition all original outputs')
        # Each local window is the original affine source window. Equal X/Y
        # strides are needed because the selected pool traversal has one stride
        # field. These finite identities include the last partial rectangle.
        for output_rows in {self.tile_height, (oh - 1) % self.tile_height + 1}:
            height = (output_rows - 1) * sh + ph
            for j in range(output_rows):
                for ky in range(ph):
                    if not 0 <= j * sh + ky < height:
                        raise ValueError('incomplete pool rectangle vertical staging')
        # Source Y uses SH independently; output row offset keeps original OW.
        if (oh - 1) * sh + ph + s.padding[0] >= 1 << 63:
            raise ValueError('pool vertical address arithmetic exceeds signed i64')
        result = dict(schema='backend.pool_blocks.v1',
            selection=self.schedule, source=asdict(s),
            batches=self.batches, channels=self.channels, stripe=self.stripe,
            stripes=[list(v) for v in self.stripes], accumulator_rows=rows,
            source_partition='each patch cell loaded once from source or exact immutable pad',
            output_partition='all batch/y/x/channel outputs once; original OW row pitch',
            vertical_projection='source y*SH-top+ky; local output_j*SH+ky',
            output_tile_height=self.tile_height,
            horizontal_projection='source x*SW-left+kx; local j*SW+kx',
            completion='drain loads before store; drain store before ACC reuse/config changes',
            maximum_row_dma=isa.DIM, maximum_staging_calls_per_output_row=loads,
            timing='UNKNOWN')
        result['source_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result


def _positive_progression(total, first, step):
    count = min(total, max(0, (first - 1) // step + 1))
    return count * first - step * count * (count - 1) // 2


def rectangle_metrics(plan):
    """Exact dynamic command/traffic multiplicities without image expansion."""
    s = plan.source
    h, _ = s.input_hw
    oh, _ = s.output_hw
    ph, _ = s.window
    sh, _ = s.stride
    top = s.padding[0]
    full, tail = divmod(oh, plan.tile_height)
    height = (plan.tile_height - 1) * sh + ph
    step = plan.tile_height * sh
    def clipped(first, extent):
        return (_positive_progression(full, first, step)
                - _positive_progression(full, first - extent, step))
    valid = clipped(h + top, height) - clipped(top, height)
    staged = full * height
    if tail:
        last_height = (tail - 1) * sh + ph
        base = full * step - top
        valid += max(0, min(last_height, h - base) - max(0, -base))
        staged += last_height
    panels = (plan.channels + isa.DIM - 1) // isa.DIM
    loads = cells = blocks = configs = 0
    for x, outputs in plan.stripes:
        width, intervals = plan.horizontal(x, outputs)
        row_calls = sum((stop - start + isa.DIM - 1) // isa.DIM
                        for start, stop, _ in intervals)
        loads += valid * row_calls + (staged - valid) * ((width + isa.DIM - 1) // isa.DIM)
        cells += staged * width
        blocks += full + bool(tail)
        configs += ((full + bool(tail)) if plan.schedule == 'baseline'
                    else bool(full) + bool(tail))
    n = plan.batches
    stores = blocks * panels * n
    return dict(loads=loads * panels * n, stores=stores,
        configurations=configs * n + 1 + bool(any(s.padding)),
        commands=(loads * panels + blocks * panels + configs) * n + 1 + bool(any(s.padding)),
        fences=2 + 2 * stores + (blocks * n if plan.schedule == 'baseline' else 0),
        staged_cells=cells * plan.channels * n,
        valid_staged_rows=valid, total_staged_rows=staged)


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
        if not 1 <= ow <= 255:
            return None
        if len(shape) == 4 and shape != (batch, h, w, n):
            return None
        if output not in ((batch * oh * ow, n), (batch, oh, ow, n)):
            return None
        if max(prod(shape), prod(output)) >= 1 << 63:
            return None
        # ConfigST local dimensions and output row pitch are 8-bit fields.
        max_width = min(255, isa.ACC_ROWS // ph)
        stripe = min(ow, 255, (max_width - pw) // sw + 1)
        tile_height = 1
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        blocks = knob('POOL_BLOCKS', default='off', values=('off', 'on'),
                      root=Path(__file__).resolve().parent.parent)
        schedule = knob('POOL_BLOCK_SCHEDULE', default='baseline',
                        values=('baseline', 'issue', 'traffic'),
                        root=Path(__file__).resolve().parent.parent)
        if blocks == 'on' and sh == sw:
            # Enumerate capacity-legal rectangles. Rank actual DMA-call and
            # readout multiplicities; this structural ranking is not timing.
            choices = []
            for tw in range(1, stripe + 1):
                width = (tw - 1) * sw + pw
                thmax = min(oh, 255, (min(255, isa.ACC_ROWS // width) - ph) // sh + 1)
                for th in range(1, thmax + 1):
                    q = PoolBlocks(s, batch, n, tw, th, schedule)
                    calls = 0
                    for x, outputs in q.stripes:
                        _, intervals = q.horizontal(x, outputs)
                        row_calls = sum((stop - start + isa.DIM - 1) // isa.DIM
                                        for start, stop, _ in intervals)
                        blocks_y = (oh + th - 1) // th
                        staged_rows = (oh - blocks_y) * sh + blocks_y * ph
                        calls += staged_rows * row_calls + blocks_y
                    metrics = rectangle_metrics(q) if schedule != 'baseline' else {}
                    objective = ((calls,) if schedule == 'baseline' else
                        (metrics['commands'], metrics['fences']) if schedule == 'issue' else
                        (metrics['staged_cells'], metrics['commands'], metrics['fences']))
                    choices.append((objective, -tw * th, -tw, tw, th))
            if choices:
                _, _, _, stripe, tile_height = min(choices)
        p = PoolBlocks(s, batch, n, stripe, tile_height, schedule)
        proof = dict(p.projection(), realization=rectangle_metrics(p))
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

        def configure(output_rows):
            local_height = (output_rows - 1) * sh + ph
            ctx.emit(*isa.config_st(n, scale=1.0, activation=0, pool_stride=sw,
                pool_size=ph, pool_out_dim=ow, porows=output_rows, pocols=outputs,
                orows=local_height, ocols=local_width))

        def height_body(y_start, output_rows, image):
            local_height = (output_rows - 1) * sh + ph
            if p.schedule == 'baseline':
                ctx.fence()
                configure(output_rows)
            base_y = op('sub', op('mul', y_start, sh), top)

            def panel(channel, columns):
                def chunks(start, stop, body):
                    full, tail = divmod(stop - start, isa.DIM)
                    if full:
                        control.loop(0, full, 1,
                            lambda i: body(op('add', start, op('mul', i, isa.DIM)), isa.DIM))
                    if tail:
                        body(start + full * isa.DIM, tail)

                def stage_row(ky):
                    iy = op('add', base_y, ky)
                    local_base = op('mul', ky, local_width)
                    def pad_row(start, stop):
                        if start == stop:
                            return
                        if pad_name is None:
                            raise ValueError('unproved padding path reached emission')
                        chunks(start, stop, lambda local, count: ctx.emit(*isa.mvin(
                            ctx.ptr(pad_name), isa.acc_addr(op('add', local_base, local), fb=ctx.fb),
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
                                        isa.acc_addr(op('add', local_base, local), fb=ctx.fb),
                                        count, columns, fb=ctx.fb))
                                chunks(start, stop, load)
                    if any(s.padding):
                        control.branch(control.cmp('ult', iy, h), valid_row,
                                       lambda: pad_row(0, local_width))
                    else:
                        valid_row()
                control.loop(0, local_height, 1, stage_row)
                ctx.fence()
                pixel = op('add', op('mul', op('add', op('mul', image, oh), y_start), ow), x)
                offset = op('add', op('mul', pixel, n), channel)
                ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0), 0, columns))
                ctx.fence()

            full, tail = divmod(n, isa.DIM)
            if full:
                control.loop(0, full, 1, lambda block: panel(op('mul', block, isa.DIM), isa.DIM))
            if tail:
                panel(full * isa.DIM, tail)

        def image_body(image):
            full, tail = divmod(oh, p.tile_height)
            if full:
                if p.schedule != 'baseline':
                    configure(p.tile_height)
                control.loop(0, full, 1, lambda i: height_body(
                    op('mul', i, p.tile_height), p.tile_height, image))
            if tail:
                if p.schedule != 'baseline':
                    configure(tail)
                height_body(full * p.tile_height, tail, image)
        control.loop(0, p.batches, 1, image_body)
    ctx.fence()
    receipt = dict(receipt, input=src, output=dst, payload_arithmetic='accelerator_store_maxpool',
                   target_encoder='backend.isa', hardware_implicit_padding=False)
    metadata = getattr(ctx, 'metadata', None)
    if metadata is not None:
        metadata.setdefault('pool_blocks', []).append(receipt)
    return receipt
