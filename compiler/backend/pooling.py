"""Device maxpool over flattened NHWC i8 spatial rows.

The terminal pool command consumes one bounded accumulator patch per output
pixel. Counted host loops generate addresses only; every value is loaded by DMA
and compared by the accelerator's enabled store-pooling path. Explicit halo
loads preserve the source pad value rather than assuming implicit hardware
padding. No full image must fit accumulator storage.

The adapter accepts already requantized i8 values. For positive finite scales,
round-half-even, signed saturation, and ReLU are monotone: for any nonempty
finite window W and monotone f, max(f(W)) = f(max(W)), because an element that
attains max(W) attains the right side and bounds all other mapped elements.
This also covers zero scale. Each source padding value must be interpreted in
the same i8 domain as the supplied intermediate; the caller owns any conversion
of an earlier-domain pad. Per-column scales commute independently by channel.
"""
from __future__ import annotations

from math import isfinite, prod

from merlin.compile.logical_abi import pool_out_dims
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter

from . import isa


def lower_pool(ctx, src, dst, attributes):
    """Emit accelerator maxpool from ``src[batch*h*w,n]`` to ``dst``.

    Required geometry: pool_in_dims=[h,w], pool_size=[ph,pw],
    pool_stride=[sh,sw]. Optional pool_padding=[top,left,bottom,right];
    pool_pad_value is required when any padding is nonzero. The selected store
    protocol admits square windows of size 1..3. Other source strides are host
    address increments and are independent of the unit-stride local patch.
    """
    attrs = attributes or {}
    # The producer has already applied its numerical epilogue to this i8 seam.
    # Pass only the standalone spatial contract to the alternate store path.
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    stripes = knob('POOL_STRIPES', default='off', values=('off', 'on'),
                   root=Path(__file__).resolve().parent.parent)
    row_cache = knob('POOL_ROW_CACHE', default='off', values=('off', 'on', 'auto'),
                     root=Path(__file__).resolve().parent.parent)
    spatial = {key: value for key, value in attrs.items() if key.startswith('pool_')}
    spatial.update(layout=attrs.get('layout', 'nhwc'), output_dtype='i8')
    if row_cache in ('on', 'auto'):
        from .pool_row_cache import lower as lower_row_cache
        if lower_row_cache(ctx, src, dst, spatial, require_full_rows=row_cache == 'auto'):
            return
    if stripes == 'on':
        from .pool_stripes import lower as lower_stripes
        if lower_stripes(ctx, src, dst, spatial):
            return
    h, w = map(int, attrs['pool_in_dims'])
    ph, pw = map(int, attrs['pool_size'])
    sh, sw = map(int, attrs['pool_stride'])
    padding = tuple(map(int, attrs.get('pool_padding', [0, 0, 0, 0])))
    if len(padding) != 4 or min(h, w, ph, pw, sh, sw) < 1 or min(padding) < 0:
        raise ValueError('invalid spatial pooling geometry')
    if ph != pw or ph > 3:
        raise ValueError('selected pooling register requires a square window of size 1..3')
    if ph * pw > isa.ACC_ROWS:
        raise ValueError('pool window exceeds selected accumulator capacity')
    if any(padding) and 'pool_pad_value' not in attrs:
        raise ValueError('source must declare its spatial padding value')
    pad_value = int(attrs.get('pool_pad_value', 0))
    if not -128 <= pad_value <= 127:
        raise ValueError('pool pad must be represented in the intermediate i8 domain')
    scales = attrs.get('acc_scale_by_column') or [attrs.get('acc_scale', 1.0)]
    if any(not isfinite(float(s)) or float(s) < 0 for s in scales):
        raise ValueError('narrow-before-pool requires finite nonnegative channel scales')
    ss, ds = ctx.spec(src), ctx.spec(dst)
    if ss['dtype'] != 'i8' or ds['dtype'] != 'i8':
        raise ValueError('device pool adapter requires i8 input and output')
    shape = tuple(map(int, ss['shape']))
    if len(shape) < 2:
        raise ValueError('pool intermediate must have spatial rows and a channel axis')
    n = shape[-1]
    rows = prod(shape[:-1])
    if rows % (h * w):
        raise ValueError('pool geometry does not divide the intermediate rows')
    batch = rows // (h * w)
    oh, ow = pool_out_dims(h, w, (ph, pw), (sh, sw), padding)
    if min(oh, ow) < 1 or prod(ds['shape']) != batch * oh * ow * n:
        raise ValueError('pool output extent disagrees with declared geometry')
    top, left, _, _ = padding
    fb = ctx.fb
    control = FnBuilderControlAdapter(fb)
    ctx.fence()

    pad_name = None
    if any(padding) and pad_value:
        # Immutable literal source values, used only as DMA operands. The host
        # never reads, compares, scales or computes a tensor payload.
        pad_name = '_pool_padding_' + str(len(ctx.tensors))
        ctx.constant(pad_name, [isa.DIM], 'i8', [pad_value] * isa.DIM)

    # The local patch is ph x pw, with one output pixel. No implicit halo is
    # used: all patch rows are initialized exactly once from a source or pad.
    ctx.emit(*isa.config_ld(n, shrink=True))
    ctx.emit(*isa.config_st(n, scale=1.0, activation=0,
                           pool_stride=1, pool_size=ph, pool_out_dim=1,
                           porows=1, pocols=1, orows=ph, ocols=pw))

    def channel_panel(channel, width):
        def output_pixel(index):
            image = control.op('udiv', index, oh * ow)
            spatial = control.op('urem', index, oh * ow)
            oy = control.op('udiv', spatial, ow)
            ox = control.op('urem', spatial, ow)
            iy0 = control.op('sub', control.op('mul', oy, sh), top)
            ix0 = control.op('sub', control.op('mul', ox, sw), left)
            image_base = control.op('mul', image, h * w)
            for ky in range(ph):
                iy = control.op('add', iy0, ky)
                for kx in range(pw):
                    ix = control.op('add', ix0, kx)
                    local = isa.acc_addr(ky * pw + kx)

                    def valid_load(iy=iy, ix=ix, local=local):
                        pixel = control.op('add', image_base,
                                           control.op('add', control.op('mul', iy, w), ix))
                        offset = control.op('add', control.op('mul', pixel, n), channel)
                        ctx.emit(*isa.mvin(ctx.ptr(src, offset), local, 1, width))

                    def halo_load(local=local):
                        pointer = 0 if pad_name is None else ctx.ptr(pad_name)
                        ctx.emit(*isa.mvin(pointer, local, 1, width))

                    if any(padding):
                        # Unsigned comparison also rejects negative halo indices.
                        inside = control.op('and', control.cmp('ult', iy, h),
                                            control.cmp('ult', ix, w), dtype='i1')
                        control.branch(inside, valid_load, halo_load)
                    else:
                        valid_load()
            ctx.fence()
            offset = control.op('add', control.op('mul', index, n), channel)
            # rows=0 selects the configured store-pool traversal. The controller
            # scans the local ph*pw rows and commits one output pixel.
            ctx.emit(*isa.mvout(ctx.ptr(dst, offset), isa.acc_addr(0), 0, width))
            ctx.fence()

        control.loop(0, batch * oh * ow, 1, output_pixel)

    full = n // isa.DIM
    if full:
        control.loop(0, full, 1,
                     lambda block: channel_panel(control.op('mul', block, isa.DIM), isa.DIM))
    if n % isa.DIM:
        channel_panel(full * isa.DIM, n % isa.DIM)
    ctx.fence()
    metadata = getattr(ctx, 'metadata', None)
    if metadata is not None:
        metadata.setdefault('device_pool', []).append({
            'input': src, 'output': dst, 'batch': batch,
            'input_spatial': [h, w], 'output_spatial': [oh, ow],
            'window': [ph, pw], 'accumulator_rows': ph * pw,
            'payload_arithmetic': 'accelerator_store_maxpool',
            'loops': 'counted_host_address_control',
        })
