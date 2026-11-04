"""Retain an explicitly padded i8 plane for StoreController maxpool.

The selected output-path contract leaves implicit pool padding unknown. This
route therefore sets upad=lpad=0 and materializes declared padding by device DMA
into otherwise dead scratchpad rows. It pools the exact converted source bytes;
no second scale, clamp or activation is applied.
"""
from __future__ import annotations

from xdsl.dialects.builtin import i8
from . import isa


def eligible(h, w, ho, wo, size, sh, sw, padding):
    """Pure capacity/register checks for an ordinary unpadded plane traversal."""
    if (len(padding) != 4 or any(type(x) is not int or x < 0 for x in padding)
            or any(type(x) is not int or x <= 0 for x in (h, w, ho, wo, size, sh, sw))):
        return False
    pt, pl, pb, pr = padding
    hp, wp = h + pt + pb, w + pl + pr
    return (any(padding) and size <= 3 and sh == sw and sh <= 3
            and max(hp, wp, ho, wo) <= 255 and hp * wp <= 16384
            and ho == (hp - size) // sh + 1
            and wo == (wp - size) // sw + 1)


def try_retained_padded(ctx, source, destination, batch, h, w, ho, wo,
                        channels, size, sh, sw, padding, pad_value):
    """Emit an exact explicit-padding retained route, or None without effects.

    Each channel tile owns scratch rows [0,(H+Pt+Pb)*(W+Pl+Pr)). Fences close
    producer use before ownership, all DMA writes before hidden-footprint
    pooling, and pooling before scratch reuse by another tile or operation.
    """
    if (not eligible(h, w, ho, wo, size, sh, sw, padding)
            or type(batch) is not int or batch <= 0
            or type(channels) is not int or channels <= 0 or channels >= 1 << 32
            or type(pad_value) is not int or not -128 <= pad_value <= 127):
        return None
    from .pooling import _padding_buffer
    constant = _padding_buffer(ctx, pad_value)
    f = ctx.f
    pt, pl, pb, pr = padding
    hp, wp = h + pt + pb, w + pl + pr
    plane, out_plane = h * w, ho * wo
    # Same known unpadded traversal configuration as pooling._retained_plane.
    # All input padding is real scratchpad data, never implicit hardware padding.
    rs1 = (2 | (sh << 4) | (size << 6) | (wo << 24) |
           (ho << 32) | (wo << 40) | (hp << 48) | (wp << 56))
    isa.fence(f)
    isa.emit(f, 0, rs1, (isa.f32_bits(1.0) << 32) | channels)

    def one_batch(bn):
        def channel_tile(channel, count):
            isa.config_ld(f, 0, state=0)
            def padding_interval(start, length):
                if not length:
                    return
                def load(offset, rows):
                    isa.mvin(f, constant, f.add_i(start, offset), count, rows)
                full = length // 16 * 16
                if full:
                    f.for_loop(0, full, 16, lambda offset: load(offset, 16), retain_loop=True)
                if full < length:
                    load(f.const(full), length - full)
            padding_interval(f.const(0), pt * wp)
            padding_interval(f.const((pt + h) * wp), pb * wp)
            if pl or pr:
                def border(y):
                    start = f.mul_i(f.add_i(f.const(pt), y), f.const(wp))
                    padding_interval(start, pl)
                    padding_interval(f.add_i(start, f.const(pl + w)), pr)
                f.for_loop(0, h, 1, border, retain_loop=True)
            # Border writes and real data occupy disjoint local rows. State0's
            # stride-zero padding reloads the immutable vector for every row;
            # the following source DMA uses the true inter-pixel byte stride.
            isa.config_ld(f, channels, state=0)
            def image_row(y):
                def load(x, rows):
                    spatial = f.add_i(f.mul_i(y, f.const(w)), x)
                    index = f.add_i(f.mul_i(f.add_i(f.mul_i(bn, f.const(plane)), spatial),
                                            f.const(channels)), channel)
                    local = f.add_i(f.mul_i(f.add_i(y, f.const(pt)), f.const(wp)),
                                    f.add_i(x, f.const(pl)))
                    isa.mvin(f, f.gep(source, index, i8), local, count, rows)
                full = w // 16 * 16
                if full:
                    f.for_loop(0, full, 16, lambda x: load(x, 16), retain_loop=True)
                if full < w:
                    load(f.const(full), w - full)
            f.for_loop(0, h, 1, image_row, retain_loop=True)
            out_index = f.add_i(f.mul_i(bn, f.const(out_plane * channels)), channel)
            isa.fence(f)
            isa.mvout(f, f.gep(destination, out_index, i8), 0, count, 0)
            isa.fence(f)
        if channels <= 16:
            channel_tile(f.const(0), channels)
        else:
            f.for_loop(0, channels, 16, lambda channel: channel_tile(channel,
                f.smin(f.const(16), f.sub_i(f.const(channels), channel))), retain_loop=True)
    if batch == 1:
        one_batch(f.const(0))
    else:
        f.for_loop(0, batch, 1, one_batch, retain_loop=True)
    return batch * out_plane, channels
