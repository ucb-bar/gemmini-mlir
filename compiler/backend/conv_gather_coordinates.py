"""Loop-invariant original NHWC gather coordinates and tap transfer bands.

This is a factoring of the shared affine gather's integer address program.
It changes neither its source windows nor its physical transfer effects.
Every original output-row fragment stays within one padded source row for
every original kernel tap. Thus image/Y/X decomposition is invariant in K,
and zero/valid/zero row bands are invariant in the input-channel chunk.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs


@dataclass(frozen=True)
class SpatialFragment:
    image: object
    padded_y: object
    padded_x: object
    local: object
    rows: object


@dataclass(frozen=True)
class TapTransferBand:
    zero: bool
    first_element: object
    local_row: object
    rows: object
    row_stride_elements: int


def spatial_fragments(prepared,builder,first,count):
    """Consume the original shared gather once per original spatial M tile."""
    gather=prepared.gather
    gather.__post_init__()
    b,ph,pw,ci=gather.input_shape
    fragments=[]
    for origin,local,rows in emit_convolution_gather_runs(gather,builder,first,count):
        fragments.append(SpatialFragment(
            builder.op('udiv',origin,ph*pw*ci),
            builder.op('urem',builder.op('udiv',origin,pw*ci),ph),
            builder.op('urem',builder.op('udiv',origin,ci),pw),local,rows))
    return tuple(fragments)


def tap_bands(prepared,builder,fragments,ky,kx):
    """Factor halo and physical row geometry before the channel-chunk loop.

    Window2D and ConvolutionGatherRuns validate the complete source geometry.
    Their valid-window inequalities bound each original output fragment's
    padded X+tap by PW-1 and padded Y+tap by PH-1. Adding a valid channel
    offset therefore cannot carry into another spatial coordinate or image.
    Invalid source rows produce only one zero band; each active local row is
    written exactly once, with no zero-fill-then-overwrite sequence.
    """
    w=prepared.source.plan.window
    _,h,width,ci=w.input_shape
    pt,pl,_,_=w.padding;sw=w.stride[1];dh,dw=w.dilation
    lhs=prepared.group.lhs_row(0)
    bands=[]
    for f in fragments:
        vy=builder.op('add',f.padded_y,builder.op('mul',ky,dh))
        vx=builder.op('add',f.padded_x,builder.op('mul',kx,dw))
        valid=builder.op('and',builder.cmp('ule',pt,vy),builder.cmp('ult',vy,pt+h),'i1')
        def before(limit):
            delta=builder.select(builder.cmp('ult',vx,limit),builder.op('sub',limit,vx),0)
            n=builder.op('udiv',builder.op('add',delta,sw-1),sw)
            return builder.select(builder.cmp('ult',n,f.rows),n,f.rows)
        prefix=before(pl);end=before(pl+width)
        local=builder.op('add',lhs,f.local)
        prefix_rows=builder.select(valid,prefix,f.rows)
        valid_rows=builder.select(valid,builder.op('sub',end,prefix),0)
        suffix_rows=builder.select(valid,builder.op('sub',f.rows,end),0)
        source_y=builder.op('sub',vy,pt)
        source_x=builder.op('sub',builder.op('add',vx,builder.op('mul',prefix,sw)),pl)
        base=builder.op('add',builder.op('mul',
            builder.op('add',builder.op('mul',f.image,h),source_y),width*ci),
            builder.op('mul',source_x,ci))
        bands.extend((
            TapTransferBand(True,0,local,prefix_rows,sw*ci),
            TapTransferBand(False,base,builder.op('add',local,prefix),valid_rows,sw*ci),
            TapTransferBand(True,0,builder.op('add',local,end),suffix_rows,sw*ci)))
    return tuple(bands)


def emit_channel_bands(builder,bands,channel,depth,transfer):
    """Only original channel offset and actual DMA issue remain inside CI."""
    for band in bands:
        def issue(band=band):
            transfer(dict(zero=band.zero,
                first_element=0 if band.zero else builder.op('add',band.first_element,channel),
                local_row=band.local_row,rows=band.rows,columns=depth,
                row_stride_elements=band.row_stride_elements))
        builder.branch(builder.cmp('ult',0,band.rows),issue,lambda:None)
