"""Private primitive-row windows shared across original horizontal taps.

Window2D proves the two endpoint rectangles in the ORIGINAL NHWC source.
For stride=dilation=1 their union is contiguous. The encoder stages that union
once per ky/channel block, with primitive transfers only, then reads its kx
subwindows in unchanged ky/kx/channel reduction order. This is not a packed
wide transfer or a host im2col buffer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import isa


@dataclass(frozen=True)
class RetainedStrip:
    rows: int
    window_rows: int
    channel_blocks: int
    input_rows: int
    rhs_base: int
    source_sha256: str
    ordering_sha256: str


def prepare(prepared):
    """Read-only capacity/source/selected-address-law admission."""
    p=prepared.source.plan;w=p.window;panel=prepared.group.panel
    ci=w.input_shape[-1]
    if (w.kernel!=(3,3) or w.stride!=(1,1) or w.dilation!=(1,1)
            or ci<isa.DIM or w.output_spatial[1]<isa.DIM):return None
    blocks=(ci+isa.DIM-1)//isa.DIM
    rows=isa.DIM+w.kernel[1]-1
    # All channel blocks must coexist to preserve ky/kx/channel, rather than
    # changing the original reduction to ky/channel/kx. RHS remains untouched.
    if rows*blocks>panel.rhs_base:return None
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=hashlib.sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'):
        return None
    # Selected row-address operand and a_stride-scaled extent, not a new
    # tile-alignment or wide-column map. Unit a_stride/non-transposed mode is
    # emitted by the existing terminal setup. Each address lies inside the
    # ordinary scratchpad row space and each packet has <=DIM rows/columns.
    access=ordering['accesses']['COMPUTE_PRELOADED.a_spad']
    extent=ordering['extents']['COMPUTE_PRELOADED.a_spad']
    if (access['operand']!='rs1' or access['offset']!=0 or access['width']!=32
            or extent['status']!='derived' or extent['space']!='scratchpad'
            or extent['rows']['offset']!=48 or extent['scale']['offset']!=16
            or extent['blocks'] is not None or isa.DIM!=16):return None
    return RetainedStrip(isa.DIM,rows,blocks,rows*blocks,panel.rhs_base,
        prepared.source.identity,digest)


def emit_products(plan,prepared,builder,first,rows,outputs,locations,lower,canonical):
    """Emit actual effects, with the unchanged original gather as fallback."""
    if rows!=plan.rows:return canonical()
    p=prepared.source.plan;w=p.window
    _,_,width,ci=w.input_shape;kh,kw=w.kernel
    # Same-row and in-bounds endpoints imply every original union cell exists.
    # Both extents stay at DIM: no fictitious output extent M+KW-1 is needed.
    start=w.emit_row_run(builder,first,rows,0,0,0,isa.DIM)
    end=w.emit_row_run(builder,first,rows,kh-1,kw-1,0,isa.DIM)
    interior=builder.op('and',start['predicate'],end['predicate'],'i1')
    def retained():
        def channels(callback):
            full,tail=divmod(ci,isa.DIM)
            if full:builder.loop(0,full,1,lambda c:callback(c,isa.DIM))
            if tail:callback(full,tail)
        def y(ky):
            origin=builder.op('add',start['first_element'],builder.op('mul',ky,width*ci))
            def stage(chunk,depth):
                base=builder.op('mul',chunk,plan.window_rows)
                source=builder.op('add',origin,builder.op('mul',chunk,isa.DIM))
                lower('load_input',dict(zero=False,first_element=source,local_row=base,
                    rows=rows,columns=depth,row_stride_elements=ci))
                lower('load_input',dict(zero=False,
                    first_element=builder.op('add',source,rows*ci),
                    local_row=builder.op('add',base,rows),rows=kw-1,columns=depth,
                    row_stride_elements=ci))
            channels(stage)
            def x(kx):
                def product(chunk,depth):
                    slot,_=locations(ky,kx,chunk)
                    initial=builder.op('and',builder.cmp('eq',ky,0),builder.cmp('eq',kx,0),'i1')
                    initial=builder.op('and',initial,builder.cmp('eq',chunk,0),'i1')
                    active=len(outputs);panel=prepared.group.panel
                    for n,out in enumerate(outputs):
                        lower('compute',dict(out,
                            lhs_row=builder.op('add',builder.op('mul',chunk,plan.window_rows),kx),
                            rhs_row=builder.op('add',panel.rhs_base,builder.op('mul',
                                builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                            depth=depth,first_reduction=initial,bias=p.bias,fresh_weights=True))
                channels(product)
            builder.loop(0,kw,1,x)
        builder.loop(0,kh,1,y)
    return builder.branch(interior,retained,canonical)
