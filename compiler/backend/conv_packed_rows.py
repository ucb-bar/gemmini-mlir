"""Retained KX-trimmed source planes, with contiguous original flat M windows.

Narrow original MVIN constructs disjoint zero/valid bands in each plane. No
host payload packing, blocked transfer mapping, or nonunit A stride is used.
"""
from dataclasses import dataclass
from . import isa


def footprint(window):
    if window.stride!=(1,1) or window.dilation!=(1,1) or min(window.padding)<0:
        return None
    oh,ow=window.output_spatial;kh,kw=window.kernel
    chunks=(window.input_shape[-1]+isa.DIM-1)//isa.DIM
    return (oh+kh-1)*ow*kw*chunks


def profitable_geometry(window):
    """Select the validated low-width class; storage legality is independent."""
    return window.output_spatial[1] <= isa.DIM//2


def column_capacity(window, expansion_budget=32):
    if not profitable_geometry(window):return None
    rows=footprint(window)
    if rows is None:return None
    reserved=((max(2*isa.DIM,rows)+isa.DIM-1)//isa.DIM)*isa.DIM
    kh,kw=window.kernel;chunks=(window.input_shape[-1]+isa.DIM-1)//isa.DIM
    columns=(isa.SPAD_ROWS-reserved)//(kh*kw*chunks*isa.DIM)
    return min(expansion_budget,columns) if columns>0 else None


@dataclass(frozen=True)
class PackedRows:
    packed_rows: int

    def emit(self,prepared,b,lower,complete):
        p=prepared.source.plan;w=p.window;panel=prepared.group.panel
        batch,h,width,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
        pt,pl,pb,pr=w.padding;chunks=panel.channel_tiles;height=oh+kh-1
        slots=prepared.group.row_blocks;pixels=oh*ow
        input_once=batch==1 and p.output_channels>isa.DIM*panel.column_blocks
        def add(a,c):return b.op('add',a,c)
        def mul(a,c):return b.op('mul',a,c)
        def minimum(a,c):return b.select(b.cmp('ult',a,c),a,c)
        def reduction(callback):
            def yy(ky):
                def xx(kx):
                    full,tail=divmod(ci,isa.DIM)
                    if full:b.loop(0,full,1,lambda c:callback(ky,kx,c,isa.DIM))
                    if tail:callback(ky,kx,full,tail)
                b.loop(0,kw,1,xx)
            b.loop(0,kh,1,yy)
        def stage_input(image_index):
            def channel(chunk,depth):
                def tap_x(kx):
                    plane=mul(add(mul(chunk,kw),kx),height*ow)
                    def row(py):
                        base=add(plane,mul(py,ow))
                        # Clamp before subtraction: no unsigned underflow is
                        # interpreted as an in-bounds source coordinate.
                        left=minimum(b.select(b.cmp('ult',kx,pl),b.op('sub',pl,kx),0),ow)
                        end=minimum(b.select(b.cmp('ult',kx,pl+width),b.op('sub',pl+width,kx),0),ow)
                        def piece(x,rows):
                            prefix=minimum(b.select(b.cmp('ult',x,left),b.op('sub',left,x),0),rows)
                            finish=minimum(b.select(b.cmp('ult',x,end),b.op('sub',end,x),0),rows)
                            def band(start,count,zero):
                                def issue():
                                    px=add(x,start)
                                    host=add(mul(add(mul(image_index,h),b.op('sub',py,pt)),width*ci),
                                        add(mul(b.op('sub',add(px,kx),pl),ci),mul(chunk,isa.DIM)))
                                    lower('load_input',dict(zero=zero,first_element=host,
                                        local_row=add(base,px),rows=count,columns=depth,row_stride_elements=ci))
                                b.branch(b.cmp('ult',0,count),issue,lambda:None)
                            def valid():
                                band(0,prefix,True);band(prefix,b.op('sub',finish,prefix),False)
                                band(finish,b.op('sub',rows,finish),True)
                            in_bounds=b.op('and',b.cmp('ule',pt,py),b.cmp('ult',py,pt+h),'i1')
                            b.branch(in_bounds,valid,lambda:band(0,rows,True))
                        full,tail=divmod(ow,isa.DIM)
                        if full:b.loop(0,full,1,lambda x:piece(mul(x,isa.DIM),isa.DIM))
                        if tail:piece(full*isa.DIM,tail)
                    b.loop(0,height,1,row)
                b.loop(0,kw,1,tap_x)
            full,tail=divmod(ci,isa.DIM)
            if full:b.loop(0,full,1,lambda c:channel(c,isa.DIM))
            if tail:channel(full,tail)
            complete('packed_image_input_ready',{})
        def phase(column,widths):
            active=len(widths)
            prepared.epoch.require(max(2,slots*panel.reduction_tiles*active))
            def weight(ky,kx,chunk,depth):
                tap=add(mul(ky,kw),kx);red=add(mul(tap,ci),mul(chunk,isa.DIM))
                ordinal=add(mul(tap,chunks),chunk)
                for n,cols in enumerate(widths):
                    lower('load_weight',dict(reduction_row=red,output_column=add(column,n*isa.DIM),
                        local_row=add(panel.rhs_base,mul(add(mul(ordinal,active),n),isa.DIM)),rows=depth,columns=cols))
            reduction(weight);complete('packed_rhs_ready',{})
            def image(image_index):
                if not input_once:stage_input(image_index)
                def spatial(x,rows,acc_slot):
                    first=add(mul(image_index,pixels),x)
                    outputs=tuple(dict(output_row=first,output_column=add(column,n*isa.DIM),rows=rows,
                        columns=cols,accumulator_row=prepared.group.accumulator_row(acc_slot,n,active_column_blocks=active))
                        for n,cols in enumerate(widths))
                    if p.bias:
                        for out in outputs:lower('initialize_bias',out)
                    def product(ky,kx,chunk,depth):
                        lhs=add(mul(add(mul(chunk,kw),kx),height*ow),add(x,mul(ky,ow)))
                        ordinal=add(mul(add(mul(ky,kw),kx),chunks),chunk)
                        initial=b.op('and',b.cmp('eq',ky,0),b.cmp('eq',kx,0),'i1')
                        fc=b.cmp('eq',chunk,0)
                        initial=(initial if fc else False)if type(fc)is bool else b.op('and',initial,fc,'i1')
                        for n,out in enumerate(outputs):
                            lower('compute',dict(out,lhs_row=lhs,rhs_row=add(panel.rhs_base,
                                mul(add(mul(ordinal,active),n),isa.DIM)),depth=depth,
                                first_reduction=initial,bias=p.bias,fresh_weights=True))
                    reduction(product)
                    for out in outputs:lower('store',out)
                full,tail=divmod(pixels,isa.DIM);groups,remainder=divmod(full,slots)
                def group(g):
                    complete('packed_acc_group_reuse',{})
                    for slot in range(slots):spatial(add(mul(g,slots*isa.DIM),slot*isa.DIM),isa.DIM,slot)
                if groups:b.loop(0,groups,1,group)
                if remainder or tail:
                    complete('packed_acc_group_reuse',{})
                    for slot in range(remainder):spatial((groups*slots+slot)*isa.DIM,isa.DIM,slot)
                    if tail:spatial(full*isa.DIM,tail,remainder)
                complete('packed_image_last_use',{})
            b.loop(0,batch,1,image);complete('packed_phase_last_use',{})
        # The checked plane region is disjoint from every RHS phase. With
        # one image, no producer or consumer changes these source rows.
        # Keep the real input-ready drain before any original consumer.
        if input_once:stage_input(0)
        full,tail=divmod(p.output_channels,isa.DIM);phases,rem=divmod(full,panel.column_blocks)
        if phases:b.loop(0,phases,1,lambda n:phase(mul(n,panel.column_blocks*isa.DIM),(isa.DIM,)*panel.column_blocks))
        if rem or tail:phase(phases*panel.column_blocks*isa.DIM,(isa.DIM,)*rem+((tail,)if tail else()))
        return dict(mechanism='retained_kx_trimmed_image_planes',source_sha256=prepared.source.identity,
            packed_rows=self.packed_rows,plane_height=height,plane_width=ow,channel_planes=chunks,
            tap_x_planes=kw,rhs_base=panel.rhs_base,rhs_rows=panel.rhs_rows,column_blocks=panel.column_blocks,
            source_order=['kernel_y','kernel_x','input_channel'],a_stride=1,
            transport='narrow original MVIN; disjoint zero/valid bands; no payload packing',
            mapping='plane(chunk,kx) + flat image output row + ky*OW',
            completion='full input-ready; two-ACC-slot reuse; full last-use before image or phase overwrite',
            input_staging_scope='whole invocation' if input_once else 'image within column phase',
            array_retention=False)


def prepare(prepared):
    if not profitable_geometry(prepared.source.plan.window):return None
    rows=footprint(prepared.source.plan.window)
    if rows is None or prepared.pipeline is None or prepared.group.row_blocks!=2:return None
    if not 0<rows<=prepared.group.panel.rhs_base:return None
    return PackedRows(rows)
