"""Positive-stride row retention using narrow MVIN and unit A stride.

Each CI chunk has one contiguous local plane per horizontal source residue.
Ordinary strided host MVIN stages these planes; each tap then reads unit local
row stride. A KH-row ring retains overlapping vertical windows. Output primitive
windows never cross source output-row boundaries. No blocked wide load mapping,
nonunit execute stride, or array-weight retention is assumed.
"""
from dataclasses import dataclass
from . import isa

@dataclass(frozen=True)
class StridedRows:
    padded_width: int
    channel_chunks: int
    ring_height: int
    ring_rows: int
    plane_pitch: int
    horizontal_stride: int

    def emit(self, prepared, b, lower, complete):
        p=prepared.source.plan;w=p.window;panel=prepared.group.panel
        batch,h,width,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
        pt,pl,pb,pr=w.padding;chunks=self.channel_chunks;pw=self.padded_width
        slots=prepared.group.row_blocks;sh,sw=w.stride;pitch=self.plane_pitch
        def add(a,c):return b.op('add',a,c)
        def mul(a,c):return b.op('mul',a,c)
        def reduction(callback):
            def yy(ky):
                def xx(kx):
                    full,tail=divmod(ci,isa.DIM)
                    if full:b.loop(0,full,1,lambda c:callback(ky,kx,c,isa.DIM))
                    if tail:callback(ky,kx,full,tail)
                b.loop(0,kw,1,xx)
            b.loop(0,kh,1,yy)
        def phase(column,widths):
            active=len(widths)
            prepared.epoch.require(max(2,slots*panel.reduction_tiles*active))
            def weight(ky,kx,chunk,depth):
                tap=add(mul(ky,kw),kx);red=add(mul(tap,ci),mul(chunk,isa.DIM))
                ordinal=add(mul(tap,chunks),chunk)
                for n,cols in enumerate(widths):
                    lower('load_weight',dict(reduction_row=red,output_column=add(column,n*isa.DIM),
                        local_row=add(panel.rhs_base,mul(add(mul(ordinal,active),n),isa.DIM)),rows=depth,columns=cols))
            reduction(weight);complete('row_strided_rhs_ready',{})
            def image(image_index):
                def stage(padded_y):
                    ring_slot=b.op('urem',padded_y,kh)
                    def channel(chunk,depth):
                        # Distinct residue planes partition the original padded
                        # row. Only actual plane elements are written; padded
                        # capacity at each plane end is never consumed.
                        for residue in range(sw):
                            length=(pw-residue+sw-1)//sw
                            if length<=0:continue
                            base=mul(add(mul(add(mul(ring_slot,chunks),chunk),sw),residue),pitch)
                            prefix=max(0,min(length,(pl-residue+sw-1)//sw))
                            end=max(0,min(length,(pl+width-residue+sw-1)//sw))
                            def band(first,count,zero):
                                if not count:return
                                def piece(x,rows):
                                    source_x=add(mul(x,sw),residue-pl)
                                    host=add(mul(add(mul(image_index,h),b.op('sub',padded_y,pt)),width*ci),
                                        add(mul(source_x,ci),mul(chunk,isa.DIM)))
                                    lower('load_input',dict(zero=zero,first_element=host,local_row=add(base,x),
                                        rows=rows,columns=depth,row_stride_elements=sw*ci))
                                full,tail=divmod(count,isa.DIM)
                                if full:b.loop(0,full,1,lambda x:piece(add(first,mul(x,isa.DIM)),isa.DIM))
                                if tail:piece(first+full*isa.DIM,tail)
                            def valid():
                                band(0,prefix,True);band(prefix,end-prefix,False);band(end,length-end,True)
                            in_bounds=b.op('and',b.cmp('ule',pt,padded_y),b.cmp('ult',padded_y,pt+h),'i1')
                            b.branch(in_bounds,valid,lambda:band(0,length,True))
                    full,tail=divmod(ci,isa.DIM)
                    if full:b.loop(0,full,1,lambda c:channel(c,isa.DIM))
                    if tail:channel(full,tail)
                b.loop(0,kh,1,stage)
                def output_y(oy):
                    # Previous row's complete ownership drain precedes this
                    # SH overwritten ring rows; initial KH rows are disjoint.
                    def advance():
                        b.loop(0,sh,1,lambda offset:stage(add(add(mul(oy,sh),kh-sh),offset)))
                    b.branch(b.cmp('ult',0,oy),advance,lambda:None)
                    complete('row_strided_input_ready',{})
                    def spatial(x,rows,acc_slot):
                        first=add(mul(add(mul(image_index,oh),oy),ow),x)
                        outputs=tuple(dict(output_row=first,output_column=add(column,n*isa.DIM),rows=rows,
                            columns=cols,accumulator_row=prepared.group.accumulator_row(acc_slot,n,active_column_blocks=active))
                            for n,cols in enumerate(widths))
                        if p.bias:
                            for out in outputs:lower('initialize_bias',out)
                        def product(ky,kx,chunk,depth):
                            ring_slot=b.op('urem',add(mul(oy,sh),ky),kh)
                            residue=b.op('urem',kx,sw)
                            plane=add(mul(add(mul(ring_slot,chunks),chunk),sw),residue)
                            lhs=add(mul(plane,pitch),add(x,b.op('udiv',kx,sw)))
                            ordinal=add(mul(add(mul(ky,kw),kx),chunks),chunk)
                            initial=b.op('and',b.cmp('eq',ky,0),b.cmp('eq',kx,0),'i1')
                            fc=b.cmp('eq',chunk,0)
                            initial=(initial if fc else False) if type(fc)is bool else b.op('and',initial,fc,'i1')
                            for n,out in enumerate(outputs):
                                lower('compute',dict(out,lhs_row=lhs,rhs_row=add(panel.rhs_base,
                                    mul(add(mul(ordinal,active),n),isa.DIM)),depth=depth,
                                    first_reduction=initial,bias=p.bias,fresh_weights=True))
                        reduction(product)
                        for out in outputs:lower('store',out)
                    full,tail=divmod(ow,isa.DIM);groups,remainder=divmod(full,slots)
                    def group(g):
                        complete('row_strided_acc_group_reuse',{})
                        for slot in range(slots):spatial(add(mul(g,slots*isa.DIM),slot*isa.DIM),isa.DIM,slot)
                    if groups:b.loop(0,groups,1,group)
                    if remainder or tail:
                        complete('row_strided_acc_group_reuse',{})
                        for slot in range(remainder):spatial((groups*slots+slot)*isa.DIM,isa.DIM,slot)
                        if tail:spatial(full*isa.DIM,tail,remainder)
                    complete('row_strided_ring_last_use',{})
                b.loop(0,oh,1,output_y)
                complete('row_strided_image_last_use',{})
            b.loop(0,batch,1,image);complete('row_strided_phase_last_use',{})
        full,tail=divmod(p.output_channels,isa.DIM);phases,rem=divmod(full,panel.column_blocks)
        if phases:b.loop(0,phases,1,lambda n:phase(mul(n,panel.column_blocks*isa.DIM),(isa.DIM,)*panel.column_blocks))
        if rem or tail:phase(phases*panel.column_blocks*isa.DIM,(isa.DIM,)*rem+((tail,)if tail else()))
        return dict(mechanism='retained_strided_input_residue_planes',source_sha256=prepared.source.identity,
            ring_rows=self.ring_rows,ring_height=kh,padded_width=pw,channel_planes=chunks,
            horizontal_residues=sw,plane_pitch=pitch,source_stride=[sh,sw],rhs_base=panel.rhs_base,
            source_order=['kernel_y','kernel_x','input_channel'],a_stride=1,
            transport='narrow MVIN; host stride SW*CI; unit local rows within each horizontal residue',
            completion='input ready; two-ACC-slot reuse; full ring last use before any row overwrite',
            mapping='split output primitive tiles at every original output-row boundary',array_retention=False)

def prepare(prepared):
    w=prepared.source.plan.window
    kh,kw=w.kernel;sh,sw=w.stride
    if (w.dilation!=(1,1) or w.stride==(1,1) or prepared.pipeline is None
            or not 0<sh<=kh or not 0<sw<=kw):return None
    kh,kw=w.kernel;batch,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
    if min(pt,pl,pb,pr)<0:return None
    pw=width+pl+pr;chunks=(ci+isa.DIM-1)//isa.DIM
    pitch=(pw+sw-1)//sw;rows=kh*chunks*sw*pitch
    # Original resident RHS is immutable above rhs_base. Ring slots and all
    # channel planes are injective, private and bounded by physical row count.
    if not 0<rows<=prepared.group.panel.rhs_base:return None
    if prepared.group.row_blocks!=2:return None
    return StridedRows(pw,chunks,kh,rows,pitch,sw)
