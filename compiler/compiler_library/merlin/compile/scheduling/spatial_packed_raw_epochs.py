"""Share original input preparation across jointly live raw-ACC channel halos.

This is a new raw-boundary composition, not the staged i8 readout cache. Every
channel plane owns disjoint assigned ACC rows until its original pooled stores
complete. The single caller-owned input panel is fully consumed before reuse.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from .spatial_producer import SpatialProducerPlan
from .spatial_raw_retention import SpatialRawRetention
from .spatial_packed_raw import PackedRawConvolution


@dataclass(frozen=True)
class PackedRawHaloEpochs:
    source: SpatialProducerPlan
    packed: PackedRawConvolution
    output_rows: int

    def __post_init__(self):
        if type(self.source) is not SpatialProducerPlan or type(self.packed) is not PackedRawConvolution:
            raise ValueError('shared raw packing requires the complete original source and panel')
        self.packed.__post_init__()
        if self.packed.original != self.source.producer:
            raise ValueError('shared raw packing and halo have different sources')
        SpatialRawRetention(self.source,self.output_rows).__post_init__()
        p=self.source.producer
        if p.accumulator_row + self.required_rows > p.accumulator_rows:
            raise ValueError('jointly live raw channel planes exceed assigned ACC rows')

    @property
    def planes(self):
        p=self.source.producer
        return (p.output_channels+p.column_tile-1)//p.column_tile

    @property
    def plane_rows(self):
        return ((self.output_rows-1)*2+3)*((self.source.output_tile_width-1)*2+3)

    @property
    def required_rows(self):
        return self.planes*self.plane_rows

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def propose_packed_raw_halo_epochs(source,packed):
    if type(source) is not SpatialProducerPlan or type(packed) is not PackedRawConvolution:
        raise ValueError('shared raw packing requires typed original source and panel')
    p,s=source.producer,source.spatial
    if source.readout_mode!='single_readout' or s.window!=(3,3) or s.stride!=(2,2):
        return None
    planes=(p.output_channels+p.column_tile-1)//p.column_tile
    width=(source.output_tile_width-1)*2+3
    rows=min(s.output_hw[0],((p.accumulator_rows-p.accumulator_row)//(planes*width)-3)//2+1)
    return None if rows<2 else PackedRawHaloEpochs(source,packed,rows)


def emit_packed_raw_halo_epochs(plan,builder,issue,complete,*,law,epoch_law,compute_pairs):
    if type(plan) is not PackedRawHaloEpochs:
        raise ValueError('shared raw packing requires a checked joint halo epoch')
    plan.__post_init__()
    source,panel=plan.source,plan.packed
    p,s=source.producer,source.spatial
    law.__post_init__()
    epoch_law.require(compute_pairs)
    if law.source_identity!=p.identity or epoch_law.source_identity!=p.identity:
        raise ValueError('joint raw halo requires the exact original source laws')
    before=(plan.identity,asdict(law),asdict(epoch_law))
    op=builder.op
    h,w=s.input_hw;oh,ow=s.output_hw;pt,pl,_,_=s.padding

    def split(extent,tile,body):
        whole,tail=divmod(extent,tile)
        if whole:builder.loop(0,whole,1,lambda i:body(op('mul',i,tile),tile))
        if tail:body(whole*tile,tail)

    def clip(v,extent):
        v=builder.select(builder.cmp('slt',v,0),0,v)
        return builder.select(builder.cmp('ult',v,extent),v,extent-1)

    def image(image):
        def stripe(x,count):
            width=(count-1)*2+3
            base_x=op('sub',op('mul',x,2),pl)
            def epoch(first_y,count_y):
                def output(dy):
                    y=op('add',first_y,dy)
                    base_y=op('sub',op('mul',y,2),pt)
                    def row(ky):
                        iy=clip(op('add',base_y,ky),h)
                        local_y=op('add',op('mul',dy,2),ky)
                        def chunk(local_x,rows):
                            raw_x=op('add',base_x,local_x)
                            last_x=op('add',raw_x,rows-1)
                            valid=builder.op('and',builder.cmp('ult',raw_x,w),builder.cmp('ult',last_x,w),'i1')
                            def produce(ix,nrows,local):
                                global_row=op('add',op('mul',op('add',op('mul',image,h),iy),w),ix)
                                prepared=panel.prepare(issue,output_row=global_row,rows=nrows)
                                def channel(column,columns):
                                    plane=op('mul',op('udiv',column,p.column_tile),plan.plane_rows)
                                    acc=op('add',p.accumulator_row,op('add',plane,op('add',op('mul',local_y,width),local)))
                                    panel.emit_prepared(builder,
                                        lambda kind,f:issue('finish_raw_tile' if kind=='store' else kind,f),
                                        complete,panel=prepared,output_column=column,columns=columns,
                                        accumulator_row=acc,law=law,epoch_law=epoch_law,compute_pairs=compute_pairs)
                                split(p.output_channels,p.column_tile,channel)
                            builder.branch(valid,lambda:produce(raw_x,rows,local_x),
                                lambda:builder.loop(0,rows,1,lambda i:produce(clip(op('add',raw_x,i),w),1,op('add',local_x,i))))
                        split(width,p.row_tile,chunk)
                    builder.branch(builder.cmp('eq',dy,0),lambda:builder.loop(0,3,1,row),lambda:builder.loop(1,3,1,row))
                    def store(column,columns):
                        plane=op('mul',op('udiv',column,p.column_tile),plan.plane_rows)
                        f=dict(output_row=op('add',op('mul',op('add',op('mul',image,oh),y),ow),x),
                            output_column=column,columns=columns,output_pixels=count,
                            accumulator_row=op('add',p.accumulator_row,op('add',plane,op('mul',op('mul',dy,2),width))),
                            local_height=3,local_width=width,window_height=3,window_width=3,
                            stride_width=2,readout_tile=op('udiv',column,p.column_tile))
                        issue('store_spatial_max',f);complete('store_spatial_max',f)
                    split(p.output_channels,p.column_tile,store)
                builder.loop(0,count_y,1,output)
            split(oh,plan.output_rows,epoch)
        begin=min(ow,max(0,(pl+1)//2));end=min(ow,max(begin,(w+pl-3)//2+1))
        builder.loop(0,begin,1,lambda x:stripe(x,1))
        split(end-begin,source.output_tile_width,lambda x,count:stripe(op('add',x,begin),count))
        builder.loop(end,ow,1,lambda x:stripe(x,1))
    builder.loop(0,p.window.input_shape[0],1,image)
    plan.__post_init__()
    if before!=(plan.identity,asdict(law),asdict(epoch_law)):
        raise ValueError('joint raw halo source/storage/laws changed during emission')
    return dict(schema='merlin.spatial_packed_raw_epochs.v1',identity=plan.identity,
        source_identity=source.identity,packed_source=panel.identity,
        jointly_live_channel_planes=plan.planes,output_rows_per_epoch=plan.output_rows,
        private_accumulator_rows=plan.required_rows,private_accumulator_bytes=plan.required_rows*p.primitive*4,
        plane_stride_rows=plan.plane_rows,private_input_panel_bytes=panel.workspace_bytes,
        readout_boundary='original raw i32 -> scalar Q1/RNE/ReLU -> spatial maximum; no staged i8',
        packing_lifetime='one preparation per original row run, all CO products fully complete before reuse',
        raw_overlap='slot(n,dy,2)=slot(n,dy+1,0); next output writes only rows1,2',
        completion='selected source-index K epochs; full raw tile and original spatial store drains',
        scope='checked counted emission; no target timing qualification')
