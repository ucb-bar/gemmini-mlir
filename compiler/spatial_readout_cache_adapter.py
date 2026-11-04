"""Default-off exact Q1 ring and shared spatial readout-cache consumer.

The producer is the incumbent window-transfer convolution, not a packed/rolling
input implementation. RollingPackedSpatialPlan supplies the checked original
Q1 ring layout required by the shared consumer; its panel/input-cache regions
are assigned but unused. Every actual producer, reload and readout completes.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from pathlib import Path
import math
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_readout import RoundedReadout,SpatialReadoutChain
from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan,PackedSpatialProducerPlan
from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan
from merlin.compile.scheduling.spatial_readout_cache import propose_spatial_readout_cache,emit_cached_spatial_max
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan,emit_convolution_tile
from isa import ACC,f32_bits

VALUES=('off','q1_ring_cache')
DECLARATION=dict(values=list(VALUES),mechanism='shared spatial_readout_cache consumer with incumbent convolution producer and private Q1 ring',note='Off incumbent. Only original ReLU, scalar positive f32 Q1, 3x3 stride2 pad1 max, no external Q1 observers. No timing claim.')

def selected():return knob('spatial_readout_cache',default='off',values=VALUES,root=Path(__file__).parent)

@dataclass(frozen=True)
class CachePlan:
    cache:object
    scale:float
    proof:dict
    @property
    def producer(self):return self.cache.source.packed.source.producer
    @property
    def workspace_bytes(self):return self.cache.source.required_bytes
    def revalidate(self):
        self.cache.__post_init__()
        if self.producer.accumulator_row!=0 or self.cache.producer.size<self.producer.row_tile*64:raise ValueError('producer leaves assigned ACC window')
        return self
    def report(self):
        self.revalidate()
        return dict(shared_cache=self.cache.report(),shared_cache_module_sha256=sha256(Path(__import__('merlin.compile.scheduling.spatial_readout_cache',fromlist=['x']).__file__).read_bytes()).hexdigest(),
            plan_sha256=self.cache.identity,source_plan_sha256=self.cache.source.packed.source.identity,
            private_workspace_bytes=self.workspace_bytes,numerical_proof=self.proof,
            producer='incumbent counted window-transfer convolution; no packed or rolling input transform claimed',
            unused_assigned_regions='packed panel and rolling input-cache regions reserved; only original Q1 ring is accessed',
            completion='every primitive producer effect, Q1 store, cache reload and pooled store completes before reuse',
            performance='UNKNOWN; matched complete-call timing required')

def prepare(plan,command):
    """Source admission and all numerical/resource checks before target issue."""
    if type(plan)is not CountedConvolutionPlan:return None
    plan.__post_init__();a=command.get('attributes',{})
    stages=['bias_add' if s=='bias' else s for s in a.get('epilogue',())]
    canonical=['bias_add','acc_scale','relu','maxpool']
    if (command.get('opcode')not in ('CONV','CONV2D') or a.get('output_dtype')!='i8'
        or stages!=[s for s in canonical if s in stages] or not {'relu','maxpool'}<=set(stages)
        or a.get('acc_scale_by_column') is not None or 'requant_shift'in a
        or tuple(a.get('pool_size',()))!=(3,3) or tuple(a.get('pool_stride',()))!=(2,2)
        or tuple(a.get('pool_padding',()))!=(1,1,1,1) or a.get('pool_pad_value',-128)!=-128
        or a.get('external_observers') or a.get('external_q1_observers')
        or plan.column_tile!=plan.primitive or plan.primitive!=16 or plan.accumulator_row!=0):return None
    scale=float(a.get('acc_scale',1.0))
    if not math.isfinite(scale)or scale<=0:return None
    from column_readout import _numerical_proof
    proof=_numerical_proof(f32_bits(scale),True)
    from merlin.llvmlower.requantization import quantized
    if any(quantized(value,(1.0,),False)!=value for value in range(-128,128)):
        raise ValueError('selected Q2 readout does not preserve every signed-i8 value')
    s=SpatialMaxReadout(plan.window.output_spatial,(3,3),(2,2),(1,1,1,1),-128)
    # A primitive-width stripe bounds static emission and pool-controller fields.
    stripe=min(plan.primitive,s.output_hw[1]);width=2*stripe+1
    if min(s.output_hw)<2 or max(*s.output_hw,width)>=256 or plan.output_channels>=1<<32:return None
    planes=(plan.output_channels+plan.column_tile-1)//plan.column_tile
    # At least two output rows need five source rows per complete channel plane.
    # Reject insufficient capacity before constructing any per-plane metadata.
    if planes*5*width>plan.accumulator_rows-plan.row_tile:return None
    chains=tuple(SpatialReadoutChain(RoundedReadout((scale,),True),RoundedReadout((1.0,),False),None)
        for _ in range((plan.output_channels+plan.column_tile-1)//plan.column_tile))
    source=SpatialProducerPlan(plan,s,chains,stripe,readout_mode='staged_readouts',intermediate=AddressWindow('private_q1',0,3*width*plan.column_tile))
    packed=PackedSpatialProducerPlan(source,1,AddressWindow('private_q1',0,1<<40),16)
    rolling=RollingPackedSpatialPlan(packed)
    rolling=RollingPackedSpatialPlan(replace(packed,workspace=replace(packed.workspace,size=rolling.required_bytes)))
    producer_rows=plan.row_tile
    consumer=propose_spatial_readout_cache(rolling,AddressWindow('accumulator',0,producer_rows*64),
        AddressWindow('accumulator',producer_rows*64,(plan.accumulator_rows-producer_rows)*64),row_bytes=64)
    if consumer is None:return None
    return CachePlan(consumer,scale,proof).revalidate()

def emit_schedule(plan,builder,produce,pool_effect,*,cache_completion=None):
    """Emit actual shared consumer and source-exact Q1 frontier order.

    produce(image,y,output_x,pixels,halo_offset,rows,width,column,count)
    completes original conv/Q1 into the given ring. pool_effect binds the shared
    reload/store effects. This control is also executable by the integer fixture.
    """
    plan.revalidate();p=plan.producer;s=plan.cache.source.packed.source.spatial
    h,w=s.input_hw;oh,ow=s.output_hw;op=builder.op
    def split(extent,tile,body):
        whole,tail=divmod(extent,tile)
        if whole:builder.loop(0,whole,1,lambda i:body(op('mul',i,tile),tile))
        if tail:body(whole*tile,tail)
    def stripe(x,pixels):
        width=2*pixels+1
        def image_body(image):
            def source_row(y):
                def chunk(offset,rows):
                    split(p.output_channels,p.column_tile,lambda col,count:produce(image,y,x,pixels,offset,rows,width,col,count))
                split(width,p.row_tile,chunk)
                def pool(py):
                    if cache_completion is None:
                        emit_cached_spatial_max(plan.cache,builder,pool_effect,image=image,pool_y=py,output_x=x,output_pixels=pixels,synchronous_completion=True)
                    else:
                        from merlin.compile.scheduling.spatial_cache_epochs import emit_spatial_cache_epochs
                        emit_spatial_cache_epochs(plan.cache,builder,pool_effect,image=image,pool_y=py,output_x=x,output_pixels=pixels,law=cache_completion)

                def ordinary():
                    numerator=op('sub',y,1)
                    ready=op('and',builder.cmp('ult',numerator,oh*2),builder.cmp('eq',op('urem',numerator,2),0),'i1')
                    builder.branch(ready,lambda:pool(op('udiv',numerator,2)),lambda:None)
                final=max(0,(h-1+1-2+1)//2)
                builder.branch(builder.cmp('eq',y,h-1),lambda:builder.loop(final,oh,1,pool) if final<oh else None,ordinary)
            builder.loop(0,h,1,source_row)
        builder.loop(0,p.window.input_shape[0],1,image_body)
    split(ow,plan.cache.source.packed.source.output_tile_width,stripe)

def emit(builder,conv_plan,command,pointers,lower,*,output):
    """Return None before issue for unadmitted source; never host payload math."""
    if selected()=='off':return None
    plan=prepare(conv_plan,command)
    if plan is None:return None
    from spatial_cache_completion import selected as completion_selected,prepare as prepare_completion
    completion=None;completion_proof=None
    if completion_selected()=='reload_epochs':
        completion,completion_proof=prepare_completion(plan)
    ring=builder.global_buffer(plan.workspace_bytes,'i8');p=plan.producer;h,w=p.window.output_spatial;co=p.output_channels
    builder.fence()
    def produce(image,y,x,pixels,offset,rows,width,col,count):
        first=builder.op('add',builder.op('sub',builder.op('mul',x,2),1),offset)
        def run(start,nrows,ring_column):
            def effect(kind,f):
                if kind=='store':
                    builder.config_st(co,activation=1,scale=plan.scale)
                    logical=builder.op('add',builder.op('mul',builder.op('urem',y,3),width),ring_column)
                    byte=builder.op('add',builder.op('mul',logical,co),col)
                    builder.mvout(builder.ptr_offset(ring,byte),ACC,cols=count,rows=nrows);builder.fence()
                else:
                    if kind=='initialize_bias':builder.config_ld(0,id=2)
                    lower(kind,f)
            row=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),y),w),start)
            emit_convolution_tile(p,builder,effect,output_row=row,output_column=col,rows=nrows,columns=count,accumulator_row=0,synchronous_completion=True)
        valid=builder.op('and',builder.cmp('sge',first,0),builder.cmp('sle',builder.op('add',first,rows),w),'i1')
        def boundary():
            def one(i):
                pos=builder.op('add',first,i)
                clipped=builder.select(builder.cmp('slt',pos,0),0,pos)
                clipped=builder.select(builder.cmp('slt',clipped,w),clipped,w-1)
                run(clipped,1,builder.op('add',offset,i))
            builder.loop(0,rows,1,one)
        builder.branch(valid,lambda:run(first,rows,offset),boundary)
    def consumer(kind,f):
        if kind=='begin_reload_epoch':
            builder.config_ld(co,shrink=True,id=2)
        elif kind in ('complete_reload_epoch','complete_store_epoch'):
            builder.fence()
        elif kind=='reload_intermediate':
            if completion is None:builder.config_ld(co,shrink=True,id=2)
            builder.mvin(builder.ptr_offset(ring,f['scratch_offset']),builder.op('or',ACC,f['accumulator_row']),cols=f['columns'],rows=f['rows'],id=2)
            if completion is None:builder.fence()
        elif kind=='store_spatial_max':
            builder.config_st(co,activation=0,scale=1.0,pool_stride=2,pool_size=3,pool_out_dim=plan.cache.source.packed.source.spatial.output_hw[1],
                porows=1,pocols=f['output_pixels'],orows=3,ocols=f['local_width'],upad=0,lpad=0)
            byte=builder.op('add',builder.op('mul',f['output_row'],co),f['output_column'])
            builder.mvout(builder.ptr_offset(pointers[output],byte),builder.op('or',ACC,f['accumulator_row']),cols=f['columns'],rows=0)
            if completion is None:builder.fence()
        else:raise ValueError('unsupported shared spatial-cache effect')
    emit_schedule(plan,builder,produce,consumer,cache_completion=completion);builder.fence()
    receipt=dict(plan.report(),route='shared_spatial_readout_cache_q1_ring',output=output,output_elements=p.window.input_shape[0]*math.prod(plan.cache.source.packed.source.spatial.output_hw)*co)
    if completion is not None:
        from dataclasses import asdict
        receipt['cache_completion_epochs']=dict(law=asdict(completion),proof=completion_proof)
        receipt['completion']='incumbent producer primitives synchronous; disjoint reload epoch full drain before original pool and full store drain before ring/producer reuse'
    # Reuse the backend's existing bounded/full Q1 materialization receipt channel.
    owner=builder
    while hasattr(owner,'builder'):owner=owner.builder
    if not hasattr(owner,'conv_pool_materialization_receipts'):owner.conv_pool_materialization_receipts=[]
    owner.conv_pool_materialization_receipts.append(receipt)
    return receipt


def admit_source(command,tensors,pointers,*,residents=None):
    """Pure conservative source admission before competing pool routes issue.

    Reuse the existing source/tensor/attribute validation of the panel adapter,
    without selecting or emitting its panel route. The returned checked cache
    owner uses exactly the incumbent convolution primitive and initialization.
    Unsupported source semantics retain the ordinary dispatch unchanged.
    """
    if selected()=='off':return None
    from conv_pool_panel import prepare as validate_source
    source=validate_source(command,tensors,pointers,residents=residents)
    if source is None:return None
    from convolution import _capacities
    from conv_policy import accumulator_init_selection
    edge,operand_rows,accumulator_rows=_capacities()
    w=source.window
    reduction=math.prod(w.kernel)*w.input_shape[-1]
    initialization=accumulator_init_selection()
    explicit=bool(source.bias) or (not source.bias and
        (initialization=='zero_accumulate' or initialization=='zero_single_slice' and reduction<=edge))
    primitive=CountedConvolutionPlan(w,source.channels,edge,min(edge,w.output_spatial[1]),
        edge,min(edge,w.input_shape[-1]),operand_rows,accumulator_rows,0,edge,0,explicit)
    try:return prepare(primitive,command)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError):return None
