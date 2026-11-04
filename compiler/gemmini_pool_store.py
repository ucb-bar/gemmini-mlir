"""Source-qualified ordinary pooled stores over shared contiguous raw strips.

The optimized producer executes once. Its original raw i32 spill remains;
there is no packed host convolution or materialized full Q1 i8 image. Shared
strip control owns borders/tails and selected ordinary CONFIG_ST/MVOUT owns
Q1. A nonidentity Q2 is a separate exact device readout, never scale-folded.
"""
from merlin.compile.scheduling.spatial_max_stripes import plan_striped_spatial_max
from merlin.compile.scheduling.spatial_readout import RoundedReadout,SpatialReadoutChain,prove_spatial_readout
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
from gemmini_conv import emit_conv,_resources
from gemmini_pool import pool_geometry
from gemmini_readout import config_pool_store
from gemmini_support import scratch_buffer
import gemmini_encode as E


def select_pool_store_route(*,kind,input_shape,weight_shape,attrs,bias_present=False,
                            external_observers=(),pool_readout=None):
    """Pure auto admission; explicit readout_cache keeps its original legality.

    The size guard is one full selected ACC capacity in i8 elements. It is a
    conservative hardware-derived region, not a measured crossover: only one
    public large geometry currently has matched complete-call cycle evidence.
    """
    from gemmini_composition import CompositionUnavailable
    edge,_,acc_rows=_resources()
    receipt=dict(schema='gemmini.pool_store_auto.v1',route='spill',
        array_dimension=edge,accumulator_rows=acc_rows,
        minimum_pooled_elements=acc_rows*edge,
        guard='batch * pooled_height * pooled_width * channels >= ACC_rows * DIM',
        timing_basis='one public large timed geometry; neighbor timing UNKNOWN')
    try:
        plan=plan_pool_store(kind=kind,input_shape=input_shape,weight_shape=weight_shape,
            attrs=attrs,bias_present=bias_present,external_observers=external_observers,
            pool_readout=pool_readout)
    except (CompositionUnavailable,ValueError,TypeError) as refusal:
        return dict(receipt,reason='source/numeric/capacity refusal: '+str(refusal))
    source=plan['source'];oh,ow=source.spatial.output_hw
    elements=source.producer.window.input_shape[0]*oh*ow*source.producer.output_channels
    admitted=elements>=receipt['minimum_pooled_elements']
    return dict(receipt,route='readout_cache' if admitted else 'spill',
        pooled_output_elements=elements,pooled_hw=(oh,ow),
        strip_plan=plan['strip'].revalidate(),
        reason='source-qualified full-capacity pooled output' if admitted else
            'pooled output below conservative full-ACC-capacity guard')


def plan_pool_store(*,kind,input_shape,weight_shape,attrs,bias_present=False,
                    external_observers=(),pool_readout=None):
    from gemmini_composition import plan_producer_pool,CompositionUnavailable
    source,geometry,actual_shape=plan_producer_pool(kind=kind,input_shape=input_shape,
        weight_shape=weight_shape,attrs=attrs,bias_present=bias_present,
        external_observers=external_observers)
    p,s=source.producer,source.spatial
    edge,_,acc_rows=_resources()
    width=(s.output_hw[1]-1)*s.stride[1]+s.window[1]
    if width>255 or s.output_hw[1]>255 or s.window[0]*width>acc_rows:
        raise CompositionUnavailable('complete original pooled-store strip exceeds selected spatial fields/ACC rows')
    q1=source.readouts[0].producer
    q2=pool_readout or RoundedReadout((1.0,),False)
    if type(q2) is not RoundedReadout or len(q2.scales)!=1:
        raise CompositionUnavailable('ordinary strip Q2 needs one original scalar readout')
    q2.__post_init__()
    # This only moves positive monotone Q1 across exact signed maximum.
    # The original Q2 is handled separately, including its own rounding/clamp.
    proof=prove_spatial_readout(SpatialReadoutChain(q1,RoundedReadout((1.0,),False),q1))
    strip=plan_striped_spatial_max(s,batches=p.window.input_shape[0],
        channels=p.output_channels,channel_block=edge,transfer_rows=edge,capacity_rows=acc_rows)
    return dict(source=source,geometry=geometry,actual_shape=actual_shape,strip=strip,
        q1=q1,q2=q2,proof=proof,q2_identity=q2.scales==(1.0,) and not q2.relu)


def emit_strip_readout(fn,src,dst,*,plan,dtype,readout):
    """Consume the canonical shared row-strip program through ordinary ISA."""
    p=plan;control=FnBuilderIndexAdapter(fn);channels=p.channels
    width=4 if dtype=='i32' else 1
    if dtype not in ('i32','i8'):raise ValueError('strip source needs original i32 or exact i8')
    record=p.revalidate()
    E.fence(fn);E.config_load(fn,channels*width,shrunk=dtype=='i8')
    h,w=p.stripe_hw;oh,ow=p.source.output_hw
    config_pool_store(fn,channels,pool_stride=p.source.stride[1],pool_size=p.source.window[1],
        pool_out_dim=ow,porows=1,pocols=ow,orows=h,ocols=w,
        scale=readout.scales[0],activation=int(readout.relu))
    def lower(kind,f):
        if kind=='stage_spatial_rows':
            offset=control.op('mul',control.op('add',control.op('mul',f['source_row'],channels),f['first_column']),width)
            E.mvin(fn,E.pointer_offset(fn,src,offset),control.op('add',E.ACC,f['local_row']),f['rows'],f['columns'])
            E.fence(fn)
        elif kind=='read_spatial_strip':
            offset=control.op('add',control.op('mul',f['output_row'],channels),f['first_column'])
            E.mvout(fn,E.pointer_offset(fn,dst,offset),E.ACC,0,f['columns'])
            E.fence(fn)
        else:raise ValueError('unknown shared ordinary pooled-store effect '+kind)
    emit_index_program(p.kernel,control,lower)
    E.fence(fn)
    return dict(**record,input_dtype=dtype,original_readout=dict(scales=readout.scales,relu=readout.relu),
        terminal='ordinary CONFIG_ST spatial fields + MVOUT rows=0; each DMA/readout completed',
        source_element_bytes=width,host_payload_arithmetic=False)


def emit_pool_store(fn,input_ptr,weight_ptr,dst_ptr,*,kind,input_shape,weight_shape,attrs,
                    bias=None,external_observers=(),pool_readout=None,readout_schedule="strips"):
    """Optimized original producer plus ordinary pooled stripe readouts."""
    if readout_schedule not in ("strips","epochs"):
        raise ValueError("unknown ordinary pooled readout schedule")
    readout = emit_strip_epoch_readout if readout_schedule == "epochs" else emit_strip_readout
    plan=plan_pool_store(kind=kind,input_shape=input_shape,weight_shape=weight_shape,
        attrs=attrs,bias_present=bias is not None,external_observers=external_observers,
        pool_readout=pool_readout)
    source=plan['source'];p=source.producer;b=p.window.input_shape[0];h,w=p.window.output_spatial;co=p.output_channels
    raw=scratch_buffer(fn,b*h*w*co*4,'ordinary_pool_raw')
    producer_attrs=dict(plan['geometry'],epilogue=['bias_add'] if bias is not None else [],output_dtype='i32')
    if kind in ('conv','CONV2D'):
        producer=emit_conv(fn,input_ptr,weight_ptr,raw,input_shape=plan['actual_shape'],
            weight_shape=weight_shape,attrs=producer_attrs,bias=bias,output_dtype='i32',
            source_destination_disjoint=True)
    else:
        from gemmini_dense import emit_dense
        producer=emit_dense(fn,input_ptr,weight_ptr,raw,m=b*h*w,k=input_shape[-1],n=co,
            attrs=producer_attrs,bias=bias,output_dtype='i32')
    oh,ow=source.spatial.output_hw
    pooled=dst_ptr if plan['q2_identity'] else scratch_buffer(fn,b*oh*ow*co,'ordinary_pool_q1')
    first=readout(fn,raw,pooled,plan=plan['strip'],dtype='i32',readout=plan['q1'])
    second=None
    if not plan['q2_identity']:
        spatial=pool_geometry((b,oh,ow,co),dict(pool_size=1,pool_stride=1))['source']
        second_plan=plan_striped_spatial_max(spatial,batches=b,channels=co,
            channel_block=p.primitive,transfer_rows=p.primitive,capacity_rows=p.accumulator_rows)
        second=readout(fn,pooled,dst_ptr,plan=second_plan,dtype='i8',readout=plan['q2'])
    return dict(schema='gemmini.ordinary_pool_store.v1',route='raw_producer_striped_ordinary_pool_store',
        producer=producer,pool=first,q2=second,readout_proof=plan['proof'],
        original_q2=dict(scales=plan['q2'].scales,relu=plan['q2'].relu),
        raw_intermediate_bytes=b*h*w*co*4,full_q1_image_bytes=0,
        pooled_q1_intermediate_bytes=0 if plan['q2_identity'] else b*oh*ow*co,
        host_payload_arithmetic=False,target_execution_verified=False,timing=None)


def plan_strip_readout_epochs(original):
    """Checked optional vertical reuse of a canonical raw-readout strip.

    The existing shared epoch planner owns every local-row/source-pixel join.
    Channel intervals use the original common transfer masks independently; no
    lane remapping, readout-scale reassociation or completion removal occurs.
    """
    from merlin.compile.scheduling.spatial_max_stripes import StripedSpatialMax
    from gemmini_pool_pipeline import plan_producer_pool_epochs
    if type(original) is not StripedSpatialMax:
        raise ValueError('strip epochs require the original checked spatial plan')
    baseline = original.revalidate()
    if original.capacity_rows > _resources()[2]:
        raise ValueError('strip epochs exceed selected physical accumulator rows')
    if (original.source.window[0] != original.source.window[1]
            or original.source.stride[0] != original.source.stride[1]):
        raise ValueError('ordinary pooling terminal requires square windows and equal strides')
    if original.channel_block != E.DIM:
        raise ValueError('strip epochs require the selected terminal column width')
    lane_width = min(original.channels, original.channel_block)
    interval = plan_striped_spatial_max(original.source, batches=original.batches,
        channels=lane_width, channel_block=lane_width,
        transfer_rows=original.transfer_rows, capacity_rows=original.capacity_rows)
    epochs = plan_producer_pool_epochs(interval)
    checked = epochs.revalidate()
    groups, tail = divmod(original.channels, original.channel_block)
    return epochs, dict(schema='gemmini.ordinary_pool_store_epochs.v1',
        original=baseline, interval=checked, full_channel_intervals=groups,
        tail_columns=tail, channels=original.channels,
        source_correspondence='shared checked spatial epoch repeated over original disjoint channel intervals',
        completion='every actual stage and readout completes before proceeding',
        target_execution_verified=False, timing=None)


def emit_strip_epoch_readout(fn,src,dst,*,plan,dtype,readout):
    """Optional completed raw-strip epochs using the same ordinary terminal.

    This is not selected by auto. The original emit_strip_readout remains the
    incumbent until this distinct product has full native/cycle qualification.
    """
    if dtype not in ('i32','i8'):
        raise ValueError('strip epoch source needs original i32 or exact i8')
    if type(readout) is not RoundedReadout or len(readout.scales)!=1:
        raise ValueError('strip epoch needs one original scalar readout')
    readout.__post_init__()
    epochs,receipt=plan_strip_readout_epochs(plan)
    control=FnBuilderIndexAdapter(fn)
    channels=plan.channels;width=4 if dtype=='i32' else 1
    h,w=plan.stripe_hw;_,ow=plan.source.output_hw
    E.fence(fn);E.config_load(fn,channels*width,shrunk=dtype=='i8')
    config_pool_store(fn,channels,pool_stride=plan.source.stride[1],
        pool_size=plan.source.window[1],pool_out_dim=ow,porows=1,pocols=ow,
        orows=h,ocols=w,scale=readout.scales[0],activation=int(readout.relu))

    def interval(first,columns):
        def lower(kind,f):
            if kind=='stage_spatial_rows':
                offset=control.op('mul',control.op('add',
                    control.op('mul',f['source_row'],channels),first),width)
                E.mvin(fn,E.pointer_offset(fn,src,offset),
                    control.op('add',E.ACC,f['local_row']),f['rows'],columns)
                E.fence(fn)
            elif kind=='read_spatial_strip':
                offset=control.op('add',control.op('mul',f['output_row'],channels),first)
                E.mvout(fn,E.pointer_offset(fn,dst,offset),
                    control.op('add',E.ACC,f['local_row']),0,columns)
                E.fence(fn)
            else:
                raise ValueError('unknown checked strip epoch effect '+kind)
        emit_index_program(epochs.kernel,control,lower)

    full,tail=divmod(channels,plan.channel_block)
    if full:
        control.loop(0,full,1,lambda i: interval(control.op('mul',i,plan.channel_block),plan.channel_block))
    if tail:
        interval(full*plan.channel_block,tail)
    E.fence(fn)
    return dict(receipt,input_dtype=dtype,
        original_readout=dict(scales=readout.scales,relu=readout.relu),
        terminal='ordinary CONFIG_ST + completed MVIN/MVOUT at checked epoch row offsets',
        source_element_bytes=width,host_payload_arithmetic=False)
