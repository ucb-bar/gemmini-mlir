"""Optional original Q1 ring and exact ACC reload retention for spatial max.

Shared rolling transport owns input and producer row frontiers. Shared readout
cache control owns actual retained Q1 reloads. Q1 and Q2 stay separate device
readouts; host work only copies input bytes through the shared packer.
"""
from dataclasses import replace
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan
from merlin.compile.scheduling.spatial_readout import RoundedReadout,SpatialReadoutChain
from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan,emit_rolling_spatial_producer
from merlin.compile.scheduling.spatial_readout_cache import propose_spatial_readout_cache,emit_cached_spatial_max
from merlin.compile.scheduling.window_transfer import Window2D,CountedConvolutionPlan,emit_convolution_tile
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.llvmlower.rolling_window import emit_rolling_row_copy,emit_cached_spatial_row_panel
from gemmini_conv import _resources,make_conv_lowerer
from gemmini_readout import config_pool_store
from gemmini_support import scratch_buffer
import gemmini_encode as E


def plan_readout_cache_pool(*,kind,input_shape,weight_shape,attrs,bias_present=False,
                            external_observers=(),pool_readout=None):
    """Admit all geometry/storage/numerics before any target effect.

    The ordinary convolution epilogue supplies Q1 and its original pool has
    identity Q2. A caller with an independently declared Q2 may supply its typed
    RoundedReadout; it is never folded with Q1. Observed Q1 refuses this route.
    """
    from gemmini_composition import plan_producer_pool,CompositionUnavailable
    base,geometry,actual_shape=plan_producer_pool(kind=kind,input_shape=input_shape,
        weight_shape=weight_shape,attrs=attrs,bias_present=bias_present,
        external_observers=external_observers)
    p,s=base.producer,base.spatial
    edge,operand_rows,acc_rows=_resources()
    if (s.window!=(3,3) or s.stride!=(2,2) or s.padding!=(1,1,1,1)
            or s.pad_value!=-128 or not all(c.producer.relu for c in base.readouts)):
        raise CompositionUnavailable('readout cache needs original ReLU and 3x3s2p1 signed-min pool')
    q2=pool_readout or RoundedReadout((1.0,),False)
    if type(q2) is not RoundedReadout or len(q2.scales)!=1:
        raise CompositionUnavailable('readout cache needs an original scalar single-stage Q2')
    q2.__post_init__()
    planes=(p.output_channels+edge-1)//edge
    # At least two consumer rows need a five-row union. Producer effects are
    # confined to one primitive ACC tile, independently of the retained cache.
    cache_rows=acc_rows-edge
    max_width=min(255,cache_rows//(5*planes))
    columns=min(base.output_tile_width,max(0,(max_width-3)//2+1))
    if columns<1:
        raise CompositionUnavailable('two original pooled rows do not fit disjoint selected ACC capacity')
    width=2*columns+1
    chains=tuple(SpatialReadoutChain(c.producer,q2,None) for c in base.readouts)
    source=replace(base,readouts=chains,output_tile_width=columns,readout_mode='staged_readouts',
        intermediate=AddressWindow('private_q1',0,3*width*edge))
    reduction=p.window.kernel[0]*p.window.kernel[1]*p.window.input_shape[-1]
    panel_rows=min(edge,width)
    alignment=edge
    aligned=lambda n:(n+alignment-1)//alignment*alignment
    row_cache=((p.window.kernel[0]-1)*p.window.dilation[0]+1)*aligned(p.window.input_shape[2]*p.window.input_shape[3])
    budget=aligned(aligned(3*width*p.output_channels)+panel_rows*reduction)+row_cache
    packed=PackedSpatialProducerPlan(source,panel_rows,AddressWindow('private_host',0,budget),alignment)
    rolling=RollingPackedSpatialPlan(packed)
    producer=AddressWindow('acc',0,edge*edge*4)
    cache=AddressWindow('acc',producer.size,(acc_rows-edge)*edge*4)
    retained=propose_spatial_readout_cache(rolling,producer,cache,row_bytes=edge*4)
    if retained is None:
        raise CompositionUnavailable('shared exact readout-cache planner refused source/capacity')
    return retained,geometry,actual_shape


def emit_readout_cache_pool(fn,input_ptr,weight_ptr,dst_ptr,*,kind,input_shape,
                           weight_shape,attrs,bias=None,external_observers=(),pool_readout=None):
    """Consume the actual shared rolling producer and cached max schedules."""
    plan,geometry,actual_shape=plan_readout_cache_pool(kind=kind,input_shape=input_shape,
        weight_shape=weight_shape,attrs=attrs,bias_present=bias is not None,
        external_observers=external_observers,pool_readout=pool_readout)
    rolling=plan.source;packed=rolling.packed;source=packed.source;p=source.producer
    co=p.output_channels;edge=p.primitive;k=packed.reduction
    control=FnBuilderControlAdapter(fn)
    workspace=scratch_buffer(fn,packed.workspace.size,'spatial_readout_cache')
    producer_receipts=[];packing_receipts=[]
    def lower(kind,f):
        if kind=='begin_transport':return object()
        if kind=='stage_row':
            emit_rolling_row_copy(fn,input_ptr,workspace,f,volatile=True)
        elif kind in ('await_transport','consumer_issued'):
            return f.get('transfer',object())
        elif kind=='complete':E.fence(fn)
        elif kind=='pack_cached_input_panel':
            packing_receipts.append(emit_cached_spatial_row_panel(rolling,fn,workspace,
                E.pointer_offset(fn,workspace,f['panel_offset']),producer_y=f['producer_y'],
                output_x=f['output_x'],output_pixels=f['output_pixels'],
                first_halo_column=f['first_halo_column'],rows=f['rows'],
                cache_destination_disjoint=True,maximum_panel_bytes=packed.panel_bytes,
                factor_channel_guards=True))
        elif kind=='produce_packed_panel':
            # The shared packer preserves original kh/kw/CI order. Interpret
            # only that complete K panel as a 1x1 product; reserve ACC[0,edge).
            rows=f['rows'];window=Window2D((1,1,rows,k),(1,1))
            tile=CountedConvolutionPlan(window,co,edge,min(edge,rows),edge,min(edge,k),
                p.operand_rows,edge,0,edge,0,bias is not None)
            q1=source.readouts[0].producer
            flat_attrs=dict(kernel=(1,1,k,co),layout='nhwc',scale=q1.scales[0],relu=q1.relu)
            terminal=make_conv_lowerer(fn,E.pointer_offset(fn,workspace,f['panel_offset']),
                weight_ptr,input_shape=(1,1,rows,k),weight_shape=(k,co),attrs=flat_attrs,
                bias=bias,output_dtype='i8',dst_ptr=E.pointer_offset(fn,workspace,f['stripe_offset']),
                synchronous_completion=True)
            for column in range(0,co,edge):
                emit_convolution_tile(tile,control,terminal,output_row=0,output_column=column,
                    rows=rows,columns=min(edge,co-column),accumulator_row=0,synchronous_completion=True)
            E.fence(fn)
            producer_receipts.append(dict(panel_rows=rows,reduction=k,channels=co,
                accumulator_rows=[0,edge],q1_scales=q1.scales,q1_relu=q1.relu))
        elif kind=='reload_intermediate':
            E.config_load(fn,f['scratch_row_stride'],scale=1.0,shrunk=True)
            E.mvin(fn,E.pointer_offset(fn,workspace,f['scratch_offset']),
                control.op('add',E.ACC,f['accumulator_row']),f['rows'],f['columns'])
            E.fence(fn)
        elif kind=='store_spatial_max':
            q2=source.readouts[0].pool
            config_pool_store(fn,co,pool_stride=f['stride_width'],pool_size=f['window_width'],
                pool_out_dim=f['output_pixels'],porows=1,pocols=f['output_pixels'],
                orows=f['local_height'],ocols=f['local_width'],scale=q2.scales[0],activation=int(q2.relu))
            offset=control.op('add',control.op('mul',f['output_row'],co),f['output_column'])
            E.mvout(fn,E.pointer_offset(fn,dst_ptr,offset),
                control.op('add',E.ACC,f['accumulator_row']),0,f['columns'])
            E.fence(fn)
        else:raise ValueError('unknown shared cached spatial effect '+str(kind))
    def consume(**fields):
        emit_cached_spatial_max(plan,control,lower,**fields,synchronous_completion=True)
    receipt=emit_rolling_spatial_producer(rolling,control,lower,
        synchronous_completion=True,pool_consumer=consume)
    E.fence(fn)
    return dict(schema='gemmini.quantized_spatial_readout_cache.v1',rolling=receipt,
        cache=plan.report(),producer_tiles=producer_receipts,packing=packing_receipts,
        original_output_retained=True,original_q1_boundary='actual device i8 store and exact i8-to-ACC reload',
        original_q2_boundary='separate original device pool readout',
        host_payload_arithmetic=False,target_execution_verified=False,timing=None)
