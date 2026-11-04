"""Stride-two pointwise producer with resident weights and exact column readout.

The source stays NHWC; the native convolution gather maps output rows directly
onto strided source DMA runs. No sampled payload tensor is materialized. The
existing column-readout consumer owns all narrowing and per-column scales.
Raw i32 and uniform/scalar i8 destinations use original attributes and the
common terminal store; mixed-column i8 keeps its existing exact consumer.
"""
from __future__ import annotations
from math import prod
from . import isa
from .conv_spatial_panels import _prepare, lower as lower_panels


def _product_attributes(attributes):
    attrs=dict(attributes or {})
    for key in ('acc_scale_by_column','acc_scale','relu','requant_shift'):
        attrs.pop(key,None)
    attrs['epilogue']=[x for x in attrs.get('epilogue',()) if x=='bias_add']
    attrs['output_dtype']='i32'
    attrs['conv_column_readout']='recompute'
    return attrs


def can_lower(ctx,ifm,rhs,dst,attributes,*,allow_uniform=False):
    """Read-only admission against actual native capacity and source geometry."""
    try:
        from .pointwise import _strided_geometry
        window,co,_,_=_strided_geometry(ctx,ifm,rhs,dst,attributes)
        attrs=dict(attributes or {})
        if attrs.get('accumulator_init') or attrs.get('requant_shift') is not None:
            return False
        if ctx.spec(dst)['dtype']=='i32':
            # No epilogue attribute is stripped: ordinary prepare validates raw
            # scale/activation identity and consumes the original bias/source.
            return _prepare(ctx,ifm,rhs,dst,attrs,allow_wide=True,
                            max_row_blocks=4) is not None
        if allow_uniform and ctx.spec(dst)['dtype']=='i8' and _prepare(
                ctx,ifm,rhs,dst,attrs,allow_wide=True,max_row_blocks=4) is not None:
            return True
        scales=attrs.get('acc_scale_by_column')
        if (ctx.spec(dst)['dtype']!='i8' or scales is None
                or len(set(map(float,scales)))<=1 or attrs.get('accumulator_init')
                or attrs.get('requant_shift') is not None):
            return False
        oh,ow=window.output_spatial
        m=window.input_shape[0]*oh*ow
        # Virtual output changes only the producer dtype. It allocates nothing;
        # the common prepare checks the same native panel and disjoint buffers.
        raw='_stride_panels_admission_raw'
        class ProductContext:
            def spec(self,name):
                return {'shape':(m,co),'dtype':'i32'} if name==raw else ctx.spec(name)
            def __getattr__(self,name):return getattr(ctx,name)
        prepared=_prepare(ProductContext(),ifm,rhs,raw,_product_attributes(attrs),
                          allow_wide=True,max_row_blocks=4)
        return prepared is not None
    except (KeyError,TypeError,ValueError,OverflowError):
        return False


def lower(ctx,ifm,rhs,dst,attributes,*,allow_uniform=False):
    """Consume real native resident/group products with the common encoder."""
    if not can_lower(ctx,ifm,rhs,dst,attributes,allow_uniform=allow_uniform):return False
    if ctx.spec(dst)['dtype']=='i32':
        receipt=lower_panels(ctx,ifm,rhs,dst,attributes,allow_wide=True,
            row_group_policy='capacity_guarded',readout_epoch='capacity_guarded')
        if not receipt:raise ValueError('prepared raw stride output lost native admission')
        receipt=dict(receipt)
        receipt['stride_source']={'route':'native_resident_panel_direct_i32',
            'input':ifm,'weights':rhs,'output':dst,'payload_copies':0,
            'source_mapping':'original Window2D + ConvolutionGatherRuns',
            'raw_readout':'existing full i32 store with original attributes'}
        ctx.metadata.setdefault('stride_resident_panels',[]).append(receipt)
        return receipt
    if allow_uniform and _prepare(ctx,ifm,rhs,dst,attributes,allow_wide=True,max_row_blocks=4) is not None:
        receipt=lower_panels(ctx,ifm,rhs,dst,attributes,allow_wide=True,
            row_group_policy='capacity_guarded',readout_epoch='capacity_guarded')
        if not receipt:raise ValueError('prepared uniform stride output lost native admission')
        receipt=dict(receipt)
        receipt['stride_source']={'route':'native_resident_panel_direct_i8',
            'input':ifm,'weights':rhs,'output':dst,'payload_copies':0,
            'source_mapping':'original Window2D + ConvolutionGatherRuns',
            'uniform_readout':'existing terminal i8 store with original attributes'}
        ctx.metadata.setdefault('stride_resident_panels',[]).append(receipt)
        return receipt
    from .conv_column import lower_conv_column_readout
    def producer(ctx,ifm,rhs,raw,attrs):
        receipt=lower_panels(ctx,ifm,rhs,raw,attrs,allow_wide=True,
            row_group_policy='capacity_guarded',readout_epoch='capacity_guarded')
        if not receipt:raise ValueError('prepared stride producer lost native admission')
        return receipt
    receipt=lower_conv_column_readout(ctx,ifm,rhs,dst,attributes,producer=producer)
    receipt=dict(receipt)
    receipt['stride_source']={'route':'native_resident_panel_raw_producer',
        'input':ifm,'weights':rhs,'output':dst,'payload_copies':0,
        'source_mapping':'original Window2D + ConvolutionGatherRuns',
        'column_readout':'existing exact device palette consumer'}
    ctx.metadata.setdefault('stride_resident_panels',[]).append(receipt)
    return receipt
