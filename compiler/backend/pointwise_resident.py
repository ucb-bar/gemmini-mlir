"""Source-qualified positive-stride 1x1 NHWC resident panels without SMT.

The original window owns batch/spatial coordinates. Shared gather fragments
split every original output row and image, then directly load sampled source
rows into a full primitive M tile. Complete-K RHS phases, accumulator epochs
and the terminal contraction/readout encoder remain shared with convolution.
"""
from __future__ import annotations
from dataclasses import asdict
from math import isfinite,prod
from pathlib import Path

from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.compile.tuning_knobs import knob
from . import isa,conv_spatial_panels


class _RawOutput:
    def __init__(self,ctx,dst):self.ctx,self.dst=ctx,dst
    def spec(self,name):
        spec=self.ctx.spec(name)
        return dict(spec,dtype='i32')if name==self.dst else spec
    def __getattr__(self,name):return getattr(self.ctx,name)


def _product_attributes(attrs):
    result=dict(attrs)
    for key in ('acc_scale_by_column','acc_scale','relu','requant_shift'):
        result.pop(key,None)
    result['epilogue']=[s for s in attrs.get('epilogue',())if s=='bias_add']
    result['output_dtype']='i32'
    return result


def _prepare(ctx,ifm,rhs,dst,attributes):
    """Refuse unsupported source, numeric and capacity contracts before effects."""
    try:
        attrs=dict(attributes or {})
        shape=tuple(ctx.spec(ifm)['shape']);kernel=tuple(attrs['kernel'])
        stride=tuple(attrs.get('stride',(1,1)))
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        if (len(shape)!=4 or len(kernel)!=4 or kernel[:2]!=(1,1)
                or kernel[2]!=shape[-1] or len(stride)!=2
                or any(type(x)is not int or x<=0 for x in (*shape,*kernel,*stride))
                or tuple(attrs.get('dilation',(1,1)))!=(1,1)
                or len(padding)!=4 or any(padding)):
            return None
        # Preserve the measured incumbent for subprimitive contractions.
        if min(kernel[2:])<isa.DIM:return None
        window=Window2D(shape,(1,1),stride,(1,1),padding)
        dtype=ctx.spec(dst)['dtype']
        if attrs.get('output_dtype',dtype)!=dtype:return None
        scales=attrs.get('acc_scale_by_column')
        mixed=scales is not None and len(set(map(float,scales)))>1
        if scales is not None and (len(scales)!=kernel[3]
                or any(not isfinite(float(s))or float(s)<0 for s in scales)):
            return None
        if mixed:
            if (dtype!='i8' or attrs.get('requant_shift')is not None
                    or any(s not in ('bias_add','acc_scale','relu')for s in attrs.get('epilogue',()))):return None
            route=attrs.get('conv_column_readout',knob('CONV_COLUMN_READOUT',default='recompute',
                values=('recompute','raw_reload'),root=Path(__file__).resolve().parents[1]))
            if route!='raw_reload':return None
            selection=attrs.get('column_readout_batching',knob('COLUMN_READOUT_BATCHING',default='off',
                values=('off','on','palette'),root=Path(__file__).resolve().parents[1]))
            if selection not in ('off','on','palette'):return None
            if selection!='off':
                from .column_batches import _admission
                _admission()
            view=_RawOutput(ctx,dst);producer=_product_attributes(attrs)
        else:
            view=ctx;producer=attrs
        prepared=conv_spatial_panels._prepare(view,ifm,rhs,dst,producer,
            allow_wide=True,allow_initializer=True)
        if prepared is None:return None
        return window,mixed,prepared
    except(KeyError,TypeError,ValueError,OverflowError):
        return None


def can_lower(ctx,ifm,rhs,dst,attributes):
    return _prepare(ctx,ifm,rhs,dst,attributes)is not None


def _input_fragments(prepared,builder,first,count,ky,kx,channel,depth,transfer):
    window=prepared.source.plan.window
    if window.kernel!=(1,1)or any(window.padding):
        raise ValueError('resident pointwise transfer requires its original 1x1 source')
    for origin,local,rows in emit_convolution_gather_runs(prepared.gather,builder,first,count):
        def active(origin=origin,local=local,rows=rows):
            transfer(dict(zero=False,first_element=builder.op('add',origin,channel),
                local_row=builder.op('add',prepared.group.lhs_row(0),local),
                rows=rows,columns=depth,row_stride_elements=window.stride[1]*window.input_shape[-1]))
        builder.branch(builder.cmp('ult',0,rows),active,lambda:None)


def lower(ctx,ifm,rhs,dst,attributes):
    admission=_prepare(ctx,ifm,rhs,dst,attributes)
    if admission is None:return False
    window,mixed,prepared=admission
    attrs=dict(attributes or {});n=prepared.source.plan.output_channels
    if mixed:
        from .pointwise import _lower_strided_column_readout
        result=_lower_strided_column_readout(ctx,ifm,rhs,dst,attrs,window,n,producer=lower)
    else:
        result=conv_spatial_panels.lower(ctx,ifm,rhs,dst,attrs,
            allow_wide=True,allow_initializer=True,input_fragments=_input_fragments)
        if result is False:raise RuntimeError('resident pointwise admission changed during emission')
    receipt=dict(schema='merlin.pointwise_resident_panels.v1',window=asdict(window),
        dimensions=[window.input_shape[0]*prod(window.output_spatial),window.input_shape[-1],n],
        source_row_stride_elements=window.stride[1]*window.input_shape[-1],
        payload_copies=0,optional_solver=False,product=result,
        initializer='matrix_and_bias'if attrs.get('accumulator_init')and attrs.get('bias')else
            ('original_matrix'if attrs.get('accumulator_init')else
            ('original_bias'if attrs.get('bias')else'original_additive_identity')),
        source_mapping='shared original-row/image fragments; one disjoint direct DMA per positive fragment')
    ctx.metadata.setdefault('pointwise_resident_panels',[]).append(receipt)
    return receipt


lower_pointwise_resident=lower
