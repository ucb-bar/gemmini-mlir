"""Opt-in resource class over the existing exact convolution column readouts.

The frozen performance hypothesis selects disjoint-half readout only for a
mixed bounded configuration palette, multiple full spatial tiles, and a group
which cannot fit a full ACC batch. Both registered actions invoke the original
source-checked producer/readout and its common terminal encoder. This policy
adds no hardware permission or numerical transform and runs no timing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineSession, PipelineContext
from . import isa
from .column_batches import _palette, _admission

LEVEL='authored_conv_column_source'
OUTPUT='lowered_conv_column_readout'


@dataclass(frozen=True)
class ConvColumnPolicySource:
    input_shape: tuple
    kernel: tuple
    stride: tuple
    dilation: tuple
    padding: tuple
    output_rows: int
    output_channels: int
    input_dtype: str
    rhs_dtype: str
    output_dtype: str
    batching_selection: str
    attributes_json: str
    operands: tuple

    def __post_init__(self):
        if (len(self.input_shape)!=4 or len(self.kernel)!=4
                or any(type(v) is not int or v<=0 for v in (*self.input_shape,*self.kernel))
                or self.kernel[2]!=self.input_shape[-1]
                or min(self.output_rows,self.output_channels)<=0):
            raise ValueError('convolution column policy requires original positive static geometry')
        window=Window2D(self.input_shape,self.kernel[:2],self.stride,self.dilation,self.padding)
        oh,ow=window.output_spatial
        if self.output_rows!=self.input_shape[0]*oh*ow or self.output_channels!=self.kernel[-1]:
            raise ValueError('convolution column policy output differs from original window')
        json.loads(self.attributes_json)


def column_policy_facts(session, attributes):
    source=session.module
    m,n=source.output_rows,source.output_channels
    # Capacity and lane ordering remain checked independently by the actual
    # emitter. This binding permits resource predicates, not queue reordering.
    admission=_admission()
    ep=attributes.get('epilogue',())
    exact=(source.input_dtype==source.rhs_dtype==source.output_dtype=='i8'
        and attributes.get('layout','nhwc').lower()=='nhwc'
        and source.batching_selection=='palette'
        and attributes.get('acc_scale_by_column') is not None
        and not any('pool' in str(x).lower() for x in ep)
        and not (attributes.get('requant_shift') is not None and 'requant' in ep))
    groups=_palette(n,attributes) if exact else ()
    capacity=isa.ACC_ROWS//min(m,isa.DIM)
    resources=dict(dim=isa.DIM,accumulator_rows=isa.ACC_ROWS,
        full_spatial_tiles=m//isa.DIM,spatial_tail_rows=m%isa.DIM,
        configuration_groups=len(groups),mixed_bounded_palette=int(2<=len(groups)<=64),
        exact_column_readout=int(exact),
        full_batch_columns=capacity,
        groups_exceeding_full_batch=sum(len(columns)>capacity for _,columns in groups))
    facts=PolicyFacts(session.stamp(),shapes=(source.input_shape,source.kernel),
        element_types=(source.input_dtype,source.rhs_dtype),layouts=('nhwc','kernel'),
        operations=frozenset({'conv2d','column_readout'}),
        resources=tuple(sorted(resources.items())))
    return facts,admission


def lower_conv_column_policy(ctx,ifm,rhs,dst,attributes,*,lowering):
    """Execute one registered original lowering; return its actual receipt.

    The callback is the non-dispatching incumbent conv_column lowering. Passing
    it explicitly avoids recursion and keeps the original source validation,
    bias producer, f32 scale/half-even/activation/saturation and terminal ISA.
    """
    attrs=dict(attributes or {})
    batching=attrs.get('column_readout_batching',knob(
        'COLUMN_READOUT_BATCHING',default='off',values=('off','on','palette'),
        root=Path(__file__).resolve().parents[1]))
    shape=tuple(map(int,ctx.spec(ifm)['shape']))
    kernel=tuple(map(int,attrs['kernel']))
    padding=tuple(map(int,attrs.get('padding',(0,0,0,0))))
    if len(padding)==2:padding=(*padding,*padding)
    stride=tuple(attrs.get('stride',(1,1)))
    dilation=tuple(attrs.get('dilation',(1,1)))
    window=Window2D(shape,kernel[:2],stride,dilation,padding)
    oh,ow=window.output_spatial
    m,n=shape[0]*oh*ow,kernel[-1]
    source=ConvColumnPolicySource(shape,kernel,stride,dilation,padding,m,n,
        ctx.spec(ifm)['dtype'],ctx.spec(rhs)['dtype'],
        attrs.get('output_dtype',ctx.spec(dst)['dtype']),batching,
        json.dumps(attrs,sort_keys=True,separators=(',',':')),(ifm,rhs,dst))
    admission=_admission()
    fingerprint=lambda value:sha256(json.dumps(asdict(value),sort_keys=True).encode()).hexdigest()
    def verify(value):
        if type(value) is not ConvColumnPolicySource:
            raise ValueError('convolution column policy source type changed')
        value.__post_init__()
    session=PipelineSession(source,
        context=PipelineContext(resources=admission['base_facts_sha256']),
        ir_level=LEVEL,fingerprint=fingerprint,verify_ir=verify)
    facts,admission=column_policy_facts(session,attrs)
    policy=PipelinePolicy((
        PipelineVariant('raw_pipeline',('conv_column_raw_pipeline',),(
            ResourcePredicate('exact_column_readout',1),
            ResourcePredicate('mixed_bounded_palette',1),
            ResourcePredicate('full_spatial_tiles',2),
            ResourcePredicate('groups_exceeding_full_batch',1))),
        PipelineVariant('incumbent',('conv_column_incumbent',))), 'incumbent')
    def entry(route):
        name='conv_column_'+route
        def action(current):
            if current.module!=source or fingerprint(current.module)!=fingerprint(source):
                raise ValueError('convolution column policy source changed')
            selected_attrs=dict(attrs,column_readout_route=route) if route=='raw_pipeline' else attrs
            receipt=lowering(ctx,ifm,rhs,dst,selected_attrs)
            batching=(receipt.get('column_readout') or {}).get('batching') or {}
            if route=='raw_pipeline' and batching.get('route')!='identical_config_disjoint_acc_double_buffer':
                raise ValueError('selected convolution pipeline did not reach the common emitter')
            return {OUTPUT:receipt}
        return name,RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),
            mutates_ir=False,ir_level=LEVEL),action)
    result=execute_policy(session,policy,dict(entry(r) for r in ('raw_pipeline','incumbent')),facts,output=OUTPUT)
    receipt=dict(result.artifact)
    receipt['conv_column_policy']=dict(schema='backend.conv_column_policy.v1',
        selection=result.receipt['selected_variant'],policy=result.receipt,
        source_sha256=fingerprint(source),admission=admission,
        performance='frozen opt-in resource hypothesis; no online measurement or default promotion',
        numerical_contract='original bias producer and exact complete per-column CONFIG_ST words')
    return receipt
