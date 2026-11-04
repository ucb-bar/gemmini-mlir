"""Explicit across-M readout epochs over the existing complete-K RHS panel.

The shared output-group artifact owns distinct ACC rows. Every M producer keeps
its original full-K drain before scratch LHS reuse; only initialized independent
outputs and ready stores are grouped. No retained-array/STAY law is inferred.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import prod

from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group
from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs
from . import isa
from .conv_regional import _binding
from .conv_transfer_regional import plan_transfer_regional
from .conv_transfer_runs import plan_transfer_runs
from .conv_spatial_panels import _emit_prepared, _input_fragments


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ConvWeightPanelEpochs:
    original: object
    prepared: object
    transfers: object
    binding_sha256: str
    group_sha256: str

    def revalidate(self,ctx=None):
        if ctx is not None:
            self.original.revalidate(ctx)
            names=self.original.regional._names()
            if _binding(ctx,names,self.prepared)!=self.binding_sha256:
                raise ValueError('M-group convolution source or allocation changed')
        old=self.original.regional.prepared;p=self.prepared
        expected=plan_convolution_output_group(old.group.panel,prod(old.gather.output_spatial),max_row_blocks=16)
        p.group.__post_init__()
        if (p.group!=expected or p.group.row_blocks<=1 or
                _digest(asdict(p.group))!=self.group_sha256 or
                replace(p,group=old.group)!=old):
            raise ValueError('M-group convolution changed its exact shared source/group')
        p.epoch.require(max(2,p.group.panel.reduction_tiles*p.group.panel.column_blocks))
        if self.transfers is not None:self.transfers.revalidate(p)

    def emit_schedule(self,prepared,builder,lower,complete,*,peel_first_reduction=False,
                      uniform_store_epoch=False,hoist_origins=False,input_fragment_emitter=None):
        self.revalidate()
        if (prepared is not self.prepared or peel_first_reduction or not uniform_store_epoch
                or input_fragment_emitter is not None):
            raise ValueError('M-group schedule requires exact prepared source and uniform ready stores')
        p=prepared.source.plan;w=p.window;panel=prepared.group.panel;group=prepared.group
        ci=w.input_shape[-1];kh,kw=w.kernel;pixels=prod(prepared.gather.output_spatial)
        def reduction(callback):
            def y(ky):
                def x(kx):
                    full,tail=divmod(ci,isa.DIM)
                    if full:builder.loop(0,full,1,lambda c:callback(ky,kx,c,isa.DIM))
                    if tail:callback(ky,kx,full,tail)
                builder.loop(0,kw,1,x)
            builder.loop(0,kh,1,y)
        def locations(ky,kx,chunk):
            tap=builder.op('add',builder.op('mul',ky,kw),kx)
            return (builder.op('add',builder.op('mul',tap,panel.channel_tiles),chunk),
                    builder.op('add',builder.op('mul',tap,ci),builder.op('mul',chunk,isa.DIM)))
        def phase(column,widths):
            active=len(widths)
            def weights(ky,kx,chunk,depth):
                slot,start=locations(ky,kx,chunk)
                for n,columns in enumerate(widths):
                    lower('load_weight',dict(reduction_row=start,output_column=builder.op('add',column,n*isa.DIM),
                        local_row=builder.op('add',panel.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),rows=depth,columns=columns))
            reduction(weights);complete('spatial_phase_rhs_ready',{})
            def spatial_group(first,row_counts):
                outputs=tuple(tuple(dict(output_row=builder.op('add',first,m*isa.DIM),
                    output_column=builder.op('add',column,n*isa.DIM),rows=rows,columns=columns,
                    accumulator_row=group.accumulator_row(m,n,active_column_blocks=active))
                    for n,columns in enumerate(widths))for m,rows in enumerate(row_counts))
                if p.bias:
                    for row in outputs:
                        for out in row:lower('initialize_bias',out)
                    complete('spatial_group_initializer_ready',{})
                for m,(rows,row) in enumerate(zip(row_counts,outputs)):
                    source_first=builder.op('add',first,m*isa.DIM);origins=None
                    if hoist_origins:
                        _,height,width,channels=w.input_shape;pt,pl,pb,pr=w.padding
                        ph,pw=height+pt+pb,width+pl+pr
                        origins=tuple((origin,local,count,builder.op('udiv',origin,ph*pw*channels),
                            builder.op('urem',builder.op('udiv',origin,pw*channels),ph),
                            builder.op('urem',builder.op('udiv',origin,channels),pw))
                            for origin,local,count in emit_convolution_gather_runs(prepared.gather,builder,source_first,rows))
                    def products(ky,kx,chunk,depth):
                        fragments=self.transfers.emit if self.transfers is not None else _input_fragments
                        fragments(prepared,builder,source_first,rows,ky,kx,builder.op('mul',chunk,isa.DIM),depth,
                            lambda f:lower('load_input',f),origins=origins)
                        slot,_=locations(ky,kx,chunk)
                        first_k=builder.op('and',builder.cmp('eq',ky,0),builder.cmp('eq',kx,0),'i1')
                        first_k=builder.op('and',first_k,builder.cmp('eq',chunk,0),'i1')
                        for n,out in enumerate(row):
                            lower('compute',dict(out,lhs_row=group.lhs_row(0),rhs_row=builder.op('add',panel.rhs_base,
                                builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                                depth=depth,first_reduction=first_k,bias=p.bias,fresh_weights=True))
                    reduction(products)
                    # Retained real full-K completion: no new LHS reuse law.
                    complete('spatial_reduction_epoch_complete',{})
                lower('store_epoch_begin',dict(active=active*len(row_counts)))
                for row in outputs:
                    for out in row:lower('store',out)
                complete('spatial_group_output_last_use',{})
            full,tail=divmod(pixels,isa.DIM);groups,left=divmod(full,group.row_blocks)
            if groups:builder.loop(0,groups,1,lambda g:spatial_group(builder.op('mul',g,group.row_blocks*isa.DIM),(isa.DIM,)*group.row_blocks))
            remainder=(isa.DIM,)*left+((tail,)if tail else ())
            if remainder:spatial_group(groups*group.row_blocks*isa.DIM,remainder)
            complete('spatial_phase_rhs_last_use',{})
        full_columns,last_columns=divmod(p.output_channels,isa.DIM)
        phases,tail=divmod(full_columns,panel.column_blocks)
        if phases:builder.loop(0,phases,1,lambda n:phase(builder.op('mul',n,panel.column_blocks*isa.DIM),(isa.DIM,)*panel.column_blocks))
        widths=(isa.DIM,)*tail+((last_columns,)if last_columns else ())
        if widths:phase(phases*panel.column_blocks*isa.DIM,widths)
        self.revalidate()
        return dict(mechanism='complete_rhs_across_m_store_epochs',source_sha256=prepared.source.identity,
            rhs_rows=panel.rhs_rows,rhs_base=panel.rhs_base,n_phase_tiles=panel.column_blocks,
            original_output_hw=list(w.output_spatial),spatial_tiles=(pixels+isa.DIM-1)//isa.DIM,
            m_group_slots=group.row_blocks,shared_group_sha256=self.group_sha256,
            shared_group=asdict(group),uniform_store_epoch=True,affine_origins_hoisted=hoist_origins,
            full_k_completion_per_m=True,lhs_row=group.lhs_row(0),fresh_weights=True,
            ownership='disjoint shared M/N ACC intervals; all producers ready before uniform stores; real group-end drain',
            compute_epoch=asdict(prepared.epoch),transfer_product=self.transfers.receipt()if self.transfers else None,
            timing=None)

    def emit(self,ctx):
        self.revalidate(ctx);r=self.original.regional
        result=_emit_prepared(ctx,r.ifm,r.rhs,r.dst,self.prepared,
            trial_mode='affine_uniform_stores'if self.original.actual_mode=='affine_uniform_stores'else'uniform_stores',
            schedule_emitter=self.emit_schedule)
        self.revalidate(ctx)
        ctx.metadata.setdefault('convolution_m_store_epochs',[]).append(dict(
            binding_sha256=self.binding_sha256,shared_group_sha256=self.group_sha256,
            original_regional_mode=self.original.actual_mode,explicit_option=True,emitted=result))
        return result


def plan_weight_panels(ctx,ifm,rhs,dst,attributes):
    original=plan_transfer_regional(ctx,ifm,rhs,dst,attributes)
    if original is None:return None
    old=original.regional.prepared
    group=plan_convolution_output_group(old.group.panel,prod(old.gather.output_spatial),max_row_blocks=16)
    if group.row_blocks<=1:return None
    prepared=replace(old,group=group)
    result=ConvWeightPanelEpochs(original,prepared,plan_transfer_runs(prepared),
        _binding(ctx,original.regional._names(),prepared),_digest(asdict(group)))
    result.revalidate(ctx);return result


def lower_weight_panels(ctx,ifm,rhs,dst,attributes):
    """Optional across-M store epochs; None refusal occurs before emission."""
    product=plan_weight_panels(ctx,ifm,rhs,dst,attributes)
    return None if product is None else product.emit(ctx)
