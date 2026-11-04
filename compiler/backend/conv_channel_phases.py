"""Complete-K NHWC output-channel phases through the common convolution owner.

Only logical N is partitioned. Real source pointers and original DRAM pitches
remain explicit: no weight/output payload is copied, reshaped, or computed on
CPU. Each phase completes before the next reuses private accelerator rows.
The caller selects this alternative; full-fit and unsupported sources decline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan, Window2D, ConvolutionInputRetentionLaw,
)
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionScheduleChoice,
    ConvolutionRhsResidencyLaw, ConvolutionRhsPhaseLaw, ConvolutionRhsPhases,
    plan_convolution_schedule, plan_convolution_rhs_residency,
    plan_convolution_rhs_phases,
)
from . import isa


@dataclass(frozen=True)
class ConvPhysicalViews:
    """Physical element pitches of bounded logical channel views.

    The context view supplies real source pointers advanced by channel_base;
    these pitches remain those of the original complete source tensors.
    This object never grants storage ownership or changes a tensor's identity.
    """
    rhs_row_stride: int
    output_row_stride: int
    source_channels: int
    logical_channels: int
    channel_base: int

    def __post_init__(self):
        if (any(type(v) is not int for v in asdict(self).values())
                or self.source_channels <= 0 or self.logical_channels <= 0
                or self.channel_base < 0
                or self.channel_base+self.logical_channels > self.source_channels
                or self.rhs_row_stride != self.source_channels
                or self.output_row_stride != self.source_channels):
            raise ValueError('channel view must retain exact original bounded row pitches')


class _PhaseContext:
    """Logical shape view with genuine original buffer identities and pointers."""
    _is_conv_channel_view = True

    def __init__(self, original, ifm, rhs, dst, bias, views):
        self._original = original
        self._rhs, self._dst, self._bias = rhs, dst, bias
        self.physical_views = views
        self._specs = {}
        for name in (rhs, dst, bias):
            if name is not None:
                spec = dict(original.spec(name))
                spec['shape'] = [*spec['shape'][:-1], views.logical_channels]
                self._specs[name] = spec

    def __getattr__(self, key):
        return getattr(self._original, key)

    def spec(self, name):
        return self._specs.get(name, self._original.spec(name))

    def buffer(self, name):
        # Alias admission sees the actual base SSA pointer, never a fabricated
        # pointer for a logical view and never an independently allocated copy.
        return self._original.buffer(name)

    def ptr(self, name, elem_offset=0):
        if name in (self._rhs, self._dst, self._bias):
            base = self.physical_views.channel_base
            elem_offset = elem_offset+base if isinstance(elem_offset,int) else self.fb.add_i(elem_offset,self.fb.const(base))
        return self._original.ptr(name, elem_offset)


def _schedule(plan, selection):
    """Reopen the exact common owner's checked schedule before any emission."""
    intervals = tuple((column,min(plan.column_tile,plan.output_channels-column))
                      for column in range(0,plan.output_channels,plan.column_tile))
    source = ConvolutionExplorationSource(plan, intervals, plan.identity)
    hardware = 'gemmini-fire-aligned-b8ff28edc096392c'
    numeric = 'signed-i8-products-i32-source-ordered-readout'
    qualification = 'synchronous-disjoint-lhs-rhs-acc-private-rows-v1'
    ids = (hardware,numeric,source.identity,qualification)
    retention = ConvolutionInputRetentionLaw(*ids,
        max_live_tiles=min(len(intervals),(plan.accumulator_rows-plan.accumulator_row)//plan.row_tile),
        input_preserved=True,independent_outputs=True,immutable_mapped_disjoint_buffers=True)
    issue = epoch = None; pairs = 1
    if selection in ('compute_epochs','reduction_epochs'):
        from .conv_epochs import epoch_admissions
        issue,epoch,pairs,_ = epoch_admissions(source,hardware,numeric,qualification,
            complete_reduction=selection=='reduction_epochs')
    schedule = plan_convolution_schedule(source,
        ConvolutionScheduleChoice(live_tiles=None,load_issue=issue is not None,compute_pairs=pairs),
        retention_law=retention,issue_law=issue,epoch_law=epoch)
    law = ConvolutionRhsResidencyLaw(*ids,weight_transfer_only=True,
        immutable_mapped_disjoint_buffers=True,other_effects_preserve_rhs=True,
        exclusive_private_rows=True,entry_quiescence_and_complete_preparation=True)
    return source,schedule,law


def _plan(ctx, ifm, rhs, dst, attributes):
    """Build all original/phase capacity and numeric checks without ctx writes."""
    if getattr(ctx,'_is_conv_channel_view',False):
        return None
    attrs = dict(attributes or {})
    selection = knob('CONV_INPUT_RETENTION',default='off',
        values=('off','on','resident','compute_epochs','reduction_epochs'),root=Path(__file__).resolve().parents[1])
    if selection not in ('resident','compute_epochs','reduction_epochs'):
        return None
    try:
        inputs = ctx.spec(ifm),ctx.spec(rhs),ctx.spec(dst)
        a,b,c = inputs
        shape = tuple(map(int,a['shape']))
        kh,kw,ci,co = tuple(map(int,attrs['kernel']))
        if (len(shape)!=4 or attrs.get('layout','nhwc').lower()!='nhwc'
                or min(*shape,kh,kw,ci,co)<=0 or ci!=shape[-1]
                or a['dtype']!='i8' or b['dtype']!='i8' or c['dtype'] not in ('i8','i32')
                or attrs.get('output_dtype',c['dtype'])!=c['dtype']):return None
        if attrs.get('accumulator_init') is not None or attrs.get('requant_shift') is not None:return None
        if any(e not in ('bias_add','acc_scale','relu') for e in attrs.get('epilogue',())):return None
        if any(key.startswith('pool') for key in attrs):return None
        padding = tuple(map(int,attrs.get('padding',(0,0,0,0))))
        if len(padding)==2:padding=(*padding,*padding)
        window = Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),tuple(attrs.get('dilation',(1,1))),padding)
        oh,ow = window.output_spatial
        if min(oh,ow)<=0:return None
        if tuple(b['shape']) not in ((kh*kw*ci,co),(kh,kw,ci,co)):return None
        if tuple(c['shape']) not in ((shape[0]*oh*ow,co),(shape[0],oh,ow,co)):return None
        bias = attrs.get('bias')
        if 'bias_add' in attrs.get('epilogue',()) and not bias:return None
        if bias and (ctx.spec(bias)['dtype']!='i32' or ctx.spec(bias)['shape']!=[co]):return None
        names = {ifm,rhs,dst,bias}-{None}
        if len(names)!=3+int(bias is not None):return None
        if dst in {ifm,rhs,bias}:return None
        aliases = ctx.cb.get('params',{}).get('tensor_aliases',{})
        if any(name in aliases for name in names):return None
        # Existing registered views may share pointer SSA values without an ABI
        # alias declaration. Reject output overlap using their real identities.
        buffers = getattr(ctx,'buffers',{})
        output = buffers.get(dst)
        if output is not None and any(buffers.get(name) is output for name in names-{dst}):return None
        # Strided/nested source views need a separate physical-view contract.
        if any(any(key in spec for key in ('strides','physical_views','physical_strides','offset')) for spec in inputs):return None
        scales = attrs.get('acc_scale_by_column')
        if scales is not None and len(scales)!=co:return None
        values = tuple(map(float,scales)) if scales is not None else (float(attrs.get('acc_scale',1.0)),)
        if any(not isfinite(v) or v<0 for v in values):return None
        relu = attrs.get('relu',False) or 'relu' in attrs.get('epilogue',())
        if c['dtype']=='i32' and (scales is not None or values!=(1.0,) or relu):return None
        mixed = scales is not None and len(set(values))>1
        if mixed:
            route = attrs.get('conv_column_readout',knob('CONV_COLUMN_READOUT',default='recompute',
                values=('recompute','raw_reload'),root=Path(__file__).resolve().parents[1]))
            if route!='recompute':return None
        tile_n = 1 if mixed else min(isa.DIM,co)
        channel_tile = min(isa.DIM,ci)
        seeded = bool(bias) or (kh*kw==1 and ci<=isa.DIM)
        plan = CountedConvolutionPlan(window,co,isa.DIM,min(isa.DIM,ow),tile_n,
            channel_tile,isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.SPAD_ROWS//2,0,bias=seeded)
        source,schedule,law = _schedule(plan,selection)
        phase_law = ConvolutionRhsPhaseLaw(source.identity,'source-closed-channel-views-v1',
            independent_output_columns=True,immutable_inputs=True,no_intermediate_observers=True,
            full_completion_before_reprepare=True)
        rhs_base = ((plan.lhs_row+plan.row_tile+channel_tile-1)//channel_tile)*channel_tile
        if plan_convolution_rhs_residency(schedule,operand_rows=isa.SPAD_ROWS,law=law,rhs_base=rhs_base) is not None:
            return None
        phases = plan_convolution_rhs_phases(schedule,operand_rows=isa.SPAD_ROWS,law=law,
            phase_law=phase_law,rhs_base=rhs_base,max_column_blocks=32)
        if type(phases) is not ConvolutionRhsPhases or len(phases.phases)>32:return None
        prepared = []
        end = 0
        for phase in phases.phases:
            intervals = phase.column_intervals
            base = intervals[0][0]; width = sum(count for _,count in intervals)
            if base!=end:return None
            end += width
            views = ConvPhysicalViews(co,co,co,width,base)
            local_attrs = dict(attrs,kernel=[kh,kw,ci,width])
            if scales is not None:local_attrs['acc_scale_by_column']=list(scales[base:base+width])
            local_scales = local_attrs.get('acc_scale_by_column')
            local_mixed = local_scales is not None and len(set(map(float,local_scales)))>1
            local_tile = 1 if local_mixed else min(isa.DIM,width)
            local_plan = CountedConvolutionPlan(window,width,isa.DIM,min(isa.DIM,ow),local_tile,
                channel_tile,isa.SPAD_ROWS,isa.ACC_ROWS,0,isa.SPAD_ROWS//2,0,bias=seeded)
            local_source,local_schedule,local_law = _schedule(local_plan,selection)
            owner = plan_convolution_rhs_residency(local_schedule,operand_rows=isa.SPAD_ROWS,law=local_law,rhs_base=rhs_base)
            if owner is None or owner.layout.rhs_rows>phase.layout.rhs_rows:return None
            owner.verify_current()
            prepared.append((views,local_attrs,local_source.identity,owner.identity))
        if end!=co:return None
        phases.verify_current()
        return dict(phases=phases,prepared=tuple(prepared),bias=bias,selection=selection,
            original_shape=shape,output_spatial=(oh,ow),kernel=(kh,kw,ci,co),dtype=c['dtype'])
    except (ValueError,TypeError,KeyError,OverflowError):
        return None


def can_lower(ctx, ifm, rhs, dst, attrs):
    """Whether all channel phases admit before any pointer/storage mutation."""
    return _plan(ctx,ifm,rhs,dst,attrs) is not None


def lower(ctx, ifm, rhs, dst, attrs):
    """Emit each checked phase through lower_conv, or False before mutation.

    Caller gates this alternative. It requires an existing resident/compute/
    reduction-epoch convolution choice; all other sources keep the incumbent.
    """
    prepared = _plan(ctx,ifm,rhs,dst,attrs)
    if prepared is None:return False
    from .conv import lower_conv
    receipts = []
    ctx.fence()
    for views,phase_attrs,source_identity,owner_identity in prepared['prepared']:
        phase_ctx = _PhaseContext(ctx,ifm,rhs,dst,prepared['bias'],views)
        receipt = lower_conv(phase_ctx,ifm,rhs,dst,phase_attrs,physical_views=views)
        receipts.append(dict(physical_views=asdict(views),source_identity=source_identity,
            rhs_residency_identity=owner_identity,emission=receipt))
        ctx.fence()
    kh,kw,ci,co = prepared['kernel']
    oh,ow = prepared['output_spatial']; batch = prepared['original_shape'][0]
    receipt = dict(mechanism='source_nhwc_complete_k_channel_phases',
        source_phases=prepared['phases'].summary(),selection=prepared['selection'],members=receipts,
        original_source=dict(ifm=ifm,rhs=rhs,dst=dst,kernel=list(prepared['kernel']),
            output_elements=batch*oh*ow*co,source_macs=batch*oh*ow*kh*kw*ci*co),
        projection='every original output channel once; original full reduction order and scalar readout preserved',
        pointers='same original source buffers; actual pointers advanced by channel_base; original physical row pitches',
        completion='common owner preparation/readout/final drains plus real phase boundaries',timing='UNKNOWN')
    ctx.metadata.setdefault('convolution_channel_phases',[]).append(receipt)
    return receipt
