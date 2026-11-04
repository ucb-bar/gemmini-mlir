"""Original NHWC row ring feeding the shared counted convolution consumer.

The cache stores channel blocks in separate primitive-width row windows. Each
source row or zero halo is staged once per output-row advance and channel
phase. Complete-K immutable weights occupy disjoint remaining scratchpad rows.
"""
from __future__ import annotations
from dataclasses import replace

from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan, ConvolutionResidentInputRows,
)
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhaseLaw, plan_convolution_rhs_phases,
    emit_convolution_schedule,
)
from . import isa


def nominal_rhs_row(window):
    """Keep the original transient RHS view disjoint from the row ring too."""
    pitch=window.input_shape[2]+window.padding[1]+window.padding[3]
    slots=(window.kernel[0]-1)*window.dilation[0]+1
    rows=pitch*slots*((window.input_shape[-1]+isa.DIM-1)//isa.DIM)
    return max(isa.SPAD_ROWS//2,((rows+isa.DIM-1)//isa.DIM)*isa.DIM)


def prepare(plan):
    """Read-only source/capacity admission; no target effect before success."""
    from .conv_channel_phases import _schedule
    p=plan; w=p.window
    if p.column_tile != isa.DIM or p.channel_tile != isa.DIM:
        return None
    # The target terminal supports an explicit row step in config_ex. All
    # ordinary products reset that state before returning to other operators.
    if not 0 < w.stride[1] < 65536:
        return None
    source,schedule,law=_schedule(p,'reduction_epochs')
    # The shared phased invocation statically emits each original N phase.
    # Reuse the incumbent's explicit 32-interval expansion budget so a wide
    # source cannot turn a capacity-bounded ring into unbounded target text.
    if len(source.column_intervals)>32:
        return None
    pitch=w.input_shape[2]+w.padding[1]+w.padding[3]
    slots=(w.kernel[0]-1)*w.dilation[0]+1
    footprint=pitch*slots*((w.input_shape[-1]+isa.DIM-1)//isa.DIM)
    if footprint+isa.DIM >= isa.SPAD_ROWS:
        return None
    rows=ConvolutionResidentInputRows(w,source.identity,isa.DIM,0,pitch,slots,
        0,footprint,min(isa.DIM,pitch),halo_value=0,stage_channels=isa.DIM,
        compute_row_step=w.stride[1])
    # Issued work per original row-ready epoch, independently of the finite
    # pending queue capacity. The same selected ready/valid and ordering law
    # admits backpressure; every row-ready/last-use callback still drains.
    schedule=replace(schedule,issue_law=replace(schedule.issue_law,
        load_issue_budget=max(schedule.issue_law.load_issue_budget,rows.initial_load_commands)))
    rhs_base=((rows.end+isa.DIM-1)//isa.DIM)*isa.DIM
    phase_law=ConvolutionRhsPhaseLaw(source.identity,'original-row-ring-private-weights-v1',
        independent_output_columns=True,immutable_inputs=True,
        no_intermediate_observers=True,full_completion_before_reprepare=True)
    owner=plan_convolution_rhs_phases(schedule,operand_rows=isa.SPAD_ROWS,
        law=law,phase_law=phase_law,rhs_base=rhs_base,input_rows=rows,
        max_column_blocks=32)
    if owner is None:
        return None
    return schedule,rows,owner


def can_lower(ctx,ifm,rhs,dst,attributes):
    from .conv_spatial_panels import _prepare
    admission=_prepare(ctx,ifm,rhs,dst,attributes,allow_wide=True)
    if admission is None:
        return False
    # Share the incumbent source plan exactly, including its primitive row
    # partition; the spatial-panel admission independently checks aliases,
    # shapes, source quantization and supported epilogues before any mutation.
    q=admission.source.plan
    if q.window.kernel != (3,3):
        return False
    try:
        p=CountedConvolutionPlan(q.window,q.output_channels,isa.DIM,
            min(isa.DIM,q.window.output_spatial[1]),q.column_tile,q.channel_tile,
            isa.SPAD_ROWS,isa.ACC_ROWS,0,nominal_rhs_row(q.window),0,bias=q.bias)
        return prepare(p) is not None
    except (ValueError,TypeError,OverflowError):
        return False


def lower(ctx,ifm,rhs,dst,attributes):
    if not can_lower(ctx,ifm,rhs,dst,attributes):
        return False
    from .conv import lower_conv
    return lower_conv(ctx,ifm,rhs,dst,attributes,rolling_rows=True)


def prepare_epochs(plan,attributes,dtype,pitch):
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    if knob('CONV_ROLLING_EPOCHS',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])!='on':
        return None
    prepared=prepare(plan)
    if prepared is None:
        return None
    from .conv_rolling_epochs import prepare as prepare_epoch
    return prepare_epoch(*prepared,attributes,dtype,pitch)


def emit(plan,control,effect,complete,*,epochs=None):
    prepared=prepare(plan)
    if prepared is None:
        raise ValueError('original row-ring admission changed before emission')
    schedule,rows,owner=prepared
    if epochs is not None and epochs.source_identity!=schedule.source.identity:
        raise ValueError('rolling epoch original source changed before emission')
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    if knob('CONV_FUTURE_ROWS',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])=='on':
        from .conv_future_rows import prepare as prepare_future, emit as emit_future
        future=prepare_future(schedule,rows,owner,None if epochs is None else epochs.law)
        if future is not None:
            return emit_future(future,control,effect,complete)
    if knob('CONV_INPUT_READY_DEPENDENCIES',default='off',values=('off','on'),
            root=Path(__file__).resolve().parents[1])=='on':
        from .conv_input_dependencies import prepare as prepare_input, emit as emit_input
        alternative=prepare_input(schedule,rows,owner,None if epochs is None else epochs.law)
        if alternative is not None:
            return emit_input(alternative,control,effect,complete)
    result=emit_convolution_schedule(schedule,control,effect,complete=complete,
        synchronous_completion=True,rhs=owner,input_rows=rows,
        readout_law=None if epochs is None else epochs.law)
    return dict(result,mechanism='original_nhwc_row_ring',
        rolling_readout_epochs=None if epochs is None else epochs.admission,
        input_cache=rows.record(),rhs_owner=owner.summary() if hasattr(owner,'summary') else owner.identity,
        completion='original row-ready, row-last-use, RHS-phase, compute-epoch and terminal readout drains',
        timing='UNKNOWN')
