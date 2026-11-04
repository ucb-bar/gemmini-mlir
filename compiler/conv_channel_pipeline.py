"""Opt-in high-channel convolution with complete-K resident RHS column phases.

The original counted NHWC plan and terminal effect encoder remain the source of
all coordinates, bias and readout semantics. This adapter changes only checked
storage/issue scheduling, never tensor payload arithmetic or source axes.
"""
from dataclasses import dataclass
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import (
    CountedConvolutionPlan, ConvolutionInputRetentionLaw,
)
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionScheduleChoice,
    ConvolutionRhsResidencyLaw, ConvolutionRhsPhaseLaw,
    plan_convolution_schedule, plan_convolution_rhs_phases,
    emit_convolution_schedule,
)

VALUES=('off','resident_channel_phases')
DECLARATION=dict(values=list(VALUES),mechanism='complete original K resident convolution RHS with capacity-derived original N phases and existing retained-input compute epochs',note='Default off. Explicit scheduling alternative; no timing winner inferred. Original encoder, readout and source K order retained.')


def selected():
    return knob('conv_channel_pipeline',default='off',values=VALUES,root=Path(__file__).parent)


@dataclass(frozen=True)
class ChannelPipeline:
    schedule: object
    rhs: object
    ordering: dict

    def revalidate(self):
        self.schedule.verify_current();self.rhs.verify_current()
        return dict(source_sha256=self.schedule.source.identity,
                    schedule_sha256=self.schedule.identity,
                    rhs=self.rhs.summary(),ordering_evidence=self.ordering)


def prepare(plan,command,*,mode=None):
    """Admit after the parent's original integer, window and readout checks.

    None retains the incumbent before any new allocation or target issue.
    Unknown/stale hardware laws are never replaced with caller assertions.
    """
    mode=selected() if mode is None else mode
    if mode not in VALUES:raise ValueError('unknown convolution channel pipeline')
    if mode=='off':return None
    if type(plan) is not CountedConvolutionPlan:raise ValueError('requires original counted convolution')
    plan.__post_init__()
    w=plan.window;a=command.get('attributes',{})
    if command.get('opcode') not in ('CONV','CONV2D'):return None
    if (w.input_shape[-1]<=plan.primitive or plan.channel_tile!=plan.primitive
        or plan.column_tile!=plan.primitive or 'maxpool' in a.get('epilogue',())
        or any(k.startswith('pool_') for k in a)):return None
    kernel=tuple(a.get('kernel',()))
    if (str(a.get('layout','nhwc')).lower()!='nhwc' or kernel[:2]!=w.kernel
        or len(kernel) not in (2,4)
        or len(kernel)==4 and kernel[2:]!=(w.input_shape[-1],plan.output_channels)
        or tuple(a.get('stride',(1,1)))!=w.stride
        or tuple(a.get('padding',(0,0,0,0)))!=w.padding
        or tuple(a.get('dilation',(1,1)))!=w.dilation):return None
    if a.get('output_dtype') not in ('i8','i32') or 'requant_shift' in a:return None
    count=(plan.output_channels+plan.column_tile-1)//plan.column_tile
    if not 2<=count<=32:return None
    intervals=tuple((c,min(plan.column_tile,plan.output_channels-c))
                    for c in range(0,plan.output_channels,plan.column_tile))
    source=ConvolutionExplorationSource(plan,intervals,plan.identity)
    from conv_policy import completion_laws
    laws=completion_laws(plan,source.identity,count,mode='pairs16')
    if laws is None:return None
    issue,epoch,pairs,evidence=laws
    if epoch is None:return None
    ids=(issue.hardware_identity,issue.numerical_policy_identity,source.identity,
         'original-immutable-disjoint-convolution-inputs-complete-output-columns-v1')
    retention=ConvolutionInputRetentionLaw(*ids,max_live_tiles=count,
        input_preserved=True,independent_outputs=True,immutable_mapped_disjoint_buffers=True)
    schedule=plan_convolution_schedule(source,
        ConvolutionScheduleChoice(live_tiles=None,load_issue=True,compute_pairs=pairs),
        retention_law=retention,issue_law=issue,epoch_law=epoch)
    # The caller's ordinary encoder writes only the selected original weight
    # rectangle. LHS/ACC/readouts are disjoint from resident RHS, and source
    # logical buffers are mapped immutable/disjoint by the ordinary kernel ABI.
    rhs_law=ConvolutionRhsResidencyLaw(*ids,True,True,True,True,True)
    phase_law=ConvolutionRhsPhaseLaw(source.identity,
        'original-independent-output-columns-no-intermediate-observer-full-phase-fence-v1',
        independent_output_columns=True,immutable_inputs=True,
        no_intermediate_observers=True,full_completion_before_reprepare=True)
    rhs=plan_convolution_rhs_phases(schedule,operand_rows=plan.operand_rows,
        law=rhs_law,phase_law=phase_law)
    if rhs is None:return None
    result=ChannelPipeline(schedule,rhs,evidence);result.revalidate();return result


def emit(plan,builder,lower,*,command,issue,complete,mode=None):
    """Consume the actual checked owner through the existing target encoder.

    Hook after all parent source checking, before its ordinary nonpool schedule
    choice. `issue` is the parent's unchanged non-draining effect callback;
    `complete` is its actual fence. Final whole-call completion stays with parent.
    """
    owner=prepare(plan,command,mode=mode)
    if owner is None:return None
    if not callable(issue) or not callable(complete):raise ValueError('requires real target issue/completion callbacks')
    before=owner.revalidate()
    actual=emit_convolution_schedule(owner.schedule,builder,issue,complete=complete,
        synchronous_completion=True,rhs=owner.rhs)
    owner.revalidate()
    result=dict(actual,conv_channel_pipeline=dict(mechanism='full_k_rhs_channel_phases',
        admission=before,source_k_order='kernel_y,kernel_x,input_channel',
        source_layout='unchanged NHWC source and flattened HWIO weights',
        readout='unchanged source bias/scales/activation and original column exits',
        host_payload_arithmetic=False,whole_phase_owner_consumed=True,timing=None))
    if not hasattr(builder,'conv_channel_pipeline_receipts'):builder.conv_channel_pipeline_receipts=[]
    builder.conv_channel_pipeline_receipts.append(result)
    return result
