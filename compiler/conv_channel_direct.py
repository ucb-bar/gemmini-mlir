"""Opt-in original NHWC ring transport with bounded real channel DMA phases.

Uses the public resident-input owner, not the kx-replicated CI<=16 owner.
All original ky,kx,ci products/readouts use the ordinary terminal callback.
"""
from dataclasses import dataclass, replace, field
from pathlib import Path
from hashlib import sha256
import json
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import ConvolutionResidentInputRows
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhaseLaw, plan_convolution_schedule,
    plan_convolution_rhs_phases, emit_convolution_schedule,
)

VALUES=('off','resident_channel_rows')
DECLARATION=dict(values=list(VALUES),mechanism='original NHWC resident row ring with capacity-derived RHS phases and exact channel-partitioned direct source DMA',note='Default off. No pooling or horizontal subsampling; no performance claim. Source K order, common products/readout and rolling ready/last-use completion are preserved.')

def _transport_law():
    from conv_horizontal import _load_law
    # Reuse only the selected coordinate/extent authority. Its unrelated
    # synchronous horizontal-emitter description is not this route's law.
    value=_load_law()
    return None if value is None else {k:v for k,v in value.items() if k not in ('completion','array')}

def selected():
    return knob('conv_channel_direct',default='off',values=VALUES,root=Path(__file__).parent)

@dataclass(frozen=True)
class ChannelRows:
    schedule: object
    input_rows: object
    rhs: object
    original: object
    source_plan: object
    transfer: dict
    source_plan_sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self,'source_plan_sha256',self.source_plan.identity)

    def revalidate(self):
        self.original.revalidate();self.schedule.verify_current();self.input_rows.revalidate();self.rhs.verify_current()
        p=self.schedule.source.plan;r=self.input_rows
        self.source_plan.__post_init__()
        if self.source_plan.identity!=self.source_plan_sha256:
            raise ValueError('original source/placement changed after direct-channel admission')
        if replace(self.source_plan,rhs_row=p.rhs_row)!=p:
            raise ValueError('direct channel changes original source beyond private RHS placement')
        if (r.source_identity!=self.schedule.source.identity or r.window!=p.window
            or r.channel_tile!=p.channel_tile or r.row_step!=1 or r.end>p.operand_rows
            or r.stage_channels>self.transfer['maximum_columns']):
            raise ValueError('direct channel source/transport ownership changed')
        # Phased owners carry and verify the very same input owner; full-fit
        # owners are checked for explicit physical disjointness here and again
        # by the library emitter.
        phases=getattr(self.rhs,'phases',(self.rhs,))
        if any(not r.disjoint_from(x.layout.rhs_base,x.layout.rhs_rows) for x in phases):
            raise ValueError('direct channel input overlaps immutable resident RHS')
        import conv_horizontal
        root=Path(conv_horizontal.__file__).parent
        selection=json.loads((root/'command_ordering.json').read_text())
        if (sha256((root/'facts.json').read_bytes()).hexdigest()!=self.transfer['facts_sha256']
            or sha256(json.dumps(selection['ordering'],sort_keys=True).encode()).hexdigest()!=self.transfer['ordering_sha256']
            or selection['ordering_sha256']!=self.transfer['ordering_sha256']):
            raise ValueError('direct channel selected facts/ordering changed')
        if _transport_law()!=self.transfer:raise ValueError('selected direct transport law changed')
        return dict(source_sha256=self.schedule.source.identity,schedule_sha256=self.schedule.identity,
            original_plan_sha256=self.source_plan.identity,private_rhs_relocation=p.rhs_row,
            input_rows=r.record(),rhs=self.rhs.summary(),selected_transport=self.transfer,
            load_issue_budget=self.schedule.issue_law.load_issue_budget)

def prepare(plan,command,*,mode=None):
    mode=selected() if mode is None else mode
    if mode not in VALUES:raise ValueError('unknown direct channel mode')
    if mode=='off':return None
    # Existing061 reopens original source geometry and selected ordering laws.
    from conv_channel_pipeline import prepare as prepare_original
    w=plan.window;a=command.get('attributes',{})
    if w.stride[1]!=1 or w.dilation!=(1,1):return None
    z=a.get('zero_points',{})
    if not isinstance(z,dict) or any(v!=0 for v in z.values()):return None
    transfer=_transport_law()
    if transfer is None:return None
    edge=plan.primitive;pitch=w.input_shape[2]+w.padding[1]+w.padding[3]
    # CONFIG_LD block_stride is a selected sixteen-bit row distance. It must
    # describe the owner's actual channel-block placement, never a fake alias.
    if not 0<pitch<1<<16:return None
    slots=w.kernel[0];end=slots*pitch*((w.input_shape[-1]+edge-1)//edge)
    if end>=plan.operand_rows:return None
    mapped=replace(plan,rhs_row=(end+edge-1)//edge*edge)
    if mapped.rhs_row+edge>plan.operand_rows:return None
    original=prepare_original(mapped,command,mode='resident_channel_phases')
    if original is None:return None
    rows=ConvolutionResidentInputRows(w,original.schedule.source.identity,edge,
        base=0,row_pitch=pitch,slots=slots,private_first=0,private_rows=end,
        stage_rows=min(edge,pitch),halo_value=0,stage_channels=transfer['maximum_columns'])
    # This field bounds the finite command batch; it grants no ordering law.
    # Selected ordering/queue/configuration premises are unchanged from061.
    budget=max(original.schedule.issue_law.load_issue_budget,rows.initial_load_commands)
    issue=replace(original.schedule.issue_law,load_issue_budget=budget)
    schedule=plan_convolution_schedule(original.schedule.source,original.schedule.choice,
        retention_law=original.schedule.retention_law,issue_law=issue,
        epoch_law=original.schedule.epoch_law)
    old=original.rhs
    rhs_law=old.phases[0].law if hasattr(old,'phases') else old.law
    phase_law=ConvolutionRhsPhaseLaw(schedule.source.identity,
        'original-independent-output-columns-no-intermediate-observer-full-phase-fence-v1',
        independent_output_columns=True,immutable_inputs=True,no_intermediate_observers=True,
        full_completion_before_reprepare=True)
    rhs=plan_convolution_rhs_phases(schedule,operand_rows=plan.operand_rows,
        law=rhs_law,phase_law=phase_law,input_rows=rows)
    if rhs is None:return None
    owner=ChannelRows(schedule,rows,rhs,original,plan,transfer);owner.revalidate();return owner

def emit(plan,builder,lower,*,command,issue,complete,input_pointer,mode=None):
    """Hook before061, after the parent's source/numerical/ABI admission.

    input_pointer is the original mapped immutable NHWC kernel argument. Only
    two new transport effects are encoded here; every weight/product/readout
    passes unchanged to the parent's ordinary non-draining issue callback.
    """
    owner=prepare(plan,command,mode=mode)
    if owner is None:return None
    if input_pointer is None or not callable(issue) or not callable(complete):
        raise ValueError('direct channel requires original source pointer and real target callbacks')
    proof=owner.revalidate()
    zero=builder.constant_i32_table([0]*((owner.transfer['maximum_columns']+3)//4))
    def target(kind,fields):
        if kind not in ('stage_input_rows','fill_input_rows'):
            return issue(kind,fields)
        if fields['block_stride_rows']!=owner.input_rows.row_pitch:
            raise ValueError('direct channel stage changes assigned packet block stride')
        if not 0<fields['columns']<=owner.transfer['maximum_columns'] or not 0<fields['rows']<=plan.primitive:
            raise ValueError('direct channel DMA extent exceeds selected law')
        fill=kind=='fill_input_rows'
        if fill and fields['value']!=0:raise ValueError('direct channel halo is not original integer zero')
        builder.config_ld(0 if fill else fields['row_stride_elements'],
            block_stride=fields['block_stride_rows'],id=0)
        ptr=zero if fill else builder.ptr_offset(input_pointer,fields['first_element'])
        builder.mvin(ptr,fields['local_row'],cols=fields['columns'],rows=fields['rows'],id=0)
    actual=emit_convolution_schedule(owner.schedule,builder,target,complete=complete,
        synchronous_completion=True,rhs=owner.rhs,input_rows=owner.input_rows)
    owner.revalidate()
    receipt=dict(actual,conv_channel_direct=dict(mode='resident_channel_rows',admission=proof,
        source_k_order='kernel_y,kernel_x,input_channel',host_payload_arithmetic=False,
        completion='real RHS preparation/phase drains, rolling ready/last-use drains, bounded compute epochs and unchanged original readout/final completion',timing='UNKNOWN'))
    if not hasattr(builder,'conv_channel_direct_receipts'):builder.conv_channel_direct_receipts=[]
    builder.conv_channel_direct_receipts.append(receipt)
    return receipt
