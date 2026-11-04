"""Bounded unary i32 reload/readout epochs over disjoint accumulator windows.

The loaded-sum epoch owner requires two source inputs and an accumulation; the
copy pipeline requires token completions and separately qualified concurrent
copies. Neither contract describes this unary source. This owner therefore uses
shared schedule IR and its checked counted-effect emission bridge directly.
It batches independent windows under one immutable original column scale, with
real whole-device drains between load/store phases and before any row reuse.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select, check_structure
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_effect_program
from gemmini_column_device import DeviceColumnPlan, selected_device_evidence, lower_device_column_effect
import gemmini_encode as E


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def selected_unary_epoch_law():
    evidence = selected_device_evidence()
    root = Path(__file__).resolve().parent/'data'
    order = json.loads((root/'COMMAND_ORDERING_SELECTION.json').read_text())['ordering']
    # Within a phase, every command addresses different physical rows. DMA
    # response ordering is irrelevant. Store release=completion is conservative
    # backpressure, not permission for outstanding same-cell reads/writes.
    for queue in ('load','store'):
        if order['same_queue'][queue]['release'] not in ('issue','completion'):
            raise ValueError('unary epochs lack finite same-queue issue semantics')
        if evidence['resources']['queue_entries'][queue] < 1:
            raise ValueError('unary epochs lack finite selected queue capacity')
    if order['completion_order']['store']['order'] != 'in_order':
        raise ValueError('unary epochs lack qualified store completion')
    return dict(selected=evidence,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        load_order=order['completion_order']['load']['order'],
        store_release=order['same_queue']['store']['release'],
        configuration='same load stride/width and same column scale throughout each epoch; complete before change',
        load_raw='disjoint row rectangles only; complete all reloads before any readout',
        store_war='complete all readouts before any private row reuse',
        progress='selected finite queues backpressure command issue; independent disjoint commands',
        ownership='exclusive private ACC rows after raw producer complete_device_fence')


def _kernel(source, live):
    rows,columns,tile=source.rows,source.columns,source.row_tile
    chunks=(rows+tile-1)//tile
    whole=(chunks-1)//live
    last=(chunks-1)%live+1
    def group(first,count,last_group):
        chunk=first+Var('slot')
        height=select('slot',count-1,(rows-1)%tile+1,tile) if last_group else Const(tile)
        fields=(('row',chunk*tile),('column',Var('column')),('rows',height),
                ('columns',Const(1)),('local_row',Var('slot')*tile))
        return (Loop('slot',count,(Call('reload_i32',fields),)),Call('complete',()),
                Loop('slot',count,(Call('readout_i8',fields),)),Call('complete',()))
    groups=(() if not whole else (Loop('epoch',whole,group(Var('epoch')*live,live,False)),))
    groups+=group(Const(whole*live),last,True)
    kernel=Kernel('original_column_unary_device_epochs',(
        TensorArg('raw',(rows,columns),'i32','read'),TensorArg('output',(rows,columns),'i8','write')),
        (Loop('column',columns,(Call('configure_column',(('column',Var('column')),)),
            Call('complete',()),*groups)),))
    errors=check_structure(kernel)
    if errors:raise ValueError('invalid unary epoch source '+repr(errors))
    return kernel


@dataclass(frozen=True)
class UnaryColumnEpochPlan:
    source: DeviceColumnPlan
    epoch_tiles: int
    live_tiles: int
    kernel: Kernel
    law: dict

    def revalidate(self):
        if type(self.source) is not DeviceColumnPlan:
            raise ValueError('unary epochs require an admitted original-column source')
        if selected_unary_epoch_law()!=self.law or selected_device_evidence()!=self.source.evidence:
            raise ValueError('unary epoch target or original source evidence changed')
        if type(self.epoch_tiles) is not int or self.epoch_tiles<=0:
            raise ValueError('unary epochs require a positive bound')
        capacity=self.source.evidence['resources']['accumulator_rows']//self.source.row_tile
        chunks=(self.source.rows+self.source.row_tile-1)//self.source.row_tile
        if self.live_tiles!=min(self.epoch_tiles,capacity,chunks):
            raise ValueError('unary epoch live windows exceed capacity or source')
        if self.source.row_tile!=E.DIM or self.source.readout.columns!=self.source.columns:
            raise ValueError('unary epoch primitive or source columns differ')
        if self.kernel.digest()!=_kernel(self.source,self.live_tiles).digest():
            raise ValueError('unary epoch source coordinates, phase order or completion changed')
        return self


def plan_device_columns_epochs(plan, *, epoch_tiles=64):
    """Admit all row ownership and complete source cover before any mutation."""
    if type(plan) is not DeviceColumnPlan:
        raise ValueError('unary epochs require an admitted original-column plan')
    if type(epoch_tiles) is not int or not 1<=epoch_tiles<=1024:
        raise ValueError('unary epoch issue bound must be in 1..1024')
    law=selected_unary_epoch_law()
    if plan.evidence!=law['selected']:
        raise ValueError('unary epoch source uses different selected evidence')
    capacity=plan.evidence['resources']['accumulator_rows']//plan.row_tile
    live=min(epoch_tiles,capacity,(plan.rows+plan.row_tile-1)//plan.row_tile)
    return UnaryColumnEpochPlan(plan,epoch_tiles,live,_kernel(plan,live),law).revalidate()


def emit_device_columns_epochs(fn,raw,dst,plan,*,epoch_tiles=64,epoch_plan=None):
    """Emit the exact unary epoch program; no source/format conversion on CPU.

    Call plan_device_columns_epochs first when the caller needs admission before
    emitting the raw producer. Otherwise this entry admits before its own first
    builder mutation. A supplied plan is revalidated against the actual source.
    """
    epochs=epoch_plan if epoch_plan is not None else plan_device_columns_epochs(plan,epoch_tiles=epoch_tiles)
    if type(epochs) is not UnaryColumnEpochPlan or epochs.source!=plan:
        raise ValueError('unary epoch plan is not for the supplied original source')
    epochs.revalidate()
    from gemmini_column_readout import source_scale_bits_buffer
    scales=source_scale_bits_buffer(fn,plan.readout.column_scale_bits)
    control=FnBuilderControlAdapter(fn)
    E.fence(fn)
    E.config_load(fn,plan.columns*4,stream=0,scale=1.0,shrunk=False)
    E.fence(fn)
    def lower(kind,fields):
        return lower_device_column_effect(fn,raw,dst,plan,scales,control,kind,fields)
    receipt=emit_index_effect_program(epochs.kernel,control,lower,
        complete_epoch=lambda:E.fence(fn),completion_instruction='complete')
    E.fence(fn)
    chunks=(plan.rows+plan.row_tile-1)//plan.row_tile
    groups=(chunks+epochs.live_tiles-1)//epochs.live_tiles
    return dict(receipt,route='original_column_unary_device_epochs',
        source_kernel_sha256=plan.kernel.digest(),source_readout_identity=plan.readout.identity,
        epoch_kernel_sha256=epochs.kernel.digest(),selected_law_sha256=_digest(epochs.law),
        live_tiles=epochs.live_tiles,private_accumulator_rows=epochs.live_tiles*plan.row_tile,
        original_output_elements=plan.rows*plan.columns,device_reload_pairs=chunks*plan.columns,
        phase_drains=2*groups*plan.columns,column_config_drains=plan.columns,
        serial_phase_drains=2*chunks*plan.columns,host_payload_loads=0,host_payload_stores=0,
        phase_order=['reload_disjoint_i32','complete','readout_disjoint_i8','complete'],
        configuration_capture='immutable throughout epoch; full drain before any change',
        device_execution_verified=False)
