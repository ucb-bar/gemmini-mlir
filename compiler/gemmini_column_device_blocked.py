"""Source-only row-window traversal for unary original-column device readout.

Shared loop interchange proposes the row-window/column order. Independent unary
source and selected epoch laws admit it: each complete phase is disjoint, every
output has one writer, and configuration changes only after complete drains.
A window is an opt-in schedule bound; no cache capacity or traffic is assumed.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select, check_structure
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_effect_program
from gemmini_column_device import DeviceColumnPlan, lower_device_column_effect, selected_device_evidence
from gemmini_column_device_epochs import selected_unary_epoch_law
import gemmini_encode as E


def _digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _kernels(plan,live):
    rows,columns,tile=plan.rows,plan.columns,plan.row_tile
    chunks=(rows+tile-1)//tile
    bulk=(chunks-1)//live
    last=(chunks-1)%live+1
    def region(window_name,column_name,first,count,last_group,extent):
        chunk=first+Var('slot')
        height=select('slot',count-1,(rows-1)%tile+1,tile) if last_group else Const(tile)
        args=(('row',chunk*tile),('column',Var(column_name)),('rows',height),
              ('columns',Const(1)),('local_row',Var('slot')*tile))
        body=(Call('configure_column',(('column',Var(column_name)),)),Call('complete',()),
              Loop('slot',count,(Call('reload_i32',args),)),Call('complete',()),
              Loop('slot',count,(Call('readout_i8',args),)),Call('complete',()))
        # Configure inside the perfect pair before interchange. Each instance
        # owns its own unchanged complete window and numerical coefficient.
        return Loop(column_name,columns,(Loop(window_name,extent,body),))
    body=[];requests=[]
    if bulk:
        body.append(region('window','column',Var('window')*live,live,False,bulk))
        requests.append(LoopTransform('reorder',('column',)))
    body.append(region('tail_window','tail_column',Const(bulk*live),last,True,1))
    requests.append(LoopTransform('reorder',('tail_column',)))
    source=Kernel('original_column_window_source',(
        TensorArg('raw',(rows,columns),'i32','read'),TensorArg('output',(rows,columns),'i8','write')),tuple(body))
    actual=propose_loop_transforms(source,tuple(requests),index_bits=64,max_static_operations=256)
    if check_structure(actual):raise ValueError('blocked device source has invalid structure')
    return source,actual,tuple(requests)


@dataclass(frozen=True)
class BlockedColumnPlan:
    source: DeviceColumnPlan
    window_tiles: int
    live_tiles: int
    source_kernel: Kernel
    kernel: Kernel
    transforms: tuple
    law: dict
    implementation_sha256: str

    def revalidate(self):
        if type(self.source) is not DeviceColumnPlan or self.source.evidence!=selected_device_evidence():
            raise ValueError('blocked device source or selected evidence changed')
        if selected_unary_epoch_law()!=self.law:
            raise ValueError('blocked device phase law changed')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=self.implementation_sha256:
            raise ValueError('blocked device implementation changed')
        if type(self.window_tiles) is not int or not 1<=self.window_tiles<=1024:
            raise ValueError('blocked device window needs an explicit bound in 1..1024')
        chunks=(self.source.rows+self.source.row_tile-1)//self.source.row_tile
        capacity=self.source.evidence['resources']['accumulator_rows']//self.source.row_tile
        if self.live_tiles!=min(chunks,capacity,self.window_tiles):
            raise ValueError('blocked device private row window exceeds admitted capacity')
        if self.source.row_tile!=E.DIM or self.source.columns!=self.source.readout.columns:
            raise ValueError('blocked device source primitive/readout changed')
        source,actual,transforms=_kernels(self.source,self.live_tiles)
        if (self.source_kernel,self.kernel,self.transforms)!=(source,actual,transforms):
            raise ValueError('blocked device source/loop decisions/actual program changed')
        return self


def plan_device_columns_blocked(plan,*,window_tiles=8):
    """Admit a complete bounded row window before any producer/builder mutation."""
    if type(plan) is not DeviceColumnPlan:
        raise ValueError('blocked device readout requires an admitted original-column plan')
    if type(window_tiles) is not int or not 1<=window_tiles<=1024:
        raise ValueError('blocked device window bound must be in 1..1024')
    law=selected_unary_epoch_law()
    capacity=plan.evidence['resources']['accumulator_rows']//plan.row_tile
    chunks=(plan.rows+plan.row_tile-1)//plan.row_tile
    live=min(capacity,chunks,window_tiles)
    source,actual,transforms=_kernels(plan,live)
    return BlockedColumnPlan(plan,window_tiles,live,source,actual,transforms,law,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).revalidate()


def emit_device_columns_blocked(fn,raw,dst,plan,*,window_tiles=8,blocked_plan=None):
    """Emit every actual interchanged effect with real configuration/phase fences."""
    blocked=blocked_plan if blocked_plan is not None else plan_device_columns_blocked(plan,window_tiles=window_tiles)
    if type(blocked) is not BlockedColumnPlan or blocked.source!=plan:
        raise ValueError('blocked device plan differs from actual source')
    blocked.revalidate()
    from gemmini_column_readout import source_scale_bits_buffer
    scales=source_scale_bits_buffer(fn,plan.readout.column_scale_bits)
    control=FnBuilderControlAdapter(fn)
    E.fence(fn)
    E.config_load(fn,plan.columns*4,stream=0,scale=1.0,shrunk=False)
    E.fence(fn)
    def lower(kind,fields):
        return lower_device_column_effect(fn,raw,dst,plan,scales,control,kind,fields)
    receipt=emit_index_effect_program(blocked.kernel,control,lower,
        complete_epoch=lambda:E.fence(fn),completion_instruction='complete')
    E.fence(fn)
    chunks=(plan.rows+plan.row_tile-1)//plan.row_tile
    windows=(chunks+blocked.live_tiles-1)//blocked.live_tiles
    return dict(receipt,route='original_column_device_blocked',window_tiles=blocked.live_tiles,
        window_rows=blocked.live_tiles*plan.row_tile,windows=windows,
        source_kernel_sha256=plan.kernel.digest(),source_readout_identity=plan.readout.identity,
        loop_source_kernel_sha256=blocked.source_kernel.digest(),blocked_kernel_sha256=blocked.kernel.digest(),
        loop_transforms=[dict(action=t.action,path=t.path) for t in blocked.transforms],
        selected_law_sha256=_digest(blocked.law),
        original_output_elements=plan.rows*plan.columns,device_reload_pairs=plan.columns*chunks,
        configuration_drains=plan.columns*windows,phase_drains=2*plan.columns*windows,
        private_accumulator_rows=blocked.live_tiles*plan.row_tile,
        host_payload_loads=0,host_payload_stores=0,
        traversal='row_window,column,load_slots,complete,store_slots,complete',
        cache_effect='unknown; no selected cache facts or traffic reduction claimed',
        device_execution_verified=False)
