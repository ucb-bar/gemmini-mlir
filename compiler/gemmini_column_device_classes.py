"""Exact source-scale classes for original-order unary device readout.

Source coordinates are selected from exact scalar configuration constants. Raw
payload stays i32 and is moved with accumulator DMA. Shared loop hoisting proposes invariant store
configuration once per class/window; exact bit equality and full phase drains
provide the independent target/source admission for that proposal.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import struct

from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select, check_structure
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_effect_program
from gemmini_column_device import DeviceColumnPlan, lower_device_column_effect, selected_device_evidence
from gemmini_column_device_epochs import selected_unary_epoch_law
from gemmini_column_grouping import plan_grouping
from gemmini_column_constants import source_configuration_word
import gemmini_encode as E


@dataclass(frozen=True)
class SourceScaleClass:
    bits: int
    columns: tuple
    table_first: int
    affine_step: int | None


def _classes(source):
    scales=tuple(struct.unpack('<f',struct.pack('<I',b))[0] for b in source.readout.column_scale_bits)
    grouping=plan_grouping(scales,columns=source.columns)
    grouping.verify()
    classes=[];first=0
    while first<source.columns:
        b=grouping.grouped_bits[first];stop=first+1
        while stop<source.columns and grouping.grouped_bits[stop]==b:stop+=1
        columns=grouping.permutation[first:stop]
        step=columns[1]-columns[0] if len(columns)>1 else 1
        affine=step if all(c==columns[0]+i*step for i,c in enumerate(columns)) else None
        classes.append(SourceScaleClass(b,columns,first,affine));first=stop
    if tuple(c for group in classes for c in group.columns)!=grouping.permutation:
        raise ValueError('scale classes lost original column bijection')
    return grouping,tuple(classes)


def _kernels(source,live,classes):
    rows,columns,tile=source.rows,source.columns,source.row_tile
    chunks=(rows+tile-1)//tile;bulk=(chunks-1)//live;last=(chunks-1)%live+1
    requests=[]
    def region(name,extent,count,last_group):
        body=[]
        first=Var(name)*live if not last_group else Const(bulk*live)
        for index,group in enumerate(classes):
            member=f'{name}_member_{index}'
            column=(('column',Const(group.columns[0])+Var(member)*group.affine_step)
                    if group.affine_step is not None else ('column_table_index',Const(group.table_first)+Var(member)))
            height=select('slot',count-1,(rows-1)%tile+1,tile) if last_group else Const(tile)
            fields=(('row',(first+Var('slot'))*tile),column,('rows',height),
                    ('columns',Const(1)),('local_row',Var('slot')*tile))
            calls=(Call('configure_column',(('column',Const(group.columns[0])),)),Call('complete',()),
                   Loop('slot',count,(Call('reload_i32',fields),)),Call('complete',()),
                   Loop('slot',count,(Call('readout_i8',fields),)),Call('complete',()))
            body.append(Loop(member,len(group.columns),calls))
            # Both the actual original scale configuration and its full drain
            # are invariant across exactly the checked equal-bit source class.
            requests.extend((LoopTransform('hoist',(name,member),index=0),
                             LoopTransform('hoist',(name,member),index=0)))
        return Loop(name,extent,tuple(body))
    body=[]
    if bulk:body.append(region('window',bulk,live,False))
    body.append(region('tail_window',1,last,True))
    original=Kernel('original_column_class_source',(
        TensorArg('raw',(rows,columns),'i32','read'),TensorArg('output',(rows,columns),'i8','write')),tuple(body))
    # The generic mechanism bounds each proposal at32 decisions. Classes are
    # at most16, and bulk/tail each get their own recorded bounded application.
    actual=original
    for start in range(0,len(requests),32):
        actual=propose_loop_transforms(actual,tuple(requests[start:start+32]),index_bits=64,max_static_operations=4096)
    if check_structure(actual):raise ValueError('invalid source scale-class program')
    return original,actual,tuple(requests)


@dataclass(frozen=True)
class ClassColumnPlan:
    source: DeviceColumnPlan
    window_tiles: int
    live_tiles: int
    grouping: object
    classes: tuple
    source_kernel: Kernel
    kernel: Kernel
    transforms: tuple
    law: dict
    implementation_sha256: str

    def revalidate(self):
        if type(self.source) is not DeviceColumnPlan or self.source.evidence!=selected_device_evidence():
            raise ValueError('class device source evidence changed')
        if selected_unary_epoch_law()!=self.law:
            raise ValueError('class device phase law changed')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=self.implementation_sha256:
            raise ValueError('class device implementation changed')
        if type(self.window_tiles) is not int or not 1<=self.window_tiles<=1024:
            raise ValueError('class device window bound is outside1..1024')
        chunks=(self.source.rows+self.source.row_tile-1)//self.source.row_tile
        capacity=self.source.evidence['resources']['accumulator_rows']//self.source.row_tile
        if self.live_tiles!=min(chunks,capacity,self.window_tiles) or self.source.row_tile!=E.DIM:
            raise ValueError('class device physical window exceeds admitted rows')
        grouping,classes=_classes(self.source)
        if not 1<len(classes)<=16 or len(classes)>=self.source.columns:
            raise ValueError('class device source lacks useful bounded repeated scale classes')
        if (grouping,classes)!=(self.grouping,self.classes):
            raise ValueError('class device source scale bits or column bijection changed')
        original,actual,transforms=_kernels(self.source,self.live_tiles,classes)
        if (original,actual,transforms)!=(self.source_kernel,self.kernel,self.transforms):
            raise ValueError('class device source, hoist decisions or emitted program changed')
        return self


def plan_device_columns_classes(plan,*,window_tiles=8):
    """Return admitted classes, or None when no bounded useful class reuse exists."""
    if type(plan) is not DeviceColumnPlan:
        raise ValueError('class device readout needs admitted original-column source')
    if type(window_tiles) is not int or not 1<=window_tiles<=1024:
        raise ValueError('class device window bound must be in1..1024')
    grouping,classes=_classes(plan)
    if not 1<len(classes)<=16 or len(classes)>=plan.columns:return None
    law=selected_unary_epoch_law()
    capacity=plan.evidence['resources']['accumulator_rows']//plan.row_tile
    live=min(capacity,(plan.rows+plan.row_tile-1)//plan.row_tile,window_tiles)
    original,actual,transforms=_kernels(plan,live,classes)
    return ClassColumnPlan(plan,window_tiles,live,grouping,classes,original,actual,transforms,law,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).revalidate()


def lower_class_device_effect(fn,raw,dst,source,table,scales,control,kind,fields):
    """Select immutable original coordinates then use the common target encoder."""
    if 'column_table_index' in fields:
        fields=dict(fields)
        fields['column']=source_configuration_word(control,
            fields.pop('column_table_index'),table)
    return lower_device_column_effect(fn,raw,dst,source,scales,control,kind,fields)


def emit_device_columns_classes(fn,raw,dst,plan,*,window_tiles=8,class_plan=None):
    """Emit the admitted class traversal; caller retains epoch64 on None admission."""
    classes=class_plan if class_plan is not None else plan_device_columns_classes(plan,window_tiles=window_tiles)
    if classes is None:return None
    if type(classes) is not ClassColumnPlan or classes.source!=plan:
        raise ValueError('class device plan differs from actual original source')
    classes.revalidate()
    from gemmini_column_readout import source_scale_bits_buffer
    scales=source_scale_bits_buffer(fn,plan.readout.column_scale_bits)
    table=(classes.grouping.permutation
           if any(c.affine_step is None for c in classes.classes) else None)
    control=FnBuilderControlAdapter(fn)
    E.fence(fn);E.config_load(fn,plan.columns*4,stream=0,scale=1.0,shrunk=False);E.fence(fn)
    def lower(kind,fields):
        return lower_class_device_effect(fn,raw,dst,plan,table,scales,control,kind,fields)
    receipt=emit_index_effect_program(classes.kernel,control,lower,
        complete_epoch=lambda:E.fence(fn),completion_instruction='complete')
    E.fence(fn)
    chunks=(plan.rows+plan.row_tile-1)//plan.row_tile
    windows=(chunks+classes.live_tiles-1)//classes.live_tiles
    return dict(receipt,route='original_column_device_scale_classes',window_tiles=classes.live_tiles,
        source_kernel_sha256=plan.kernel.digest(),source_readout_identity=plan.readout.identity,
        class_source_kernel_sha256=classes.source_kernel.digest(),class_kernel_sha256=classes.kernel.digest(),
        grouping_identity=classes.grouping.identity,source_scale_classes=len(classes.classes),
        affine_classes=sum(c.affine_step is not None for c in classes.classes),
        metadata_classes=sum(c.affine_step is None for c in classes.classes),
        original_output_elements=plan.rows*plan.columns,device_reload_pairs=plan.columns*chunks,
        configuration_drains=len(classes.classes)*windows,phase_drains=2*plan.columns*windows,
        private_accumulator_rows=classes.live_tiles*plan.row_tile,
        host_payload_loads=0,host_payload_stores=0,
        selected_law_sha256=hashlib.sha256(json.dumps(classes.law,sort_keys=True).encode()).hexdigest(),
        configuration='shared hoist of exact equal-bit class config+drain; no motion across payload phases',
        device_execution_verified=False)
