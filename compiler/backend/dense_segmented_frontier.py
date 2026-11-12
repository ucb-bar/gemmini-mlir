"""Checked bounded future-slot issue groups, without new execution laws.

The input already has a current M-compute loop then one future RHS load per N
column. Grouping moves only those actual future-slot loads before a bounded group of current
columns. Original segmented source, ownership and finite epoch checks decide
legality; no token wait, independent launch or synchronous DMA is introduced.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compiler.schedule.blocks import Compute, Load, Preload
from merlin.compiler.schedule.compact_dense import _digest
from merlin.compiler.schedule.dense_panel_pipeline import _calls
from merlin.compiler.schedule.dense_segmented_pipeline import (
    DenseSegmentedPipelineResult, verify_dense_segmented_pipeline,
)
from merlin.compiler.schedule.tile_load_coalescing import _plain, _replace_call
from merlin.compiler.kernel_ir.ir import Call, Const, Loop, Var, check_structure


@dataclass(frozen=True)
class DenseGroupedFrontierResult:
    original: object
    kernel: object
    realization: dict
    checked_kernel_sha256: str
    checked_report_sha256: str
    checked_owner_sha256: str

    @property
    def admission(self): return self.original.admission
    @property
    def output_readout(self): return self.original.output_readout
    @property
    def bias_source(self): return self.original.bias_source
    @property
    def readout_qualification_identity(self): return self.original.readout_qualification_identity

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest()!=self.checked_kernel_sha256
                or _digest(self.realization['grouped_frontier_proof'])!=self.checked_report_sha256
                or sha256(Path(__file__).read_bytes()).hexdigest()!=self.checked_owner_sha256):
            raise ValueError('checked grouped frontier or its actual proof changed')

    def materialize_scf(self):
        from merlin.compiler.schedule.scf_commands import materialize_scf_commands
        self.revalidate(); result=materialize_scf_commands(self.kernel); self.revalidate()
        return result

    def emit_llvm(self,builder,lower_call):
        from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate(); bridge=FnBuilderIndexAdapter(builder)
        result=emit_index_program(self.kernel,bridge,
            lambda name,args:lower_call(name,{k:bridge.value(v) for k,v in args.items()}))
        self.revalidate(); return result


def group_future_frontier(original,problem,block,resources,storage,adapter,*,group_extent=None):
    if group_extent is None:
        from merlin.compiler.selection.knobs import knob
        group_extent=knob('DENSE_SEGMENTED_FRONTIER_GROUP',default=4,
            values=(2,4,8),root=Path(__file__).resolve().parents[1])
    if type(group_extent) is not int or group_extent not in (2,4,8):
        raise ValueError('segmented frontier group must be exactly 2, 4 or 8 columns')
    if type(original) is not DenseSegmentedPipelineResult:
        raise ValueError('grouped frontier requires its original typed segmented owner')
    original.revalidate()
    used=set(); changes=[]
    def names(body):
        for statement in body:
            if type(statement) is Loop:
                used.add(statement.var); names(statement.body)
    names(original.kernel.body)
    def fresh(stem):
        while stem in used: stem+='_' 
        used.add(stem); return stem
    def substitute(body,old,value):
        return tuple(replace(s,body=substitute(s.body,old,value)) if type(s) is Loop
                     else _replace_call(s,old,value) for s in body)
    def walk(body):
        out=[]
        for statement in body:
            if type(statement) is not Loop:
                out.append(statement); continue
            s=statement
            mixed=(s.extent>=2 and len(s.body)==2 and type(s.body[0]) is Loop
                   and type(s.body[1]) is Call and _plain(s.body[1]))
            if mixed:
                calls=tuple(_calls(s.body[0].body))
                mixed=(len(calls)==2 and all(_plain(c) for c in calls)
                       and type(adapter.project(calls[0])) is Preload
                       and type(adapter.project(calls[1])) is Compute
                       and type(adapter.project(s.body[1])) is Load
                       and adapter.project(s.body[1]).role=='weight')
            if not mixed:
                out.append(replace(s,body=walk(s.body))); continue
            groups,tail=divmod(s.extent,group_extent)
            if not groups:
                out.append(s); continue
            suffix=str(group_extent)
            band=fresh(s.var+'_group'+suffix); future=fresh(s.var+'_future'+suffix); current=fresh(s.var+'_current'+suffix)
            try:
                loads=substitute((s.body[1],),s.var,Var(band)*group_extent+Var(future))
                computes=substitute((s.body[0],),s.var,Var(band)*group_extent+Var(current))
            except ValueError:
                # Variable-dependent column tails retain the complete incumbent
                # loop; never discard a selector or guess its reachable arm.
                out.append(s); continue
            grouped=Loop(band,groups,(Loop(future,group_extent,loads),Loop(current,group_extent,computes)))
            remainder=tuple(c for n in range(groups*group_extent,s.extent)
                            for c in (*substitute((s.body[1],),s.var,Const(n)),
                                      *substitute((s.body[0],),s.var,Const(n))))
            out.extend((grouped,*remainder))
            changes.append(dict(original_column_loop=s.var,columns=s.extent,
                groups=groups,tail_columns=tail,order='future_rhs_group_then_current_compute'))
        return tuple(out)
    kernel=replace(original.kernel,body=walk(original.kernel.body))
    if not changes:return original,dict(applied=False,group_extent=group_extent,
        reason='no exact selector-free mixed frontier of the requested extent')
    if check_structure(kernel):raise ValueError('grouped frontier changed lexical completion ownership')
    proof=verify_dense_segmented_pipeline(kernel,problem,block,resources,storage,adapter,
        original.admission,static_budget=128,timeout_ms=30000,
        output_readout=original.output_readout,bias_source=original.bias_source)
    receipt=dict(applied=True,source_sha256=original.kernel.digest(),kernel_sha256=kernel.digest(),
        group_extent=group_extent,regions=changes,
        completion='original explicit full drains and conservative issued-band bounds retained')
    realization=dict(original.realization,grouped_frontier=receipt,grouped_frontier_proof=proof,
        kernel_sha256=kernel.digest())
    result=DenseGroupedFrontierResult(original,kernel,realization,kernel.digest(),_digest(proof),
        sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate(); return result,receipt
