"""Typed original-row destinations for the ordinary dense terminal emitter.

The dense schedule, contraction, scale words and completion policy are retained.
Only terminal row intervals are intersected with a checked output partition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.ir import Block

from . import isa
from .dense_readout import store_tile


class MappedReadoutRefusal(ValueError):
    """Optional mapping rejected before any dense instruction is emitted."""


def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class RowDestination:
    dst: str
    begin: int
    end: int
    source_index: int | None


def row_intersections(partitions,row,rows,acc):
    """Exact integer projection used to check every actual scheduled store."""
    fragments=[]
    for p in partitions:
        lo=max(row,p.begin);hi=min(row+rows,p.end)
        if lo<hi:fragments.append((p.dst,lo-p.begin,hi-lo,acc+lo-row))
    return tuple(fragments)


class ScatteredReadout:
    def __init__(self,partitions,m,n,dtype,attributes):
        self.partitions=tuple(partitions)
        self.m,self.n,self.dtype=m,n,dtype
        self.attributes=json.loads(json.dumps(attributes))
        self.binding=_identity(self._binding())
        self.certificate=None

    def _binding(self):
        return dict(partitions=[vars(p) for p in self.partitions],m=self.m,n=self.n,
            dtype=self.dtype,attributes=self.attributes)

    def _revalidate(self,ctx,attributes):
        if self.binding!=_identity(self._binding()) or attributes!=self.attributes:
            raise ValueError('scattered readout original partition or attributes changed')
        for p in self.partitions:
            if ctx.spec(p.dst)['shape']!=[p.end-p.begin,self.n] or ctx.spec(p.dst)['dtype']!=self.dtype:
                raise ValueError('scattered readout typed output changed')

    def bind_schedule(self,ctx,schedule,attributes,*,max_calls=1_000_000,max_cells=1_048_576):
        """Check actual store operands/coverage and retain every epoch drain.

        Refuse before any dense emission if exact finite verification exceeds
        its explicit compiler-work budget. Library revalidation proves producer
        readiness and reuse; this certificate checks the unchanged terminal
        scope and its output mapping independently on the actual call stream.
        """
        from merlin.sched.ir import Loop,instances,evaluate
        from .dense_store_epochs import BEGIN,END,STORE
        from .dense_panel_epochs import READY_BEGIN
        self._revalidate(ctx,attributes)
        schedule.revalidate()
        kernel=schedule.kernel
        def count(items):return sum(x.extent*count(x.body) if isinstance(x,Loop) else 1 for x in items)
        calls=count(kernel.body)
        if calls>max_calls or self.m*self.n>max_cells:
            raise ValueError('scattered readout finite proof budget exceeded')
        covered=bytearray(self.m*self.n);stores=fragments=0;epoch=False
        full_rows = self.m <= isa.DIM
        expected=isa.config_st(self.n*(4 if self.dtype=='i32' else 1),
            scale=self.attributes.get('acc_scale',1.0),activation=int(bool(
                self.attributes.get('relu',False) or 'relu' in self.attributes.get('epilogue',()))))
        for call,env in instances(kernel):
            if call.instr in (BEGIN,READY_BEGIN):
                if epoch or tuple(schedule.command)!=expected:
                    raise ValueError('scattered readout store epoch configuration changed')
                epoch=True;continue
            if call.instr==END:
                if not epoch:raise ValueError('scattered readout unmatched epoch drain')
                epoch=False;continue
            if call.instr not in ('gemmini_mvout',STORE):
                if epoch:raise ValueError('scattered readout ACC producer inside store epoch')
                continue
            if (call.instr==STORE)!=epoch:
                raise ValueError('scattered readout store completion scope changed')
            a={key:evaluate(value,env) for key,value in call.args}
            row,col,rows,cols,acc=(a[key] for key in ('dram_row','dram_col','rows','cols','acc'))
            if not (0<rows<=isa.DIM and 0<cols<=isa.DIM and 0<=row<=self.m-rows
                    and 0<=col<=self.n-cols and 0<=acc<=isa.ACC_ROWS-rows):
                raise ValueError('scattered readout terminal bounds changed')
            full_rows = full_rows and row == 0 and rows == self.m
            split=row_intersections(self.partitions,row,rows,acc)
            if sum(f[2] for f in split)!=rows:
                raise ValueError('scattered readout row coverage changed')
            for dst,local,height,local_acc in split:
                p=next(p for p in self.partitions if p.dst==dst)
                if not (0<=local<=p.end-p.begin-height and acc<=local_acc<=acc+rows-height):
                    raise ValueError('scattered readout fragment escaped source tile')
            for r in range(row,row+rows):
                for c in range(col,col+cols):
                    index=r*self.n+c
                    if covered[index]:raise ValueError('scattered readout duplicated live output')
                    covered[index]=1
            stores+=1;fragments+=len(split)
        if epoch or not all(covered):raise ValueError('scattered readout missing completion or live output')
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        static_selection=knob('ATTENTION_STATIC_FRAGMENTS',default='off',values=('off','on'),root=Path(__file__).resolve().parent.parent)
        self.static_fragments = bool(full_rows and stores and static_selection=='on')
        self.static_binding = (self.binding,self.m,kernel.digest()) if self.static_fragments else None
        self.certificate=dict(binding_sha256=self.binding,kernel_sha256=kernel.digest(),
            checked_actual_calls=calls,original_terminal_stores=stores,split_terminal_stores=fragments,
            output_elements=len(covered),write_multiplicity=1,
            row_destinations=[vars(p) for p in self.partitions],
            proof='actual finite terminal calls: bounded complete unique source coverage; row intersections preserve every ACC lane and original output column; unchanged epoch configuration and closing drains',
            completion='ordinary store_tile fences retained per fragment; configured store fragments remain inside their original checked epoch and closing full drain',
            eliminated_joined_result_bytes=self.m*self.n*(4 if self.dtype=='i32' else 1))
        self.certificate['static_full_row_fragments'] = dict(applied=self.static_fragments,all_actual_terminal_rows_zero=bool(full_rows),actual_rows=self.m if full_rows else None,terminal_calls=stores,proof='finite actual stream: every terminal store has dram_row=0,rows=M<=DIM; exact constant member boundaries preserve ACC lane and output column')
        return self.certificate

    def emit(self,ctx,a,n,attributes,*,configured=False):
        """Split one checked terminal tile using the existing terminal encoder."""
        self._revalidate(ctx,attributes)
        if self.certificate is None or n!=self.n:
            raise ValueError('scattered readout lacks actual schedule certificate')
        fb=ctx.fb
        def value(x):return fb.const(x) if isinstance(x,int) else x
        if self.static_fragments:
            if self.static_binding!=(self.binding,self.m,self.certificate['kernel_sha256']):
                raise ValueError('static full-row fragment binding changed')
            self._emit_static(ctx,a,n,attributes,configured=configured)
            return
        row,rows,acc=(value(a[key]) for key in ('dram_row','rows','acc'))
        end=fb.add_i(row,rows)
        source_index=ctx.current_source_index
        for p in self.partitions:
            lo=fb.smax(row,fb.const(p.begin));hi=fb.smin(end,fb.const(p.end))
            body,after=Block(),Block()
            fb.region.add_block(body);fb.region.add_block(after)
            valid=fb.add(llvm.ICmpOp(lo,hi,IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
            fb.add(llvm.CondBrOp(valid,body,[],after,[]));fb.blk=body
            local=fb.sub_i(lo,fb.const(p.begin));height=fb.sub_i(hi,lo)
            local_acc=fb.add_i(acc,fb.sub_i(lo,row))
            ctx.current_source_index=p.source_index if p.source_index is not None else source_index
            if configured:
                offset=fb.add_i(fb.mul_i(local,fb.const(n)),value(a['dram_col']))
                ctx.emit(*isa.mvout(ctx.ptr(p.dst,offset),isa.acc_addr(local_acc,
                    full=self.dtype=='i32',fb=fb),height,value(a['cols']),fb=fb))
            else:
                store_tile(ctx,p.dst,local_acc,height,a['cols'],n,local,a['dram_col'],attributes)
            fb.add(llvm.BrOp(after));fb.blk=after
        ctx.current_source_index=source_index


    def _emit_static(self,ctx,a,n,attributes,*,configured):
        """Same terminal encoder/configuration/drains with proven constant rows."""
        fb=ctx.fb
        def value(x):return fb.const(x) if isinstance(x,int) else x
        source_index=ctx.current_source_index
        for p in self.partitions:
            acc=value(a['acc'])
            local_acc=acc if p.begin==0 else fb.add_i(acc,fb.const(p.begin))
            height=p.end-p.begin
            ctx.current_source_index=p.source_index if p.source_index is not None else source_index
            if configured:
                ctx.emit(*isa.mvout(ctx.ptr(p.dst,value(a['dram_col'])),isa.acc_addr(local_acc,
                    full=self.dtype=='i32',fb=fb),height,value(a['cols']),fb=fb))
            else:
                store_tile(ctx,p.dst,local_acc,height,a['cols'],n,0,a['dram_col'],attributes)
        ctx.current_source_index=source_index


def make_readout(ctx,members,*,m,k,n,attributes):
    """Admit original distinct typed outputs without allocating joined output."""
    if any(type(v) is not int or v<=0 for v in (m,k,n)):
        raise ValueError('scattered readout requires positive static joined geometry')
    if ctx.cb.get('params',{}).get('tensor_aliases'):
        raise ValueError('scattered readout refuses explicit pointer aliases')
    if (len(members)<2 or len(members)>32 or attributes.get('bias')
            or attributes.get('accumulator_init') or attributes.get('acc_scale_by_column') is not None
            or attributes.get('requant_shift') is not None
            or any(key.startswith('pool') for key in attributes)
            or any(e not in ('acc_scale','relu') for e in attributes.get('epilogue',()))):
        raise ValueError('scattered readout requires uniform scalar terminal readout')
    outputs=[row['dst'] for row in members]
    inputs={row[key] for row in members for key in ('lhs','rhs')}
    if len(set(outputs))!=len(outputs) or set(outputs)&inputs:
        raise ValueError('scattered readout needs distinct outputs disjoint from inputs')
    rhs=members[0]['rhs'];dtype=ctx.spec(outputs[0])['dtype'];cursor=0;parts=[]
    if dtype not in ('i8','i32') or attributes.get('output_dtype',dtype)!=dtype:
        raise ValueError('scattered readout output dtype changed')
    if dtype=='i32' and (attributes.get('acc_scale',1.0)!=1.0 or attributes.get('relu') or 'relu' in attributes.get('epilogue',())):
        raise ValueError('raw scattered readout cannot apply narrowing arithmetic')
    strip=('merlin.source_op_index','merlin.global_task','prov.region_id')
    canonical={key:value for key,value in attributes.items() if key not in strip}
    for member in members:
        left,right,out=(ctx.spec(member[key]) for key in ('lhs','rhs','dst'))
        attrs={key:value for key,value in member.get('attributes',{}).items() if key not in strip}
        if (member['rhs']!=rhs or attrs!=canonical or left['dtype']!='i8' or right['dtype']!='i8'
                or len(left['shape'])!=2 or left['shape'][1]!=k or right['shape']!=[k,n]
                or out['shape']!=[left['shape'][0],n] or out['dtype']!=dtype or left['shape'][0]<=0):
            raise ValueError('scattered readout original product binding changed')
        rows=left['shape'][0]
        parts.append(RowDestination(member['dst'],cursor,cursor+rows,member.get('attributes',{}).get('merlin.source_op_index')))
        cursor+=rows
    if cursor!=m or min(m,k,n)<=0 or max(m*k,m*n)>=1<<32:
        raise ValueError('scattered readout joined geometry changed')
    return ScatteredReadout(parts,m,n,dtype,attributes)
