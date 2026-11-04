"""Checked palette-major column readout with one complete store word per group.

Raw i32 values are immutable and output rectangles are disjoint. Every MVIN
uses a private accumulator row interval, all loads fully drain before MVOUT,
and all stores fully drain before accumulator reuse. Different store words
are separated by a real complete drain; configuration snapshots are not used.
"""
from dataclasses import dataclass
from merlin.sched.ir import Call, Kernel, Loop, TensorArg, Var, as_expr, add, mul, check_structure
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_effect_program
from . import isa

_LOAD='column_palette_load'
_STORE='column_palette_store'
_CONFIG='column_palette_complete_config'
_COMPLETE='column_palette_full_completion'


def _call(name, **args):
    return Call(name,tuple((key,as_expr(value))for key,value in args.items()))


def _program(m,n,groups,*,hoist):
    def batch(group,row,rows,first,count):
        slot=Var('column_slot')
        args=dict(group=group,position=add(first,slot),row=row,rows=rows,acc_row=mul(slot,rows))
        load=Loop('column_slot',count,(_call(_LOAD,**args),))
        store=Loop('column_slot',count,(_call(_STORE,**args),))
        config=()if hoist else(_call(_CONFIG,group=group),)
        return(load,_call(_COMPLETE),*config,store,_call(_COMPLETE))
    def columns(group,row,rows):
        capacity=isa.ACC_ROWS//rows
        batches,tail=divmod(len(groups[group][1]),capacity)
        result=()
        if batches:
            result+=(Loop('column_batch',batches,batch(group,row,rows,mul(Var('column_batch'),capacity),capacity)),)
        if tail:result+=batch(group,row,rows,batches*capacity,tail)
        return result
    full,tail=divmod(m,isa.DIM)
    def rows(group):
        result=()
        if full:result+=(Loop('spatial_tile',full,columns(group,mul(Var('spatial_tile'),isa.DIM),isa.DIM)),)
        if tail:result+=columns(group,full*isa.DIM,tail)
        return result
    if hoist:
        body=tuple(node for group in range(len(groups))for node in(_call(_CONFIG,group=group),*rows(group)))
    else:
        body=()
        if full:
            body+=(Loop('spatial_tile',full,tuple(node for group in range(len(groups))
                for node in columns(group,mul(Var('spatial_tile'),isa.DIM),isa.DIM))),)
        if tail:body+=tuple(node for group in range(len(groups))for node in columns(group,full*isa.DIM,tail))
    # Explicit source publication boundaries enclose both exact alternatives.
    return Kernel('column_palette_readout',(TensorArg('raw',(m,n),'i32','read'),
        TensorArg('dst',(m,n),'i8','write')),(_call(_COMPLETE),*body,_call(_COMPLETE)),
        (('offset_unit','elements'),('raw_state','immutable_i32')))


@dataclass(frozen=True)
class ColumnConfigPlan:
    m:int
    n:int
    groups:tuple
    source:object
    kernel:object

    def revalidate(self):
        if min(self.m,self.n)<=0 or self.m*self.n*4>=1<<63:
            raise ValueError('palette hoist requires bounded positive raw/output geometry')
        if not 1<=len(self.groups)<=32:
            raise ValueError('palette hoist retains the existing bounded palette admission')
        if sorted(c for _,columns in self.groups for c in columns)!=list(range(self.n)):
            raise ValueError('palette hoist must partition every original output column')
        for command,columns in self.groups:
            if not columns or len(command)!=3 or any(type(x)is not int for x in command):
                raise ValueError('palette hoist requires immutable complete configuration words')
        if (self.source!=_program(self.m,self.n,self.groups,hoist=False)
                or self.kernel!=_program(self.m,self.n,self.groups,hoist=True)
                or check_structure(self.source)or check_structure(self.kernel)):
            raise ValueError('palette hoist source offsets, drains or transformed kernel changed')
        return dict(source_kernel_sha256=self.source.digest(),kernel_sha256=self.kernel.digest(),
            semantic_law='immutable raw input; disjoint output rectangles; identical full store word within group',
            load_completion='full completion before every store batch',
            store_completion='full completion before every accumulator reuse and different configuration',
            publication='full completion at entry and exit; no external observer crosses this consumer')

    def emit(self,ctx,raw,dst,column_at):
        proof=self.revalidate()
        bridge=FnBuilderIndexAdapter(ctx.fb)
        def lower(tag,args):
            group=args['group']
            if type(group)is not int or not 0<=group<len(self.groups):
                raise ValueError('actual palette selector changed')
            if tag==_CONFIG:
                ctx.emit(*self.groups[group][0]);return
            if tag not in(_LOAD,_STORE):raise ValueError('unknown checked column effect')
            column=column_at(ctx.fb,bridge.value(args['position']),self.groups[group][1])
            offset=ctx.fb.add_i(ctx.fb.mul_i(bridge.value(args['row']),ctx.fb.const(self.n)),column)
            local=isa.acc_addr(bridge.value(args['acc_row']),fb=ctx.fb)
            # Rows are the actual primitive full/tail extent; never a wide MVIN.
            rows=args['rows']
            if type(rows)is not int or not 1<=rows<=isa.DIM:raise ValueError('invalid primitive row extent')
            if tag==_LOAD:
                ctx.emit(*isa.mvin(ctx.ptr(raw,offset),local,rows,1,channel=2,fb=ctx.fb))
            else:ctx.emit(*isa.mvout(ctx.ptr(dst,offset),local,rows,1,fb=ctx.fb))
        emission=emit_index_effect_program(self.kernel,bridge,lower,
            complete_epoch=ctx.fence,completion_instruction=_COMPLETE)
        self.revalidate()
        return dict(proof,emission=emission)


def plan_column_config_hoist(m,n,groups):
    groups=tuple((tuple(command),tuple(columns))for command,columns in groups)
    plan=ColumnConfigPlan(m,n,groups,_program(m,n,groups,hoist=False),_program(m,n,groups,hoist=True))
    plan.revalidate()
    return plan
