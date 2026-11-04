"""Exact convolution readout loop interchange over immutable raw accumulators.

The shared counted-loop transform moves each complete configuration palette
outside spatial row tiles. Target loads/stores still own disjoint ACC slots and
both real completion boundaries; no scale-dependent contraction is repeated.
"""
from dataclasses import dataclass
from math import prod
from merlin.sched.ir.expr import Sum, Scaled, evaluate
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, check_structure
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from . import isa
from .column_batches import _admission, _palette, _column_at


@dataclass(frozen=True)
class ConvPaletteReadout:
    rows: int
    columns: int
    groups: tuple
    source: Kernel
    transformed: Kernel
    block_tiles: int = 0

    def validate(self):
        if min(self.rows,self.columns)<=0 or check_structure(self.transformed):
            raise ValueError('invalid convolution readout domain')
        if sorted(c for _,cs in self.groups for c in cs)!=list(range(self.columns)):
            raise ValueError('readout palettes must partition all original columns')
        tiles=(self.rows+isa.DIM-1)//isa.DIM
        def epochs(var, extent, row):
            return tuple(Loop(var,extent,(Call('column_readout_epoch',(
                ('row_tile',row),('palette',Const(group)))),))
                for group in range(len(self.groups)))
        if self.block_tiles:
            full,tail=divmod(tiles,self.block_tiles)
            expected=(Loop('spatial_block',full,epochs('within_block',self.block_tiles,
                Var('spatial_block')*self.block_tiles+Var('within_block'))),)
            if tail:
                expected+=epochs('spatial_tile',tail,Var('spatial_tile')+full*self.block_tiles)
        else:
            expected=epochs('spatial_tile',tiles,Var('spatial_tile'))
        if self.transformed.body!=expected:
            raise ValueError('palette transformation changed the original readout index partition')
        # Each epoch loads immutable raw [row,column] once, writes a unique
        # original output element, and completes its private ACC ownership.
        # Their domain is a Cartesian partition, so the shared interchange
        # preserves effects without moving any pending access across a drain.


def plan_conv_palette_readout(m,n,attributes,block_tiles=0):
    groups=_palette(n,attributes)
    if not 1<len(groups)<=32:return None
    source=Kernel('conv_column_readout',(),(Loop('spatial_tile',(m+isa.DIM-1)//isa.DIM,
        (Loop('palette',len(groups),(Call('column_readout_epoch',(
            ('row_tile',Var('spatial_tile')),('palette',Var('palette')))),)),)),))
    tiles=(m+isa.DIM-1)//isa.DIM
    if block_tiles and tiles>=block_tiles:
        transforms=[LoopTransform('split',('spatial_tile',),factor=block_tiles,
                    names=('spatial_block','within_block')),
            LoopTransform('reorder',('spatial_block','within_block')),
            LoopTransform('unroll',('spatial_block','palette'),factor=len(groups))]
        if tiles%block_tiles:
            transforms.extend((LoopTransform('reorder',('spatial_tile',)),
                LoopTransform('unroll',('palette',),factor=len(groups))))
    else:
        block_tiles=0
        transforms=[LoopTransform('reorder',('spatial_tile',)),
            LoopTransform('unroll',('palette',),factor=len(groups))]
    transformed=propose_loop_transforms(source,tuple(transforms),
        index_bits=64,max_static_operations=512)
    result=ConvPaletteReadout(m,n,groups,source,transformed,block_tiles)
    result.validate()
    return result


def emit_conv_palette_readout(ctx,raw,dst,plan):
    plan.validate()
    m,n=plan.rows,plan.columns
    if (tuple(ctx.spec(raw)['shape'])!=(m,n) or ctx.spec(raw)['dtype']!='i32'
            or ctx.spec(dst)['dtype']!='i8' or ctx.buffer(raw)is ctx.buffer(dst)
            or prod(ctx.spec(dst)['shape'])!=m*n or ctx.spec(dst)['shape'][-1]!=n):
        raise ValueError('convolution readout requires disjoint original i32/i8 buffers')
    admission=_admission();fb=ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))
    def expression(value,env):
        if type(value)is Const:return fb.const(value.value)
        if type(value)is Var:return env[value.name]
        if type(value)is Scaled:return fb.mul_i(expression(value.expr,env),fb.const(value.factor))
        if type(value)is Sum:
            out=fb.const(0)
            for term in value.terms:out=fb.add_i(out,expression(term,env))
            return out
        raise ValueError('unsupported transformed readout expression')
    def emit_spatial(node,env,static):
        call=node.body[0]
        group=call.arg('palette').value
        command,columns=plan.groups[group]
        ctx.emit(*command)
        def epoch(row,rows):
            capacity=isa.ACC_ROWS//rows
            def batch(first,count):
                def transfer(slot,store):
                    column=_column_at(fb,fb.add_i(first,slot),columns)
                    offset=fb.add_i(fb.mul_i(row,fb.const(n)),column)
                    local=isa.acc_addr(fb.mul_i(slot,fb.const(rows)),fb=fb)
                    if store:
                        ctx.emit(*isa.mvout(ctx.ptr(dst,offset),local,rows,1,fb=fb))
                    else:
                        ctx.emit(*isa.mvin(ctx.ptr(raw,offset),local,rows,1,channel=2,fb=fb))
                fb.for_loop(0,count,1,lambda slot:transfer(slot,False),retain_loop=True)
                ctx.fence()
                fb.for_loop(0,count,1,lambda slot:transfer(slot,True),retain_loop=True)
                ctx.fence()
            whole=len(columns)//capacity*capacity
            if whole:
                fb.for_loop(0,whole,capacity,lambda first:batch(first,capacity),retain_loop=True)
            if whole<len(columns):batch(fb.const(whole),len(columns)-whole)
        row_expr=call.arg('row_tile')
        last_partial=bool(m%isa.DIM and evaluate(row_expr,{**static,node.var:node.extent-1})==m//isa.DIM)
        full=node.extent-int(last_partial)
        def emit_tile(tile,rows):
            row=fb.mul_i(expression(row_expr,{**env,node.var:tile}),fb.const(isa.DIM))
            epoch(row,rows)
        if full:
            fb.for_loop(0,full,1,lambda tile:emit_tile(tile,isa.DIM),retain_loop=True)
        if last_partial:emit_tile(fb.const(full),m%isa.DIM)
    for node in plan.transformed.body:
        if node.var=='spatial_block':
            # Only the final block can contain the partial original row tile.
            # Peel it before code emission so all row counts stay literal.
            peel=bool(m%isa.DIM and ((m+isa.DIM-1)//isa.DIM)%plan.block_tiles==0)
            full=node.extent-int(peel)
            def block(index,static_index):
                for child in node.body:
                    emit_spatial(child,{node.var:index},{node.var:static_index})
            if full:
                fb.for_loop(0,full,1,lambda index:block(index,0),retain_loop=True)
            if peel:block(fb.const(full),full)
        else:emit_spatial(node,{}, {})
    ctx.fence()
    plan.validate()
    receipt=dict(selection=('blocked_palette'+str(plan.block_tiles)) if plan.block_tiles else 'palette_major',source_schedule_sha256=plan.source.digest(),
        transformed_schedule_sha256=plan.transformed.digest(),
        source_schedule=plan.source.text(),transformed_schedule=plan.transformed.text(),
        passes=(['loop_schedule.split'] if plan.block_tiles else [])+['loop_schedule.reorder','loop_schedule.unroll','literal_config_hoist','backend.isa'],
        palette_groups=len(plan.groups),configuration_commands=len(plan.groups)*(((m+isa.DIM-1)//isa.DIM+plan.block_tiles-1)//plan.block_tiles if plan.block_tiles else 1),
        maximum_live_accumulator_rows=isa.ACC_ROWS,
        source_coverage='every original row,column exactly once; unchanged f32 configuration bits',
        completion='full load drain; identical-config stores; full drain before ACC reuse or scale change',
        admission=admission)
    ctx.metadata.setdefault('conv_column_readout_schedules',[]).append(receipt)
    return receipt


def emit_conv_column_readout(ctx,raw,dst,m,n,attributes):
    """Shared optional consumer for canonical and strided convolution products."""
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    from .column_batches import emit_batched_column_readout
    order=attributes.get('conv_column_readout_order',knob(
        'CONV_COLUMN_READOUT_ORDER',default='row_major',
        values=('row_major','palette_major','blocked_palette2','blocked_palette4','blocked_palette8'),root=Path(__file__).resolve().parents[1]))
    if order in ('palette_major','blocked_palette2','blocked_palette4','blocked_palette8'):
        block_tiles=int(order[-1]) if order.startswith('blocked_palette') else 0
        plan=plan_conv_palette_readout(m,n,attributes,block_tiles)
        if plan is not None:return emit_conv_palette_readout(ctx,raw,dst,plan)
    elif order!='row_major':
        raise ValueError('convolution column readout order must be row_major or palette_major')
    return emit_batched_column_readout(ctx,raw,dst,m,n,attributes)
