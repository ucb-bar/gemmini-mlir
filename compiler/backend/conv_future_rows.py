"""Disjoint next-row transport after last-panel launch, before its readout.

Reverse projection is the checked library lookahead SCF. The new operation's
region executes after current complete-K products have been issued; the same
real block-final drain completes its transport and current ACC readout.
"""
from dataclasses import dataclass,replace
import hashlib,json
from pathlib import Path
from xdsl.ir import Block,Region
from xdsl.irdl import IRDLOperation,irdl_op_definition,operand_def,result_def,prop_def,region_def,traits_def
from xdsl.dialects import arith,func,scf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.rewriter import Rewriter
from xdsl.traits import NoTerminator
from merlin.xdsl_dialects.rolling_window import AwaitOp,BeginOp,CompleteOp,ConsumeGroupOp,ConsumerType,ResidentType,StageRowOp
from merlin.compile.scheduling.rolling_lookahead import materialize_lookahead_rolling_window,verify_lookahead_rolling_window,RollingLookaheadAdmission
from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsResidency,plan_convolution_rhs_residency,emit_convolution_rhs_preparation
from merlin.compile.scheduling.window_transfer import emit_convolution_compute_epoch_column_block
from .conv_input_dependencies import DependencyOp,DependencyType,text,_law
from . import isa

@irdl_op_definition
class FutureConsumeOp(IRDLOperation):
    name='conv_future.consume_last_panel'
    traits=traits_def(NoTerminator())
    resident=operand_def(DependencyType)
    image=operand_def()
    output_y=operand_def()
    consumer=result_def(ConsumerType)
    output_rows=prop_def(IntegerAttr)
    future=region_def('single_block')


def transform(module,reverse=False):
    m=module.clone()
    for op in tuple(m.walk()):
        if not reverse and type(op)is ConsumeGroupOp:
            body=Block();prior=op.prev_op;move=[]
            while prior is not None and type(prior)not in (AwaitOp,CompleteOp,ConsumeGroupOp):
                if type(prior)is BeginOp:break
                move.append(prior);prior=prior.prev_op
            if prior is not None and type(prior)is BeginOp:
                for item in reversed(move):op.parent.detach_op(item);body.add_op(item)
            new=FutureConsumeOp.create(operands=[op.resident,op.image,op.output_y],result_types=[ConsumerType()],properties=dict(op.properties),regions=[Region(body)])
            Rewriter.replace_op(op,new)
        elif reverse and type(op)is FutureConsumeOp:
            if op.attributes or set(op.properties)!={'output_rows'}:raise ValueError('unprojected future consumer effect')
            for item in tuple(op.future.block.ops):
                op.future.block.detach_op(item);op.parent.insert_op_before(item,op)
            Rewriter.replace_op(op,ConsumeGroupOp.create(operands=[op.resident,op.image,op.output_y],result_types=[ConsumerType()],properties=dict(op.properties)))
    for op in tuple(m.walk()):
        if not reverse and type(op)is AwaitOp:
            Rewriter.replace_op(op,DependencyOp.create(operands=[op.transfer],result_types=[DependencyType()]))
        elif reverse and type(op)is DependencyOp:
            if op.attributes or op.properties or op.regions:raise ValueError('unprojected pending transport effect')
            Rewriter.replace_op(op,AwaitOp.create(operands=[op.transfer],result_types=[ResidentType()]))
    # The actual carried readiness handles change type with the boundary.
    for operation in m.walk():
        for value in operation.results:
            if type(value.type)is (DependencyType if reverse else ResidentType):
                value._type=ResidentType() if reverse else DependencyType()
        for region in operation.regions:
            for block in region.blocks:
                for value in block.args:
                    if type(value.type)is (DependencyType if reverse else ResidentType):
                        value._type=ResidentType() if reverse else DependencyType()
    m.verify();return m

def future_admission(schedule):
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    cross={(e['older'],e['younger'],e['space']):e for e in record['ordering']['cross_queue']}
    for reader in ('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad'):
        edge=cross[(reader,'MVIN.spad_addr','scratchpad')]
        if edge.get('tracked')is not True or edge.get('disjoint')!='issue' or edge.get('overlap')!='completion':
            raise ValueError('future-row requires qualified disjoint execute-reader to MVIN issue law')
    law=RollingLookaheadAdmission(schedule.issue_law.hardware_identity,'original-final-panel-before-readout-transport-v1',True,True,True,True,True)
    law.require();return law

@dataclass(frozen=True)
class Prepared:
    schedule:object
    rows:object
    rhs:object
    readout:object
    module:object
    identity:str
    source_identity:str
    law:object
    lookahead:object
    def revalidate(self):
        self.schedule.verify_current();self.rows.revalidate();self.rhs.verify_current();self.lookahead.require()
        if self.lookahead!=future_admission(self.schedule):raise ValueError("future-row transport admission changed")
        if _law(self.schedule,self.rows,self.rhs)!=self.law:raise ValueError('future-row original target law changed')
        if hashlib.sha256(text(self.module).encode()).hexdigest()!=self.identity:raise ValueError('future-row actual IR changed')
        projected=transform(self.module,True)
        verify_lookahead_rolling_window(projected,self.rows.rolling_plan(),output_rows=1)
        if hashlib.sha256(text(projected).encode()).hexdigest()!=self.source_identity:raise ValueError('future row source projection changed')
        if len(self.schedule.column_blocks)!=1:raise ValueError('future row needs one complete last-panel readout block')
        for block in self.schedule.column_blocks:self.readout.require(self.schedule.source.plan,self.schedule.source.plan.row_tile,block)


def prepare(schedule,rows,rhs,readout):
    if type(rhs)is not ConvolutionRhsResidency or readout is None:return None
    try:
        w=rows.window
        if w.output_spatial[0]<2 or len(schedule.column_blocks)!=1:return None
        # Modulo is injective on the union of adjacent windows, including halo.
        slots=rows.slots+w.stride[0]
        rows=replace(rows,slots=slots,private_rows=slots*rows.rows_per_source_row)
        if not rows.disjoint_from(schedule.source.plan.rhs_row,schedule.source.plan.channel_tile):return None
        rhs=plan_convolution_rhs_residency(schedule,operand_rows=isa.SPAD_ROWS,law=rhs.law,rhs_base=(rows.end+15)//16*16)
        if rhs is None or not rows.disjoint_from(rhs.layout.rhs_base,rhs.layout.rhs_rows):return None
        law=_law(schedule,rows,rhs)
        admit=future_admission(schedule)
        admit.require();original=materialize_lookahead_rolling_window(rows.rolling_plan(),output_rows=1)
        verify_lookahead_rolling_window(original,rows.rolling_plan(),output_rows=1)
        actual=transform(original)
        p=Prepared(schedule,rows,rhs,readout,actual,hashlib.sha256(text(actual).encode()).hexdigest(),hashlib.sha256(text(original).encode()).hexdigest(),law,admit);p.revalidate();return p
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError):return None


def emit(p,builder,effect,complete):
    p.revalidate();schedule=p.schedule;rows=p.rows;plan=schedule.source.plan;oh,ow=plan.window.output_spatial
    lower,finish=emit_convolution_rhs_preparation(p.rhs,builder,effect,complete=complete)
    def consume(image,y,future):
        def column(x,count,last):
            issued=[False]
            def output(kind,fields):
                if kind=='store' and last and not issued[0]:
                    future();issued[0]=True
                lower(kind,fields)
            output_row=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,oh),y),ow),x)
            emit_convolution_compute_epoch_column_block(plan,builder,output,complete,output_row=output_row,rows=count,output_tiles=schedule.column_blocks[0],law=schedule.retention_law,issue_law=schedule.issue_law,epoch_law=schedule.epoch_law,compute_pairs=schedule.choice.compute_pairs,input_rows=rows,readout_law=p.readout)
            if last and not issued[0]:raise ValueError('future transport lacks actual final-panel producer/readout seam')
        whole,tail=divmod(ow,plan.row_tile);prefix=whole if tail else whole-1
        if prefix:builder.loop(0,prefix,1,lambda x:column(builder.op('mul',x,plan.row_tile),plan.row_tile,False))
        column(prefix*plan.row_tile,tail or plan.row_tile,True)
    def walk(block,inherited):
        env=dict(inherited)
        for op in block.ops:
            if type(op)is arith.ConstantOp:env[op.result]=op.value.value.data
            elif type(op)in(arith.AddiOp,arith.SubiOp,arith.MuliOp,arith.RemUIOp):env[op.result]=builder.op({arith.AddiOp:'add',arith.SubiOp:'sub',arith.MuliOp:'mul',arith.RemUIOp:'urem'}[type(op)],env[op.lhs],env[op.rhs])
            elif type(op)is arith.CmpiOp:env[op.result]=builder.cmp({0:'eq',6:'ult'}[op.predicate.value.data],env[op.lhs],env[op.rhs])
            elif type(op)is scf.ForOp:
                def body(index,carried=()):return walk(op.body.block,{**env,**dict(zip(op.body.block.args,(index,*carried),strict=True))})
                if op.iter_args:
                    values=builder.loop_carried(env[op.lb],env[op.ub],env[op.step],tuple(env[x]for x in op.iter_args),body);env.update(zip(op.results,values,strict=True))
                else:builder.loop(env[op.lb],env[op.ub],env[op.step],body)
            elif type(op)is BeginOp:env[op.transfer]=rows.base
            elif type(op)is StageRowOp:rows.emit_stage(builder,{k:env[getattr(op,k)]for k in('transfer','image','input_row','source_byte','cache_byte','valid')},lower)
            elif type(op)is DependencyOp:env[op.dependency]=env[op.transfer]
            elif type(op)is FutureConsumeOp:
                consume(env[op.image],env[op.output_y],lambda:walk(op.future.block,env));env[op.consumer]=rows.base
            elif type(op)is CompleteOp:complete('input_last_use',dict(local_row=rows.base))
            elif type(op)is scf.YieldOp:return tuple(env[v]for v in op.arguments)
            elif type(op)is not func.ReturnOp:raise ValueError('unsupported future-row operation '+op.name)
        return ()
    walk(tuple(p.module.body.block.ops)[0].body.block,{});finish();p.revalidate()
    return dict(mechanism='disjoint_future_row_last_panel_transport',actual_ir_sha256=p.identity,source_ir_sha256=p.source_identity,input_cache=rows.record(),first_reader_law=p.law.record(),completion='real original last-panel block drain completes readout and future row; every input-last-use retained',timing='UNKNOWN')
