"""Original primitive-width row staging ordered by each first device reader.

This is a distinct lowered IR: Await/Consume are replaced by Dependency/Read.
It does not bind the synchronous rolling emitter's await callback to a no-op.
The exact reverse projection proves the original counted source/lifetime law;
the selected RAW law supplies the only weakened edge. Complete remains a real
full drain before every slot rewrite, including image backedges.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

from xdsl.dialects import arith, func, scf
from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, operand_def, result_def
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from merlin.xdsl_dialects.rolling_window import (
    AwaitOp, BeginOp, CompleteOp, ConsumeOp, ConsumerType, ResidentType, StageRowOp, TransferType,
)
from merlin.compile.scheduling.rolling_window import materialize_rolling_window, verify_rolling_window
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsResidency, emit_convolution_rhs_preparation,
)
from merlin.compile.scheduling.window_transfer import emit_convolution_compute_epoch_column_block
from . import isa


@irdl_attr_definition
class DependencyType(ParametrizedAttribute, TypeAttribute):
    name = 'conv_input.dependency'


@irdl_op_definition
class DependencyOp(IRDLOperation):
    name = 'conv_input.dependency'
    transfer = operand_def(TransferType)
    dependency = result_def(DependencyType)


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = 'conv_input.read'
    dependency = operand_def(DependencyType)
    image = operand_def()
    output_y = operand_def()
    consumer = result_def(ConsumerType)


def text(module):
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def _rewrite(module, reverse=False):
    actual = module.clone()
    changes = 0
    for op in tuple(actual.walk()):
        if reverse and type(op) in (DependencyOp, ReadOp):
            if op.attributes or op.properties or op.regions:
                raise ValueError('input dependency operation contains unprojected effects')
        if not reverse and type(op) is AwaitOp:
            Rewriter.replace_op(op, DependencyOp.create(operands=[op.transfer], result_types=[DependencyType()]))
            changes += 1
        elif not reverse and type(op) is ConsumeOp:
            Rewriter.replace_op(op, ReadOp.create(operands=[op.resident, op.image, op.output_y], result_types=[ConsumerType()]))
        elif reverse and type(op) is DependencyOp:
            Rewriter.replace_op(op, AwaitOp.create(operands=[op.transfer], result_types=[ResidentType()]))
            changes += 1
        elif reverse and type(op) is ReadOp:
            Rewriter.replace_op(op, ConsumeOp.create(operands=[op.dependency, op.image, op.output_y], result_types=[ConsumerType()]))
    if changes < 1:
        raise ValueError('input dependency transformation found no actual transport boundary')
    actual.verify()
    return actual


@dataclass(frozen=True)
class FirstReaderLaw:
    """Exact selected RAW premises, bound to primitive original encodings."""
    ordering_identity: str
    facts_identity: str
    records_json: str
    producer_funct: int = 2
    lhs_reader_funct: int = 4
    weight_reader_funct: int = 6

    def record(self):
        if (self.producer_funct,self.lhs_reader_funct,self.weight_reader_funct)!=(2,4,6):
            raise ValueError('first-reader law changed actual MVIN/COMPUTE_PRELOADED/PRELOAD encodings')
        return dict(ordering_sha256=self.ordering_identity,base_facts_sha256=self.facts_identity,
            producer_funct=self.producer_funct,lhs_reader_funct=self.lhs_reader_funct,
            weight_reader_funct=self.weight_reader_funct,**json.loads(self.records_json))


def _law(schedule, rows, rhs):
    # The incumbent independently verifies this exact fact selection for its
    # compute epochs. Recheck the actual MVIN/first-reader records here too.
    record = json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    record_facts=record['base_facts_sha256']
    ordering = record['ordering']
    digest = hashlib.sha256(json.dumps(ordering, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != 'ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780':
        raise ValueError('input first-reader law differs from selected ordering')
    if record['base_facts_sha256'] != 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af':
        raise ValueError('input first-reader law differs from selected RTL facts')
    cross = {(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    selected = []
    for reader in ('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad'):
        edge = cross[('MVIN.spad_addr',reader,'scratchpad')]
        if edge.get('overlap') != 'completion' or not (edge.get('tracked') is True or (edge.get('tracked') is None and edge.get('disjoint') == 'completion')):
            raise ValueError('input first-reader completion dependency is not qualified')
        selected.append(dict(edge))
    issue = schedule.issue_law
    epoch = schedule.epoch_law
    if (issue is None or epoch is None or not issue.local_range_dependencies
            or not issue.ordered_load_configuration or not issue.finite_queue_backpressure
            or not epoch.local_read_after_write or not epoch.complete_device_drain
            or not epoch.operand_consumption_retirement):
        raise ValueError('input first-reader stream lacks original issue/completion premises')
    if (rows.stage_channels != isa.DIM or rows.stage_rows > isa.DIM
            or rows.channel_tile != isa.DIM or rows.channel_segments is not None
            or rows.image_windows != 1 or rows.base != 0 or rows.row_pitch >= 65536
            or not rows.disjoint_from(rhs.layout.rhs_base,rhs.layout.rhs_rows)):
        raise ValueError('input first-reader law requires original disjoint primitive-width staging')
    if not 0 < rows.row_step < 65536:
        raise ValueError('input reader row-step is not the original encoded extent')
    rows.revalidate()
    rhs.verify_current()
    record = dict(selected_first_reader_dependencies=selected,
        original_issue_law=issue.__dict__,original_compute_law=epoch.__dict__,
        primitive_rows=isa.DIM,primitive_columns=isa.DIM,
        partition='disjoint zero/valid/zero X bands; disjoint channel blocks at row_pitch; no duplicate local writes',
        reuse='every original complete remains a real full drain before slot/image reuse',
        configuration='original ordered load-channel capture; no wide transfers or new configuration values')
    return FirstReaderLaw(digest,record_facts,json.dumps(record,sort_keys=True))


@dataclass(frozen=True)
class Prepared:
    schedule: object
    rows: object
    rhs: object
    readout_law: object
    module: object
    identity: str
    source_module_identity: str
    admission: FirstReaderLaw

    def revalidate(self):
        self.admission.record()
        self.schedule.verify_current()
        self.rows.revalidate()
        self.rhs.verify_current()
        for block in self.schedule.column_blocks:
            self.readout_law.require(self.schedule.source.plan,self.schedule.source.plan.row_tile,block)
        if hashlib.sha256(text(self.module).encode()).hexdigest() != self.identity:
            raise ValueError('actual dependent-input IR changed')
        projected = _rewrite(self.module, reverse=True)
        verify_rolling_window(projected, self.rows.rolling_plan())
        if hashlib.sha256(text(projected).encode()).hexdigest() != self.source_module_identity:
            raise ValueError('dependent-input IR lost exact original source projection')
        if _law(self.schedule,self.rows,self.rhs) != self.admission:
            raise ValueError('dependent-input source/target admission changed')


def prepare(schedule, rows, rhs, readout_law):
    """Before alternative effects, admit full-fit immutable RHS and exact IR."""
    if type(rhs) is not ConvolutionRhsResidency or readout_law is None:
        return None
    try:
        admission = _law(schedule,rows,rhs)
        for block in schedule.column_blocks:
            readout_law.require(schedule.source.plan,schedule.source.plan.row_tile,block)
        original = materialize_rolling_window(rows.rolling_plan())
        verify_rolling_window(original,rows.rolling_plan())
        actual = _rewrite(original)
        result = Prepared(schedule,rows,rhs,readout_law,actual,
            hashlib.sha256(text(actual).encode()).hexdigest(),
            hashlib.sha256(text(original).encode()).hexdigest(),admission)
        result.revalidate()
        return result
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError):
        return None


def emit(prepared,builder,effect,complete):
    """Lower actual dependent-input SCF through original product/readout APIs."""
    prepared.revalidate()
    schedule=prepared.schedule;rows=prepared.rows;p=schedule.source.plan
    lower,finish=emit_convolution_rhs_preparation(prepared.rhs,builder,effect,complete=complete)
    oh,ow=p.window.output_spatial
    def consume(image,y):
        def column(x,count):
            output_row=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,oh),y),ow),x)
            for block in schedule.column_blocks:
                emit_convolution_compute_epoch_column_block(p,builder,lower,complete,
                    output_row=output_row,rows=count,output_tiles=block,
                    law=schedule.retention_law,issue_law=schedule.issue_law,
                    epoch_law=schedule.epoch_law,compute_pairs=schedule.choice.compute_pairs,
                    input_rows=rows,readout_law=prepared.readout_law)
        whole,tail=divmod(ow,p.row_tile)
        if whole:builder.loop(0,whole,1,lambda x:column(builder.op('mul',x,p.row_tile),p.row_tile))
        if tail:column(whole*p.row_tile,tail)
    def walk(block,inherited):
        env=dict(inherited)
        for op in block.ops:
            if type(op) is arith.ConstantOp:env[op.result]=op.value.value.data
            elif type(op) in (arith.AddiOp,arith.SubiOp,arith.MuliOp,arith.RemUIOp):
                name={arith.AddiOp:'add',arith.SubiOp:'sub',arith.MuliOp:'mul',arith.RemUIOp:'urem'}[type(op)]
                env[op.result]=builder.op(name,env[op.lhs],env[op.rhs])
            elif type(op) is arith.CmpiOp:env[op.result]=builder.cmp('ult',env[op.lhs],env[op.rhs])
            elif type(op) is scf.ForOp:
                builder.loop(env[op.lb],env[op.ub],env[op.step],lambda index:walk(op.body.block,{**env,op.body.block.args[0]:index}))
            elif type(op) is BeginOp:env[op.transfer]=rows.base
            elif type(op) is StageRowOp:
                rows.emit_stage(builder,{name:env[getattr(op,name)]for name in
                    ('transfer','image','input_row','source_byte','cache_byte','valid')},lower)
            elif type(op) is DependencyOp:
                # Pending DMA is ordered at each actual reader, not declared
                # globally complete and not substituted for an Await callback.
                env[op.dependency]=env[op.transfer]
            elif type(op) is ReadOp:
                if env[op.dependency]!=rows.base:raise ValueError('first-reader ownership handle changed')
                consume(env[op.image],env[op.output_y]);env[op.consumer]=rows.base
            elif type(op) is CompleteOp:
                if env[op.consumer]!=rows.base:raise ValueError('consumer last-use handle changed')
                complete('input_last_use',dict(local_row=rows.base))
            elif type(op) not in (scf.YieldOp,func.ReturnOp):
                raise ValueError('unsupported dependent-input effect '+op.name)
    walk(tuple(prepared.module.body.block.ops)[0].body.block,{})
    finish()
    prepared.revalidate()
    return dict(mechanism='original_row_ring_first_reader_dependencies',
        actual_ir_sha256=prepared.identity,source_ir_sha256=prepared.source_module_identity,
        admission=prepared.admission.record(),completion='real block-final, input-last-use, RHS preparation and invocation drains',
        source_projection='identical original source addresses, primitive input/weight/product/store effects; only input-ready edge changes',timing='UNKNOWN')
