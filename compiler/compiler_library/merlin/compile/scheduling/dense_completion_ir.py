"""Executable completion decision in the original schedule dialect.

This is a post-storage consumer of an existing checked command kernel. It does
not reconstruct data calls to match a different pipeline or change its kind.
"""
from xdsl.dialects.builtin import ModuleOp, StringAttr, BoolAttr
from xdsl.ir import Block, Region
from merlin.xdsl_dialects.schedule import PlanOp, BindOp, HandleType, CompletionEpochsOp
from .compact_dense import _digest
from .dense_completion import _pin
from .schedule_ir import dense_schedule_text, parse_dense_schedule_plan


def _binding(original,context):
    return 'dense-completion:'+_digest((original.kernel.digest(),_pin(context)))


def make_dense_completion_plan(original,context,*,strategy='ordered_k'):
    bind = BindOp.create(properties={'target':StringAttr(_binding(original,context))},result_types=[HandleType()])
    decision = CompletionEpochsOp.create(operands=[bind.handle],properties={
        'strategy':StringAttr(strategy),'scope':StringAttr('output_block'),
        'preserve_readout':BoolAttr.from_bool(True)})
    plan = PlanOp.create(properties={'sym_name':StringAttr('dense_completion')},
        regions=[Region(Block([bind,decision]))])
    module = ModuleOp([plan])
    module.verify()
    return module


def consume_dense_completion_plan(plan,original,context):
    if type(plan) is str:
        plan = parse_dense_schedule_plan(plan)
    if (type(plan) is not ModuleOp or plan.attributes or plan.properties
            or len(plan.regions)!=1 or len(plan.body.blocks)!=1 or plan.body.block.args):
        raise ValueError('completion plan requires an ordinary unannotated module')
    plan.verify()
    ops = list(plan.body.block.ops)
    if len(ops)!=1 or type(ops[0]) is not PlanOp:
        raise ValueError('completion plan requires one original schedule.plan')
    owner = ops[0]
    if (set(owner.properties)!={'sym_name'} or owner.attributes or owner.operands or owner.results
            or len(owner.body.blocks)!=1 or owner.body.block.args):
        raise ValueError('completion plan has unsupported policy or region effects')
    decisions = list(owner.body.block.ops)
    if len(decisions)!=2 or type(decisions[0]) is not BindOp or type(decisions[1]) is not CompletionEpochsOp:
        raise ValueError('completion plan supports only bind then completion_epochs')
    bind,decision = decisions
    if (bind.target.data!=_binding(original,context) or bind.attributes
            or set(bind.properties)!={'target'} or bind.operands or bind.regions
            or decision.target is not bind.handle or decision.attributes or decision.results or decision.regions
            or set(decision.properties)!={'strategy','scope','preserve_readout'}):
        raise ValueError('completion plan differs from the current original command/storage binding')
    return decision.strategy.data,dense_schedule_text(plan)
