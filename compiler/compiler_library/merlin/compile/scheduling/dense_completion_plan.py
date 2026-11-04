"""Consume existing schedule.plan/bind/apply_policy for completion decisions.

This is a post-construction decision over an actual counted Kernel, distinct
from the earlier blocking/placement/pipeline plan. The binding pins its exact
payload and frozen checking context; it grants no ordering/ownership law.
No new dialect, data-command recipe or ignored schedule annotation is added.
"""
from __future__ import annotations

from hashlib import sha256
from io import StringIO

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from merlin.xdsl_dialects.schedule import ApplyPolicyOp, BindOp, HandleType, PlanOp

PREFIX = 'dense_k_stream_completion:'
MODES = ('original', 'period', 'output_block')


def _binding(kernel, context_identity):
    if (type(context_identity) is not str or len(context_identity)!=64
            or any(c not in '0123456789abcdef' for c in context_identity)):
        raise ValueError('completion schedule requires a frozen original context SHA256')
    return f'dense-completion:{kernel.digest()}:{context_identity}'


def completion_plan_text(plan):
    stream=StringIO();Printer(stream=stream,print_generic_format=True).print_op(plan)
    return stream.getvalue()


def make_completion_plan(kernel, context_identity, *, mode):
    if mode not in MODES:
        raise ValueError('unknown executable completion policy')
    bind=BindOp(result_types=[HandleType()],properties={'target':StringAttr(_binding(kernel,context_identity))})
    apply=ApplyPolicyOp(operands=[[bind.handle]],properties={'policy':StringAttr(PREFIX+mode)})
    owner=PlanOp(properties={'sym_name':StringAttr('dense_completion')},
                 regions=[Region([Block([bind,apply])])])
    plan=ModuleOp([owner]);plan.verify()
    return plan


def read_completion_plan(plan, kernel, context_identity):
    """Verify and consume every actual decision; reject unknown effects."""
    if type(plan) is not ModuleOp:
        raise ValueError('completion consumer requires the original scheduling dialect ModuleOp')
    plan.verify()
    if (plan.attributes or plan.properties or len(plan.regions)!=1
            or len(plan.body.blocks)!=1 or plan.body.block.args):
        raise ValueError('completion plan has unknown module effects or bindings')
    members=list(plan.body.block.ops)
    if len(members)!=1 or type(members[0]) is not PlanOp:
        raise ValueError('completion requires exactly one schedule.plan')
    owner=members[0]
    if (owner.attributes or set(owner.properties)!={'sym_name'} or owner.operands or owner.results
            or len(owner.body.blocks)!=1 or owner.body.block.args):
        raise ValueError('completion schedule.plan contains unknown policy effects')
    ops=list(owner.body.block.ops)
    if len(ops)!=2 or type(ops[0]) is not BindOp or type(ops[1]) is not ApplyPolicyOp:
        raise ValueError('completion consumer supports only bind followed by apply_policy')
    bind,apply=ops
    if (bind.attributes or set(bind.properties)!={'target'} or bind.operands or bind.regions
            or bind.target.data!=_binding(kernel,context_identity)
            or apply.attributes or set(apply.properties)!={'policy'} or apply.regions or apply.results
            or tuple(apply.handles)!=(bind.handle,)):
        raise ValueError('completion decision has a different payload/context binding or unknown effects')
    name=apply.policy.data
    if not name.startswith(PREFIX) or (mode:=name[len(PREFIX):]) not in MODES:
        raise ValueError('completion decision names an unsupported executable policy')
    text=completion_plan_text(plan)
    return mode,dict(text=text,sha256=sha256(text.encode()).hexdigest(),
        source_kernel_sha256=kernel.digest(),context_sha256=context_identity,
        decisions_consumed=('bind','apply_policy'),mode=mode)
