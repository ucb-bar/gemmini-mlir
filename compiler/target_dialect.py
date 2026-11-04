"""Typed logical schedule commands, prior to storage and ISA encoding."""
import json
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class CommandOp(IRDLOperation):
    name = "gemmini_schedule.command"
    opcode = attr_def(StringAttr)
    operands_json = attr_def(StringAttr)
    options_json = attr_def(StringAttr)

    def verify_(self):
        if not self.opcode.data:
            raise VerifyException("schedule command must have an opcode")
        for field in (self.operands_json, self.options_json):
            if not isinstance(json.loads(field.data), dict):
                raise VerifyException("schedule operands/options must be dictionaries")


GemminiSchedule = Dialect("gemmini_schedule", [CommandOp], [])


def schedule_module(cb):
    commands = [CommandOp.create(attributes={
        "opcode": StringAttr(c["opcode"]),
        "operands_json": StringAttr(json.dumps(c.get("operands") or {}, sort_keys=True)),
        "options_json": StringAttr(json.dumps(c.get("attributes") or {}, sort_keys=True)),
    }) for c in cb["commands"]]
    return ModuleOp(commands, attributes={"gemmini_schedule.tensors": StringAttr(json.dumps(cb.get("tensors", {}))),
                                        "gemmini_schedule.params": StringAttr(json.dumps(cb.get("params", {})))})
