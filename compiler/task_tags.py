"""Attach source task ownership to the imported LLVM dialect operations."""
from io import StringIO
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr
from xdsl.dialects.llvm import LLVM
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.parser import Parser
from xdsl.printer import Printer


class CastSyntax(IRDLOperation):
    argument = operand_def()
    result = result_def()

    @classmethod
    def parse(cls,parser):
        position = parser.pos
        value = parser.parse_unresolved_operand()
        parser.parse_punctuation(':')
        source = parser.parse_type()
        parser.parse_keyword('to')
        target = parser.parse_type()
        return cls.create(operands=[parser.resolve_operand(value,source)],result_types=[target])


CASTS = [irdl_op_definition(type('Cast_'+name,(CastSyntax,),{'name':'llvm.'+name}))
         for name in ('fptosi','fptoui','uitofp')]


class LLVMPrinter(Printer):
    def print_op(self, operation):
        previous = self.print_generic_format
        self.print_generic_format = operation.name in {'llvm.inline_asm','llvm.fptosi','llvm.fptoui','llvm.uitofp'}
        try:
            super().print_op(operation)
        finally:
            self.print_generic_format = previous


def attach(text, buffer, source):
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(LLVM)
    for operation in CASTS:
        context.load_op(operation)
    module = Parser(context,text).parse_module()
    current = -1
    for function in module.body.block.ops:
        if function.name != 'llvm.func' or not function.body.blocks:
            continue
        for block in function.body.blocks:
            for operation in list(block.ops):
                assembly = operation.properties.get('asm_string')
                if operation.name == 'llvm.inline_asm' and assembly is not None and assembly.data.startswith('# merlin.task '):
                    current = int(assembly.data.split()[-1])
                    block.erase_op(operation)
                    continue
                if operation.name == 'llvm.return':
                    operation.attributes['merlin.global_task'] = IntegerAttr(-2,64)
                    continue
                operation.attributes['merlin.global_task'] = IntegerAttr(current,64)
                if current >= 0 and operation.name == 'llvm.inline_asm':
                    operation.attributes['merlin.source_op_index'] = IntegerAttr(current,64)
                    original = source.operations[current]
                    region = original.attributes.get('prov.region_id')
                    if region is not None:
                        operation.attributes['prov.region_id'] = region
    stream = StringIO()
    LLVMPrinter(stream=stream).print_op(module)
    return stream.getvalue()+'\n'


def finalize_plan(buffer):
    ranges = buffer.pop('_source_task_ranges')
    buffer.pop('_source_instruction_tasks',None)
    plan = buffer['params']['global_program_plan']
    count = 1
    for task,(start,end) in zip(plan['tasks'],ranges):
        if end <= start:
            raise ValueError('Source operation has no executable lowering: '+str(task['source_op_indices']))
        task['instruction_start'] = count
        count += end-start
        task['instruction_end'] = count
    plan['prologue_instruction_range'] = [0,1]
    plan['epilogue_instruction_range'] = [count,count+1]
    plan['schedule_instruction_count'] = count+1
