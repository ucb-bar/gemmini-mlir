"""Real xDSL LLVM emitter for ordinary Gemmini commands and scalar control.

The exposed FnBuilder is the admitted host preparation/control builder; callers
remain responsible for deciding which tensor operations execute on the device.
"""
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, ModuleOp, StringAttr, TensorType, i64
from xdsl.ir import SSAValue, Operation, Block, Region
from xdsl.printer import Printer
from merlin.llvmlower.host_builder import FnBuilder, PTR, INT_TYPES
from merlin.compile.tuning_knobs import knob
from isa_encoder import insn_asm


class TerminalInlineAsm(llvm.InlineAsmOp):
    # Generic form keeps source/task attributes in the stock MLIR grammar.
    # The upstream xDSL custom inline-asm printer places attributes too early.
    print = Operation.print


class LLVMEmitter:
    def __init__(self, arg_names, *, symbol='kernel_entry',
                 completion_symbol='kernel_complete', completion_asm='fence',
                 entry_context=None, control_unroll=None):
        if entry_context is None:
            entry_context = knob('entry_context', default='retained',
                                 values=('retained', 'flush0'), root=Path(__file__).parent)
        if entry_context not in ('retained', 'flush0'):
            raise ValueError('entry_context must be retained or flush0')
        if control_unroll is None:
            control_unroll = knob('control_unroll', default='incumbent',
                                  values=('incumbent', 4, 8), root=Path(__file__).parent)
        if not (control_unroll == 'incumbent' or
                type(control_unroll) is int and control_unroll in (4, 8)):
            raise ValueError('control_unroll must be incumbent, 4, or 8')
        self.control_unroll = control_unroll
        # Budget predicted additional LLVM operations, shared across the kernel.
        # Metadata remains an optimizer request, not a promised machine size.
        self._control_unroll_remaining = 4096
        self.control_unroll_decisions = []
        self.arg_names = list(arg_names)
        if len(set(self.arg_names)) != len(self.arg_names):
            raise ValueError('duplicate logical pointer argument')
        self.b = FnBuilder([PTR] * len(self.arg_names))
        self.args = dict(zip(self.arg_names, self.b.entry.args))
        self.symbol = symbol
        self.completion_symbol = completion_symbol
        self.completion_asm = completion_asm
        self._scope = {}
        self._opened = False
        self.entry_context = entry_context
        self._entry_context_initialized = False
        self._unroll = None
        self._finished = None
        self._globals = []
        self._global_names = set()
        # FnBuilder funnels arithmetic/control through add. Tag at creation,
        # rather than rescanning the entire accumulated graph for every task.
        original_add = self.b.add
        def add(op):
            folded = self._fold_unrolled_integer(op) if self._unroll is not None else None
            return folded if folded is not None else original_add(self._tag(op))
        self.b.add = add

    def _fold_unrolled_integer(self, op):
        """Evaluate pure, literal i64 address arithmetic in unrolled bodies.

        Keeping these addresses literal also makes the retained command stream
        describe the same concrete scratch rows that LLVM's optimizer executes.
        Runtime operands and all device effects remain ordinary operations.
        """
        supported = {'llvm.add', 'llvm.sub', 'llvm.mul', 'llvm.and',
                     'llvm.or', 'llvm.xor', 'llvm.udiv', 'llvm.urem',
                     'llvm.shl', 'llvm.lshr'}
        if op.name not in supported or len(op.results) != 1 or op.results[0].type != i64:
            return None
        if len(op.operands) != 2:
            return None
        values = []
        mask = (1 << 64) - 1
        for operand in op.operands:
            owner = operand.owner
            if not isinstance(owner, Operation) or owner.name != 'llvm.mlir.constant':
                return None
            value = owner.properties.get('value')
            if not isinstance(value, IntegerAttr):
                return None
            values.append(value.value.data & mask)
        a, b = values
        if op.name in {'llvm.udiv', 'llvm.urem'} and b == 0:
            return None
        if op.name in {'llvm.shl', 'llvm.lshr'} and b >= 64:
            return None
        if op.name == 'llvm.add': result = a + b
        elif op.name == 'llvm.sub': result = a - b
        elif op.name == 'llvm.mul': result = a * b
        elif op.name == 'llvm.and': result = a & b
        elif op.name == 'llvm.or': result = a | b
        elif op.name == 'llvm.xor': result = a ^ b
        elif op.name == 'llvm.udiv': result = a // b
        elif op.name == 'llvm.urem': result = a % b
        elif op.name == 'llvm.shl': result = a << b
        else: result = a >> b
        result &= mask
        if result >= 1 << 63:
            result -= 1 << 64
        return self.b.const(result).owner

    def value(self, operand):
        return operand if isinstance(operand, SSAValue) else self.b.const(int(operand))

    def _tag(self, op, **attrs):
        selected = dict(self._scope)
        selected.update({k: v for k, v in attrs.items() if v is not None})
        for name, value in selected.items():
            if value is not None:
                key = 'prov.region_id' if name == 'region_id' else 'merlin.' + name
                if key not in op.attributes:
                    op.attributes[key] = StringAttr(str(value)) if name == 'region_id' else IntegerAttr(int(value), i64)
        return op

    @contextmanager
    def task(self, task_index, *, source_op_index=None, region_id=None):
        previous = self._scope
        self._scope = {'global_task': task_index, 'source_op_index': source_op_index, 'region_id': region_id}
        try:
            yield self
        finally:
            self._scope = previous

    def insn(self, op, rs1=0, rs2=0, *, task_index=None, source_op_index=None, region_id=None):
        # gSIM trace qualification requires an opening device-visible fence.
        # Emit lazily so refused/empty stubs do not acquire target effects,
        # and attach the first command's active task identity.
        if not self._opened:
            self.fence()
        if not self._entry_context_initialized:
            self._entry_context_initialized = True
            if self.entry_context == 'flush0':
                # Reproduce the public reference's device entry sequence:
                # three_model_elves_analysis/resnet50.elf.dis:342551,
                # PC 0x8010b7ba, word 0x0ef7b07b; both operands are zero.
                # Selected facts name funct 7 FLUSH_CMD but do not establish
                # its detailed TLB effect. Do not infer additional mode bits.
                self.insn('flush', 0, 0, task_index=task_index,
                          source_op_index=source_op_index, region_id=region_id)
        instr = TerminalInlineAsm(insn_asm(op), 'r,r,~{memory}',
                                 [self.value(rs1), self.value(rs2)], has_side_effects=True)
        self._tag(instr, global_task=task_index, source_op_index=source_op_index, region_id=region_id)
        return self.b.add(instr)

    def fence(self):
        self._opened = True
        return self.b.add(self._tag(TerminalInlineAsm('fence', '~{memory}', [], has_side_effects=True)))

    def ptr(self, base, index=0, dtype='i8'):
        if isinstance(base, str):
            base = self.args[base]
        return self.b.gep(base, self.value(index), INT_TYPES[dtype])

    def address(self, ptr):
        return self.b.add(llvm.PtrToIntOp(ptr, i64)).results[0]

    def alloca(self, count, dtype='i8', *, alignment=64):
        # Allocate in entry so dynamic tile loops never grow the stack per trip.
        op = self._tag(llvm.AllocaOp(self.value(count), INT_TYPES[dtype], alignment))
        terminator = self.b.entry.last_op
        if terminator is not None and terminator.has_trait(llvm.IsTerminator):
            self.b.entry.insert_op_before(op, terminator)
        else:
            self.b.entry.add_op(op)
        return op.results[0]

    def _global(self, count, dtype, name, *, values=None, alignment=64):
        if self._finished is not None:
            raise ValueError('global storage must be declared before module finalization')
        count = int(count)
        if count <= 0:
            raise ValueError('global storage requires a positive element count')
        name = name or f'__gemmini_storage_{len(self._globals)}'
        if name in self._global_names or name in (self.symbol, self.completion_symbol):
            raise ValueError(f'duplicate global symbol {name}')
        self._global_names.add(name)
        ty = INT_TYPES[dtype]
        arr_ty = llvm.LLVMArrayType(count, ty)
        if values is None:
            zero = llvm.ZeroOp(result_types=[arr_ty])
            block = Block([zero, llvm.ReturnOp(zero.res)])
            glob = llvm.GlobalOp(arr_ty, name, 'internal', alignment=alignment, body=Region(block))
        else:
            width = ty.width.data
            mask = (1 << width) - 1
            values = [(int(v) & mask) for v in values]
            # The LLVM global dense initializer has tensor syntax and array
            # storage. It becomes .rodata, with no per-invocation host stores.
            dense = DenseIntOrFPElementsAttr.from_list(TensorType(ty, [count]), values)
            glob = llvm.GlobalOp(arr_ty, name, 'internal', constant=True,
                                 alignment=alignment, value=dense)
        self._globals.append(glob)
        return self.b.add(llvm.AddressOfOp(name, PTR)).results[0]

    def scratch(self, count, dtype='i8', *, name=None, alignment=64):
        """Persistent aligned storage; caller establishes writes and lifetimes."""
        return self._global(count, dtype, name, alignment=alignment)

    def constant_array(self, values, dtype='i32', *, name=None, alignment=64):
        values = list(values)
        return self._global(len(values), dtype, name, values=values, alignment=alignment)

    @contextmanager
    def bounded_unroll(self, *, max_trip_count=2, budget=64):
        """Remove small fixed control loops within a bounded emission budget.

        This is ordinary constant loop unrolling: operands are recomputed from
        each actual induction constant and all device effects remain in order.
        Large loops retain runtime control, bounding code growth. In particular,
        reusable small RHS tiles then carry concrete local addresses to both
        loads and their consumers, rather than opaque induction expressions.
        """
        if max_trip_count < 1 or budget < 0:
            raise ValueError('invalid bounded-unroll limits')
        previous = self._unroll
        self._unroll = {'limit': int(max_trip_count), 'remaining': int(budget)}
        try:
            yield self
        finally:
            self._unroll = previous

    def for_loop(self, start, stop, step, body):
        if step <= 0 or start < 0 or stop < start:
            raise ValueError('invalid CPU loop bounds')
        count = (stop - start + step - 1) // step
        control = self._unroll
        if control is not None and count <= control['limit'] and count <= control['remaining']:
            control['remaining'] -= count
            for iv in range(start, stop, step):
                body(self.value(iv))
            return None
        if self.control_unroll == 'incumbent' or control is not None or count == 0:
            return self.b.for_loop(start, stop, step, body, retain_loop=True)
        before = len(self.b.region.blocks)
        result = self.b.for_loop(start, stop, step, body, retain_loop=True)
        blocks = list(self.b.region.blocks)[before:]
        # An ordinary builder loop adds header/body/exit. Nested runtime loops
        # add further blocks; retaining those outer loops prevents multiplying
        # unroll factors across a loop nest. Literal resident scopes above keep
        # their existing folding and branch-removal behavior.
        if len(blocks) != 3 or count < self.control_unroll:
            return result
        header, loop_body, _ = blocks
        operations = list(loop_body.ops)
        if not any(isinstance(op, TerminalInlineAsm) and
                   '.insn' in op.asm_string.data for op in operations):
            return result
        cost = len(operations) * (self.control_unroll - 1)
        if len(operations) > 256 or cost > self._control_unroll_remaining:
            return result
        latch = operations[-1]
        if not isinstance(latch, llvm.BrOp) or latch.successors[0] is not header:
            return result
        annotation = Context(allow_unregistered=True).get_attr('llvm.loop_annotation')
        latch.attributes['loop_annotation'] = annotation(
            'llvm.loop_annotation', False, False,
            f'unroll = <count = {self.control_unroll} : i32>')
        self._control_unroll_remaining -= cost
        self.control_unroll_decisions.append({
            'start': start, 'stop': stop, 'step': step, 'trip_count': count,
            'factor': self.control_unroll, 'body_operations': len(operations),
            'predicted_extra_operations': cost})
        return result

    def module(self):
        if self._finished is None:
            if self._opened:
                self.fence()
            fn = llvm.FuncOp(self.symbol, llvm.LLVMFunctionType([PTR] * len(self.arg_names), llvm.LLVMVoidType()),
                             linkage=llvm.LinkageAttr('external'), body=self.b.finish())
            functions = [fn]
            if self.completion_symbol is not None:
                if self.completion_symbol == self.symbol:
                    raise ValueError('completion symbol must differ from the entry')
                completion = Block([
                    TerminalInlineAsm(self.completion_asm, '~{memory}', [], has_side_effects=True),
                    llvm.ReturnOp(),
                ])
                functions.append(llvm.FuncOp(
                    self.completion_symbol, llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
                    linkage=llvm.LinkageAttr('external'), body=Region(completion)))
            self._finished = ModuleOp([*self._globals, *functions])
            self._finished.verify()
        return self._finished

    def finish(self):
        stream = StringIO()
        Printer(stream=stream).print_op(self.module())
        return stream.getvalue() + '\n'
