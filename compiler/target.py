"""Gemmini instruction operands and logical-pointer LLVM emission.

Packed fields follow rtl_facts/facts.json register_bundle_layouts. Scheduling
and numerical decisions belong to the calling compiler passes.
"""
from __future__ import annotations
import io
import struct
from pathlib import Path
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8, i32, i64
from xdsl.printer import Printer
from xdsl.ir import Block, Region, SSAValue
from merlin.llvmlower.host_builder import FnBuilder, PTR, INT_TYPES
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter

GARBAGE = 0xffffffff
ACC = 0x80000000
ACCUMULATE = 0x40000000
FULL = 0x20000000
DIM = 16

class Builder(FnBuilder):
    def __init__(self, arg_names, symbol='kernel_entry'):
        self.arg_names = tuple(arg_names)
        super().__init__([PTR] * len(self.arg_names))
        self.args = dict(zip(self.arg_names, self.entry.args))
        self.symbol = symbol
        self.globals = []
        self._module = None
        self._control = FnBuilderControlAdapter(self)

    @classmethod
    def for_logical_kernel(cls, command_buffer, *, abi=None):
        from merlin.compile.logical_abi import kernel_arg_order, logical_abi
        if abi is None:
            abi = logical_abi(Path(__file__).resolve().parent.parent / 'mlir_oot_backend_contract.yaml')
        b = cls(kernel_arg_order(command_buffer, abi))
        return b, b.args

    def value(self, value):
        return self.const(value) if isinstance(value, int) else value

    def ptr(self, base, byte_offset=0):
        if isinstance(base, str):
            base = self.args[base]
        value = self.add(llvm.PtrToIntOp(base)).results[0] if base.type == PTR else base
        return value if byte_offset == 0 and isinstance(byte_offset, int) else self.add_i(value, self.value(byte_offset))

    def pointer(self, value):
        return value if value.type == PTR else self.add(llvm.IntToPtrOp(value)).results[0]

    def alloca(self, count, dtype='i8', *, name=None):
        """Private static storage, kept outside the bounded kernel stack."""
        if not isinstance(count, int) or count <= 0:
            raise ValueError('scratch storage requires positive static element count')
        ty = INT_TYPES[dtype]
        symbol = name or f'gemmini_scratch_{len(self.globals)}'
        self.globals.append(llvm.GlobalOp(llvm.LLVMArrayType(count, ty), symbol, 'internal', alignment=64))
        return self.add(llvm.AddressOfOp(symbol, PTR)).results[0]

    def rocc(self, funct, rs1=0, rs2=0):
        if funct not in (0,1,2,3,4,5,6,7,14,23):
            raise ValueError('unsupported primitive Gemmini instruction')
        self.add(llvm.InlineAsmOp(f'.insn r 0x7b, 3, {funct}, x0, $0, $1',
            'r,r,~{memory}', [self.value(rs1), self.value(rs2)], has_side_effects=True))

    def fence(self):
        self.add(llvm.InlineAsmOp('fence', '~{memory}', [], has_side_effects=True))

    def op(self, opcode, lhs, rhs, dtype='i64'):
        return self._control.op(opcode, lhs, rhs, dtype)

    def cmp(self, predicate, lhs, rhs):
        return self._control.cmp(predicate, lhs, rhs)

    def branch(self, predicate, yes, no):
        return self._control.branch(predicate, yes, no)

    def select(self, predicate, yes, no):
        return self._control.select(predicate, yes, no)

    def loop(self, start, stop, step, body):
        return self.for_loop(start, stop, step, body, retain_loop=True)

    def module_text(self):
        if self._module is None:
            region = super().finish()
            fn = llvm.FuncOp(self.symbol, llvm.LLVMFunctionType([PTR] * len(self.arg_names)),
                             linkage=llvm.LinkageAttr('external'), body=region)
            # The selected target contract supplies these exported harness symbols.
            complete = llvm.FuncOp('kernel_complete', llvm.LLVMFunctionType([]),
                linkage=llvm.LinkageAttr('external'), body=Region([Block([
                    llvm.InlineAsmOp('fence', '~{memory}', [], has_side_effects=True),
                    llvm.ReturnOp(),
                ])]))
            self._module = ModuleOp([*self.globals, fn, complete])
            self._module.verify()
        stream = io.StringIO()
        Printer(stream=stream).print_op(self._module)
        return stream.getvalue() + '\n'


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def packed(b, addr, rows=16, cols=16):
    # Dynamic local row/shape expressions are emitted as integer arithmetic.
    return b.or_i(b.value(addr), b.or_i(b.shl_i(b.value(cols),32), b.shl_i(b.value(rows),48)))


def cfg_ld(b, stride, stream=0, scale=1.0, shrunk=False, block_stride=16, pixel_repeats=1):
    if stream not in (0,1,2): raise ValueError('load stream must be 0,1,2')
    word = 1 | (int(shrunk)<<2) | (stream<<3) | (pixel_repeats<<8) | (block_stride<<16) | (f32_bits(scale)<<32)
    b.rocc(0, word, stride)


def cfg_ex(b, dataflow=1, activation=0, a_stride=1, c_stride=1,
           a_transpose=False, b_transpose=False, scale=1., shift=0):
    word = (dataflow<<2) | (activation<<3) | (int(a_transpose)<<8) | (int(b_transpose)<<9) | (a_stride<<16) | (f32_bits(scale)<<32)
    b.rocc(0, word, shift | (c_stride<<48))


def cfg_st(b, stride, scale=1., activation=0, *, pool_size=0,
           pool_stride=0, pool_out_dim=0, porows=0, pocols=0,
           orows=0, ocols=0, upad=0, lpad=0):
    # ConfigMvoutRs1, as declared in the selected register bundle facts.
    fields = ((activation, 2, 2), (pool_stride, 4, 2),
              (pool_size, 6, 2), (upad, 8, 2), (lpad, 10, 2),
              (pool_out_dim, 24, 8), (porows, 32, 8),
              (pocols, 40, 8), (orows, 48, 8), (ocols, 56, 8))
    word = 2
    for value, offset, width in fields:
        if type(value) is not int or not 0 <= value < 1 << width:
            raise ValueError(f'store configuration field at bit {offset} exceeds {width} bits')
        word |= value << offset
    b.rocc(0, word, b.or_i(b.value(stride), b.value(f32_bits(scale)<<32)))


def mvin(b, ptr, addr, rows=16, cols=16, stream=0):
    b.rocc((2,1,14)[stream], ptr, packed(b,addr,rows,cols))


def mvout(b, ptr, addr, rows=16, cols=16):
    b.rocc(3, ptr, packed(b,addr,rows,cols))


def preload(b, baddr, caddr, brows=16, bcols=16, crows=16, ccols=16):
    b.rocc(6, packed(b,baddr,brows,bcols), packed(b,caddr,crows,ccols))


def compute(b, aaddr, bdaddr=GARBAGE, arows=16, acols=16, bdrows=16, bdcols=16, flip=True):
    b.rocc(4 if flip else 5, packed(b,aaddr,arows,acols), packed(b,bdaddr,bdrows,bdcols))
