"""Terminal Gemmini encoder shared by all source and schedule lowerings.

Register layouts and local geometry come from the selected public RTL facts.
Only addressing and control arithmetic is performed on the scalar core.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8, i32, i64
from xdsl.ir import SSAValue
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter


FACTS = json.loads((Path(__file__).parent / 'data/encoding_facts.json').read_text())
BUNDLES = next(x['bundles'] for x in FACTS['interfaces'] if x['name'] == 'register_bundle_layouts')
FUNCTS = next(x for x in FACTS['interfaces'] if x['name'] == 'funct_decode_table')
FUNCTIONS = {name: int(code) for code, name in FUNCTS['names'].items()}
ACC = 1 << 31
ACCUMULATE = 1 << 30
FULL = 1 << 29
GARBAGE = (1 << 32) - 1


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def kernel_module(function, extras=()):
    """Expose exactly the selected target's entry and completion symbols."""
    from merlin.llvmlower.host_builder import FnBuilder
    from xdsl.dialects.builtin import StringAttr
    function.properties['sym_name']=StringAttr('kernel_entry')
    function.properties['linkage']=llvm.LinkageAttr('external')
    completion = FnBuilder([])
    completion.add(llvm.InlineAsmOp('fence', '~{memory}', [], has_side_effects=True))
    complete = llvm.FuncOp('kernel_complete', llvm.LLVMFunctionType([],llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr('external'),body=completion.finish())
    module = ModuleOp([*extras,function,complete])
    module.verify()
    return module


class Encoder:
    def __init__(self, fn):
        self.fn = fn
        self.index_builder = FnBuilderControlAdapter(fn)
        self.receipts = []
        self.config_cache = {}
        # The selected call contract starts with a completion boundary and
        # clears stale address-translation state before issuing tensor work.
        self.fence()
        self.issue('FLUSH_CMD', 0, 0)

    def val(self, value):
        return value if isinstance(value, SSAValue) else self.fn.const(int(value))

    def add(self, a, b):
        return self.fn.add_i(self.val(a), self.val(b))

    def mul(self, a, b):
        return self.fn.mul_i(self.val(a), self.val(b))

    def flag(self, address, bits):
        return self.fn.or_i(self.val(address), self.val(bits))

    def bundle(self, name, **fields):
        result = self.val(0)
        for key, value in fields.items():
            spec = BUNDLES[name]['fields'][key]
            if isinstance(value, int):
                width = spec.get('width') or spec.get('slot_width')
                if value < 0 or width and value >= (1 << width):
                    raise ValueError(f'{name}.{key} exceeds its encoding field')
            value = self.val(value)
            if spec['offset']:
                value = self.fn.shl_i(value, spec['offset'])
            result = self.fn.or_i(result, value)
        return result

    def issue(self, name, rs1=0, rs2=0):
        if name.startswith('LOOP_'):
            raise ValueError('hardware loop descriptors are prohibited')
        funct = FUNCTIONS[name]
        self.fn.add(llvm.InlineAsmOp(
            f'.insn r 0x{FUNCTS["custom_opcode"]:02x}, {FUNCTS["funct3"]}, {funct}, x0, $0, $1',
            'r,r,~{memory}', [self.val(rs1), self.val(rs2)], has_side_effects=True))

    def fence(self):
        self.fn.add(llvm.InlineAsmOp('fence', '~{memory}', [], has_side_effects=True))
        # A runtime branch can configure one side only. Conservatively forget
        # cached state at schedule boundaries instead of assuming dominance.
        self.config_cache.clear()

    def offset(self, ptr, byteoffset):
        return self.fn.gep(ptr, self.val(byteoffset), i8)

    ptr_offset = offset

    def ptr_zero(self):
        return self.fn.add(llvm.IntToPtrOp(self.val(0))).results[0]

    def address(self, ptr):
        if isinstance(ptr, int):
            return self.val(ptr)
        if ptr.type == i64:
            return ptr
        return self.fn.add(llvm.PtrToIntOp(ptr, i64)).results[0]

    def configure_execute(self):
        self.issue('CONFIG_CMD', self.bundle('ConfigExRs1', cmd_type=0, dataflow=1,
            activation=0, a_stride=1, acc_scale=float_bits(1.0)),
            self.bundle('ConfigExRs2', in_shift=0, c_stride=1))

    def load(self, ptr, localrow, rows, cols, stride, stream=0, scale=1.0,
             shrunk=False, block_stride=16, **kwargs):
        if kwargs:
            raise TypeError(f'unsupported load options {kwargs}')
        config = self.bundle('ConfigMvinRs1', shrink=int(shrunk), state_id=stream,
            pixel_repeats=1, stride=block_stride, scale=float_bits(scale))
        # ConfigMvinRs1 omits the command selector, which occupies bits [1:0].
        self.issue('CONFIG_CMD', self.flag(config, 1), stride)
        self.issue(('LOAD_CMD', 'LOAD2_CMD', 'LOAD3_CMD')[stream], self.address(ptr),
            self.bundle('MvinRs2', local_addr=localrow, num_rows=rows, num_cols=cols))

    def preload(self, rhsrow, accrow, krows, ncols, mrows, accumulate=False):
        accrow = self.flag(accrow, ACC | (ACCUMULATE if accumulate else 0))
        self.issue('PRELOAD_CMD', self.bundle('PreloadRs', local_addr=rhsrow,
            num_rows=krows, num_cols=ncols), self.bundle('PreloadRs', local_addr=accrow,
            num_rows=mrows, num_cols=ncols))

    def compute(self, lhsrow, kcols, mrows, accumulate=False):
        self.issue('COMPUTE_AND_FLIP_CMD', self.bundle('ComputeRs', local_addr=lhsrow,
            num_rows=mrows, num_cols=kcols), self.bundle('ComputeRs', local_addr=GARBAGE,
            num_rows=16, num_cols=16))

    def store(self, ptr, accrow, rows, cols, stride, dtype='i32', scale=1.0,
              relu=False, **pool):
        st = dict(cmd_type=2, activation=int(relu))
        st.update(pool)
        self.issue('CONFIG_CMD', self.bundle('ConfigMvoutRs1', **st),
            self.bundle('ConfigMvoutRs2', stride=stride, acc_scale=float_bits(scale)))
        addr = self.flag(accrow, ACC | (FULL if dtype == 'i32' else 0))
        self.issue('STORE_CMD', self.address(ptr), self.bundle('MvoutRs2',
            local_addr=addr, num_rows=rows, num_cols=cols))

    def dense_effect(self, instruction, fields, pointers, problem):
        f = fields
        if instruction in ('lhs_load', 'weight_load'):
            side = 'lhs' if instruction == 'lhs_load' else 'rhs'
            pitch = problem.k if side == 'lhs' else problem.n
            byteoffset = self.add(self.mul(f['src_row'], pitch), f['src_col'])
            self.load(self.offset(pointers[side], byteoffset), f['dst'], f['rows'],
                f['cols'], self.mul(pitch, f.get('step',1)), stream=0 if side == 'lhs' else 1)
        elif instruction == 'bias_load':
            self.load(self.offset(pointers['bias'], self.mul(f['col'],4)),
                self.flag(f['acc'], ACC), f['rows'], f['cols'], 0,
                stream=2, shrunk=False)
        elif instruction == 'preload':
            addr = self.flag(f['acc'], ACC)
            continuation = f.get('continue', 0)
            if isinstance(continuation, int):
                if continuation:
                    addr = self.flag(addr, ACCUMULATE)
            else:
                addr = self.flag(addr, self.fn.shl_i(continuation, 30))
            self.issue('PRELOAD_CMD', self.bundle('PreloadRs', local_addr=f['weight'],
                num_rows=f['depth'], num_cols=f['cols']), self.bundle('PreloadRs',
                local_addr=addr, num_rows=f['rows'], num_cols=f['cols']))
        elif instruction == 'compute':
            self.compute(f['lhs'], f['depth'], f['rows'])
        elif instruction == 'store':
            dtype = pointers.get('out_dtype','i32')
            byteoffset = self.mul(self.add(self.mul(f['row'], problem.n), f['col']),
                4 if dtype == 'i32' else 1)
            self.store(self.offset(pointers['out'], byteoffset), f['acc'], f['rows'],
                f['cols'], problem.n * (4 if dtype == 'i32' else 1), dtype=dtype,
                scale=pointers.get('scale',1.0), relu=pointers.get('relu',False))
        else:
            raise ValueError(f'unknown actual dense effect {instruction}')
