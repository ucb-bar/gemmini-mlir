"""Small structured LLVM emitter for Gemmini dispatch and layout loops."""
import struct


class Builder:
    def __init__(self, arguments):
        self.arguments = arguments
        self.lines = ['entry:']
        self.globals = []
        self.serial = 0
        self.label = 'entry'
        self.lines.append('  call void asm sideeffect "fence", "~{memory}"()')

    def name(self, prefix='v'):
        self.serial += 1
        return f'{prefix}{self.serial}'

    def emit(self, text):
        self.lines.append('  ' + text)

    def value(self, instruction):
        result = '%' + self.name()
        self.emit(f'{result} = {instruction}')
        return result

    def const(self, value, ty='i64'):
        if isinstance(value, str):
            return value
        if ty.startswith('i'):
            return str(int(value))
        return str(float(value))

    def op(self, opcode, lhs, rhs, ty='i64'):
        if isinstance(lhs,int) and isinstance(rhs,int) and ty.startswith('i'):
            operations = {'add': lambda: lhs+rhs, 'sub': lambda: lhs-rhs,
                          'mul': lambda: lhs*rhs, 'and': lambda: lhs&rhs,
                          'or': lambda: lhs|rhs, 'xor': lambda: lhs^rhs,
                          'shl': lambda: lhs<<rhs, 'lshr': lambda: (lhs % (1 << int(ty[1:])))>>rhs,
                          'udiv': lambda: lhs//rhs, 'urem': lambda: lhs%rhs,
                          'sdiv': lambda: int(lhs/rhs), 'srem': lambda: lhs-int(lhs/rhs)*rhs}
            if opcode in operations:
                return operations[opcode]()
        return self.value(f'{opcode} {ty} {self.const(lhs, ty)}, {self.const(rhs, ty)}')

    def cmp(self, pred, lhs, rhs, ty='i64'):
        return self.value(f'icmp {pred} {ty} {self.const(lhs, ty)}, {self.const(rhs, ty)}')

    def select(self, cond, lhs, rhs, ty='i64'):
        return self.value(f'select i1 {self.const(cond, "i1")}, {ty} {self.const(lhs,ty)}, {ty} {self.const(rhs,ty)}')

    def cast(self, opcode, value, src, dst):
        return self.value(f'{opcode} {src} {self.const(value,src)} to {dst}')

    def gep(self, ptr, index, ty='i8'):
        return self.value(f'getelementptr {ty}, ptr {ptr}, i64 {self.const(index)}')

    def load(self, ptr, ty):
        return self.value(f'load {ty}, ptr {ptr}, align 1')

    def store(self, value, ptr, ty):
        self.emit(f'store {ty} {self.const(value,ty)}, ptr {ptr}, align 1')

    def alloc(self, count, ty='i8', align=64):
        name = '@scratch_' + self.name('g')
        self.globals.append(f'{name} = internal global [{int(count)} x {ty}] zeroinitializer, align {align}')
        return name

    def rocc(self, funct, rs1, rs2):
        self.emit(f'call void asm sideeffect ".insn r 0x7b, 3, {int(funct)}, x0, $0, $1", "r,r,~{{memory}}"(i64 {self.const(rs1)}, i64 {self.const(rs2)})')

    def fence(self):
        self.emit('call void asm sideeffect "fence", "~{memory}"()')

    def block(self, label):
        self.lines.append(label + ':')
        self.label = label

    def jump(self, label):
        self.emit(f'br label %{label}')

    def loop(self, start, end, step, callback):
        before = self.label
        head = self.name('loop')
        body = self.name('body')
        latch = self.name('latch')
        done = self.name('done')
        iv = '%' + self.name('iv')
        nxt = '%' + self.name('next')
        self.jump(head)
        self.block(head)
        self.emit(f'{iv} = phi i64 [ {self.const(start)}, %{before} ], [ {nxt}, %{latch} ]')
        cond = self.cmp('slt',iv,end)
        self.emit(f'br i1 {cond}, label %{body}, label %{done}')
        self.block(body)
        callback(iv)
        self.jump(latch)
        self.block(latch)
        self.emit(f'{nxt} = add i64 {iv}, {self.const(step)}')
        self.jump(head)
        self.block(done)

    def branch_if(self, cond, then_callback, else_callback=None):
        yes = self.name('if')
        no = self.name('else')
        done = self.name('endif')
        self.emit(f'br i1 {self.const(cond,"i1")}, label %{yes}, label %{no}')
        self.block(yes)
        then_callback()
        self.jump(done)
        self.block(no)
        if else_callback is not None:
            else_callback()
        self.jump(done)
        self.block(done)

    def finish(self):
        self.emit('ret void')
        args = ', '.join(f'ptr %arg{i}' for i in range(len(self.arguments)))
        return '\n'.join(['target triple = "riscv64-unknown-unknown"',
                          'module asm ".text\\0A.p2align 2\\0A.globl kernel_complete\\0A.type kernel_complete,@function\\0Akernel_complete:\\0Afence\\0Aret\\0A.size kernel_complete,.-kernel_complete\\0A"', *self.globals,
                          f'define void @kernel_entry({args}) {{', *self.lines, '}']) + '\n'


def float_bits(value):
    return struct.unpack('<I',struct.pack('<f',float(value)))[0]
