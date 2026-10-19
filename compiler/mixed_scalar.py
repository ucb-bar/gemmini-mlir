"""Scalar expression lowering shared by tensor-region LLVM emitters."""
from __future__ import annotations

import struct

from xdsl.dialects.builtin import FloatAttr, IntegerAttr


def llvm_type(ty):
    return {'f16': 'half', 'f32': 'float', 'f64': 'double',
            'index': 'i64'}.get(str(ty), str(ty))


def float_literal(value, ty='double'):
    """Spell floating constants exactly, including infinity and signed zero."""
    ty = llvm_type(ty)
    value = float(value)
    if ty == 'half':
        return '0xH' + struct.pack('>e', value).hex().upper()
    if ty == 'float':
        value = struct.unpack('>f', struct.pack('>f', value))[0]
    return '0x' + struct.pack('>d', value).hex().upper()


def _name(operation):
    name = operation.name
    if name == 'builtin.unregistered':
        return operation.attributes['op_name__'].data
    return name


def _attr(operation, key):
    return operation.properties.get(key, operation.attributes.get(key))


def _integer(attribute):
    value = attribute.value if hasattr(attribute, 'value') else attribute
    return int(value.data if hasattr(value, 'data') else value)


class ScalarEmitter:
    def __init__(self, builder, pointer_callback):
        self.b = builder
        self.pointer = pointer_callback

    def _declaration(self, declaration):
        if declaration not in self.b.globals:
            self.b.globals.append(declaration)

    def intrinsic(self, name, operands, ty):
        suffix = {'half': 'f16', 'float': 'f32', 'double': 'f64'}[ty]
        symbol = 'llvm.' + name + '.' + suffix
        signature = ', '.join([ty] * len(operands))
        self._declaration(f'declare {ty} @{symbol}({signature})')
        args = ', '.join(f'{ty} {value}' for value in operands)
        return self.b.value(f'call {ty} @{symbol}({args})')

    def _poly(self, x, coefficients, ty='double'):
        value = float_literal(coefficients[-1], ty)
        for coefficient in reversed(coefficients[:-1]):
            value = self.b.op('fadd', self.b.op('fmul', value, x, ty),
                              float_literal(coefficient, ty), ty)
        return value

    def _exp(self, value, ty):
        """Range-reduced exponential without a target libc dependency."""
        b = self.b
        original = value
        x = value if ty == 'double' else b.cast('fpext', value, ty, 'double')
        low, high = float_literal(-745.2), float_literal(709.79)
        below = b.value(f'fcmp olt double {x}, {low}')
        above = b.value(f'fcmp ogt double {x}, {high}')
        nan = b.value(f'fcmp uno double {x}, {x}')
        x = b.select(below, low, x, 'double')
        x = b.select(above, high, x, 'double')
        x = b.select(nan, float_literal(0), x, 'double')
        scaled = b.op('fmul', x, float_literal(1.4426950408889634074), 'double')
        negative = b.value(f'fcmp olt double {scaled}, {float_literal(0)}')
        rounding = b.select(negative, float_literal(-0.5), float_literal(0.5), 'double')
        exponent = b.cast('fptosi', b.op('fadd', scaled, rounding, 'double'), 'double', 'i64')
        nf = b.cast('sitofp', exponent, 'i64', 'double')
        remainder = b.op('fsub', x, b.op('fmul', nf, float_literal(0.6931471803691238), 'double'), 'double')
        remainder = b.op('fsub', remainder, b.op('fmul', nf, float_literal(1.9082149292705877e-10), 'double'), 'double')
        factorial = 1
        coefficients = [1.0]
        for power in range(1, 15):
            factorial *= power
            coefficients.append(1.0 / factorial)
        polynomial = self._poly(remainder, coefficients)
        # Splitting the exponent keeps both scaling factors normal even when
        # the final product is subnormal or overflows.
        half = b.op('ashr', exponent, 1)
        other = b.op('sub', exponent, half)
        first_bits = b.op('shl', b.op('add', half, 1023), 52)
        other_bits = b.op('shl', b.op('add', other, 1023), 52)
        first_scale = b.cast('bitcast', first_bits, 'i64', 'double')
        other_scale = b.cast('bitcast', other_bits, 'i64', 'double')
        out = b.op('fmul', b.op('fmul', polynomial, first_scale, 'double'), other_scale, 'double')
        out = b.select(below, float_literal(0), out, 'double')
        out = b.select(above, float_literal(float('inf')), out, 'double')
        if ty != 'double':
            out = b.cast('fptrunc', out, 'double', ty)
        return b.select(nan, original, out, ty)

    def _sincos(self, value, ty, cosine=False):
        b = self.b
        x = value if ty == 'double' else b.cast('fpext', value, ty, 'double')
        scaled = b.op('fmul', x, float_literal(0.63661977236758134308), 'double')
        negative = b.value(f'fcmp olt double {scaled}, {float_literal(0)}')
        rounding = b.select(negative, float_literal(-0.5), float_literal(0.5), 'double')
        q = b.cast('fptosi', b.op('fadd', scaled, rounding, 'double'), 'double', 'i64')
        qf = b.cast('sitofp', q, 'i64', 'double')
        r = b.op('fsub', x, b.op('fmul', qf, float_literal(1.57079632673412561417), 'double'), 'double')
        r = b.op('fsub', r, b.op('fmul', qf, float_literal(6.07710050650619224932e-11), 'double'), 'double')
        r2 = b.op('fmul', r, r, 'double')
        sine = b.op('fmul', r, self._poly(r2, [1.0, -1/6, 1/120, -1/5040,
                                              1/362880, -1/39916800, 1/6227020800,
                                              -1/1307674368000]), 'double')
        cos = self._poly(r2, [1.0, -1/2, 1/24, -1/720, 1/40320,
                              -1/3628800, 1/479001600, -1/87178291200])
        neg_sine = b.value(f'fneg double {sine}')
        neg_cos = b.value(f'fneg double {cos}')
        quadrant = b.op('and', q, 3)
        options = [cos, neg_sine, neg_cos, sine] if cosine else [sine, cos, neg_sine, neg_cos]
        out = options[3]
        for index in (2, 1, 0):
            out = b.select(b.cmp('eq', quadrant, index), options[index], out, 'double')
        return out if ty == 'double' else b.cast('fptrunc', out, 'double', ty)

    def _tanh(self, value, ty):
        b = self.b
        x = value if ty == 'double' else b.cast('fpext', value, ty, 'double')
        neg = b.value(f'fcmp olt double {x}, {float_literal(0)}')
        minus_x = b.value(f'fneg double {x}')
        magnitude = b.select(neg, minus_x, x, 'double')
        e = self._exp(b.op('fmul', magnitude, float_literal(-2), 'double'), 'double')
        denominator = b.op('fadd', float_literal(1), e, 'double')
        out = b.op('fsub', b.op('fdiv', float_literal(2), denominator, 'double'), float_literal(1), 'double')
        negative_out = b.value(f'fneg double {out}')
        out = b.select(neg, negative_out, out, 'double')
        # Preserve tiny arguments and the sign of zero without cancellation.
        tiny = b.value(f'fcmp olt double {magnitude}, {float_literal(1e-8)}')
        out = b.select(tiny, x, out, 'double')
        return out if ty == 'double' else b.cast('fptrunc', out, 'double', ty)

    def _maximum(self, lhs, rhs, ty, minimum=False):
        """IEEE max/min with propagated NaNs and deterministic signed zero."""
        b = self.b
        pred = 'olt' if minimum else 'ogt'
        greater = b.value(f'fcmp {pred} {ty} {lhs}, {rhs}')
        chosen = b.select(greater, lhs, rhs, ty)
        lhs_nan = b.value(f'fcmp uno {ty} {lhs}, {lhs}')
        chosen = b.select(lhs_nan, lhs, chosen, ty)
        lhs_zero = b.value(f'fcmp oeq {ty} {lhs}, {float_literal(0, ty)}')
        rhs_zero = b.value(f'fcmp oeq {ty} {rhs}, {float_literal(0, ty)}')
        both_zero = b.op('and', lhs_zero, rhs_zero, 'i1')
        integer_type = {'half': 'i16', 'float': 'i32', 'double': 'i64'}[ty]
        lhs_bits = b.cast('bitcast', lhs, ty, integer_type)
        rhs_bits = b.cast('bitcast', rhs, ty, integer_type)
        zero_bits = b.op('or' if minimum else 'and', lhs_bits, rhs_bits, integer_type)
        zero = b.cast('bitcast', zero_bits, integer_type, ty)
        return b.select(both_zero, zero, chosen, ty)

    def emit(self, operation, env, indices=()):
        """Emit one typed scalar operation and add its results to ``env``."""
        name = _name(operation)
        b = self.b
        if name in ('linalg.yield', 'tensor.yield', 'scf.yield'):
            return {}
        if len(operation.results) != 1:
            raise ValueError('Unsupported scalar result arity for ' + name)
        result = operation.results[0]
        ty = llvm_type(result.type)

        def operand(value):
            if value not in env:
                owner = value.owner
                if hasattr(owner, 'name') and _name(owner) == 'arith.constant':
                    self.emit(owner, env, indices)
                else:
                    raise ValueError('Scalar operand has not been lowered: ' + str(value))
            return env[value]

        if name == 'arith.constant':
            attribute = _attr(operation, 'value')
            if isinstance(attribute, FloatAttr):
                value = float_literal(attribute.value.data, ty)
            elif isinstance(attribute, IntegerAttr):
                value = str(attribute.value.data)
            else:
                raise ValueError('Scalar constant requires a scalar typed attribute')
        elif name == 'linalg.index':
            value = indices[_integer(_attr(operation, 'dim'))]
        elif name == 'tensor.extract':
            tensor = operation.operands[0]
            shape = tensor.type.get_shape()
            offset = '0'
            for dimension, index in zip(shape, operation.operands[1:]):
                index_value = operand(index)
                src_ty = llvm_type(index.type)
                if src_ty != 'i64':
                    index_value = b.cast('sext', index_value, src_ty, 'i64')
                offset = b.op('add', b.op('mul', offset, dimension), index_value)
            value = b.load(b.gep(self.pointer(tensor), offset, ty), ty)
        else:
            args = [operand(item) for item in operation.operands]
            binary = {
                'arith.addf': 'fadd', 'arith.subf': 'fsub',
                'arith.mulf': 'fmul', 'arith.divf': 'fdiv', 'arith.remf': 'frem',
                'arith.addi': 'add', 'arith.subi': 'sub', 'arith.muli': 'mul',
                'arith.divsi': 'sdiv', 'arith.divui': 'udiv',
                'arith.remsi': 'srem', 'arith.remui': 'urem',
                'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor',
                'arith.shli': 'shl', 'arith.shrsi': 'ashr', 'arith.shrui': 'lshr',
            }
            casts = {'arith.extsi': 'sext', 'arith.extui': 'zext',
                     'arith.trunci': 'trunc', 'arith.sitofp': 'sitofp',
                     'arith.uitofp': 'uitofp', 'arith.fptosi': 'fptosi',
                     'arith.fptoui': 'fptoui', 'arith.extf': 'fpext',
                     'arith.truncf': 'fptrunc', 'arith.bitcast': 'bitcast'}
            if name in binary:
                value = b.op(binary[name], args[0], args[1], ty)
            elif name == 'arith.negf':
                value = b.value(f'fneg {ty} {args[0]}')
            elif name in casts:
                src_ty = llvm_type(operation.operands[0].type)
                value = args[0] if src_ty == ty else b.cast(casts[name], args[0], src_ty, ty)
            elif name in ('arith.index_cast', 'arith.index_castui'):
                src_ty = llvm_type(operation.operands[0].type)
                if src_ty == ty:
                    value = args[0]
                else:
                    growing = int(src_ty[1:]) < int(ty[1:])
                    opcode = ('zext' if name.endswith('ui') else 'sext') if growing else 'trunc'
                    value = b.cast(opcode, args[0], src_ty, ty)
            elif name in ('arith.cmpf', 'arith.cmpi'):
                predicate = _integer(_attr(operation, 'predicate'))
                if name == 'arith.cmpf':
                    predicates = ['false', 'oeq', 'ogt', 'oge', 'olt', 'ole',
                                  'one', 'ord', 'ueq', 'ugt', 'uge', 'ult',
                                  'ule', 'une', 'uno', 'true']
                    opcode = 'fcmp'
                else:
                    predicates = ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge',
                                  'ult', 'ule', 'ugt', 'uge']
                    opcode = 'icmp'
                src_ty = llvm_type(operation.operands[0].type)
                value = b.value(f'{opcode} {predicates[predicate]} {src_ty} {args[0]}, {args[1]}')
            elif name == 'arith.select':
                value = b.select(args[0], args[1], args[2], ty)
            elif name in ('arith.maximumf', 'arith.minimumf'):
                value = self._maximum(args[0], args[1], ty, name == 'arith.minimumf')
            elif name in ('arith.maxnumf', 'arith.minnumf'):
                value = self.intrinsic('maxnum' if name == 'arith.maxnumf' else 'minnum', args, ty)
            elif name == 'math.rsqrt':
                root = self.intrinsic('sqrt', args, ty)
                value = b.op('fdiv', float_literal(1, ty), root, ty)
            elif name == 'math.exp':
                value = self._exp(args[0], ty)
            elif name in ('math.sin', 'math.cos'):
                value = self._sincos(args[0], ty, name == 'math.cos')
            elif name in ('math.exp2', 'math.log', 'math.log2',
                          'math.sqrt', 'math.absf',
                          'math.floor', 'math.ceil', 'math.trunc', 'math.roundeven'):
                intrinsic = {'absf': 'fabs', 'roundeven': 'roundeven'}.get(name[5:], name[5:])
                value = self.intrinsic(intrinsic, args, ty)
            elif name == 'math.tanh':
                value = self._tanh(args[0], ty)
            elif name == 'math.powf':
                value = self.intrinsic('pow', args, ty)
            else:
                raise ValueError('Unsupported scalar operation ' + name)
        env[result] = value
        return {result: value}
