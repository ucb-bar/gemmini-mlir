"""Whole-source tensor lowering with explicit storage and bounded LLVM loops."""
from math import prod
from xdsl.dialects.builtin import TensorType, DenseIntOrFPElementsAttr
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr
from builder import Builder
from contraction import matmul
from linalg_frontend import op_name, contraction_kind


def llvm_type(ty):
    text = str(ty)
    return {'f32': 'float', 'f64': 'double', 'f16': 'half', 'index': 'i64'}.get(text, text)


def dtype(value):
    ty = value.type.element_type if isinstance(value.type, TensorType) else value.type
    return llvm_type(ty)


def shape(value):
    return tuple(value.type.get_shape())


def attr_ints(operation, key):
    attr = operation.properties.get(key, operation.attributes.get(key))
    if hasattr(attr, 'get_values'):
        return list(attr.get_values())
    return [item.value.data for item in attr]


class TaskBuilder(Builder):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.task = None
        self.instruction_tasks = []

    def emit(self, text):
        super().emit(text)
        self.instruction_tasks.append(self.task)


class SourceEmitter:
    def __init__(self, program, buffer):
        from mixed_scalar import ScalarEmitter
        self.program = program
        self.buffer = buffer
        self.arguments = buffer['compiler_pointer_abi']['arguments']
        self.b = TaskBuilder(self.arguments)
        self.ptrs = {name: '%arg' + str(i) for i, name in enumerate(self.arguments)}
        self.values = {}
        self.scalar = ScalarEmitter(self.b, self.pointer)
        for value, name in program.names.items():
            if name not in self.ptrs:
                self.ptrs[name] = self.b.alloc(prod(shape(value)), dtype(value))

    def pointer(self, value):
        return self.ptrs[self.program.names[value]]

    def coords(self, index, dims):
        result = []
        for dimension, size in enumerate(dims):
            stride = prod(dims[dimension + 1:])
            coordinate = self.b.op('udiv', index, stride) if stride != 1 else index
            coordinate = self.b.op('urem', coordinate, size) if dimension else coordinate
            result.append(coordinate)
        return result

    def linear(self, coordinates, dims):
        offset = 0
        for coordinate, size in zip(coordinates, dims):
            offset = self.b.op('add', self.b.op('mul', offset, size), coordinate)
        return offset

    def tensor_load(self, value, index):
        return self.b.load(self.b.gep(self.pointer(value), index, dtype(value)), dtype(value))

    def tensor_store(self, value, index, data):
        self.b.store(data, self.b.gep(self.pointer(value), index, dtype(value)), dtype(value))

    def copy(self, source, result):
        self.b.loop(0, prod(shape(result)), 1,
                    lambda i: self.tensor_store(result, i, self.tensor_load(source, i)))

    def fill(self, result, data):
        self.b.loop(0, prod(shape(result)), 1, lambda i: self.tensor_store(result, i, data))

    def affine(self, expr, indices):
        if isinstance(expr, AffineDimExpr):
            return indices[expr.position]
        if isinstance(expr, AffineConstantExpr):
            return expr.value
        if isinstance(expr, AffineBinaryOpExpr):
            lhs, rhs = self.affine(expr.lhs, indices), self.affine(expr.rhs, indices)
            op = expr.kind.name.lower()
            opcode = {'add': 'add', 'mul': 'mul', 'mod': 'srem', 'floordiv': 'sdiv'}.get(op)
            if opcode:
                return self.b.op(opcode, lhs, rhs)
        raise ValueError('Unsupported affine expression ' + str(expr))

    def mapped_index(self, value, mapping, indices):
        return self.linear([self.affine(expr, indices) for expr in mapping.results], shape(value))

    def nested(self, extents, callback, indices=()):
        if len(indices) == len(extents):
            callback(indices)
            return
        self.b.loop(0, extents[len(indices)], 1,
                    lambda index: self.nested(extents, callback, indices + (index,)))

    def region(self, operation, arguments, indices):
        env = dict(self.values)
        block = operation.regions[0].block
        env.update(zip(block.args, arguments))
        for inner in block.ops:
            if op_name(inner) == 'linalg.yield':
                return [env[value] for value in inner.operands]
            self.scalar.emit(inner, env, indices)
        raise ValueError('Scalar region has no yield')

    def generic(self, operation):
        maps = [attr.data for attr in operation.properties['indexing_maps']]
        extents = [None] * maps[0].num_dims
        for value, mapping in zip(operation.operands, maps):
            for size, expr in zip(shape(value), mapping.results):
                if isinstance(expr, AffineDimExpr):
                    old = extents[expr.position]
                    if old not in (None, size):
                        raise ValueError('Conflicting affine iteration extents')
                    extents[expr.position] = size
        if any(size is None for size in extents):
            raise ValueError('Cannot infer source generic loop extent')
        count_outputs = len(operation.results)
        count_inputs = len(operation.operands) - count_outputs
        initializers = list(operation.operands[count_inputs:])
        for initializer, result in zip(initializers, operation.results):
            self.copy(initializer, result)
        def body(indices):
            args = []
            for i, (value, mapping) in enumerate(zip(operation.operands, maps)):
                storage = value if i < count_inputs else operation.results[i - count_inputs]
                offset = self.mapped_index(value, mapping, indices)
                args.append(self.tensor_load(storage, offset))
            yielded = self.region(operation, args, indices)
            for i, (result, data) in enumerate(zip(operation.results, yielded)):
                offset = self.mapped_index(result, maps[count_inputs + i], indices)
                self.tensor_store(result, offset, data)
        self.nested(extents, body)

    def contraction(self, operation, kind):
        lhs, rhs = operation.operands[:2]
        result = operation.results[0]
        dims, rdims = shape(lhs), shape(rhs)
        m, k = dims[-2:]
        n = rdims[-2] if kind == 'ATTENTION_QK' else rdims[-1]
        attrs = {'output_dtype': str(result.type.element_type), 'epilogue': []}
        def batch(index):
            ap = self.b.gep(self.pointer(lhs), self.b.op('mul', index, m * k), 'i8')
            wp = self.b.gep(self.pointer(rhs), self.b.op('mul', index, k * n), 'i8')
            cp = self.b.gep(self.pointer(result), self.b.op('mul', index, m * n), dtype(result))
            matmul(self.b, ap, wp, cp, m, k, n, attrs, {}, {},
                   transpose_rhs=kind == 'ATTENTION_QK')
        if len(dims) > 2:
            self.b.loop(0, prod(dims[:-2]), 1, batch)
        else:
            matmul(self.b, self.pointer(lhs), self.pointer(rhs), self.pointer(result),
                   m, k, n, attrs, {}, {}, transpose_rhs=kind == 'ATTENTION_QK')

    def transpose(self, operation):
        source, result = operation.operands[0], operation.results[0]
        permutation = attr_ints(operation, 'permutation')
        def body(index):
            outcoords = self.coords(index, shape(result))
            incoords = [0] * len(permutation)
            for outdim, indim in enumerate(permutation):
                incoords[indim] = outcoords[outdim]
            offset = self.linear(incoords, shape(source))
            self.tensor_store(result, index, self.tensor_load(source, offset))
        self.b.loop(0, prod(shape(result)), 1, body)

    def reduction(self, operation):
        source, initializer = operation.operands
        result = operation.results[0]
        reductions = attr_ints(operation, 'dimensions')
        self.copy(initializer, result)
        def body(index):
            coordinates = self.coords(index, shape(source))
            outcoords = [coord for i, coord in enumerate(coordinates) if i not in reductions]
            offset = self.linear(outcoords, shape(result))
            args = [self.tensor_load(source, index), self.tensor_load(result, offset)]
            data = self.region(operation, args, coordinates)[0]
            self.tensor_store(result, offset, data)
        self.b.loop(0, prod(shape(source)), 1, body)

    def slice(self, operation, insert=False):
        source = operation.operands[0]
        result = operation.results[0]
        offsets = attr_ints(operation, 'static_offsets')
        sizes = attr_ints(operation, 'static_sizes')
        strides = attr_ints(operation, 'static_strides')
        if any(v < 0 for v in offsets + sizes + strides):
            raise ValueError('Dynamic slice parameters are unsupported')
        if insert:
            self.copy(operation.operands[1], result)
        iteration_shape = shape(source) if insert else shape(result)
        fullshape = shape(result) if insert else shape(source)
        # Rank-reduced slices retain the non-unit dimensions in source order.
        kept = []
        cursor = 0
        for i, size in enumerate(sizes):
            if cursor < len(iteration_shape) and size == iteration_shape[cursor]:
                kept.append(i)
                cursor += 1
            elif size != 1:
                raise ValueError('Unsupported rank-reduced slice')
        if cursor != len(iteration_shape):
            raise ValueError('Slice size and result shape disagree')
        def body(index):
            coords = self.coords(index, iteration_shape)
            allcoords = list(offsets)
            for dim, coord in zip(kept, coords):
                allcoords[dim] = self.b.op('add', offsets[dim], self.b.op('mul', coord, strides[dim]))
            other = self.linear(allcoords, fullshape)
            if insert:
                self.tensor_store(result, other, self.tensor_load(source, index))
            else:
                self.tensor_store(result, index, self.tensor_load(source, other))
        self.b.loop(0, prod(iteration_shape), 1, body)

    def concat(self, operation):
        result = operation.results[0]
        dim = operation.properties['dim'].value.data
        begin = 0
        for source in operation.operands:
            start = begin
            def body(index):
                coords = self.coords(index, shape(source))
                coords[dim] = self.b.op('add', coords[dim], start)
                offset = self.linear(coords, shape(result))
                self.tensor_store(result, offset, self.tensor_load(source, index))
            self.b.loop(0, prod(shape(source)), 1, body)
            begin += shape(source)[dim]

    def quantize(self, operation, dequantize=False):
        source, scale, zero = operation.operands[:3]
        result = operation.results[0]
        scale_value = self.tensor_load(scale, 0)
        zero_value = self.tensor_load(zero, 0)
        zero_float = self.b.cast('sitofp', zero_value, dtype(zero), 'float')
        qmin = operation.properties['quant_min'].value.data
        qmax = operation.properties['quant_max'].value.data
        def body(index):
            x = self.tensor_load(source, index)
            if dequantize:
                f = self.b.cast('sitofp', x, dtype(source), 'float')
                data = self.b.op('fmul', self.b.op('fsub', f, zero_float, 'float'), scale_value, 'float')
            else:
                divided = self.b.op('fdiv', x, scale_value, 'float')
                rounded = self.b.value(f'call i64 asm "fcvt.l.s $0, $1, rne", "=r,f"(float {divided})')
                z = zero_value if dtype(zero) == 'i64' else self.b.cast('sext', zero_value, dtype(zero), 'i64')
                shifted = self.b.op('add', rounded, z)
                low = self.b.cmp('slt', shifted, qmin)
                clipped_low = self.b.select(low, qmin, shifted)
                high = self.b.cmp('sgt', clipped_low, qmax)
                clipped = self.b.select(high, qmax, clipped_low)
                data = self.b.cast('trunc', clipped, 'i64', dtype(result))
            self.tensor_store(result, index, data)
        self.b.loop(0, prod(shape(result)), 1, body)

    def constant(self, operation):
        value = operation.results[0]
        if not isinstance(value.type, TensorType):
            self.scalar.emit(operation, self.values, ())
            slot = self.b.alloc(1, dtype(value), 8)
            self.b.store(self.values[value], slot, dtype(value))
            self.values[value] = self.b.load(slot, dtype(value))
            return
        attr = operation.properties.get('value', operation.attributes.get('value'))
        if not isinstance(attr, DenseIntOrFPElementsAttr):
            raise ValueError('Unsupported tensor constant')
        data = list(attr.get_values())
        if len(set(data)) == 1:
            self.fill(value, data[0])
        else:
            for i, item in enumerate(data):
                self.tensor_store(value, i, item)

    def operation(self, operation):
        name = op_name(operation)
        kind = self.program.device_kinds.get(operation)
        eligible = kind is not None
        if eligible:
            self.contraction(operation, kind)
        elif name == 'arith.constant':
            self.constant(operation)
        elif name == 'tensor.empty':
            self.fill(operation.results[0], 0)
        elif name == 'tensor.splat':
            self.fill(operation.results[0], self.values[operation.operands[0]])
        elif name in ('tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'):
            self.copy(operation.operands[0], operation.results[0])
        elif name == 'linalg.generic':
            self.generic(operation)
        elif name == 'linalg.transpose':
            self.transpose(operation)
        elif name == 'linalg.reduce':
            self.reduction(operation)
        elif name == 'tensor.extract_slice':
            self.slice(operation)
        elif name == 'tensor.insert_slice':
            self.slice(operation, True)
        elif name == 'tensor.concat':
            self.concat(operation)
        elif name in ('quant_ext.quantize_per_tensor', 'quant_ext.dequantize_per_tensor'):
            self.quantize(operation, name == 'quant_ext.dequantize_per_tensor')
        elif name == 'linalg.fill':
            self.fill(operation.results[0], self.values[operation.operands[0]])
        else:
            raise ValueError('Unsupported whole-source operation ' + name)

    def emit(self):
        ranges = []
        groups = [[operation] for operation in self.program.operations]
        for task_index, operations in enumerate(groups):
            self.b.task = 'task_' + str(task_index)
            self.b.emit(f'call void asm sideeffect "# merlin.task {task_index}", ""()')
            start = len(self.b.instruction_tasks)
            for operation in operations:
                self.operation(operation)
            end = len(self.b.instruction_tasks)
            ranges.append([start, end])
        self.b.task = None
        result = self.b.finish()
        self.buffer['_source_instruction_tasks'] = self.b.instruction_tasks
        self.buffer['_source_task_ranges'] = ranges
        return result


def emit(text, buffer):
    from mixed_frontend import parsed_source
    return SourceEmitter(parsed_source(text), buffer).emit()
