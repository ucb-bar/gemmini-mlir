"""Exact source matchers for standalone integer accumulator datapath forms."""
from __future__ import annotations
from math import prod
from xdsl.dialects import builtin
from xdsl.ir.affine import AffineMap
from merlin.llvmlower.host_tensor_codegen import attr_of


def _literal(value, wanted):
    owner = value.owner
    if getattr(owner, 'name', None) != 'arith.constant':
        return False
    attr = attr_of(owner, 'value')
    return isinstance(attr, builtin.IntegerAttr) and attr.value.data == wanted


def _bound(op, value, wanted):
    others = [v for v in op.operands if v is not value]
    return value in op.operands and len(others) == 1 and _literal(others[0], wanted)


def recognize(op):
    """Admit closed same-shape parallel bodies, never an operation-name hint."""
    if op.name != 'linalg.generic' or len(op.results) != 1 or len(op.outputs) != 1 or len(op.inputs) not in (1, 2):
        return None
    if any(not name.startswith('prov.') for name in op.attributes):
        return None
    if set(op.properties) - {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}:
        return None
    values = list(op.inputs) + list(op.outputs) + list(op.results)
    if any(not isinstance(v.type, builtin.TensorType) or not isinstance(v.type.encoding, builtin.NoneAttr) for v in values):
        return None
    shape = tuple(op.results[0].type.get_shape())
    if any(x <= 0 for x in shape) or any(tuple(v.type.get_shape()) != shape for v in values):
        return None
    if any(mp.data != AffineMap.identity(len(shape)) for mp in op.indexing_maps):
        return None
    if any('parallel' not in str(kind.data).lower() for kind in op.iterator_types):
        return None
    body = op.regions[0].block
    instructions = list(body.ops)
    # No arithmetic flags, executable library calls or unknown attributes.
    for ins in instructions:
        if any(not key.startswith('prov.') for key in ins.attributes):
            return None
        if ins.name == 'arith.addi' and getattr(ins, 'overflow_flags', None) is not None and ins.overflow_flags.data:
            return None
    args = list(body.args)
    dtype = [str(value.type.element_type) for value in op.inputs]
    output = str(op.results[0].type.element_type)
    names = [ins.name for ins in instructions]
    recipe = dict(inputs=tuple(op.inputs), output=op.results[0], shape=shape, output_dtype=output)
    if len(op.inputs) == 1:
        source = args[0]
        if names == ['linalg.yield'] and list(instructions[0].operands) == [source] and dtype[0] == output and output in ('i8', 'i32'):
            return dict(recipe, kind='movement', attrs={})
        if names == ['arith.extsi', 'linalg.yield'] and dtype == ['i8'] and output == 'i32' and list(instructions[0].operands) == [source] and list(instructions[1].operands) == list(instructions[0].results):
            return dict(recipe, kind='movement', attrs={})
        if names == ['arith.maxsi', 'linalg.yield'] and dtype == ['i8'] and output == 'i8' and _bound(instructions[0], source, 0) and list(instructions[1].operands) == list(instructions[0].results):
            return dict(recipe, kind='movement', attrs={'epilogue': ['relu']})
        if names == ['arith.maxsi', 'arith.minsi', 'arith.trunci', 'linalg.yield'] and dtype == ['i32'] and output == 'i8':
            lo, hi, cast, yld = instructions
            if _bound(lo, source, -128) and _bound(hi, lo.results[0], 127) and list(cast.operands) == list(hi.results) and list(yld.operands) == list(cast.results):
                return dict(recipe, kind='movement', attrs={})
        return None
    if dtype == ['i32', 'i32'] and output == 'i32' and names == ['arith.addi', 'linalg.yield']:
        add, yld = instructions
        if set(add.operands) == set(args[:2]) and list(yld.operands) == list(add.results):
            return dict(recipe, kind='add_i32', attrs={})
    if dtype != ['i8', 'i8'] or output not in ('i8', 'i32') or names[:3] != ['arith.extsi', 'arith.extsi', 'arith.addi']:
        return None
    left, right, add = instructions[:3]
    if list(left.operands) != [args[0]] or list(right.operands) != [args[1]] or str(left.results[0].type) != 'i32' or str(right.results[0].type) != 'i32' or set(add.operands) != {left.results[0], right.results[0]}:
        return None
    if output == 'i32' and names == ['arith.extsi', 'arith.extsi', 'arith.addi', 'linalg.yield'] and list(instructions[3].operands) == list(add.results):
        return dict(recipe, kind='add_i8', attrs={})
    if output == 'i8' and names == ['arith.extsi', 'arith.extsi', 'arith.addi', 'arith.maxsi', 'arith.minsi', 'arith.trunci', 'linalg.yield']:
        lo, hi, cast, yld = instructions[3:]
        if _bound(lo, add.results[0], -128) and _bound(hi, lo.results[0], 127) and list(cast.operands) == list(hi.results) and list(yld.operands) == list(cast.results):
            return dict(recipe, kind='add_i8', attrs={})
    return None


def emit(fn, record, pointers):
    from gemmini_pointwise import emit_movement, emit_residual, _tiles, _ptr
    import gemmini_encode as enc
    inputs = [pointers[value] for value in record['inputs']]
    destination = pointers[record['output']]
    shape, dtype = record['shape'], record['output_dtype']
    if record['kind'] == 'movement':
        return emit_movement(fn, inputs[0], destination, shape,
            dtype=str(record['inputs'][0].type.element_type), attrs=dict(record['attrs'], output_dtype=dtype))
    if record['kind'] == 'add_i8':
        return emit_residual(fn, *inputs, destination, 1, prod(shape), output_dtype=dtype, attrs=record['attrs'])
    if record['kind'] != 'add_i32':
        raise ValueError('unknown exact graph integer device recipe')
    enc.fence(fn)
    enc.config_load(fn, 64, stream=0, shrunk=False)
    enc.config_load(fn, 64, stream=1, shrunk=False)
    enc.config_store(fn, 64)
    def tile(offset, rows, cols):
        enc.mvin(fn, _ptr(fn, inputs[0], offset, 4), enc.ACC, rows, cols, stream=0)
        enc.fence(fn)
        enc.mvin(fn, _ptr(fn, inputs[1], offset, 4), enc.ACC_ADD, rows, cols, stream=1)
        enc.fence(fn)
        enc.mvout(fn, _ptr(fn, destination, offset, 4), enc.ACC_FULL, rows, cols)
        enc.fence(fn)
    _tiles(fn, prod(shape), tile)


def source_readout_decisions(module):
    """Source-aware fusion admission hook; ordinary placement stays with the graph owner.

    Calling this audit never removes observers or changes the pointwise matcher.
    A successful plan still needs explicit original producer-tile consumption.
    """
    from gemmini_source_readout import analyze
    return analyze(module)
