"""Source SSA and typed stream bindings for the tensor-wide runtime f32 gain."""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.ir import Operation


class RuntimeScaleRef(NamedTuple):
    arg_index: int
    byte_offset: int = 0


def validate_scale_ref(ref: RuntimeScaleRef, args: Sequence) -> None:
    if (not isinstance(ref, RuntimeScaleRef) or type(ref.arg_index) is not int
            or not 0 <= ref.arg_index < len(args)
            or type(ref.byte_offset) is not int or ref.byte_offset != 0):
        raise ValueError('runtime scale reference requires a valid argument and byte offset zero')
    arg = args[ref.arg_index]
    if (arg.role != 'input' or arg.dtype != 'f32' or arg.shape != (1,)
            or type(arg.shape[0]) is not int or type(arg.nbytes) is not int or arg.nbytes != 4):
        raise ValueError('runtime scale argument must be an input f32[1] with four-byte extent')


def _source_index(record):
    expected = dict(schema='merlin.readout_scale_operand.v1', kind='runtime_input',
                    dtype='f32', storage_shape=[1], granularity='tensor', domain='finite_f32')
    if not isinstance(record, Mapping) or set(record) != set(expected) | {'source'}:
        raise ValueError('invalid runtime scale operand schema')
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError('invalid runtime scale operand contract')
    source = record['source']
    if (not isinstance(source, Mapping) or set(source) != {'func_arg', 'element_index'}
            or type(source['func_arg']) is not int or source['func_arg'] < 0
            or type(source['element_index']) is not int or source['element_index'] != 0):
        raise ValueError('invalid runtime scale source argument or element index')
    if any(type(dim) is not int for dim in record['storage_shape']):
        raise ValueError('runtime scale shape must contain integer dimensions')
    return source['func_arg']


def _source_value(record, module):
    index = _source_index(record)
    functions = [op for op in module.walk() if op.name == 'func.func' and op.regions[0].blocks]
    if len(functions) != 1:
        raise ValueError('runtime scale requires one defined source function')
    args = functions[0].regions[0].blocks[0].args
    if index >= len(args):
        raise ValueError('runtime scale original function argument is out of range')
    value = args[index]
    if (not isinstance(value.type, TensorType) or value.type.get_shape() != (1,)
            or not isinstance(value.type.get_element_type(), Float32Type)):
        raise ValueError('runtime scale original argument must be tensor<1xf32>')
    return value


def bind_scale_operand(scale_operand: Mapping, module: Operation,
                       operands: list, stream_args: list) -> RuntimeScaleRef:
    """Re-resolve original SSA; append or reuse its exact input argument binding."""
    from .stream import Arg

    value = _source_value(scale_operand, module)
    matches = [name for name, source in operands if source is value]
    if len(matches) > 1:
        raise ValueError('runtime scale SSA has ambiguous kernel operand bindings')
    if matches:
        indices = [i for i, arg in enumerate(stream_args) if arg.name == matches[0]]
        if len(indices) != 1:
            raise ValueError('runtime scale binding has no unique stream argument')
        ref = RuntimeScaleRef(indices[0])
    else:
        names = {a.name for a in stream_args} | {name for name, _ in operands}
        ordinal = 0
        while f'readout_scale{ordinal}' in names:
            ordinal += 1
        name = f'readout_scale{ordinal}'
        ref = RuntimeScaleRef(len(stream_args))
        stream_args.append(Arg(name, 'f32', (1,), 'input'))
        operands.append((name, value))
    validate_scale_ref(ref, stream_args)
    return ref


def validate_runtime_config(ins, args=None):
    """Only the whole CONFIG_ST scale slot may be dynamic; DMA semantics stay empty."""
    if (ins.kind != 'config_st' or ins.arg is not None or type(ins.off) is not int or ins.off != 0
            or len(ins.attrs) != 3 or set(dict(ins.attrs)) != {'stride', 'acc_act', 'scale_operand'}):
        raise ValueError('runtime scale requires a plain CONFIG_ST without DMA or pooled fields')
    ref = ins.a('scale_operand')
    if (not isinstance(ref, RuntimeScaleRef) or type(ref.arg_index) is not int or ref.arg_index < 0
            or type(ref.byte_offset) is not int or ref.byte_offset != 0):
        raise ValueError('invalid runtime scale reference')
    if type(ins.a('stride')) is not int or type(ins.a('acc_act')) is not int:
        raise ValueError('runtime CONFIG_ST fields must be integers')
    if ins.a('acc_act') not in (0, 1):
        raise ValueError('runtime CONFIG_ST supports no activation or accumulator ReLU')
    if args is not None:
        validate_scale_ref(ref, args)
    return ref


def validate_runtime_record(rec, args):
    """Check the discriminant and every static/reserved bit before register insertion."""
    from .readout_encoding import code, config_st, unpack

    if len(rec.key) != 3 or rec.key[:2] != (code('semantic_class', 'CONFIG'), None):
        raise ValueError('runtime record must be a CONFIG with no DMA argument')
    validate_scale_ref(rec.key[2], args)
    if len(rec.values) != 2 or any(type(v) is not int or not 0 <= v < 1 << 64 for v in rec.values):
        raise ValueError('runtime record requires two unsigned register words')
    rs1, rs2 = rec.values
    act = unpack(rs1, 'ConfigMvoutRs1', 'activation')
    stride = unpack(rs2, 'ConfigMvoutRs2', 'stride')
    if act not in (0, 1) or config_st(stride, act, 0)[1:] != (rs1, rs2):
        raise ValueError('runtime CONFIG_ST has nonzero scale slot or invalid static/reserved bits')
