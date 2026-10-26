"""Resolve authenticated symbolic CONFIG_ST operands only for concrete execution."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .runtime_scale import RuntimeScaleRef, validate_scale_ref

# Bit patterns preserve signed zero and subnormals. These are backstop trials,
# never a restriction on the admitted finite-f32 source domain.
SCALE_TRIAL_BITS = (
    0,
    0x80000000,
    0x3E800000,
    0x3F000000,
    0x3F800000,
    0x3E99999A,
    0xBE800000,
    0xBF000000,
    1,
    0x7F7FFFFF,
    0x80000001,
    0xFF7FFFFF,
)


def runtime_scale_bits(
    ref: RuntimeScaleRef, args, buffers: Mapping[int, np.ndarray]
) -> int:
    validate_scale_ref(ref, args)
    if ref.arg_index not in buffers:
        raise ValueError("runtime scale input buffer is absent")
    value = np.asarray(buffers[ref.arg_index])
    if not value.flags.c_contiguous:
        raise ValueError("runtime scale input storage must be contiguous")
    if value.dtype == np.dtype(np.uint8):
        if value.ndim != 1 or value.nbytes != 4:
            raise ValueError("runtime scale byte buffer must have four bytes")
    elif value.dtype != np.dtype(np.float32) or value.shape != (1,):
        raise ValueError("runtime scale input requires f32[1] storage")
    raw = value.view(np.uint8).reshape(-1)
    bits = int.from_bytes(
        raw[ref.byte_offset : ref.byte_offset + 4].tobytes(), "little"
    )
    if bits & 0x7F800000 == 0x7F800000:
        raise ValueError("runtime scale must be finite f32")
    return bits


def runtime_scale_value(ref: RuntimeScaleRef, args, buffers) -> np.float32:
    return np.array([runtime_scale_bits(ref, args, buffers)], dtype=np.uint32).view(
        np.float32
    )[0]


def resolve_config_instruction(ins, args, buffers):
    ref = ins.a("scale_operand")
    if ref is None:
        return ins
    if (
        ins.kind != "config_st"
        or ins.arg is not None
        or type(ins.off) is not int
        or ins.off != 0
    ):
        raise ValueError("runtime scale operand is only legal on CONFIG_ST")
    attrs = dict(ins.attrs)
    if (
        len(ins.attrs) != 3
        or set(attrs) != {"stride", "acc_act", "scale_operand"}
        or type(attrs["stride"]) is not int
        or type(attrs["acc_act"]) is not int
        or attrs["acc_act"] not in (0, 1)
    ):
        raise ValueError("unknown, ambiguous or pooled runtime scale configuration")
    bits = runtime_scale_bits(ref, args, buffers)
    del attrs["scale_operand"]
    attrs.update(
        multiplier_bits=bits,
        acc_scale=np.array([bits], dtype=np.uint32).view(np.float32)[0].item(),
    )
    return ins._replace(attrs=tuple(sorted(attrs.items())))


def runtime_scale_refs(stream):
    from .stream import CompactInstructions

    refs = set()
    instructions = (
        stream.ins.templates()
        if isinstance(stream.ins, CompactInstructions)
        else stream.ins
    )
    for ins in instructions:
        ref = ins.a("scale_operand")
        if ref is not None:
            validate_scale_ref(ref, stream.args)
            if ins.kind != "config_st":
                raise ValueError("runtime scale token on a non-CONFIG_ST instruction")
            refs.add(ref)
    return frozenset(refs)


def runtime_scale_trial(trial, *, offset=0):
    if type(trial) is not int or trial < 0:
        raise ValueError("scale trial index must be nonnegative")
    bits = SCALE_TRIAL_BITS[(trial + offset) % len(SCALE_TRIAL_BITS)]
    return np.array([bits], dtype=np.uint32).view(np.float32)


def source_scale_ref(scale_operand, module, operands, args):
    """Independently authenticate an existing binding against original source SSA.

    Admission must not add a missing operand or repair an emitter binding.
    Reconstruct the source record with Merlin's public reader, then require the
    exact source value to occur once in the kernel's existing input bindings.
    """
    from merlin.xdsl_dialects.lowering.runtime_readout import scale_operand_record

    if not isinstance(scale_operand, Mapping):
        raise TypeError("runtime scale source record is absent")
    source = scale_operand.get("source", {})
    if (
        not isinstance(source, Mapping)
        or type(source.get("func_arg")) is not int
        or type(source.get("element_index")) is not int
        or any(type(dim) is not int for dim in scale_operand.get("storage_shape", []))
    ):
        raise ValueError("runtime scale source indices must be integers")
    functions = [
        op for op in module.walk() if op.name == "func.func" and op.regions[0].blocks
    ]
    if len(functions) != 1:
        raise ValueError(
            "runtime scale admission requires one original source function"
        )
    values = []
    for value in functions[0].regions[0].blocks[0].args:
        try:
            record = scale_operand_record(value)
        except ValueError:
            continue
        if record == scale_operand:
            values.append(value)
    if len(values) != 1:
        raise ValueError("runtime scale source record differs from original argument")
    names = [name for name, value in operands if value is values[0]]
    indices = [
        i for i, arg in enumerate(args) if len(names) == 1 and arg.name == names[0]
    ]
    if len(indices) != 1:
        raise ValueError("runtime scale source is not bound to the exact kernel input")
    ref = RuntimeScaleRef(indices[0])
    validate_scale_ref(ref, args)
    return ref
