"""Readout instruction admission and the exact int32 host readout ABI.

The host fallback consumes full-width signed i32. Factors are reduced rationals,
serialized as decimal numerator/denominator strings; zero point is applied AFTER
RNE. No floating intermediate participates in a code decision.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .exactness import quantized


def validate_device_readout(stream, meta, *, numeric=None, match=None, readout_options=None):
    """Prove an integer contraction's emitted stream from its replayed dataflow.

    The stream is decoded completely and replayed against the source-derived
    expectation (``contraction_proof``); any unmodelled field refuses.
    """
    if meta.get('pooled_store'):
        from .pooled_store import validate
        return validate(stream, meta, numeric, match=match, readout_options=readout_options)
    from .contraction_proof import admit

    if match is None or match.kernel.stream is not stream or match.kernel.meta is not meta:
        raise ValueError('device readout admission requires the source match of the emitted stream')
    return admit(match, readout_options=readout_options)


@dataclass
class Readout:
    out: str
    terms: list[tuple[str, object]]
    constant: object = 0
    relu: bool = False
    zero_point: int = 0
    quant_min: int = -128
    quant_max: int = 127
    permutations: dict = None
    broadcasts: dict = None
    after: int | None = None
    placement: tuple | dict | None = None  # scalar-parameter destination region, in elements
    axis: int = -1


def expanded(value, shape, axis=-1):
    a = np.asarray(value, dtype=object)
    size = int(np.prod(shape))
    if a.size == size:
        return a.reshape(-1)
    if a.ndim == 1 and shape and a.size == shape[axis]:
        dims = [1]*len(shape)
        dims[axis] = a.size
        a = a.reshape(dims)
    return np.broadcast_to(a, shape).reshape(-1)


def parameters(readout, value, shape):
    return expanded(value, shape, readout.axis)


def evaluate(
    values,
    numerator,
    denominator,
    *,
    zero_point=0,
    quant_min=-128,
    quant_max=127,
    relu=False,
):
    """Pure-Python reference, including negative ties and asymmetric parity."""
    factor = Fraction(int(numerator), int(denominator))

    def scalar(x):
        real = int(x) * factor
        if relu:
            real = max(real, 0)
        return quantized(real, quant_min, quant_max, zero_point)

    dt = np.uint8 if quant_min >= 0 else np.int8
    a = np.asarray(values)
    return np.array([scalar(x) for x in a.flat], dtype=dt).reshape(a.shape)


def apply(stream, buffers, readouts=None):
    """Instruction-model host epilogue, separate from the Gemmini datapath."""
    by_name = {a.name: buffers[i] for i, a in enumerate(stream.args)}
    for readout in stream.readouts if readouts is None else readouts:
        out = by_name[readout.out]
        if readout.placement is not None:
            if isinstance(readout.placement,dict):
                from math import prod
                p = readout.placement
                count = prod(p['shape'])
                logical = np.arange(count,dtype=np.int64)
                positions = np.full(count,p['offset'],dtype=np.int64)
                for axis,(extent,stride) in enumerate(zip(p['shape'],p['strides'])):
                    positions += (logical//prod(p['shape'][axis+1:])%extent)*stride
            else:
                count, block, offset, stride = readout.placement
                positions = np.arange(count,dtype=np.int64)
                positions = positions//block*stride + positions%block + offset
            values = np.full(count,readout.constant,dtype=object)
            for name,factor in readout.terms:
                values += by_name[name].reshape(-1)[positions].astype(object)*factor
            if readout.relu:
                values = np.maximum(values, 0)
            out.reshape(-1)[positions] = [quantized(Fraction(v),readout.quant_min,readout.quant_max,readout.zero_point) for v in values]
            continue
        values = parameters(readout, readout.constant, out.shape).copy()
        for name, factor in readout.terms:
            raw = by_name[name]
            if readout.permutations and name in readout.permutations:
                shp, perm = readout.permutations[name]
                raw = raw.reshape(shp).transpose(perm)
            if readout.broadcasts and name in readout.broadcasts:
                raw = raw.reshape(readout.broadcasts[name])
            raw_values = expanded(raw, out.shape)
            values += raw_values.astype(object)*parameters(readout, factor, out.shape)
        if readout.relu:
            values = np.maximum(values, 0)
        zps, lows, highs = [parameters(readout, v, out.shape) for v in
                           (readout.zero_point, readout.quant_min, readout.quant_max)]
        out.flat[:] = np.array([quantized(Fraction(v), int(lo), int(hi), int(zp))
                                     for v,zp,lo,hi in zip(values,zps,lows,highs)], dtype=np.int64).astype(out.dtype)
