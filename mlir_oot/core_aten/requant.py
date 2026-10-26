"""Readout instruction admission and the exact int32 host readout ABI.

The host fallback consumes full-width signed i32. Factors are reduced rationals,
serialized as decimal numerator/denominator strings; zero point is applied AFTER
RNE. No floating intermediate participates in a code decision.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .exactness import quantized


def _validate_execute_config(config):
    from ..tables import isa

    expected = dict(dataflow=isa.WEIGHT_STATIONARY, act=isa.NO_ACTIVATION,
                    acc_scale=1, a_stride=1, c_stride=1, a_transpose=0, b_transpose=0)
    if config is None or any(config.a(k) != v for k, v in expected.items()):
        raise ValueError('effective CONFIG_EX differs from the proved contraction layout/semantics')


def _validate_load_config(stream, ins, config, address_flags, centered):
    from .storage import itemsize
    from .stream import DIM

    expected = dict(scale=1, shrunk=0, block_stride=DIM, pixel_repeats=1)
    if not ins.a('zero'):
        if ins.arg is None or not 0 <= ins.arg < len(stream.args):
            raise ValueError('load has no proved source argument')
        arg = stream.args[ins.arg]
        accumulator = ins.a('local') & address_flags['acc_i8']
        # Centered byte selectors use separately proved scratchpad gathers.
        if accumulator or not centered:
            expected['stride'] = (0 if accumulator and len(arg.shape) == 1 else
                                  arg.shape[-1] * itemsize(arg.dtype))
    if config is None or any(config.a(k) != v for k, v in expected.items()):
        raise ValueError('effective CONFIG_LD differs from the proved seed/operand load layout/semantics')


def _validate_readout_configs(stream, meta, address_flags, narrow):
    """Check effective state at every use, including changes after PRELOAD.

    Keep consumed load configurations live through compute/readout: rebinding
    one without a new proved load is outside this contraction proof.
    """
    from .storage import itemsize
    from ..tables import isa

    execute, store = None, None
    loads, consumed = {}, {}
    for ins in stream.ins:
        if ins.kind == 'config_ex':
            execute = ins
        elif ins.kind == 'config_st':
            store = ins
        elif ins.kind == 'config_ld':
            loads[ins.a('load_id')] = ins
        elif ins.kind == 'mvin':
            config = loads.get(ins.a('load_id'))
            _validate_load_config(stream, ins, config, address_flags,
                                  meta.get('centered_byte_contraction'))
            consumed[ins.a('load_id')] = config
        if (ins.kind in ('preload', 'compute') or
                ins.kind == 'mvout' and ins.a('local') & address_flags['acc_i8']):
            _validate_execute_config(execute)
            if any(loads.get(state) != config for state, config in consumed.items()):
                raise ValueError('effective CONFIG_LD changed after a proved load')
        if ins.kind == 'mvout' and ins.a('local') & address_flags['acc_i8']:
            output = stream.args[ins.arg]
            if store is None or store.a('stride') != output.shape[-1] * itemsize(output.dtype):
                raise ValueError('effective CONFIG_ST differs from the proved readout layout')
            if not narrow and (store.a('acc_scale') != 1 or store.a('acc_act') != isa.NO_ACTIVATION):
                raise ValueError('effective CONFIG_ST differs from the proved full-width semantics')


def _validate_readout_seed(stream, numeric, address_flags):
    expected = numeric.get('seed')
    loads = [i for i in stream.ins if i.kind == 'mvin' and i.a('local') & address_flags['acc_i8']]
    if expected is None:
        if loads:
            raise ValueError('device readout contains an unproved accumulator seed')
        return
    _validate_seed_tiles(stream, expected, address_flags)


def _validate_seed_load(stream, ins, expected, config):
    from .emit import encode
    from .readout_encoding import code
    from .stream import DIM

    arg = stream.args[ins.arg] if ins.arg is not None and 0 <= ins.arg < len(stream.args) else None
    if (encode(ins).key[0] != code('semantic_class', 'MVIN3') or ins.a('zero') or
            arg is None or arg.dtype != 'i32' or arg.role != 'const' or
            arg.name not in stream.consts or stream.consts[arg.name].dtype != np.dtype('int32') or
            not np.array_equal(stream.consts[arg.name], expected) or
            tuple(arg.shape) != expected.shape):
        raise ValueError('MVIN3 seed differs from the source-proved bias')
    if config is None or any(config.a(k) != v for k, v in dict(
            scale=1, shrunk=0, block_stride=DIM, pixel_repeats=1).items()):
        raise ValueError('MVIN3 seed width/layout differs from the source-proved bias')


def _validate_seed_slice(stream, store, seed, config, expected):
    from .storage import itemsize

    output = stream.args[store.arg]
    columns = output.shape[-1]
    width = itemsize('i32')
    start = store.off // itemsize(output.dtype)
    broadcast = expected.shape == (columns,)
    if not broadcast and expected.shape != tuple(output.shape):
        raise ValueError('MVIN3 seed shape differs from the output region')
    offset = (start % columns if broadcast else start) * width
    stride = 0 if broadcast else columns * width
    if seed.off != offset or config.a('stride') != stride:
        raise ValueError('MVIN3 seed offset/stride differs from the source-proved bias slice')


def _seed_output_tiles(stream):
    from math import prod
    from .storage import itemsize
    from .stream import DIM

    tiles = set()
    for index, arg in enumerate(stream.args):
        if arg.role != 'output':
            continue
        rows, columns = arg.shape[-2:]
        for batch in range(prod(arg.shape[:-2])):
            for row in range(0, rows, DIM):
                for col in range(0, columns, DIM):
                    offset = (batch * rows * columns + row * columns + col) * itemsize(arg.dtype)
                    tiles.add((index, offset, min(DIM, rows-row), min(DIM, columns-col)))
    return tiles


def _validate_seed_tiles(stream, expected, address_flags):
    """Bind each seed lifetime to compute and its eventual output region.

    A store ends a tile lifetime: local accumulator rows can then be reused,
    but every subsequent output region must receive its own seed before MACs.
    """
    from .stream import DIM

    expected = np.asarray(expected)
    remaining = _seed_output_tiles(stream)
    configs, tiles = {}, {}
    preload = None
    mask = address_flags['acc_i8'] | address_flags['acc_accum'] | address_flags['full_c_bit']
    for ins in stream.ins:
        if ins.kind == 'config_ld':
            configs[ins.a('load_id')] = ins
        elif ins.kind == 'mvin' and ins.a('local') & address_flags['acc_i8']:
            config = configs.get(ins.a('load_id'))
            _validate_seed_load(stream, ins, expected, config)
            row = ins.a('local') & ~mask
            rows, cols = ins.a('rows'), ins.a('cols')
            if (ins.a('local') != (row | address_flags['acc_i8']) or
                    not 1 <= rows <= DIM or not 1 <= cols <= DIM or
                    any(row < r + t[0].a('rows') and r < row + rows for r, t in tiles.items())):
                raise ValueError('MVIN3 seed overlaps or duplicates an accumulator tile')
            tiles[row] = (ins, config, False)
        elif ins.kind == 'preload':
            row = ins.a('c') & ~mask
            tile = tiles.get(row)
            if (tile is None or preload is not None or
                    ins.a('c') != (row | address_flags['acc_i8'] | address_flags['acc_accum']) or
                    ins.a('c_rows') != tile[0].a('rows') or
                    ins.a('c_cols') != tile[0].a('cols')):
                raise ValueError('accumulator compute lacks a matching prior MVIN3 seed tile')
            preload = row
        elif ins.kind == 'compute':
            if preload is None or preload not in tiles:
                raise ValueError('accumulator compute lacks a prior seeded preload')
            seed, config, _ = tiles[preload]
            tiles[preload] = (seed, config, True)
            preload = None
        elif ins.kind == 'mvout' and ins.a('local') & address_flags['acc_i8']:
            row = ins.a('local') & ~mask
            tile = tiles.pop(row, None)
            if (tile is None or not tile[2] or ins.a('rows') != tile[0].a('rows') or
                    ins.a('cols') != tile[0].a('cols')):
                raise ValueError('output region lacks exactly one complete MVIN3 seed tile')
            region = (ins.arg, ins.off, ins.a('rows'), ins.a('cols'))
            if region not in remaining:
                raise ValueError('output region duplicates or differs from the complete seed layout')
            remaining.remove(region)
            _validate_seed_slice(stream, ins, tile[0], tile[1], expected)
    if tiles or preload is not None or remaining:
        raise ValueError('MVIN3 seed tile has no completed output region')


def _validate_readout_registers(stream, bits, relu):
    from .emit import encode
    from .readout_encoding import activation, unpack
    configured = None
    for ins in stream.ins:
        if ins.kind == 'config_st':
            rec = encode(ins)
            configured = (unpack(rec.values[1], 'ConfigMvoutRs2', 'acc_scale'),
                          unpack(rec.values[0], 'ConfigMvoutRs1', 'activation'),
                          unpack(rec.values[1], 'ConfigMvoutRs2', 'stride'))
        if ins.kind == 'mvout':
            columns = stream.args[ins.arg].shape[-1]
            start = ins.off % columns
            expected = bits[start:start + ins.a('cols')]
            if (len(expected) != ins.a('cols') or len(set(expected)) != 1 or
                    configured != (expected[0], activation(relu), columns)):
                raise ValueError('decoded readout fields differ from the proved multiplier bits')


def validate_device_readout(stream, meta, *, numeric=None):
    """Authenticate actual stores and their decoded CONFIG_ST multiplier bits."""
    from .readout_encoding import flags
    from ..tables import isa

    plan = meta.get('composition_plan')
    narrow = meta.get('readout') == 'f32_product_rne'
    if narrow and (plan is None or plan['decision']['cut'] != 'requant' or stream.readouts or stream.packs):
        raise ValueError('device readout lacks a closed composition proof')
    bits = meta.get('multiplier_bits', ())
    address_flags = flags() if narrow else dict(acc_i8=isa.ACC_ADDR_BIT, full_c_bit=isa.ACC_FULL_ROW_BIT)
    _validate_readout_configs(stream, meta, address_flags, narrow)
    if narrow:
        if list(bits) != plan['decision']['proof'].get('multiplier_bits'):
            raise ValueError('readout multiplier bits differ from the composition proof')
        _validate_readout_seed(stream, numeric or {}, address_flags)
        _validate_readout_registers(stream, bits, meta['relu'])
    stores = 0
    for ins in stream.ins:
        if ins.kind == 'config_ld' and ins.a('scale') != 1:
            raise ValueError('exact producer contains an unproved load scale')
        if ins.kind == 'config_st' and not narrow and ins.a('acc_scale') != 1:
            raise ValueError('full-width producer contains an unproved readout scale')
        if ins.kind != 'mvout':
            continue
        if not ins.a('local') & address_flags['acc_i8']:
            if not meta.get('centered_byte_contraction'):
                raise ValueError('exact producer contains an unproved scratchpad store')
            continue
        stores += 1
        full = bool(ins.a('local') & address_flags['full_c_bit'])
        if full == narrow:
            raise ValueError('accumulator store width differs from the composition exit')
    if not stores:
        raise ValueError('exact producer emitted no accumulator stores')
    return 'f32_product_rne' if narrow else 'full_i32'


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
