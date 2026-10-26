"""Scaled-load operand sums with a pooled readout tail: one kernel, proof, emission and admission.

Fixtures are retained seal #4 pre-campaign payload sources (generic form):

* ``per_operand_unpadded``: candidate 0170587dc05f..., case 65345230b42d... (aligned)
* ``tensor_unpadded_17``: candidate 08f3fb7c61f4..., case 4157a2f0770c... (partial, 17 channels)
* ``per_operand_padded``: candidate 04bec94f074a..., case 45fa4eb6a203... (aligned, -inf border)
* ``tensor_padded_17``: candidate 2ba106d16d62..., case 17d9716c74e8... (partial, 17 channels, -inf border)
"""

import copy
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten import operand_sum
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.readout_encoding import activation
from mlir_oot.core_aten.stream_proof import execute

PACKAGE = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'scaled_pool'
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32',
    readout=dict(strategy='f32_product_rne'), internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))
ROUNDING = dict(per_operand_unpadded='per_operand', per_operand_padded='per_operand',
                tensor_unpadded_17='single_rounding', tensor_padded_17='single_rounding')
PADDED = {'per_operand_padded', 'tensor_padded_17'}


def provider():
    spec = importlib.util.spec_from_file_location('scaled_pool_provider',
        PACKAGE / 'merlin-support/core_aten/execution_provider.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(rounding, *, strategy='f32_product_rne', pool=True):
    readout = [dict(kind='activation', function='relu', domain='accumulator'),
               dict(kind='convert', to='f32', rounding='half_even'),
               dict(kind='scale', dtype='f32', rounding='half_even', granularity='per_command', columns_per_scale=16),
               dict(kind='requant', to='i8', rounding='half_even', clamp=[-128, 127])]
    if pool:
        readout.append(dict(kind='pool', padding='zero_tap', reductions=['max']))
    return dict(path=dict(schema='output_path_v1', accumulator=dict(dtype='i32'),
                          accumulator_stages=[dict(kind='accumulate', dtype='i32', operand_scale='none'),
                                              dict(kind='preload', dtype='i32', range=[-(1 << 31), (1 << 31) - 1],
                                                   rounding='none')],
                          readout=readout, exits=dict(accumulator=True, requant=True)),
                scheme=dict(readout_strategy=strategy, operand_sum_rounding=rounding), facts={})


def source(name):
    return (FIXTURES / f'{name}.mlir').read_text()


def scaled(text, selected):
    _, manifest, matches = build(text, readout_options=selected)
    return manifest, [m for m in matches if m.op == 'scaled_operand_sum']


def _f32_constants(text, name):
    """The f32 column constant ``%name`` (hex dense splat or per-column), else None."""
    found = re.search(r'%' + name + r' = "arith.constant"\(\) <\{value = dense<"0x([0-9A-F]+)">', text)
    return None if found is None else np.frombuffer(bytes.fromhex(found.group(1)), '<f4').astype(np.float32)


def independent(text, a, b):
    """A numpy f32 model of the fixture source, written from the MLIR text, not from the provider."""
    signature = re.search(r'-> tensor<(\d+)x(\d+)x(\d+)x(\d+)xi8>\}', text)
    rows, cols = a.shape
    side = int(round(rows ** 0.5))

    def gains(name):
        g = _f32_constants(text, name)
        return None if g is None else np.broadcast_to(g, (cols,)).astype(np.float32)

    def requant(x, g):
        return np.clip(np.rint(x.astype(np.float32) * g), -128, 127).astype(np.int64)
    left, right = gains('left_requant_gains'), gains('right_requant_gains')
    x = requant(a, left) if left is not None else a.astype(np.int64)
    y = requant(b, right) if right is not None else b.astype(np.int64)
    total = x + y
    value = total.astype(np.float32) * gains('gains').astype(np.float32)
    if '%activation = ' in text:
        value = np.maximum(value, np.float32(0))
    image = value.reshape(1, side, side, cols).astype(np.float64)
    pad = 1 if 'tensor.insert_slice' in text else 0
    framed = np.pad(image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), constant_values=-np.inf)
    out_rows = int(signature.group(2))
    out = np.full((1, out_rows, out_rows, cols), -np.inf)
    for i in range(2):
        for j in range(2):
            out = np.maximum(out, framed[:, i:i + 2 * out_rows:2, j:j + 2 * out_rows:2])
    return np.clip(np.rint(out.astype(np.float32)), -128, 127).astype(np.int64)


def run(stream, a, b):
    names = [arg.name for arg in stream.args]
    buffers = {i: (dict(a=a, b=b)[n].astype(np.int8).copy() if n in 'ab' else np.full(arg.shape, 0x5A, np.int8))
               for i, (n, arg) in enumerate(zip(names, stream.args))}
    execute(stream, buffers)
    return buffers[names.index('y')].astype(np.int64)


@pytest.mark.parametrize('name', sorted(ROUNDING))
def test_retained_payload_is_one_admitted_pooled_kernel(name):
    from merlin_experiments.phase0.fusion_execution import prove_placement
    text, selected = source(name), options(ROUNDING[name])
    manifest, (match,) = scaled(text, selected)
    traced = [op for op in match.consumed if 'prov.trace_role' in op.attributes]
    roles = {op.attributes['prov.trace_role'].data for op in traced}
    assert roles == set(re.findall(r'prov.trace_role = "(\w+)"', text))
    from mlir_oot.core_aten.match import name_of
    assert name_of(match.root) == 'quant_ext.quantize_per_tensor'
    proof = match.kernel.meta['scaled_operand_sum']
    assert proof['form'] == 'staged_pooled_sum' and proof['operand_sum_rounding'] == ROUNDING[name]
    assert proof['pool']['zero_taps'] == (name in PADDED)
    assert proof['pool']['nonnegative_codes'] is proof['relu']  # every fixture with a rectifier
    assert not any(i.kind in ('compute', 'preload') for i in match.kernel.stream.ins)
    admitted = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                                 readout_options=selected)
    assert admitted['status'] == 'proven', admitted
    assert len(admitted['kernels']) == 1 and admitted['kernels'][0]['operand_sum'] == proof
    placement, reason = prove_placement(text, admitted)
    assert reason is None and placement is not None
    rng = np.random.default_rng(len(name))
    for trial in range(4):
        a = rng.integers(-128, 128, (proof['rows'], proof['cols'])).astype(np.int8)
        b = rng.integers(-128, 128, (proof['rows'], proof['cols'])).astype(np.int8)
        if trial == 0:
            a[:], b[:] = 127, 127
        elif trial == 1:
            a[:], b[:] = -128, -128
        np.testing.assert_array_equal(run(match.kernel.stream, a, b).reshape(-1), independent(text, a, b).reshape(-1))


def test_padded_window_uses_one_register_padded_band_and_unpadded_loads_unread_pixels():
    from mlir_oot.core_aten.stream_proof import decode
    pooled = {}
    for name in ('per_operand_padded', 'per_operand_unpadded'):
        _, (match,) = scaled(source(name), options('per_operand'))
        stores = [decode(i).fields for i in match.kernel.stream.ins if i.kind == 'config_st']
        pooled[name] = [f for f in stores if f['pool_stride']]
        assert stores[-1]['pool_stride'] == 0
        loads = [decode(i) for i in match.kernel.stream.ins if i.kind == 'mvin']
        rows = {(ld.arg, (ld.off // 16) + r) for ld in loads for r in range(ld.fields['num_rows'])}
        assert len(rows) == 2 * 289  # every pixel of both operands, the floor-mode unread row too
    padded, unpadded = pooled['per_operand_padded'][0], pooled['per_operand_unpadded'][0]
    assert (padded['upad'], padded['lpad'], padded['orows'], padded['ocols'], padded['porows']) == (1, 1, 17, 17, 9)
    assert (unpadded['upad'], unpadded['lpad'], unpadded['orows'], unpadded['ocols'], unpadded['porows']) == (
        0, 0, 16, 17, 8)


def _drop_activation(text):
    text = re.sub(r'    %activation_empty = .*?\n', '', text)
    text = re.sub(r'    %activation = "linalg.generic".*?node0_activation"\} : [^\n]*\n', '', text, flags=re.S)
    return text.replace('"tensor.expand_shape"(%activation)', '"tensor.expand_shape"(%scale)')


def _twin(kind):
    text, name = source('per_operand_padded'), 'per_operand'
    selected = options(name)
    if kind == 'padded_without_rectifier':
        text = _drop_activation(text)  # a zero tap would raise negative window maxima
    elif kind == 'quantize_scale':
        text = text.replace('%qs = "arith.constant"() <{value = dense<"0x0000803F">',
                            '%qs = "arith.constant"() <{value = dense<"0x0000003F">')
    elif kind == 'quantize_zero_point':
        text = text.replace('%qz = "arith.constant"() <{value = dense<0>', '%qz = "arith.constant"() <{value = dense<3>')
    elif kind == 'pool_init':
        text = text.replace('%ninf = "arith.constant"() <{value = 0xff800000 : f32}>',
                            '%ninf = "arith.constant"() <{value = 0.000000e+00 : f32}>')
    elif kind == 'border_value':
        text = text.replace('%pz = "arith.constant"() <{value = 0xff800000 : f32}>',
                            '%pz = "arith.constant"() <{value = 0.000000e+00 : f32}>')
    elif kind == 'host_readout_strategy':
        selected = options(name, strategy='exact_rational_host')
    elif kind == 'path_without_pool':
        selected = options(name, pool=False)
    elif kind == 'selected_single_rounding':
        selected = options('single_rounding')
    elif kind == 'operand_multiplier_above_one':
        text = text.replace('%left_requant_gains = "arith.constant"() <{value = dense<"0xABAA2A3E">',
                            '%left_requant_gains = "arith.constant"() <{value = dense<"0x0000C03F">')
    return text, selected


@pytest.mark.parametrize('kind', ['padded_without_rectifier', 'quantize_scale', 'quantize_zero_point', 'pool_init',
                                  'border_value', 'host_readout_strategy', 'path_without_pool',
                                  'selected_single_rounding', 'operand_multiplier_above_one'])
def test_refused_twins_keep_no_pooled_kernel(kind):
    text, selected = _twin(kind)
    _, matches = scaled(text, selected)
    assert not matches
    admitted = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                                 readout_options=selected)
    assert admitted['status'] != 'proven' or not any(
        k.get('operand_sum', {}).get('pool') for k in admitted.get('kernels', []))


def test_unpadded_without_rectifier_admits_signed_window_maxima():
    text = _drop_activation(source('per_operand_unpadded'))
    _, (match,) = scaled(text, options('per_operand'))
    proof = match.kernel.meta['scaled_operand_sum']
    assert proof['relu'] is False and proof['pool']['zero_taps'] is False and proof['pool']['nonnegative_codes'] is False
    a = np.full((proof['rows'], proof['cols']), -128, np.int8)
    b = np.full((proof['rows'], proof['cols']), -100, np.int8)
    out = run(match.kernel.stream, a, b)
    assert out.min() < 0
    np.testing.assert_array_equal(out.reshape(-1), independent(text, a, b).reshape(-1))


def _tamper(stream, kind):
    instructions = list(stream.ins)
    index = {k: [i for i, ins in enumerate(instructions) if ins.kind == k] for k in ('mvin', 'mvout', 'config_st', 'fence')}
    pooled = [i for i in index['config_st'] if dict(instructions[i].attrs).get('pool_stride')]

    def attr(position, **changes):
        ins = instructions[position]
        instructions[position] = ins._replace(attrs=tuple(sorted({**dict(ins.attrs), **changes}.items())))
    if kind == 'truncated_unread_load':
        del instructions[index['mvin'][-1]]
    elif kind == 'truncated_image_load':
        del instructions[index['mvin'][0]]
    elif kind == 'offset_load':
        instructions[index['mvin'][3]] = instructions[index['mvin'][3]]._replace(off=instructions[index['mvin'][3]].off + 16)
    elif kind == 'swapped_operand':
        first = instructions[index['mvin'][0]]
        other = next(i for i in index['mvin'] if instructions[i].arg != first.arg)
        instructions[index['mvin'][0]] = first._replace(arg=instructions[other].arg)
    elif kind == 'pool_stride':
        attr(pooled[0], pool_stride=1)
    elif kind == 'pool_size':
        attr(pooled[0], pool_size=3)
    elif kind == 'upad':
        current = dict(instructions[pooled[0]].attrs)['upad']
        attr(pooled[0], upad=1 - current)
    elif kind == 'porows':
        attr(pooled[0], porows=dict(instructions[pooled[0]].attrs)['porows'] - 1)
    elif kind == 'orows':
        attr(pooled[0], orows=dict(instructions[pooled[0]].attrs)['orows'] - 1)
    elif kind == 'multiplier':
        bits = dict(instructions[pooled[0]].attrs)['multiplier_bits'] + 1
        attr(pooled[0], multiplier_bits=bits, acc_scale=float(np.uint32(bits).view(np.float32)))
    elif kind == 'activation':
        attr(pooled[0], acc_act=0 if dict(instructions[pooled[0]].attrs)['acc_act'] else activation(True))
    elif kind == 'store_offset':
        instructions[index['mvout'][0]] = instructions[index['mvout'][0]]._replace(off=instructions[index['mvout'][0]].off + 1)
    elif kind == 'store_bank':
        attr(index['mvout'][0], local=dict(instructions[index['mvout'][0]].attrs)['local'] + 200)
    elif kind == 'unfenced_pair':
        first_store = index['mvout'][0]
        instructions = [ins for i, ins in enumerate(instructions)
                        if not (ins.kind == 'fence' and index['mvin'][0] < i < first_store)]
    elif kind == 'pooling_left_enabled':
        del instructions[index['config_st'][-1]]
    elif kind == 'final_fence':  # nothing orders the last drain before the kernel returns
        instructions = [ins for i, ins in enumerate(instructions) if not (ins.kind == 'fence' and i > index['mvout'][-1])]
    elif kind == 'load_scale':
        configs = [i for i, ins in enumerate(instructions) if ins.kind == 'config_ld']
        attr(configs[0], scale=0.5)
    return instructions


TAMPERING = ['truncated_unread_load', 'truncated_image_load', 'offset_load', 'swapped_operand', 'pool_stride',
             'pool_size', 'upad', 'porows', 'orows', 'multiplier', 'activation', 'store_offset', 'store_bank',
             'unfenced_pair', 'pooling_left_enabled', 'final_fence', 'load_scale']


@pytest.mark.parametrize('name', ['per_operand_padded', 'tensor_unpadded_17'])
@pytest.mark.parametrize('kind', TAMPERING)
def test_validation_rejects_altered_words(name, kind):
    _, (match,) = scaled(source(name), options(ROUNDING[name]))
    stream = copy.copy(match.kernel.stream)
    assert operand_sum.validate(stream, match.kernel.meta) == 'scaled_operand_sum'
    if kind == 'truncated_unread_load' and name in PADDED:
        kind = 'truncated_image_load'  # a padded window reads every pixel
    stream.ins = _tamper(stream, kind)
    with pytest.raises(ValueError):
        operand_sum.validate(stream, match.kernel.meta)


@pytest.mark.parametrize('kind', ['upad', 'multiplier', 'truncated_unread_load'])
def test_provider_admission_declines_altered_words(monkeypatch, kind):
    text, selected = source('per_operand_unpadded'), options('per_operand')
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            match.kernel.stream.ins = _tamper(match.kernel.stream, kind)
        return module, manifest, matches
    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=selected)
    assert proof['status'] == 'declined'


@pytest.mark.parametrize('field,value', [('pad', [0, 0, 0, 0]), ('nonnegative_codes', False), ('window', 3)])
def test_admission_rederives_the_pooled_proof(field, value):
    _, (match,) = scaled(source('per_operand_padded'), options('per_operand'))
    match.kernel.meta['scaled_operand_sum']['pool'][field] = value
    with pytest.raises(ValueError):
        operand_sum.admit(match, readout_options=options('per_operand'))
