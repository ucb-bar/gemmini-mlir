"""Scaled accumulate-on-load operand sums: proof, emission, admission and fallback."""

import copy
import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from merlin.runtime.tensor import Tensor
from merlin.targetgen.composition_capsules import _Source, _readout, _sum
from mlir_oot.core_aten import operand_sum
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import encode
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.readout_encoding import unpack
from mlir_oot.core_aten.stream import DIM, Ins, simulate

PACKAGE = Path(__file__).resolve().parents[2]
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32',
    readout=dict(strategy='f32_product_rne'), internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))
# Merlin's searched f32 disagreement multipliers (composition_corpus.scheme_scales).
GRID = (0.1666666716337204, 0.10000000149011612, 0.024390242993831635)


def provider():
    spec = importlib.util.spec_from_file_location('operand_sum_provider',
        PACKAGE / 'merlin-support/core_aten/execution_provider.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(*, accumulate=True, rounding=None, strategy='f32_product_rne'):
    stages = [dict(kind='preload', dtype='i32', range=[-(1 << 31), (1 << 31) - 1], rounding='none')]
    if accumulate:
        stages.append(dict(kind='accumulate', dtype='i32', operand_scale='none'))
    scheme = dict(readout_strategy=strategy)
    if rounding:
        scheme['operand_sum_rounding'] = rounding
    return dict(path=dict(schema='output_path_v1', accumulator=dict(dtype='i32'), accumulator_stages=stages,
        readout=[dict(kind='activation', function='relu', domain='accumulator'),
                 dict(kind='convert', to='f32', rounding='half_even'),
                 dict(kind='scale', dtype='f32', rounding='half_even', granularity='per_command', columns_per_scale=DIM),
                 dict(kind='requant', to='i8', rounding='half_even', clamp=[-128, 127])],
        exits=dict(accumulator=True, requant=True)), scheme=scheme, facts={})


def staged(m, n, rounding, operand_multipliers, multipliers, *, relu=True, block=False):
    """A composition-axis source and the capsule Merlin's reference evaluates."""
    source = _Source()
    case = dict(producer='operand_sum', operand_sum_rounding=rounding, operand_multipliers=list(operand_multipliers),
                multipliers=list(multipliers), exit='requant', accumulator_stages=[],
                readout_stages=(['activation'] if relu else []) + ['convert', 'scale', 'requant'])
    scheme = dict(weight_scale_granularity='column_block' if block else 'tensor', columns_per_scale=DIM,
                  readout_strategy='f32_product_rne', operand_sum_rounding=rounding)
    value = _readout(source, 'readout', _sum(source, case, m, n), case, scheme, m, n)
    capsule = dict(inputs=[dict(name='Left', shape=[m, n], dtype='i8', role='input'),
                           dict(name='Right', shape=[m, n], dtype='i8', role='input')],
                   stimulus_range=[-128, 127],
                   operation=dict(attributes=dict(case=case, quantization_scheme=scheme, M=m, K=m, N=n,
                                                  outputs=['readout'])))
    return source.render([value]), capsule


def dequantized(shape=(37, 21), sa=0.25, sb=0.25, so=0.5, *, relu=False, wb=None):
    t = 'x'.join(map(str, shape))
    i8, f32 = f'tensor<{t}xi8>', f'tensor<{t}xf32>'
    dims = ','.join(f'd{i}' for i in range(len(shape)))
    ident = f'affine_map<({dims})->({dims})>'
    body = '%v=arith.addf %x,%y:f32' if wb is None else (
        f'%w=arith.constant {wb}:f32 %yy=arith.mulf %y,%w:f32 %v=arith.addf %x,%yy:f32')
    body += (' %rz=arith.constant 0.0:f32 %vr=arith.maximumf %v,%rz:f32 linalg.yield %vr:f32'
             if relu else ' linalg.yield %v:f32')

    def dq(name, arg, scale):
        return (f'%{name}_s=arith.constant dense<{scale!r}>:tensor<f32>\n'
                f'%{name}_z=arith.constant dense<0>:tensor<i64>\n'
                f'%{name}="quant_ext.dequantize_per_tensor"(%{arg},%{name}_s,%{name}_z) '
                f'<{{quant_min=-128:i64,quant_max=127:i64}}>:({i8},tensor<f32>,tensor<i64>)->{f32}')
    return f'''builtin.module {{ func.func @forward(%a:{i8}, %b:{i8}) -> {i8} {{
{dq('da', 'a', sa)}
{dq('db', 'b', sb)}
%e=tensor.empty():{f32}
%s=linalg.generic {{indexing_maps=[{ident},{ident},{ident}], iterator_types=[{','.join(['"parallel"'] * len(shape))}]}}
  ins(%da,%db:{f32},{f32}) outs(%e:{f32}) {{ ^bb0(%x:f32,%y:f32,%o:f32): {body} }} -> {f32}
%qs=arith.constant dense<{so!r}>:tensor<f32>
%qz=arith.constant dense<0>:tensor<i64>
%q="quant_ext.quantize_per_tensor"(%s,%qs,%qz) <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
  ({f32},tensor<f32>,tensor<i64>)->{i8}
func.return %q:{i8}
}} }}'''


def scaled(text, selected):
    _, manifest, matches = build(text, readout_options=selected)
    return manifest, [m for m in matches if m.op == 'scaled_operand_sum']


def reference(capsule):
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate
    leaves = materialize_capsule_leaves(capsule)
    inputs = {name: np.asarray(leaves[name].to_list(), np.int8) for name in ('Left', 'Right')}
    return inputs, np.asarray(evaluate(capsule)['readout'], np.int64)


def decoded(stream, kind, bundle, field):
    return [unpack(encode(i).values[0 if bundle.endswith('Rs1') else 1], bundle, field)
            for i in stream.ins if i.kind == kind]


@pytest.mark.parametrize('m,n,loads,gains,relu,block', [
    (3, 5, GRID[:2], GRID[2:], True, False),
    (17, 35, (0.75, 1.0), GRID, True, True),
    (40, 33, GRID[1:], (0.5,), False, False),
    (1, 16, (1.0, GRID[0]), (GRID[1], 0.375), True, True),
])
def test_per_operand_double_rounding_matches_merlin_reference(m, n, loads, gains, relu, block):
    text, capsule = staged(m, n, 'per_operand', loads, gains, relu=relu, block=block)
    manifest, (match,) = scaled(text, options(rounding='per_operand'))
    stream, proof = match.kernel.stream, match.kernel.meta['scaled_operand_sum']
    assert proof['operand_sum_rounding'] == 'per_operand' and proof['relu'] == relu
    assert decoded(stream, 'config_ld', 'ConfigMvinRs1', 'scale') == [
        int(np.asarray(form_multiplier(v)).view(np.uint32)) for v in loads]
    width = DIM if block else n
    columns = [gains[min(j // width, len(gains) - 1)] for j in range(n)]
    assert set(decoded(stream, 'config_st', 'ConfigMvoutRs2', 'acc_scale')) == {
        int(np.asarray(form_multiplier(v)).view(np.uint32)) for v in columns}
    assert not stream.readouts and not any(i.kind in ('compute', 'preload') for i in stream.ins)
    inputs, expected = reference(capsule)
    actual = simulate(stream, dict(a=inputs['Left'], b=inputs['Right']))['y']
    np.testing.assert_array_equal(actual, expected)
    rewritten = rewrite(text, manifest)
    assert 'composition_stage = "operand_sum"' not in rewritten and 'composition_stage = "readout"' not in rewritten
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                              readout_options=options(rounding='per_operand'))
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'scaled_operand_sum'


def test_single_rounding_unit_operands_are_exact():
    text, capsule = staged(18, 20, 'single_rounding', (1.0, 1.0), GRID[:1])
    _, (match,) = scaled(text, options(rounding='single_rounding'))
    assert match.kernel.meta['scaled_operand_sum']['load_scale_bits'] == [
        int(np.asarray(form_multiplier(1)).view(np.uint32))] * 2
    inputs, expected = reference(capsule)
    np.testing.assert_array_equal(simulate(match.kernel.stream, dict(a=inputs['Left'], b=inputs['Right']))['y'],
                                  expected)


@pytest.mark.parametrize('loads,rounding,selected', [
    (GRID[:2], 'single_rounding', options()),                       # one rounding of a scaled sum
    (GRID[:2], 'per_operand', options(rounding='single_rounding')),  # selected scheme differs
    (GRID[:2], 'per_operand', options(strategy='exact_rational_host')),
    (GRID[:2], 'per_operand', options(accumulate=False)),            # path cannot accumulate a load
    ((1.5, 0.5), 'per_operand', options()),                          # load multiplier outside (0, 1]
])
def test_unproved_staged_sums_keep_the_existing_path(loads, rounding, selected):
    text, _ = staged(4, 6, rounding, loads, (0.5,))
    _, matches = scaled(text, selected)
    assert not matches


@pytest.mark.parametrize('shape,sa,sb,so,relu', [
    ((37, 21), 0.25, 0.25, 0.5, False),
    ((2, 3, 40), 0.25, 0.25, 0.75, True),
    ((5, 17), 0.0625, 0.0625, 0.0625, False),
])
def test_dequantized_sum_single_rounding_equals_exact_reference(shape, sa, sb, so, relu):
    text = dequantized(shape, sa, sb, so, relu=relu)
    _, (match,) = scaled(text, options())
    proof = match.kernel.meta['scaled_operand_sum']
    assert proof['reference'] == 'exact_rational_single_rounding' and proof['domain_pairs'] == 256 * 256
    rng = np.random.default_rng(sum(shape))
    a = rng.integers(-128, 128, shape, dtype=np.int8)
    b = rng.integers(-128, 128, shape, dtype=np.int8)
    exact = Tensor(shape, a.reshape(-1).tolist()).residual_add_scaled(
        Tensor(shape, b.reshape(-1).tolist()), Fraction(sa) / Fraction(so), Fraction(sb) / Fraction(so))
    expected = np.clip(np.asarray(exact.data).reshape(shape), 0 if relu else -128, 127)
    out = simulate(match.kernel.stream, dict(a=a.reshape(match.kernel.stream.args[0].shape),
                                             b=b.reshape(match.kernel.stream.args[1].shape)))['y']
    np.testing.assert_array_equal(out.reshape(shape), expected)
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                              readout_options=options())
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('kwargs,selected', [
    (dict(sa=0.5, sb=0.25), options()),          # a ratio load rounds b alone
    (dict(sa=0.0250686146, sb=0.0202060752, so=0.0302009135), options()),
    (dict(wb=-1.0), options()),                  # subtraction negates the load minimum
    (dict(), options(rounding='per_operand')),   # no explicit per-operand requantization
])
def test_unproved_dequantized_sums_keep_the_exact_integer_path(kwargs, selected):
    text = dequantized(**kwargs)
    _, manifest, matches = build(text, readout_options=selected)
    assert not [m for m in matches if m.op == 'scaled_operand_sum']
    assert [m.op for m in matches] in (['aten.add.Tensor'], ['aten.sub.Tensor'])


def test_no_readout_options_is_unchanged():
    _, _, matches = build(dequantized(), readout_options=None)
    assert [m.op for m in matches] == ['aten.add.Tensor']


def _tamper(stream, kind):
    instructions = list(stream.ins)
    loads = [i for i, ins in enumerate(instructions) if ins.kind == 'mvin']
    stores = [i for i, ins in enumerate(instructions) if ins.kind == 'mvout']
    configs = {k: [i for i, ins in enumerate(instructions) if ins.kind == k] for k in ('config_ld', 'config_st')}

    def attr(index, key, value):
        ins = instructions[index]
        instructions[index] = ins._replace(attrs=tuple((k, value if k == key else v) for k, v in ins.attrs))
    if kind == 'truncated_store':
        del instructions[stores[-1]]
    elif kind == 'truncated_load':
        del instructions[loads[-1]]
    elif kind == 'offset':
        instructions[loads[-1]] = instructions[loads[-1]]._replace(off=instructions[loads[-1]].off + 1)
    elif kind == 'swapped':
        first, last = loads[0], loads[len(loads) // 2]
        instructions[first] = instructions[first]._replace(arg=instructions[last].arg)
    elif kind == 'load_scale':
        attr(configs['config_ld'][1], 'scale', 0.5)
    elif kind == 'readout_scale':
        ins = instructions[configs['config_st'][0]]
        attrs = dict(ins.attrs)
        bits = attrs['multiplier_bits'] + 1
        attrs.update(multiplier_bits=bits, acc_scale=float(np.uint32(bits).view(np.float32)))
        instructions[configs['config_st'][0]] = ins._replace(attrs=tuple(sorted(attrs.items())))
    elif kind == 'activation':
        attr(configs['config_st'][0], 'acc_act', 0)
    elif kind == 'unfenced':
        instructions = [ins for i, ins in enumerate(instructions)
                        if not (ins.kind == 'fence' and loads[0] < i < loads[-1])]
    elif kind in ('load_config_mutation', 'store_config_mutation', 'execute_config'):
        # Change the effective state between dependent commands, then restore it.
        target = loads[-1] if kind == 'load_config_mutation' else stores[-1]
        if kind == 'load_config_mutation':
            original = instructions[configs['config_ld'][1]]
            changed = original._replace(attrs=tuple((k, 0.5 if k == 'scale' else v) for k, v in original.attrs))
        elif kind == 'store_config_mutation':
            original = instructions[configs['config_st'][-1]]
            changed = original._replace(attrs=tuple((k, v * 2 if k == 'stride' else v) for k, v in original.attrs))
        else:
            original = None
            changed = Ins('config_ex', (('a_stride', 1), ('a_transpose', 0), ('acc_scale', 1.0), ('act', 0),
                                        ('b_transpose', 0), ('c_stride', 2), ('dataflow', 1)), None, 0)
        instructions[target:target] = [changed]
        if original is not None:
            instructions.insert(target + 2, original)
    elif kind == 'full_width':
        attr(stores[0], 'local', instructions[stores[0]].a('local') | (1 << 29))
    return instructions


TAMPERING = ['truncated_store', 'truncated_load', 'offset', 'swapped', 'load_scale',
             'readout_scale', 'activation', 'unfenced', 'full_width', 'load_config_mutation',
             'store_config_mutation', 'execute_config']


@pytest.mark.parametrize('kind', TAMPERING)
def test_validation_rejects_altered_streams(kind):
    text, _ = staged(17, 20, 'per_operand', GRID[:2], GRID[2:])
    _, (match,) = scaled(text, options())
    stream = copy.copy(match.kernel.stream)
    assert operand_sum.validate(stream, match.kernel.meta) == 'scaled_operand_sum'
    stream.ins = _tamper(stream, kind)
    with pytest.raises(ValueError):
        operand_sum.validate(stream, match.kernel.meta)


@pytest.mark.parametrize('kind', ['offset', 'load_scale', 'truncated_store', 'load_config_mutation',
                                  'execute_config'])
def test_provider_admission_declines_altered_streams(monkeypatch, kind):
    text, _ = staged(17, 20, 'per_operand', GRID[:2], GRID[2:])
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            match.kernel.stream.ins = _tamper(match.kernel.stream, kind)
        return module, manifest, matches
    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined'


def test_altered_proof_is_rejected():
    text, _ = staged(3, 5, 'per_operand', GRID[:2], GRID[2:])
    _, (match,) = scaled(text, options())
    meta = copy.deepcopy(match.kernel.meta)
    meta['scaled_operand_sum']['load_scale_bits'][0] += 1
    with pytest.raises(ValueError):
        operand_sum.validate(match.kernel.stream, meta)


def test_exhaustive_proof_detects_double_rounding_against_single_rounding_source():
    from mlir_oot.core_aten.catalog import parse_module
    module = parse_module(dequantized(sa=0.5, sb=0.25))
    generic = next(op for op in module.walk() if op.name == 'linalg.generic')
    region = operand_sum.dequantized_sum(generic, options())
    assert region is not None
    with pytest.raises(operand_sum.Refused):
        operand_sum.prove(region, options())
    a = np.array([0], np.int64)
    b = np.array([1], np.int64)
    # 0 + 1 * 0.5 rounds to 0 once, but a ratio load of b rounds 0.5 first: the scheme matters.
    assert region.source(a, b, 0).tolist() == [0]
    assert f32_readout(b, np.float32(0.5), 'i8').tolist() == [0]
