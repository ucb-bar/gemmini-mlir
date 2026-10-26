"""Output-stationary and K-concat residual schedules: emission, admission, refusal twins."""

import copy

import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import OS, WS, simulate
from mlir_oot.core_aten import compositions
from test_composition_readout import PACKAGE, POLICY, provider
from test_composition_readout import options as readout_options

BLOCK = dict(schema='merlin.software_spec.compositions.v1', status='reviewed',
             grading='placement_and_instruction_audit', chains=[
    dict(id='contraction_readout', producer='contraction', accumulator_stages=dict(preload='optional'),
         readout_stages='from_output_path', exits=['requant', 'accumulator'], dataflows='from_facts',
         constraints=['activation_before_scale_requires_nonnegative_multiplier',
                      dict(os_dataflow_requires='mesh_partial_sum_bound_from_constant_operands')]),
    dict(id='contraction_operand_extension', producer='contraction', extension='constant_integer_operand',
         readout_stages='from_output_path', exits=['requant', 'accumulator'],
         constraints=['coefficient_integer_exact', dict(k_extension_within='from_facts')])])

SEMANTICS = copy.deepcopy(POLICY['numerical_semantics'])
SEMANTICS['native_unit_contract'] = dict(mesh_mac=dict(signed_bits=20, overflow='wrap_each_mac', dataflows=['WS', 'OS']))
COMPOSED = dict(POLICY, numerical_semantics=SEMANTICS, compositions=BLOCK)
UNIT, OUT_SCALE = 0.125, 7.0


def options():
    return dict(readout_options(), compositions=BLOCK, numerical_semantics=SEMANTICS)


def _generic(name, inputs, types, otype, indexing, body):
    return f'''%e_{name}=tensor.empty():{otype}
    %{name}=linalg.generic {{indexing_maps=[{','.join(indexing)}], iterator_types=["parallel","parallel"]}}
    ins({inputs}:{types}) outs(%e_{name}:{otype}) {{ {body} }} -> {otype}'''


def source(m, k, n, *, weights=None, coefficient=None, bias=True, relu=True):
    """i8 matmul -> f32 * UNIT -> [+bias] -> [+coefficient*UNIT*residual] -> [relu] -> quantize."""
    itype, otype = f'tensor<{m}x{n}xi32>', f'tensor<{m}x{n}xf32>'
    ident, col = 'affine_map<(m,n)->(m,n)>', 'affine_map<(m,n)->(n)>'
    params = [f'%a:tensor<{m}x{k}xi8>'] + ([] if weights is not None else [f'%b:tensor<{k}x{n}xi8>'])
    params += [f'%r:tensor<{m}x{n}xi8>'] if coefficient is not None else []
    text = [f'builtin.module {{ func.func @forward({", ".join(params)}) -> tensor<{m}x{n}xi8> {{']
    if weights is not None:
        rows = ','.join('[' + ','.join(str(int(v)) for v in row) + ']' for row in weights)
        text.append(f'%b=arith.constant dense<[{rows}]>:tensor<{k}x{n}xi8>')
    text.append(f'''%z=arith.constant 0:i32
    %e=tensor.empty():{itype}
    %init=linalg.fill ins(%z:i32) outs(%e:{itype})->{itype}
    %mm=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
      affine_map<(m,n,k)->(m,n)>], iterator_types=["parallel","parallel","reduction"]}}
    ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init:{itype}) {{
    ^bb0(%x:i8,%w:i8,%c:i32):
      %xx=arith.extsi %x:i8 to i32
      %ww=arith.extsi %w:i8 to i32
      %p=arith.muli %xx,%ww:i32
      %v=arith.addi %p,%c:i32
      linalg.yield %v:i32 }} -> {itype}''')
    text.append(_generic('f', '%mm', itype, otype, [ident, ident],
                         '^bb0(%x:i32,%o:f32): %v=arith.sitofp %x:i32 to f32 linalg.yield %v:f32'))
    text += [f'%s0=arith.constant {UNIT:e}:f32', f'%sc=tensor.splat %s0:{otype}',
             _generic('scaled', '%f,%sc', f'{otype},{otype}', otype, [ident] * 3,
                      '^bb0(%x:f32,%s:f32,%o:f32): %v=arith.mulf %x,%s:f32 linalg.yield %v:f32')]
    last = 'scaled'
    offsets = (np.arange(n) - 5) if bias else np.zeros(n, dtype=int)
    if bias:
        dense = ','.join(f'{float(v) * UNIT:e}' for v in offsets)
        text += [f'%bias=arith.constant dense<[{dense}]>:tensor<{n}xf32>',
                 _generic('biased', f'%{last},%bias', f'{otype},tensor<{n}xf32>', otype, [ident, col, ident],
                          '^bb0(%x:f32,%y:f32,%o:f32): %v=arith.addf %x,%y:f32 linalg.yield %v:f32')]
        last = 'biased'
    if coefficient is not None:
        text += [f'%rs=arith.constant dense<{coefficient * UNIT:e}>:tensor<f32>',
                 '%rz=arith.constant dense<0>:tensor<i64>',
                 f'''%rd="quant_ext.dequantize_per_tensor"(%r,%rs,%rz) <{{quant_min=-128:i64,quant_max=127:i64}}>:
                 (tensor<{m}x{n}xi8>,tensor<f32>,tensor<i64>)->{otype}''',
                 _generic('sum', f'%{last},%rd', f'{otype},{otype}', otype, [ident] * 3,
                          '^bb0(%x:f32,%y:f32,%o:f32): %v=arith.addf %x,%y:f32 linalg.yield %v:f32')]
        last = 'sum'
    if relu:
        text.append(_generic('relu', f'%{last}', otype, otype, [ident, ident],
                             '^bb0(%x:f32,%o:f32): %rz0=arith.constant 0.0:f32 %v=arith.maximumf %x,%rz0:f32 '
                             'linalg.yield %v:f32'))
        last = 'relu'
    text += [f'%qs=arith.constant dense<{OUT_SCALE:e}>:tensor<f32>', '%qz=arith.constant dense<0>:tensor<i64>',
             f'''%q="quant_ext.quantize_per_tensor"(%{last},%qs,%qz)
             <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
             ({otype},tensor<f32>,tensor<i64>)->tensor<{m}x{n}xi8>
             func.return %q:tensor<{m}x{n}xi8>''', '}}']
    return '\n'.join(text), offsets


def expected(inputs, weights, offsets, *, coefficient=0, relu=True):
    acc = inputs['a'].astype(np.int64) @ np.asarray(weights, np.int64) + offsets
    if coefficient:
        acc = acc + coefficient * inputs['residual'].astype(np.int64)
    return f32_readout(acc, form_multiplier(UNIT, OUT_SCALE), 'i8', relu=relu)


def composed(text, selected=None):
    _, manifest, matches = build(text, readout_options=options() if selected is None else selected)
    match, = [m for m in matches if m.kernel.meta.get('integer_producer')]
    return manifest, match


def admit(text, selected=None, policy=COMPOSED):
    return provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=policy,
                                             readout_options=options() if selected is None else selected)


def dataflows(stream):
    return {i.a('dataflow') for i in stream.ins if i.kind == 'config_ex'}


def weights_of(k, n, low=-3, high=3, seed=1):
    return np.random.default_rng(seed).integers(low, high + 1, (k, n))


# ------------------------------------------------------------------------------- facts

def test_facts_derive_os_and_digits_from_contract_and_policy():
    facts = compositions.facts(SEMANTICS)
    assert facts['dataflows'] == ['WS', 'OS'] and facts['mesh_signed_bits'] == 20
    assert facts['digit_bits'] == 8 and facts['accumulator_bits'] == 32
    assert compositions.facts(POLICY['numerical_semantics'])['dataflows'] == ['WS']
    wrong = copy.deepcopy(SEMANTICS)
    wrong['native_unit_contract']['mesh_mac']['signed_bits'] = 32
    assert compositions.facts(wrong)['dataflows'] == ['WS']


# ------------------------------------------------------------------- output stationary

@pytest.mark.parametrize('m,k,n,bias', [(3, 17, 5, True), (33, 40, 35, False), (16, 16, 16, True)])
def test_os_constant_weights_are_proved_simulated_and_covered(m, k, n, bias):
    weights = weights_of(k, n)
    text, offsets = source(m, k, n, weights=weights, bias=bias)
    manifest, match = composed(text)
    stream = match.kernel.stream
    assert dataflows(stream) == {OS}
    assert any(i.kind == 'compute' and i.a('bd') != 0xFFFFFFFF for i in stream.ins)
    assert match.kernel.meta['readout'] == 'f32_product_rne'
    inputs = dict(a=np.random.default_rng(m).integers(-128, 128, (m, k)).astype(np.int8),
                  b=weights.astype(np.int8))
    np.testing.assert_array_equal(simulate(stream, inputs)['out'], expected(inputs, weights, offsets))
    assert 'quant_ext.quantize_per_tensor' not in rewrite(text, manifest)
    proof = admit(text)
    assert proof['status'] == 'proven', proof
    schedule = proof['kernels'][0]['schedule']
    assert schedule['dataflow'] == 'OS'
    assert max(schedule['dataflow_plan']['mesh_partial_upper']) < 1 << 19


def test_os_refusals_keep_weight_stationary():
    k, n = 40, 5
    runtime, _ = source(3, k, n)
    _, match = composed(runtime)
    assert dataflows(match.kernel.stream) == {WS}
    assert 'dataflow_plan' not in match.kernel.meta
    wide, _ = source(3, k, n, weights=np.full((k, n), 127))
    _, match = composed(wide)
    assert dataflows(match.kernel.stream) == {WS}
    plain, _ = source(3, k, n, weights=weights_of(k, n))
    _, match = composed(plain, readout_options())   # no declared compositions
    assert dataflows(match.kernel.stream) == {WS}
    assert admit(wide)['status'] == 'proven'


# ------------------------------------------------------------------------ K-concat

@pytest.mark.parametrize('m,k,n,coefficient,digits', [
    (3, 17, 5, 3, [3]), (17, 33, 35, 200, [127, 73]), (16, 16, 16, -2, [-2]), (2, 1, 17, 254, [127, 127])])
def test_k_concat_appends_digit_tiles_and_matches_reference(m, k, n, coefficient, digits):
    text, offsets = source(m, k, n, coefficient=coefficient)
    manifest, match = composed(text)
    stream = match.kernel.stream
    assert match.kernel.meta['k_extension_plan']['decision']['tile_digits'] == digits
    tiles = stream.consts['kext_digits'].reshape(len(digits), 16, 16)
    assert [int(t[0, 0]) for t in tiles] == digits
    rng = np.random.default_rng(k + n)
    inputs = dict(a=rng.integers(-128, 128, (m, k)).astype(np.int8),
                  b=rng.integers(-128, 128, (k, n)).astype(np.int8),
                  residual=rng.integers(-128, 128, (m, n)).astype(np.int8))
    reference = expected(inputs, inputs['b'], offsets, coefficient=coefficient)
    np.testing.assert_array_equal(simulate(stream, inputs)['out'], reference)
    binding, = manifest['bindings']
    assert 'residual' in binding['operands']
    rewritten = rewrite(text, manifest)
    assert 'quant_ext.quantize_per_tensor' not in rewritten
    proof = admit(text)
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'f32_product_rne'
    assert proof['kernels'][0]['schedule']['k_extension_plan']['decision']['tile_digits'] == digits


def test_non_integer_coefficient_refuses_and_keeps_host_residual():
    text, _ = source(3, 17, 5, coefficient=2.5)
    manifest, match = composed(text)
    assert 'k_extension_plan' not in match.kernel.meta
    assert match.kernel.meta['readout'] == 'full_i32'
    assert 'kext_digits' not in match.kernel.stream.consts
    assert 'quant_ext.quantize_per_tensor' in rewrite(text, manifest)


# --------------------------------------------------------------- adversarial streams

def _tamper(monkeypatch, edit):
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            if match.kernel.meta.get('integer_producer'):
                match.kernel.stream.ins = edit(list(match.kernel.stream.ins), match.kernel.stream)
        return module, manifest, matches
    monkeypatch.setattr(backend, 'build', altered)
    return api


def _replace(ins, **fields):
    return ins._replace(attrs=tuple(sorted({**dict(ins.attrs), **fields}.items())))


def _drop_last(kind, predicate=lambda i: True):
    def edit(instructions, stream):
        index = max(n for n, i in enumerate(instructions) if i.kind == kind and predicate(i))
        return instructions[:index] + instructions[index + 1:]
    return edit


def _config_between_preload_and_compute(**fields):
    def edit(instructions, stream):
        index = next(n for n, i in enumerate(instructions) if i.kind == 'compute')
        config = next(i for i in instructions if i.kind == 'config_ex')
        return (instructions[:index] + [_replace(config, **fields)] + [instructions[index]] + [config] +
                instructions[index + 1:])
    return edit


def _shift(kind, field, delta):
    def edit(instructions, stream):
        index = next(n for n, i in enumerate(instructions) if i.kind == kind)
        changed = instructions[index]
        changed = (changed._replace(off=changed.off + delta) if field == 'off' else
                   _replace(changed, **{field: changed.a(field) + delta}))
        return instructions[:index] + [changed] + instructions[index + 1:]
    return edit


def _store_config_before_first_store(instructions, stream):
    # The first store drains a full tile of rows, so its row stride matters.
    index = min(n for n, i in enumerate(instructions) if i.kind == 'mvout')
    config = [i for i in instructions[:index] if i.kind == 'config_st'][-1]
    return instructions[:index] + [_replace(config, stride=config.a('stride') + 1)] + instructions[index:]


OS_ATTACKS = {
    'store_stride': _store_config_before_first_store,
    'truncated_k': _drop_last('compute'),
    'missing_store': _drop_last('mvout'),
    'offset_a': _shift('mvin', 'off', 1),
    'narrow_tile': _shift('compute', 'a_cols', -1),
    'c_stride_mutation': _config_between_preload_and_compute(c_stride=2),
    'dataflow_switch': _config_between_preload_and_compute(dataflow=WS),
    'load_stride': _shift('config_ld', 'stride', 1),
}


@pytest.mark.parametrize('attack', sorted(OS_ATTACKS))
def test_os_admission_rejects_tampered_streams(monkeypatch, attack):
    text, _ = source(17, 40, 20, weights=weights_of(40, 20))
    api = _tamper(monkeypatch, OS_ATTACKS[attack])
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=COMPOSED, readout_options=options())
    assert proof['status'] == 'declined', attack


def _digit_tile(value):
    def edit(instructions, stream):
        tiles = stream.consts['kext_digits'].copy()
        tiles[:16] = np.eye(16, dtype=np.int8) * value
        stream.consts['kext_digits'] = tiles
        return instructions
    return edit


def _residual_compute(predicate_index):
    def edit(instructions, stream):
        residual = stream._index['residual']
        loads = [n for n, i in enumerate(instructions) if i.kind == 'mvin' and i.arg == residual]
        return instructions[:loads[predicate_index]] + instructions[loads[predicate_index] + 1:]
    return edit


def _digit_pair(repeat):
    """Drop, or issue twice, the last digit tile's PRELOAD + COMPUTE pair."""
    def edit(instructions, stream):
        index = max(n for n, i in enumerate(instructions) if i.kind == 'compute')
        pair = instructions[index - 1:index + 1]
        assert [i.kind for i in pair] == ['preload', 'compute']
        return instructions[:index - 1] + (pair * 2 if repeat else []) + instructions[index + 1:]
    return edit


K_ATTACKS = {
    'dropped_digit_pair': _digit_pair(False),
    'repeated_digit_pair': _digit_pair(True),
    'missing_digit': _drop_last('compute'),
    'wrong_digit': _digit_tile(5),
    'missing_residual_load': _residual_compute(0),
    'c_stride_mutation': _config_between_preload_and_compute(c_stride=2),
    'dataflow_switch': _config_between_preload_and_compute(dataflow=OS),
    'residual_offset': lambda ins, s: [i._replace(off=i.off + 1) if i.arg == s._index['residual'] else i
                                       for i in ins],
}


@pytest.mark.parametrize('attack', sorted(K_ATTACKS))
def test_k_concat_admission_rejects_tampered_streams(monkeypatch, attack):
    text, _ = source(17, 33, 35, coefficient=200)
    api = _tamper(monkeypatch, K_ATTACKS[attack])
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=COMPOSED, readout_options=options())
    assert proof['status'] == 'declined', attack


def test_policy_mesh_width_mismatch_declines_os():
    text, _ = source(3, 17, 5, weights=weights_of(17, 5))
    policy = copy.deepcopy(COMPOSED)
    policy['numerical_semantics']['internal_arithmetic'].update(mac_result_bits=18, mac_result_overflow='wrap_to_18_bits')
    proof = admit(text, policy=policy)
    assert proof['status'] == 'declined'


def test_os_stream_without_a_source_bound_is_declined(monkeypatch):
    # Emission forced to OS where the source bound fails: admission re-derives the
    # licensed dataflows from the source and refuses the output-stationary replay.
    k, n = 40, 5
    wide, _ = source(3, k, n, weights=np.full((k, n), 127))
    forged = dict(plan={}, mesh_partial_lower=[0], mesh_partial_upper=[0])
    # Only this test's emitter instance is forged; the provider's admission
    # loads its own module instances and derives the dataflows afresh.
    monkeypatch.setattr(compositions, 'output_stationary', lambda *a, **kw: forged)
    _, match = composed(wide)
    assert dataflows(match.kernel.stream) == {OS}
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    assert api._core_aten_module(PACKAGE, 'compositions') is not compositions
    monkeypatch.setattr(backend, 'build', build)
    proof = api.admit_numeric_schedule(wide, package=PACKAGE, numeric_policy=COMPOSED, readout_options=options())
    assert proof['status'] == 'declined'
    assert 'dataflow' in proof['reason']
