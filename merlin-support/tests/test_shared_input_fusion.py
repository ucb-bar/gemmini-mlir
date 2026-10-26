"""Shared-input sibling fusion: one resident input, per-output readouts, panel weights."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from mlir_oot.core_aten import shared_input
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import encode
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.readout_encoding import unpack
from mlir_oot.core_aten.stream import DIM, simulate

PACKAGE = Path(__file__).resolve().parents[2]
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32',
    readout=dict(strategy='f32_product_rne'), internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def provider():
    spec = importlib.util.spec_from_file_location('shared_input_provider',
        PACKAGE / 'merlin-support/core_aten/execution_provider.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options():
    return dict(path=dict(schema='output_path_v1', accumulator=dict(dtype='i32'),
        accumulator_stages=[dict(kind='preload', dtype='i32', range=[-(1 << 31), (1 << 31)-1], rounding='none')],
        readout=[dict(kind='activation', function='relu', domain='accumulator'),
                 dict(kind='convert', to='f32', rounding='half_even'),
                 dict(kind='scale', dtype='f32', rounding='half_even', granularity='per_command', columns_per_scale=DIM),
                 dict(kind='requant', to='i8', rounding='half_even', clamp=[-128, 127])],
        exits=dict(accumulator=True, requant=True)), scheme=dict(readout_strategy='f32_product_rne'), facts={})


def _dense(values):
    rows = ','.join('[' + ','.join(str(int(v)) for v in row) + ']' for row in values)
    return f'dense<[{rows}]>'


def source(m, k, widths, *, constant=True, raw=False, relu=(True, False, True), column_block=False, seed=5,
           bias=None):
    """Sibling contractions of one input; every sibling has its own readout scale."""
    raws = raw if isinstance(raw, tuple) else (raw,) * len(widths)
    rng = np.random.default_rng(seed)
    weights = [rng.integers(-9, 10, (k, n), dtype=np.int8) for n in widths]
    params = [f'%a:tensor<{m}x{k}xi8>'] + ([] if constant else
              [f'%w{i}:tensor<{k}x{n}xi8>' for i, n in enumerate(widths)])
    results = [f'tensor<{m}x{n}x{"i32" if r else "i8"}>' for n, r in zip(widths, raws)]
    text = [f'builtin.module {{ func.func @forward({", ".join(params)}) -> ({", ".join(results)}) {{']
    if constant:
        text += [f'%w{i}=arith.constant {_dense(w)}:tensor<{k}x{n}xi8>' for i, (w, n) in enumerate(zip(weights, widths))]
    for i, n in enumerate(widths):
        ty = f'tensor<{m}x{n}xi32>'
        text.append(f'''%z{i}=arith.constant 0:i32
        %init{i}="tensor.splat"(%z{i}):(i32)->{ty}
        %mm{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
          affine_map<(m,n,k)->(m,n)>], iterator_types=["parallel","parallel","reduction"]}}
        ins(%a,%w{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init{i}:{ty}) {{
        ^bb0(%x:i8,%y:i8,%c:i32):
          %xx=arith.extsi %x:i8 to i32
          %yy=arith.extsi %y:i8 to i32
          %p=arith.muli %xx,%yy:i32
          %v=arith.addi %c,%p:i32
          linalg.yield %v:i32 }} -> {ty}''')
    scales, names = [], []
    maps, col = 'affine_map<(m,n)->(m,n)>', 'affine_map<(m,n)->(n)>'
    for i, n in enumerate(widths):
        if raws[i]:
            names.append(f'%mm{i}')
            scales.append(None)
            continue
        itype, otype = f'tensor<{m}x{n}xi32>', f'tensor<{m}x{n}xf32>'
        weight_scales = np.array([2.0 ** (-1 - i - (j // DIM if column_block else 0)) for j in range(n)])
        scales.append(weight_scales * .25)

        def generic(name, inputs, types, indexing, body):
            return f'''%e_{name}=tensor.empty():{otype}
            %{name}=linalg.generic {{indexing_maps=[{','.join(indexing)}], iterator_types=["parallel","parallel"]}}
            ins({inputs}:{types}) outs(%e_{name}:{otype}) {{ {body} }} -> {otype}'''
        text.append(generic(f'f{i}', f'%mm{i}', itype, [maps, maps],
                            '^bb0(%x:i32,%o:f32): %v=arith.sitofp %x:i32 to f32 linalg.yield %v:f32'))
        dense = ','.join(str(.25 * v) for v in weight_scales)
        text.append(f'%sc{i}=arith.constant dense<[{dense}]>:tensor<{n}xf32>')
        text.append(generic(f's{i}', f'%f{i},%sc{i}', f'{otype},tensor<{n}xf32>', [maps, col, maps],
                            '^bb0(%x:f32,%s:f32,%o:f32): %v=arith.mulf %x,%s:f32 linalg.yield %v:f32'))
        last = f's{i}'
        if bias is not None and bias[i]:
            text.append(f'%bc{i}=arith.constant dense<{.25 * weight_scales[0]}>:tensor<{n}xf32>')
            text.append(generic(f'b{i}', f'%{last},%bc{i}', f'{otype},tensor<{n}xf32>', [maps, col, maps],
                                '^bb0(%x:f32,%c:f32,%o:f32): %v=arith.addf %x,%c:f32 linalg.yield %v:f32'))
            last = f'b{i}'
        if relu[i % len(relu)]:
            text.append(generic(f'r{i}', f'%{last}', otype, [maps, maps],
                '^bb0(%x:f32,%o:f32): %rz=arith.constant 0.0:f32 %v=arith.maximumf %x,%rz:f32 linalg.yield %v:f32'))
            last = f'r{i}'
        text.append(f'''%qs{i}=arith.constant dense<7.0>:tensor<f32>
        %qz{i}=arith.constant dense<0>:tensor<i64>
        %q{i}="quant_ext.quantize_per_tensor"(%{last},%qs{i},%qz{i})
        <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
        ({otype},tensor<f32>,tensor<i64>)->tensor<{m}x{n}xi8>''')
        names.append(f'%q{i}')
    text.append(f'func.return {", ".join(names)}:{", ".join(results)}')
    text.append('}}')
    return '\n'.join(text), weights, scales


def expected(inputs, weights, scales, relu, raw):
    outs = []
    for i, w in enumerate(weights):
        acc = inputs.astype(np.int64) @ w.astype(np.int64)
        if raw:
            outs.append(acc.astype(np.int32))
        else:
            outs.append(f32_readout(acc, form_multiplier(scales[i], 7), 'i8', relu=relu[i % len(relu)]))
    return outs


def fused(text, selected=None):
    _, manifest, matches = build(text, readout_options=options() if selected is None else selected)
    found = [m for m in matches if m.kernel.meta.get('shared_input')]
    return manifest, matches, found


def run(match, a, weights):
    inputs = dict(a=a)
    for i, w in enumerate(weights):
        inputs[f'w{i}'] = w
    return simulate(match.kernel.stream, inputs)['out']


CASES = [
    # (m, k, widths, constant, raw, column_block) -- tails on every axis
    (19, 37, (21, 40, 9), True, False, False),
    (5, 16, (16, 33), False, False, True),
    (33, 70, (7, 18, 65, 3), True, False, True),
    (3, 9, (11, 5), True, True, False),
    (17, 5, (70, 2), False, True, False),
]


@pytest.mark.parametrize('m,k,widths,constant,raw,column_block', CASES)
def test_fused_siblings_equal_reference(m, k, widths, constant, raw, column_block):
    relu = (True, False, True)
    text, weights, scales = source(m, k, widths, constant=constant, raw=raw, column_block=column_block)
    manifest, matches, found = fused(text)
    match, = found
    assert len(matches) == 1 and len(manifest['bindings']) == 1
    stream = match.kernel.stream
    a_loads = [i for i in stream.ins if i.kind == 'mvin' and i.arg == stream._index['a']]
    tiles = -(-m // DIM) * -(-k // DIM)
    assert len(a_loads) == tiles  # each shared-input tile is moved in exactly once
    rng = np.random.default_rng(m * k)
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    flat = run(match, a, weights)
    offset = 0
    for i, want in enumerate(expected(a, weights, scales, relu, raw)):
        got = flat[offset:offset + want.size].reshape(want.shape)
        np.testing.assert_array_equal(got, want)
        offset += want.size
    record = match.kernel.meta['shared_input']
    if constant:
        assert record['panel_width'] == shared_input.panel_width()
        assert all(row['weight_layout']['panel_width'] == record['panel_width'] for row in record['members'])
    if not raw:
        multipliers = {unpack(encode(i).values[1], 'ConfigMvoutRs2', 'acc_scale')
                       for i in stream.ins if i.kind == 'config_st'}
        assert multipliers == {int(v) for s in scales for v in form_multiplier(s, 7).view(np.uint32)}
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                              readout_options=options())
    assert proof['status'] == 'proven', proof
    kernel, = proof['kernels']
    assert kernel['shared_input']['realization']['status'] == 'verified_finite_shared_input_effects'


def test_panel_width_is_derived_from_contract_and_facts():
    from mlir_oot.core_aten.readout_encoding import contract
    from merlin.targetgen.rtl.facts import load_facts
    payload = contract()['memory_model']['dma']['max_transfer_bytes']
    edge = load_facts(contract()['name'])['facts']['arrays'][0]['cols']
    assert shared_input.panel_width() == payload // edge * edge


def test_panel_width_fails_closed_without_declared_payload(monkeypatch):
    from mlir_oot.core_aten import readout_encoding
    declared = copy.deepcopy(readout_encoding.contract())
    declared['memory_model'].pop('dma')
    monkeypatch.setattr(readout_encoding, 'contract', lambda: declared)
    with pytest.raises(ValueError, match='UNKNOWN panel width'):
        shared_input.panel_width()
    text, weights, scales = source(4, 6, (5, 3))
    _, _, found = fused(text)
    match, = found  # still fused, but with row-major weights
    assert all(row['weight_layout'] is None for row in match.kernel.meta['shared_input']['members'])
    a = np.arange(24, dtype=np.int8).reshape(4, 6) - 12
    want = np.concatenate([w.reshape(-1) for w in expected(a, weights, scales, (True, False, True), False)])
    np.testing.assert_array_equal(run(match, a, weights), want)


def test_offload_binds_each_source_result_to_its_slice():
    text, _, _ = source(6, 20, (18, 4))
    manifest, _, _ = fused(text)
    binding, = manifest['bindings']
    assert [s['offset'] for s in binding['result_slices']] == [0, 6 * 18]
    rewritten = rewrite(text, manifest)
    assert rewritten.count('func.call') == 1
    assert rewritten.count('tensor.extract_slice') == 2 and rewritten.count('tensor.expand_shape') == 2
    assert 'quant_ext.quantize_per_tensor' not in rewritten and 'linalg.generic' not in rewritten


@pytest.mark.parametrize('reason,text', [
    ('different inputs', None),
    ('bias offsets', 'bias'),
    ('mixed exits', 'mixed'),
    ('no readout options', 'none'),
])
def test_unprovable_groups_fall_back_to_separate_contractions(reason, text):
    selected = None
    if text is None:
        body, _, _ = source(4, 6, (5, 3))
        body = body.replace('%a:tensor<4x6xi8>', '%a:tensor<4x6xi8>, %other:tensor<4x6xi8>', 1)
        body = body.replace('ins(%a,%w1', 'ins(%other,%w1')
    elif text == 'bias':
        body, _, _ = source(4, 6, (5, 3), bias=(True, False))
    elif text == 'mixed':
        body, _, _ = source(4, 6, (5, 3), raw=(False, True))
    else:
        body, _, _ = source(4, 6, (5, 3))
        selected = False
    if selected is False:
        _, manifest, matches = build(body, readout_options=None)
        found = [m for m in matches if m.kernel.meta.get('shared_input')]
    else:
        manifest, matches, found = fused(body)
    assert not found, reason
    assert not any('result_slices' in b for b in manifest['bindings'])


CORRUPTIONS = {
    'skip_input_tile': 'mesh input row was never loaded',
    'offset_weight': 'stationary operand differs from the source contraction',
    'multiplier': 'proved multiplier bits',
    'activation': 'proved multiplier bits',
    'store_pitch': 'stores do not write every sibling output element',
    'weight_bytes': 'not the source weight',
    'drop_store': 'mesh result overwrites a live unstored accumulator row',
    'load_stride': 'DMA row leaves its argument',
    'store_width': 'DMA element width differs from its argument type',
    'swap_weights': 'differs from its source contraction',
    # Configuration changed between dependent commands, then restored.
    'execute_stride_restored': 'effective CONFIG_EX differs',
    'load_scale_restored': 'effective CONFIG_LD differs',
    'store_scale_mid_panel': 'proved multiplier bits',
    # CONFIG_EX changed only around a store: readout depends on the execute state too.
    'execute_changed_at_store': 'effective CONFIG_EX differs',
    # The last stores are never ordered before the kernel returns.
    'no_final_fence': 'unfenced stores',
}


def _configured(kind, *args, **kwargs):
    from mlir_oot.core_aten.stream import Stream
    probe = Stream([])
    getattr(probe, kind)(*args, **kwargs)
    return probe.ins[0]


@pytest.mark.parametrize('corruption', CORRUPTIONS)
def test_admission_declines_tampered_streams(monkeypatch, corruption):
    text, _, _ = source(19, 37, (21, 40), relu=(True, False), constant=corruption != 'swap_weights')
    api = provider()
    assert api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY,
                                      readout_options=options())['status'] == 'proven'
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def replace_attr(ins, key, value):
        return ins._replace(attrs=tuple((k, value if k == key else v) for k, v in ins.attrs))

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            if not match.kernel.meta.get('shared_input'):
                continue
            s = match.kernel.stream
            a, w1 = s._index['a'], s._index['w1']
            if corruption == 'swap_weights':
                bound = dict(match.operands)
                bound['w0'], bound['w1'] = bound['w1'], bound['w0']
                match.operands[:] = list(bound.items())
            index = lambda pred: next(n for n, i in enumerate(s.ins) if pred(i))
            if corruption == 'skip_input_tile':
                del s.ins[index(lambda i: i.kind == 'mvin' and i.arg == a)]
            elif corruption == 'offset_weight':
                n = index(lambda i: i.kind == 'mvin' and i.arg == w1 and i.off > 0)
                s.ins[n] = s.ins[n]._replace(off=s.ins[n].off + 1)
            elif corruption == 'multiplier':
                n = index(lambda i: i.kind == 'config_st')
                s.ins[n] = replace_attr(replace_attr(s.ins[n], 'acc_scale', .125), 'multiplier_bits',
                                        int(np.float32(.125).view(np.uint32)))
            elif corruption == 'activation':
                n = index(lambda i: i.kind == 'config_st')
                s.ins[n] = replace_attr(s.ins[n], 'acc_act', 1 - s.ins[n].a('acc_act'))
            elif corruption == 'store_pitch':
                n = index(lambda i: i.kind == 'config_st')
                s.ins[n] = replace_attr(s.ins[n], 'stride', s.ins[n].a('stride') + 1)
            elif corruption == 'weight_bytes':
                s.consts['w0'] = s.consts['w0'].copy()
                s.consts['w0'].flat[0] ^= 1
            elif corruption == 'drop_store':
                del s.ins[index(lambda i: i.kind == 'mvout')]
            elif corruption == 'load_stride':
                n = index(lambda i: i.kind == 'config_ld' and i.a('load_id') == 1)
                s.ins[n] = replace_attr(s.ins[n], 'stride', s.ins[n].a('stride') * 2)
            elif corruption == 'execute_stride_restored':
                n = index(lambda i: i.kind == 'compute')
                s.ins[n:n + 1] = [_configured('config_ex', c_stride=2), s.ins[n], _configured('config_ex')]
            elif corruption == 'load_scale_restored':
                n = index(lambda i: i.kind == 'mvin' and i.arg == w1)
                stride = next(i for i in reversed(s.ins[:n]) if i.kind == 'config_ld' and i.a('load_id') == 1)
                s.ins[n:n + 1] = [_configured('config_ld', 1, stride.a('stride'), scale=.5), s.ins[n], stride]
            elif corruption == 'store_scale_mid_panel':
                n = next(n for n in range(1, len(s.ins)) if s.ins[n].kind == 'mvout' and s.ins[n - 1].kind == 'mvout')
                current = next(i for i in reversed(s.ins[:n]) if i.kind == 'config_st')
                other = _configured('config_st', current.a('stride'), current.a('acc_act'),
                                    multiplier_bits=int(np.float32(.125).view(np.uint32)))
                s.ins[n:n + 1] = [other, s.ins[n], current]
            elif corruption == 'execute_changed_at_store':
                n = index(lambda i: i.kind == 'mvout')
                s.ins[n:n + 1] = [_configured('config_ex', c_stride=2), s.ins[n], _configured('config_ex')]
            elif corruption == 'no_final_fence':
                while s.ins[-1].kind == 'fence':
                    s.ins.pop()
            elif corruption == 'store_width':
                n = index(lambda i: i.kind == 'mvout')
                s.ins[n] = replace_attr(s.ins[n], 'local', s.ins[n].a('local') | (1 << 29))
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', corruption
    assert CORRUPTIONS[corruption] in proof['reason'], proof['reason']


def test_backstop_compares_against_the_exact_reference():
    """The differential model refuses when the stream and the source reference disagree."""
    from merlin.compile.scheduling.panel_layout import PanelLayout
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from mlir_oot.core_aten.shared_input import _backstop, _members
    from mlir_oot.core_aten.stream_proof import ProofError
    text, _, _ = source(6, 20, (18, 4))  # constant weights: both siblings use panels
    _, _, found = fused(text)
    match, = found
    from mlir_oot.core_aten.source import module_of
    module = module_of(match.root)
    group, = find_shared_input_groups(module)
    _, members = _members(module, group, options())
    width = match.kernel.meta['shared_input']['panel_width']
    weights = {f'w{i}': (i, PanelLayout(m.k, m.n, width)) for i, m in enumerate(members)}
    bases = [0, 6 * 18]
    _backstop(match.kernel.stream, members, weights, bases)
    members[1].bits = tuple(int(np.float32(.5).view(np.uint32)) for _ in members[1].bits)
    with pytest.raises(ProofError, match='differs from the exact reference'):
        _backstop(match.kernel.stream, members, weights, bases)
