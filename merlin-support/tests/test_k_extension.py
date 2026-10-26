"""Resident digit reuse and complete source-bound residual regions."""

import numpy as np
import pytest

from mlir_oot.frontend.parse import parse_module
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate
from test_composition_schedules import source, options, admit, composed, _digit_pair, _tamper, COMPOSED
from test_composition_readout import PACKAGE


def integer_source(m, k, n, coefficient):
    text = source(m, k, n, coefficient=coefficient, bias=False, relu=False)[0]
    text = text[:text.index('%e_f=')].replace(f'-> tensor<{m}x{n}xi8>', f'-> tensor<{m}x{n}xi32>', 1)
    wide, byte = f'tensor<{m}x{n}xi32>', f'tensor<{m}x{n}xi8>'
    return text + f'''%e_r=tensor.empty():{wide}
      %sum=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>,
        affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
        ins(%mm,%r:{wide},{byte}) outs(%e_r:{wide}) {{
        ^bb0(%acc:i32,%rv:i8,%dest:i32):
          %ext=arith.extsi %rv:i8 to i32
          %c=arith.constant {coefficient}:i32
          %p=arith.muli %ext,%c:i32
          %v=arith.addi %acc,%p:i32
          linalg.yield %v:i32 }} ->{wide}
      func.return %sum:{wide} }} }}'''


def tagged(text):
    from xdsl.dialects.builtin import StringAttr
    module = parse_module(text)
    for index, op in enumerate(module.walk()):
        if op.name == 'linalg.generic' or getattr(op, 'op_name', None) and op.op_name.data == 'quant_ext.quantize_per_tensor':
            op.attributes['prov.trace_role'] = StringAttr(f'member_{index}')
    return str(module)


@pytest.mark.parametrize('raw,m,k,n,coefficient', [
    (False, 3, 17, 5, 254), (False, 17, 33, 35, -254),
    (True, 17, 17, 17, 1), (True, 3, 1, 19, -255)])
def test_complete_region_admitted_and_executed(raw, m, k, n, coefficient):
    from merlin_experiments.phase0.fusion_execution import prove_placement
    text = integer_source(m, k, n, coefficient) if raw else source(m, k, n, coefficient=coefficient)[0]
    text = tagged(text)
    proof = admit(text)
    assert proof['status'] == 'proven', proof
    placement, reason = prove_placement(text, proof)
    assert placement is not None, reason
    assert len(proof['bindings']) == 1
    assert proof['semantic_gate']
    _, match = composed(text)
    assert match.result[1].owner.name == ('linalg.generic' if raw else 'builtin.unregistered')
    if raw:
        rng = np.random.default_rng(k)
        inputs = {name: rng.integers(-128, 128, shape).astype(np.int8) for name, shape in
                  [('a', (m, k)), ('b', (k, n)), ('residual', (m, n))]}
        expected = inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64) + coefficient * inputs['residual'].astype(np.int64)
        np.testing.assert_array_equal(simulate(match.kernel.stream, inputs)['out'], expected)
        assert match.kernel.meta['readout'] == 'full_i32'
    rewritten = rewrite(text, build(text, readout_options=options())[1])
    assert 'arith.muli' not in rewritten


def test_large_coefficient_reuses_one_resident_digit_and_keeps_all_contributions():
    text = integer_source(1, 1, 1, 259842)
    _, match = composed(text)
    stream = match.kernel.stream
    assert stream.consts['kext_digits'].shape == (16, 16)
    assert sum(i.kind == 'mvin' and i.arg == stream._index['kext_digits'] for i in stream.ins) == 1
    assert sum(i.kind == 'compute' for i in stream.ins) == 1 + 2046
    proof = admit(text)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('coefficient', [254, 259842])
def test_requant_reuses_resident_digits_without_losing_contributions(coefficient):
    text = source(1, 1, 1, coefficient=coefficient)[0]
    _, match = composed(text)
    stream = match.kernel.stream
    digits = match.kernel.meta['k_extension_plan']['decision']['tile_digits']
    assert stream.consts['kext_digits'].shape == (16, 16)
    assert sum(i.kind == 'mvin' and i.arg == stream._index['kext_digits'] for i in stream.ins) == 1
    assert sum(i.kind == 'compute' for i in stream.ins) == 1 + len(digits)
    proof = admit(text)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('repeat', [False, True])
def test_reused_digit_missing_or_extra_contribution_refuses(monkeypatch, raw, repeat):
    text = integer_source(1, 1, 1, 254) if raw else source(1, 1, 1, coefficient=254)[0]
    api = _tamper(monkeypatch, _digit_pair(repeat))
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=COMPOSED, readout_options=options())
    assert proof['status'] == 'declined'


@pytest.mark.parametrize('mutation', ['unsigned', 'wrong_add', 'wrong_map', 'overflow', 'external_observer', 'exit_unavailable'])
def test_integer_residual_refusal_retains_host_boundary(mutation):
    text = integer_source(3, 17, 3 if mutation == 'wrong_map' else 5, 2)
    selected = options()
    if mutation == 'unsigned':
        text = text.replace('arith.extsi %rv', 'arith.extui %rv')
    elif mutation == 'wrong_add':
        text = text.replace('arith.addi %acc,%p', 'arith.subi %acc,%p')
    elif mutation == 'wrong_map':
        text = text.replace('affine_map<(m,n)->(m,n)>,\n        affine_map', 'affine_map<(m,n)->(n,m)>,\n        affine_map')
    elif mutation == 'overflow':
        text = text.replace('arith.constant 2:i32', 'arith.constant 16777216:i32')
    elif mutation == 'external_observer':
        text = text.replace('%e_r=tensor.empty()', '%observed=tensor.cast %mm:tensor<3x5xi32> to tensor<3x5xi32>\n%e_r=tensor.empty()')
    else:
        selected['path']['exits']['accumulator'] = False
    _, match = composed(text, selected)
    assert 'k_extension_plan' not in match.kernel.meta
    assert match.result[1] is match.root.results[0]


def test_requant_source_inexact_f32_residual_stays_on_host(monkeypatch):
    """The unpooled twin of the pooled refusal: f32(1/6) does not reassociate over ``+ residual``."""
    from mlir_oot.core_aten import compositions
    from mlir_oot.core_aten.stream import simulate
    text = source(16, 16, 16, coefficient=1, bias=False, relu=False)[0]
    text = text.replace('1.250000e-01', '1.666666716e-01').replace('dense<7.000000e+00>', 'dense<1.000000e+00>')
    assert '1.666666716e-01:f32' in text and 'dense<1.666666716e-01>:tensor<f32>' in text
    _, match = composed(text)
    assert 'k_extension_plan' not in match.kernel.meta
    assert match.result[1] is match.root.results[0]
    # The gate is what refuses it: bypassed, the region fuses and disagrees with the source.
    monkeypatch.setattr(compositions, '_source_exact_extension', lambda *args: True)
    _, fused = composed(text)
    assert fused.kernel.meta.get('k_extension_plan') is not None
    a, b, r = (np.zeros((16, 16), np.int8) for _ in range(3))
    a[0, :2], b[:2, 0], r[0, 0] = (127, 61), (6, 1), -64
    gain = np.float32(1 / 6)
    sequential = np.rint(np.float32(np.float32(823) * gain + np.float32(-64) * gain))
    assert sequential == 127 and np.rint(np.float32(823 - 64) * gain) == 126
    out = simulate(fused.kernel.stream, dict(a=a, b=b, residual=r))['out']
    assert out[0, 0] == 126 != sequential
