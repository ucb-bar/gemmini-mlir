"""Store-path pooling, standalone pooled stores and ones-matmul means: emission and admission."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from mlir_oot.core_aten import pool_plan
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.lint import HazardError, lint
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.pooled_store import validate
from mlir_oot.core_aten.stream import DIM, Ins, simulate

PACKAGE = Path(__file__).resolve().parents[2]
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32',
    readout=dict(strategy='f32_product_rne'), internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def provider():
    spec = importlib.util.spec_from_file_location('pooled_provider',
        PACKAGE / 'merlin-support/core_aten/execution_provider.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(pool=True):
    readout = [dict(kind='activation', function='relu', domain='accumulator'),
               dict(kind='convert', to='f32', rounding='half_even'),
               dict(kind='scale', dtype='f32', rounding='half_even', granularity='per_command', columns_per_scale=DIM),
               dict(kind='requant', to='i8', rounding='half_even', clamp=[-128, 127])]
    if pool:
        readout.append(dict(kind='pool', reductions=['max'], padding='unknown'))
    return dict(path=dict(schema='output_path_v1', accumulator=dict(dtype='i32'),
        accumulator_stages=[dict(kind='preload', dtype='i32', range=[-(1 << 31), (1 << 31)-1], rounding='none')],
        readout=readout, exits=dict(accumulator=True, requant=True)),
        scheme=dict(readout_strategy='f32_product_rne'), facts={})


_POOL = '''%ninf{t}=arith.constant 0xFF800000:f32
    %pe{t}=tensor.empty():{pooled}
    %pi{t}=linalg.fill ins(%ninf{t}:f32) outs(%pe{t}:{pooled})->{pooled}
    %win{t}=tensor.empty():tensor<{k}x{k}xf32>
    %pool{t}=linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1*{s}+d4,d2*{s}+d5,d3)>,
      affine_map<(d0,d1,d2,d3,d4,d5)->(d4,d5)>,affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3)>],
      iterator_types=["parallel","parallel","parallel","parallel","reduction","reduction"]}}
      ins({src},%win{t}:{image},tensor<{k}x{k}xf32>) outs(%pi{t}:{pooled}) {{
      ^bb0(%px:f32,%pw:f32,%po:f32): %pv=arith.maximumf %po,%px:f32 linalg.yield %pv:f32 }} -> {pooled}'''


def fused_source(n=1, h=4, w=4, k=8, c=5, *, window=2, stride=2, relu=True, bias=True, column_block=False, pad=0):
    """conv-as-matmul rows [n*h*w, k] x [k, c] -> f32 readout -> NHWC max pool -> quantize."""
    m = n * h * w
    ph, pw = (h + 2 * pad - window) // stride + 1, (w + 2 * pad - window) // stride + 1
    itype, ftype = f'tensor<{m}x{c}xi32>', f'tensor<{m}x{c}xf32>'
    image, pooled = f'tensor<{n}x{h}x{w}x{c}xf32>', f'tensor<{n}x{ph}x{pw}x{c}xf32>'
    framed = f'tensor<{n}x{h + 2 * pad}x{w + 2 * pad}x{c}xf32>'
    ident, col = 'affine_map<(m,n)->(m,n)>', 'affine_map<(m,n)->(n)>'
    scales = np.array([2.0 ** (-1 - (j // DIM if column_block else 0)) for j in range(c)])
    offsets = (np.arange(c) - 2) if bias else np.zeros(c, dtype=int)

    def generic(name, inputs, types, maps, body):
        return f'''%e_{name}=tensor.empty():{ftype}
        %{name}=linalg.generic {{indexing_maps=[{','.join(maps)}], iterator_types=["parallel","parallel"]}}
        ins({inputs}:{types}) outs(%e_{name}:{ftype}) {{ {body} }} -> {ftype}'''
    text = [f'''builtin.module {{ func.func @forward(%a:tensor<{m}x{k}xi8>, %b:tensor<{k}x{c}xi8>)
        -> tensor<{n}x{ph}x{pw}x{c}xi8> {{
        %z=arith.constant 0:i32
        %e=tensor.empty():{itype}
        %init=linalg.fill ins(%z:i32) outs(%e:{itype})->{itype}
        %mm=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
          affine_map<(m,n,k)->(m,n)>], iterator_types=["parallel","parallel","reduction"]}}
        ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{c}xi8>) outs(%init:{itype}) {{
        ^bb0(%x:i8,%wt:i8,%acc:i32):
          %xx=arith.extsi %x:i8 to i32
          %ww=arith.extsi %wt:i8 to i32
          %p=arith.muli %xx,%ww:i32
          %v=arith.addi %p,%acc:i32
          linalg.yield %v:i32 }} -> {itype}''',
        generic('f', '%mm', itype, [ident, ident], '^bb0(%x:i32,%o:f32): %v=arith.sitofp %x:i32 to f32 linalg.yield %v:f32'),
        f'%sc=arith.constant dense<[{",".join(str(.25 * v) for v in scales)}]>:tensor<{c}xf32>',
        generic('scaled', '%f,%sc', f'{ftype},tensor<{c}xf32>', [ident, col, ident],
                '^bb0(%x:f32,%s:f32,%o:f32): %v=arith.mulf %x,%s:f32 linalg.yield %v:f32')]
    last = 'scaled'
    if bias:
        text += [f'%bias=arith.constant dense<[{",".join(str(float(v)) for v in offsets * .25 * scales)}]>:tensor<{c}xf32>',
                 generic('biased', '%scaled,%bias', f'{ftype},tensor<{c}xf32>', [ident, col, ident],
                         '^bb0(%x:f32,%bv:f32,%o:f32): %v=arith.addf %x,%bv:f32 linalg.yield %v:f32')]
        last = 'biased'
    if relu:
        text += [generic('relu', f'%{last}', ftype, [ident, ident],
                         '^bb0(%x:f32,%o:f32): %rz=arith.constant 0.0:f32 %v=arith.maximumf %x,%rz:f32 linalg.yield %v:f32')]
        last = 'relu'
    text += [f'%img=tensor.expand_shape %{last} [[0,1,2],[3]] output_shape [{n},{h},{w},{c}] : {ftype} into {image}']
    if pad:
        text += [f'''%pinf=arith.constant 0xFF800000:f32
        %border=tensor.splat %pinf:{framed}
        %framed=tensor.insert_slice %img into %border[0,{pad},{pad},0][{n},{h},{w},{c}][1,1,1,1]:{image} into {framed}''']
    text += [_POOL.format(t='', pooled=pooled, image=framed if pad else image, k=window, s=stride,
                          src='%framed' if pad else '%img'),
             '%qs=arith.constant dense<7.0>:tensor<f32>', '%qz=arith.constant dense<0>:tensor<i64>',
             f'''%q="quant_ext.quantize_per_tensor"(%pool,%qs,%qz) <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
             ({pooled},tensor<f32>,tensor<i64>)->tensor<{n}x{ph}x{pw}x{c}xi8>
             func.return %q:tensor<{n}x{ph}x{pw}x{c}xi8>''', '}}']
    return '\n'.join(text), scales, offsets


def movement_source(n=1, h=5, w=6, c=20, *, window=3, stride=2, pad=(1, 1, 1, 1), out_scale=0.5):
    """dq(i8 NHWC) -> -inf border -> max pool -> quantize with the same qparams."""
    top, left, bottom, right = pad
    hp, wp = h + top + bottom, w + left + right
    ph, pw = (hp - window) // stride + 1, (wp - window) // stride + 1
    raw, padded = f'tensor<{n}x{h}x{w}x{c}xf32>', f'tensor<{n}x{hp}x{wp}x{c}xf32>'
    pooled = f'tensor<{n}x{ph}x{pw}x{c}xf32>'
    text = [f'''builtin.module {{ func.func @forward(%x:tensor<{n}x{h}x{w}x{c}xi8>) -> tensor<{n}x{ph}x{pw}x{c}xi8> {{
        %s=arith.constant dense<0.5>:tensor<f32>
        %so=arith.constant dense<{out_scale}>:tensor<f32>
        %z=arith.constant dense<0>:tensor<i64>
        %d="quant_ext.dequantize_per_tensor"(%x,%s,%z) <{{quant_min=-128:i64,quant_max=127:i64}}>:
          (tensor<{n}x{h}x{w}x{c}xi8>,tensor<f32>,tensor<i64>)->{raw}''']
    source = '%d'
    if any(pad):
        text += [f'''%pinf=arith.constant 0xFF800000:f32
        %border=tensor.splat %pinf:{padded}
        %padded=tensor.insert_slice %d into %border[0,{top},{left},0][{n},{h},{w},{c}][1,1,1,1]:{raw} into {padded}''']
        source = '%padded'
    text += [_POOL.format(t='', pooled=pooled, image=padded if any(pad) else raw, k=window, s=stride, src=source),
             f'''%q="quant_ext.quantize_per_tensor"(%pool,%so,%z) <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
             ({pooled},tensor<f32>,tensor<i64>)->tensor<{n}x{ph}x{pw}x{c}xi8>
             func.return %q:tensor<{n}x{ph}x{pw}x{c}xi8>''', '}}']
    return '\n'.join(text)


def mean_source(shape=(2, 8, 3, 4), in_scale=0.5, out_scale=0.25, count=12.0):
    kept = shape[:2]
    full, part = 'x'.join(map(str, shape)), 'x'.join(map(str, kept))
    identity = 'affine_map<(d0, d1) -> (d0, d1)>'
    return f'''builtin.module {{ func.func @forward(%x: tensor<{full}xi8>) -> tensor<{part}xi8> {{
    %si = arith.constant dense<{in_scale!r}> : tensor<f32>
    %so = arith.constant dense<{out_scale!r}> : tensor<f32>
    %mz = arith.constant dense<0> : tensor<i64>
    %xd = "quant_ext.dequantize_per_tensor"(%x, %si, %mz) <{{quant_min = -128 : i64, quant_max = 127 : i64}}> :
      (tensor<{full}xi8>, tensor<f32>, tensor<i64>) -> tensor<{full}xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %init = tensor.splat %zero : tensor<{part}xf32>
    %total = linalg.reduce ins(%xd:tensor<{full}xf32>) outs(%init:tensor<{part}xf32>) dimensions = [2, 3]
      (%mp: f32, %mq: f32) {{
        %mr = arith.addf %mp, %mq : f32
        linalg.yield %mr : f32
      }}
    %n = arith.constant {count!r} : f32
    %ns = tensor.splat %n : tensor<{part}xf32>
    %me = tensor.empty() : tensor<{part}xf32>
    %mean = linalg.generic {{indexing_maps = [{identity}, {identity}, {identity}], iterator_types = ["parallel", "parallel"]}}
      ins(%total, %ns : tensor<{part}xf32>, tensor<{part}xf32>) outs(%me : tensor<{part}xf32>) {{
    ^bb0(%dp: f32, %dq: f32, %do: f32):
      %dr = arith.divf %dp, %dq : f32
      linalg.yield %dr : f32
    }} -> tensor<{part}xf32>
    %pooled = "quant_ext.quantize_per_tensor"(%mean, %so, %mz) <{{quant_min = -128 : i64, quant_max = 127 : i64,
      output_dtype = "int8"}}> : (tensor<{part}xf32>, tensor<f32>, tensor<i64>) -> tensor<{part}xi8>
    func.return %pooled : tensor<{part}xi8>
  }}
}}'''


def window_max(y, window, stride, pad=(0, 0, 0, 0)):
    top, left, bottom, right = pad
    y = np.pad(y.astype(np.float64), ((0, 0), (top, bottom), (left, right), (0, 0)), constant_values=-np.inf)
    ph, pw = (y.shape[1] - window) // stride + 1, (y.shape[2] - window) // stride + 1
    out = np.full((y.shape[0], ph, pw, y.shape[3]), -np.inf)
    for i in range(window):
        for j in range(window):
            out = np.maximum(out, y[:, i:i + stride * (ph - 1) + 1:stride, j:j + stride * (pw - 1) + 1:stride])
    return out.astype(np.int64)


def fused(text, selected=None):
    _, manifest, matches = build(text, readout_options=options() if selected is None else selected)
    found = [m for m in matches if m.kernel.meta.get('pooled_store')]
    return manifest, found


def composition(text, kind):
    _, manifest, matches = build(text, readout_options=options())
    found = [m for m in matches if m.kernel.meta.get('composition') == kind]
    return manifest, found


FUSED_CASES = [dict(), dict(n=2, h=9, w=7, k=37, c=35, window=3, stride=2, column_block=True),
               dict(n=1, h=40, w=20, k=20, c=17, window=2, stride=1, relu=False),
               dict(n=1, h=6, w=6, k=3, c=4, window=3, stride=3, bias=False, relu=False)]


@pytest.mark.parametrize('case', FUSED_CASES)
def test_fused_pool_equals_source_readout_then_window_max(case):
    text, scales, offsets = fused_source(**case)
    manifest, (match,) = fused(text)
    stream, meta = match.kernel.stream, match.kernel.meta
    n, h, w, c = meta['pooled_store']['image']
    window, stride = meta['pooled_store']['window'], meta['pooled_store']['stride']
    k = stream.args[0].shape[1]
    lint(stream)
    assert validate(stream, meta, match.numeric) == 'f32_product_rne'
    rng = np.random.default_rng(k + c)
    inputs = dict(a=rng.integers(-40, 41, (n * h * w, k), dtype=np.int8), b=rng.integers(-9, 10, (k, c), dtype=np.int8))
    acc = inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64) + offsets
    readout = f32_readout(acc, form_multiplier(scales * .25, 7), 'i8', relu=case.get('relu', True))
    expected = window_max(readout.reshape(n, h, w, c), window, stride)
    np.testing.assert_array_equal(simulate(stream, inputs)['out'], expected)
    assert 'quant_ext.quantize' not in rewrite(text, manifest)
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'f32_product_rne'


def test_negative_readouts_are_never_padded_by_garbage_taps():
    # All-negative pooled inputs: a zero garbage tap would win every window.
    text, scales, offsets = fused_source(n=1, h=4, w=4, k=2, c=3, relu=False, bias=False)
    _, (match,) = fused(text)
    stream = match.kernel.stream
    inputs = dict(a=np.full((16, 2), 5, np.int8), b=np.full((2, 3), -7, np.int8))
    out = simulate(stream, inputs)['out']
    assert (out < 0).all()
    # Register padding would read garbage zeros; admission must refuse it.
    padded = [i._replace(attrs=tuple((key, 1 if key == 'upad' else v) for key, v in i.attrs))
              if i.kind == 'config_st' and i.a('pool_stride') else i for i in stream.ins]
    tampered = copy.copy(stream)
    tampered.ins = padded
    assert (simulate(tampered, inputs)['out'] == 0).any()
    with pytest.raises(ValueError, match='garbage'):
        validate(tampered, match.kernel.meta, match.numeric)


def test_padded_fused_pool_keeps_the_unfused_route():
    # The output path derives no pool padding value, so a -inf border refuses fusion.
    text, _, _ = fused_source(h=5, w=5, window=3, stride=2, pad=1)
    for selected in (options(), {**options(), 'path': {**options()['path'], 'readout': options(pool=False)['path']['readout']}}):
        manifest, found = fused(text, selected)
        assert not found
        assert 'quant_ext.quantize_per_tensor' in rewrite(text, manifest)


def test_register_limits_are_derived_and_refuse_wide_windows():
    from mlir_oot.core_aten.readout_encoding import field
    limits = pool_plan.register_limits()
    assert limits['pool_size'] == (1 << field('ConfigMvoutRs1', 'pool_size')[1]) - 1
    image = pool_plan.Image(1, 8, 8, DIM)
    with pytest.raises(pool_plan.PoolPlanRefused):
        pool_plan.plan(image, pool_plan.Window(limits['pool_size'] + 1, 1), capacity_rows=512)
    with pytest.raises(pool_plan.PoolPlanRefused):
        pool_plan.plan(pool_plan.Image(1, 4, limits['ocols'] + 1, DIM), pool_plan.Window(2, 2), capacity_rows=1 << 20)
    text, _, _ = fused_source(h=8, w=8, window=limits['pool_size'] + 1, stride=1)
    _, found = fused(text)
    assert not found


def test_band_planner_covers_every_pooled_row_once():
    image, window = pool_plan.Image(2, 41, 13, DIM), pool_plan.Window(3, 2, (1, 1, 1, 1))
    bands = pool_plan.plan(image, window, capacity_rows=128)
    assert len(bands) > 2 and all(b.orows * b.ocols <= 128 for b in bands)
    broken = bands[:1] + bands[2:]
    with pytest.raises(pool_plan.PoolPlanRefused):
        pool_plan.covers(broken, image, window, capacity_rows=128)
    with pytest.raises(pool_plan.PoolPlanRefused, match='only padding'):
        pool_plan.plan(pool_plan.Image(1, 4, 4, DIM), pool_plan.Window(2, 2, (2, 0, 0, 0)), capacity_rows=128)


def _tamper(stream, edit):
    changed = copy.copy(stream)
    changed.ins = edit(list(stream.ins))
    return changed


def _set(ins, **values):
    return ins._replace(attrs=tuple(sorted({**dict(ins.attrs), **values}.items())))


def _fused_adversaries():
    def truncated(ins):
        store = max(i for i, x in enumerate(ins) if x.kind == 'mvout')
        return ins[:store] + ins[store + 1:]

    def offset(ins):
        return [x._replace(off=x.off + 1) if x.kind == 'mvout' else x for x in ins]

    def stride_mid_band(ins):
        first = next(i for i, x in enumerate(ins) if x.kind == 'compute')
        ex = next(x for x in ins if x.kind == 'config_ex')
        return ins[:first] + [_set(ex, c_stride=2)] + ins[first:first + 1] + [ex] + ins[first + 1:]

    def pool_fields_mid_band(ins):
        store = next(i for i, x in enumerate(ins) if x.kind == 'mvout')
        config = ins[store - 1]
        return ins[:store] + [_set(config, pool_stride=config.a('pool_stride') + 1)] + ins[store:store + 1] + [config] + ins[store + 1:]

    def load_stride(ins):
        first = next(i for i, x in enumerate(ins) if x.kind == 'mvin' and x.a('load_id') == 1)
        ld = next(x for x in ins if x.kind == 'config_ld' and x.a('load_id') == 1)
        return ins[:first] + [_set(ld, stride=ld.a('stride') + 1)] + ins[first:first + 1] + [ld] + ins[first + 1:]

    def multiplier(ins):
        return [_set(x, multiplier_bits=x.a('multiplier_bits') + 1, acc_scale=float(np.uint32(
            x.a('multiplier_bits') + 1).view(np.float32))) if x.kind == 'config_st' and x.a('pool_stride') else x
            for x in ins]

    def activation(ins):
        return [_set(x, acc_act=0) if x.kind == 'config_st' and x.a('pool_stride') else x for x in ins]

    def missing_k(ins):
        computes = [i for i, x in enumerate(ins) if x.kind == 'compute']
        drop = computes[-1]
        return ins[:drop - 1] + ins[drop + 1:]

    def leave_pooling_on(ins):
        return ins[:-2] + ins[-1:]

    return dict(truncated=truncated, offset=offset, stride_mid_band=stride_mid_band,
                pool_fields_mid_band=pool_fields_mid_band, load_stride=load_stride, multiplier=multiplier,
                activation=activation, missing_k=missing_k, leave_pooling_on=leave_pooling_on)


@pytest.mark.parametrize('name', sorted(_fused_adversaries()))
def test_fused_admission_rejects_tampered_streams(name):
    text, _, _ = fused_source(n=1, h=6, w=6, k=37, c=20, window=3, stride=1)
    _, (match,) = fused(text)
    tampered = _tamper(match.kernel.stream, _fused_adversaries()[name])
    with pytest.raises(ValueError):
        validate(tampered, match.kernel.meta, match.numeric)


def test_provider_declines_tampered_fused_stream(monkeypatch):
    text, _, _ = fused_source()
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            if match.kernel.meta.get('pooled_store'):
                match.kernel.stream = _tamper(match.kernel.stream, _fused_adversaries()['stride_mid_band'])
        return module, manifest, matches
    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined' and 'CONFIG_EX' in proof['reason']


def test_fused_seed_tamper_and_exact_policy_decline():
    text, _, _ = fused_source()
    _, (match,) = fused(text)
    stream = copy.copy(match.kernel.stream)
    stream.consts = {**stream.consts, 'bias_c': stream.consts['bias_c'] + 1}
    with pytest.raises(ValueError, match='seed'):
        validate(stream, match.kernel.meta, match.numeric)
    exact = copy.deepcopy(POLICY)
    exact['numerical_semantics']['readout']['strategy'] = 'exact_rational_host'
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=exact, readout_options=options())
    assert proof['status'] == 'declined'


@pytest.mark.parametrize('case', [dict(), dict(pad=(0, 0, 0, 0), window=2, stride=2),
                                  dict(n=2, h=33, w=31, c=7, window=3, stride=2, pad=(1, 1, 1, 1)),
                                  dict(h=4, w=4, c=3, window=2, stride=1, pad=(1, 0, 0, 1))])
def test_standalone_pooled_store_matches_padded_window_max(case):
    text = movement_source(**case, out_scale=0.5)
    manifest, (match,) = composition(text, 'pooled_movement')
    stream, meta = match.kernel.stream, match.kernel.meta
    lint(stream)
    assert validate(stream, meta) == 'storage_codes'
    n, h, w, c = meta['pooled_store']['image']
    x = np.random.default_rng(h * w).integers(-128, -1, (n, h, w, c), dtype=np.int8)  # all negative
    expected = window_max(x, meta['pooled_store']['window'], meta['pooled_store']['stride'], tuple(case.get('pad', (1, 1, 1, 1))))
    np.testing.assert_array_equal(simulate(stream, dict(x=x))['out'], expected)
    assert 'quant_ext.quantize' not in rewrite(text, manifest)
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'storage_codes'


def test_standalone_pool_with_different_qparams_or_no_scheme_is_not_offloaded():
    for text, selected in ((movement_source(out_scale=0.25), options()), (movement_source(), None)):
        _, matches = build(text, readout_options=selected)[1:]
        assert not any(m.kernel.meta.get('composition') == 'pooled_movement' for m in matches)


def _movement_adversaries():
    def no_fence(ins):
        store = next(i for i, x in enumerate(ins) if x.kind == 'mvout')
        fence = max(i for i in range(store) if ins[i].kind == 'fence')
        return ins[:fence] + ins[fence + 1:]

    def zero_border(ins):
        return ins  # replaced through constants below

    def register_padding(ins):
        return [_set(x, upad=1, lpad=1) if x.kind == 'config_st' and x.a('pool_stride') else x for x in ins]

    def window(ins):
        return [_set(x, pool_size=x.a('pool_size') - 1) if x.kind == 'config_st' and x.a('pool_stride') else x
                for x in ins]

    def shifted_load(ins):
        index = next(i for i, x in enumerate(ins) if x.kind == 'mvin' and x.a('load_id') == 0)
        return ins[:index] + [ins[index]._replace(off=ins[index].off + 20)] + ins[index + 1:]

    return dict(no_fence=no_fence, register_padding=register_padding, window=window, shifted_load=shifted_load,
                zero_border=zero_border)


@pytest.mark.parametrize('name', sorted(_movement_adversaries()))
def test_standalone_admission_rejects_tampered_streams(name):
    text = movement_source()
    _, (match,) = composition(text, 'pooled_movement')
    tampered = _tamper(match.kernel.stream, _movement_adversaries()[name])
    if name == 'zero_border':
        tampered.consts = {'pool_border': np.zeros_like(tampered.consts['pool_border'])}
    with pytest.raises(ValueError):
        validate(tampered, match.kernel.meta)


def test_unfenced_scratchpad_pool_is_a_lint_hazard():
    _, (match,) = composition(movement_source(), 'pooled_movement')
    tampered = _tamper(match.kernel.stream, _movement_adversaries()['no_fence'])
    with pytest.raises(HazardError, match='HW4'):
        lint(tampered)


@pytest.mark.parametrize('shape,count', [((2, 8, 3, 4), 12.0), ((3, 5, 7, 9), 63.0)])
def test_window_mean_uses_the_formed_multiplier(shape, count):
    text = mean_source(shape, count=count)
    manifest, (match,) = composition(text, 'window_mean')
    stream, meta = match.kernel.stream, match.kernel.meta
    lint(stream)
    multiplier = form_multiplier(0.5, 0.25 * count)
    assert meta['multiplier_bits'] == [int(np.asarray(multiplier).view(np.uint32))]
    x = np.random.default_rng(int(count)).integers(-128, 128, shape, dtype=np.int8)
    total = x.reshape(shape[0] * shape[1], -1).astype(np.int64).sum(axis=1)
    expected = f32_readout(total, multiplier, 'i8', relu=False).reshape(shape[:2])
    np.testing.assert_array_equal(simulate(stream, dict(x=x))['y'], expected)
    assert 'quant_ext.quantize' not in rewrite(text, manifest)
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'f32_product_rne'


def test_window_mean_refuses_a_divisor_other_than_its_window_and_tampering():
    _, found = composition(mean_source(count=11.0), 'window_mean')
    assert not found
    _, (match,) = composition(mean_source(), 'window_mean')
    stream = match.kernel.stream
    for edit in (lambda ins: [_set(x, multiplier_bits=x.a('multiplier_bits') ^ 1, acc_scale=float(np.uint32(
                     x.a('multiplier_bits') ^ 1).view(np.float32))) if x.kind == 'config_st' and x.a('multiplier_bits')
                              else x for x in ins],
                 lambda ins: [x for x in ins if not (x.kind == 'compute' and x is [c for c in ins if c.kind == 'compute'][-1])]):
        with pytest.raises(ValueError):
            validate(_tamper(stream, edit), match.kernel.meta)
    ones = copy.copy(stream)
    ones.consts = {'ones16': np.full_like(stream.consts['ones16'], 2)}
    with pytest.raises(ValueError):
        validate(ones, match.kernel.meta)
