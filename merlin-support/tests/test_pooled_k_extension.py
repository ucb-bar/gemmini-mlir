"""Residual digit accumulation before store-path max pooling, and refusal twins."""
import copy

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.stream import simulate
from mlir_oot.core_aten.pooled_store import validate
from mlir_oot.core_aten.stream_proof import ProofError
from test_pooled_store import fused_source, options as pool_options, window_max
from test_composition_schedules import options as extension_options, admit


def options():
    selected = copy.deepcopy(extension_options())
    selected['path'] = pool_options()['path']
    chain = selected['compositions']['chains'][1]
    chain['reduction_kinds'] = ['max']
    chain['constraints'] += [dict(max_requires_store_path_pooling='from_facts'),
                            'scratchpad_source_pooling_fenced',
                            dict(padded_max_requires='producer_lower_bound_or_derived_pad_value')]
    return selected


def source(h=5, w=5, k=17, c=19, coefficient=254, *, pad=1, relu=True):
    text, _, _ = fused_source(h=h, w=w, k=k, c=c, window=3, stride=2,
                              bias=False, relu=relu, pad=pad)
    m = h * w
    raw, floating = f'tensor<{m}x{c}xi8>', f'tensor<{m}x{c}xf32>'
    text = text.replace(f'%b:tensor<{k}x{c}xi8>)', f'%b:tensor<{k}x{c}xi8>,%residual_arg:{raw})')
    marker = '%e_relu=' if relu else '%img='
    residual = f'''%rs=arith.constant dense<{coefficient * .125}>:tensor<f32>
    %reszero=arith.constant dense<0>:tensor<i64>
    %rd="quant_ext.dequantize_per_tensor"(%residual_arg,%rs,%reszero)
      <{{quant_min=-128:i64,quant_max=127:i64}}>:({raw},tensor<f32>,tensor<i64>)->{floating}
    %er=tensor.empty():{floating}
    %sum=linalg.generic {{indexing_maps=[affine_map<(m,n)->(m,n)>,
      affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}}
      ins(%scaled,%rd:{floating},{floating}) outs(%er:{floating}) {{
      ^bb0(%x:f32,%rv:f32,%o:f32): %v=arith.addf %x,%rv:f32 linalg.yield %v:f32 }}->{floating}
    '''
    text = text.replace(marker, residual + marker).replace('ins(%scaled:', 'ins(%sum:')
    if not relu:
        text = text.replace('tensor.expand_shape %scaled', 'tensor.expand_shape %sum')
    return text


@pytest.mark.parametrize('h,w,k,c,coefficient,pad,relu', [
    (5, 5, 17, 19, 254, 1, True), (7, 9, 33, 35, -254, 0, False),
    (40, 20, 20, 17, 1, 0, True), (1, 1, 1, 3, -255, 1, True)])
def test_pooled_residual_executes_and_is_admitted(h, w, k, c, coefficient, pad, relu):
    text = source(h, w, k, c, coefficient, pad=pad, relu=relu)
    selected = options()
    _, _, matches = build(text, readout_options=selected)
    match, = [m for m in matches if m.kernel.meta.get('pooled_store')]
    assert match.kernel.meta['k_extension_plan']['decision']['coefficient'] == coefficient
    assert validate(match.kernel.stream, match.kernel.meta, match=match, readout_options=selected) == 'f32_product_rne'
    rng = np.random.default_rng(k)
    inputs = dict(a=rng.integers(-8, 9, (h*w, k), dtype=np.int8),
                  b=rng.integers(-8, 9, (k, c), dtype=np.int8),
                  residual=rng.integers(-128, 128, (h*w, c), dtype=np.int8))
    acc = inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64)
    acc += coefficient * inputs['residual'].astype(np.int64)
    # Independent source operations, including the residual dequantization.
    scaled = (inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64)).astype(np.float32) * np.float32(.125)
    scaled = scaled + inputs['residual'].astype(np.float32) * np.float32(coefficient * .125)
    if relu:
        scaled = np.maximum(scaled, 0)
    quantized = np.clip(np.rint(scaled / np.float32(7)), -128, 127).astype(np.int8)
    expected = window_max(quantized.reshape(1,h,w,c), 3, 2, (pad,)*4)
    np.testing.assert_array_equal(simulate(match.kernel.stream, inputs)['out'], expected)
    proof = admit(text, selected)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_residual', 'wrong_digit', 'wrong_scale'])
def test_pooled_residual_tampering_refuses(mutation):
    text, selected = source(h=3,w=3,c=5), options()
    _, _, matches = build(text, readout_options=selected)
    match, = [m for m in matches if m.kernel.meta.get('pooled_store')]
    stream = match.kernel.stream
    if mutation in ('missing', 'duplicate'):
        index = next(i for i, ins in enumerate(stream.ins) if ins.kind == 'compute' and
                     i > next(j for j, v in enumerate(stream.ins) if v.kind == 'mvin' and v.arg == stream._index['residual']))
        if mutation == 'missing':
            del stream.ins[index]
        else:
            stream.ins[index:index] = [stream.ins[index]]
    elif mutation == 'wrong_residual':
        ins = next(i for i in stream.ins if i.kind == 'mvin' and i.arg == stream._index['residual'])
        index = stream.ins.index(ins)
        stream.ins[index] = ins._replace(off=ins.off + 1)
    elif mutation == 'wrong_digit':
        stream.consts['kext_digits'][0,0] -= 1
    else:
        ins = next(i for i in stream.ins if i.kind == 'config_st' and i.a('pool_stride'))
        index = stream.ins.index(ins)
        attrs = dict(ins.attrs, multiplier_bits=0x3f800000)
        stream.ins[index] = ins._replace(attrs=tuple(sorted(attrs.items())))
    with pytest.raises((ProofError, ValueError, KeyError)):
        validate(stream, match.kernel.meta, match=match, readout_options=selected)


def test_undeclared_or_nonintegral_pooled_extension_keeps_host_boundary():
    for text, selected in [(source(), pool_options()),
                           (source().replace('dense<31.75>', 'dense<31.8>'), options())]:
        _, _, matches = build(text, readout_options=selected)
        assert not any(m.kernel.meta.get('pooled_store') for m in matches)


def test_pooled_source_inexact_f32_residual_stays_on_host():
    text = source(h=5, w=5, k=7, c=3, coefficient=1, pad=0)
    text = text.replace('0.125', '0.1666666716337204').replace('dense<7.0>', 'dense<1.0>')
    selected = options()
    _, _, matches = build(text, readout_options=selected)
    assert not any(m.kernel.meta.get('pooled_store') for m in matches)
    gain = np.float32(1 / 6)
    source_value = np.float32(np.float32(823) * gain + np.float32(-64) * gain)
    extended_value = np.float32(823 - 64) * gain
    assert np.rint(source_value) == 127 and np.rint(extended_value) == 126
