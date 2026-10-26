"""Source-bound F1/F2 emission, numeric admission and conservative fallback."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from merlin.runtime.exact_readout import f32_readout, form_multiplier
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import encode
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.readout_encoding import flags, unpack
from mlir_oot.core_aten.stream import DIM, simulate

PACKAGE = Path(__file__).resolve().parents[2]
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32',
    readout=dict(strategy='f32_product_rne'), internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def provider():
    spec = importlib.util.spec_from_file_location('readout_provider',
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


def source(m=3, k=17, n=5, *, column_block=False, raw=False, bias=True, relu=True):
    itype, otype = f'tensor<{m}x{n}xi32>', f'tensor<{m}x{n}xf32>'
    maps = 'affine_map<(m,n)->(m,n)>'
    col = 'affine_map<(m,n)->(n)>'
    weight_scales = np.array([2.0 ** (-1 - (j // DIM if column_block else 0)) for j in range(n)])
    offsets = np.arange(n)-5 if bias else np.zeros(n, dtype=int)
    def generic(name, inputs, types, indexing, body):
        return f'''%e_{name}=tensor.empty():{otype}
        %{name}=linalg.generic {{indexing_maps=[{','.join(indexing)}], iterator_types=["parallel","parallel"]}}
        ins({inputs}:{types}) outs(%e_{name}:{otype}) {{ {body} }} -> {otype}'''
    text = [f'''builtin.module {{ func.func @forward(%a:tensor<{m}x{k}xi8>, %b:tensor<{k}x{n}xi8>)
        -> {'tensor<'+str(m)+'x'+str(n)+'xi8>' if not raw else itype} {{
        %z=arith.constant 0:i32
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
          linalg.yield %v:i32 }} -> {itype}''']
    if raw:
        return '\n'.join(text+[f'func.return %mm:{itype}', '}}']), weight_scales, offsets
    text += [generic('f', '%mm', itype, [maps,maps],
        '^bb0(%x:i32,%o:f32): %v=arith.sitofp %x:i32 to f32 linalg.yield %v:f32')]
    dense_scales = ','.join(str(.25 * v) for v in weight_scales)
    text += [f'%sc=arith.constant dense<[{dense_scales}]>:tensor<{n}xf32>',
        generic('scaled','%f,%sc',f'{otype},tensor<{n}xf32>',[maps,col,maps],
                '^bb0(%x:f32,%s:f32,%o:f32): %v=arith.mulf %x,%s:f32 linalg.yield %v:f32')]
    last = 'scaled'
    if bias:
        dense_bias = ','.join(str(float(v)) for v in offsets*.25*weight_scales)
        text += [f'%bias=arith.constant dense<[{dense_bias}]>:tensor<{n}xf32>',
            generic('biased', '%scaled,%bias', f'{otype},tensor<{n}xf32>',[maps,col,maps],
                '^bb0(%x:f32,%bias_value:f32,%o:f32): %v=arith.addf %x,%bias_value:f32 linalg.yield %v:f32')]
        last = 'biased'
    if relu:
        text += [generic('relu',f'%{last}',otype,[maps,maps],
            '^bb0(%x:f32,%o:f32): %rz=arith.constant 0.0:f32 %v=arith.maximumf %x,%rz:f32 linalg.yield %v:f32')]
        last = 'relu'
    text += ['%qs=arith.constant dense<7.0>:tensor<f32>', '%qz=arith.constant dense<0>:tensor<i64>',
        f'''%q="quant_ext.quantize_per_tensor"(%{last},%qs,%qz)
        <{{quant_min=-128:i64,quant_max=127:i64,output_dtype="int8"}}>:
        ({otype},tensor<f32>,tensor<i64>)->tensor<{m}x{n}xi8>
        func.return %q:tensor<{m}x{n}xi8>''', '}}']
    return '\n'.join(text), weight_scales, offsets


@pytest.mark.parametrize('m,k,n,column_block', [(3,17,5,False), (17,33,35,True), (2,1,17,True), (129,1,129,True)])
def test_f1_arbitrary_f32_and_column_blocks(m,k,n,column_block):
    text, scales, offsets = source(m,k,n,column_block=column_block)
    _, manifest, matches = build(text, readout_options=options())
    match, = [m for m in matches if m.kernel.meta.get('composition_plan')]
    stream = match.kernel.stream
    assert not stream.readouts
    assert all(not i.a('local') & flags()['full_c_bit'] for i in stream.ins if i.kind == 'mvout')
    assert any(i.kind == 'mvin' and i.a('load_id') == 2 for i in stream.ins)
    bits = form_multiplier(scales*.25, 7).view(np.uint32).tolist()
    assert match.kernel.meta['multiplier_bits'] == bits
    encoded = [unpack(encode(i).values[1], 'ConfigMvoutRs2', 'acc_scale') for i in stream.ins if i.kind == 'config_st']
    assert set(encoded) == set(bits)
    rng = np.random.default_rng(k+n)
    inputs = dict(a=rng.integers(-32,33,(m,k),dtype=np.int8), b=rng.integers(-8,9,(k,n),dtype=np.int8))
    acc = inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64) + offsets
    expected = f32_readout(acc, form_multiplier(scales*.25, 7), 'i8', relu=True)
    np.testing.assert_array_equal(simulate(stream, inputs)['out'], expected)
    assert len(manifest['bindings']) == 1
    assert 'quant_ext.quantize' not in rewrite(text,manifest)
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'f32_product_rne'


def test_f2_full_width_has_no_readout_stages():
    text, _, _ = source(raw=True)
    _, _, matches = build(text, readout_options=options())
    match, = [m for m in matches if m.kernel.meta.get('composition_plan')]
    assert match.kernel.meta['composition_plan']['route'] == 'int32_readout'
    assert all(i.a('local') & flags()['full_c_bit'] for i in match.kernel.stream.ins if i.kind == 'mvout')
    proof = provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY,readout_options=options())
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['device_readout'] == 'full_i32'


def test_unproved_and_missing_policy_keep_host_boundary():
    text, _, _ = source()
    for selected in (None, {**options(), 'scheme':dict(readout_strategy='invalid')},
                     {**options(), 'path':{**options()['path'], 'exits':dict(accumulator=True,requant=False)}}):
        _, manifest, matches = build(text,readout_options=selected)
        assert not any(m.kernel.meta.get('readout') == 'f32_product_rne' for m in matches)
        assert 'quant_ext.quantize_per_tensor' in rewrite(text, manifest)
        if selected is not None:
            producer, = [m for m in matches if m.kernel.meta.get('integer_producer')]
            assert producer.kernel.meta['readout'] == 'full_i32'
            assert all(i.a('local') & flags()['full_c_bit'] for i in producer.kernel.stream.ins if i.kind == 'mvout')


@pytest.mark.parametrize('bias,relu', [(False,False), (True,False), (False,True)])
def test_optional_bias_and_activation_follow_source(bias,relu):
    text, scales, offsets = source(m=1,k=3,n=2,bias=bias,relu=relu)
    _, _, matches = build(text,readout_options=options())
    match, = [m for m in matches if m.kernel.meta.get('composition_plan')]
    inputs = dict(a=np.array([[-7,11,23]],np.int8),b=np.array([[2,1],[3,-8],[1,-2]],np.int8))
    acc = inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64) + offsets
    expected = f32_readout(acc,form_multiplier(scales*.25,7),'i8',relu=relu)
    np.testing.assert_array_equal(simulate(match.kernel.stream,inputs)['out'],expected)


def test_numeric_proof_for_another_clamp_retains_host_readout():
    text, _, _ = source()
    text = text.replace('quant_min=-128', 'quant_min=-127')
    selected = options()
    selected['path']['readout'][-1]['clamp'] = [-127,127]
    _, manifest, matches = build(text,readout_options=selected)
    producer, = [m for m in matches if m.kernel.meta.get('integer_producer')]
    assert producer.kernel.meta['readout'] == 'full_i32'
    assert 'quant_ext.quantize_per_tensor' in rewrite(text,manifest)


def test_region_routing_reproves_and_forwards_the_selected_plan(tmp_path,monkeypatch):
    from merlin.targetgen.rtl.facts import load_facts
    text, _, _ = source()
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    (tmp_path/'spec-int8-admission.json').write_text(json.dumps(dict(selection=dict(spec=POLICY))))
    api = provider()
    emitted = []
    def emit(*args,**kwargs):
        emitted.append(kwargs)
        return ('rewritten','shim','object','manifest')
    monkeypatch.setattr(api,'core_aten_offload',emit)
    route = api.routing(tmp_path,target='gemmini',package=PACKAGE,facts=load_facts('gemmini'),
                        eligible=[],readout_options=options())
    assert route.region_offload(prepared,tmp_path/'work') == ('rewritten','shim','object','manifest')
    assert emitted[0]['readout_options'] == options()


@pytest.mark.parametrize('corruption', ['multiplier', 'seed', 'store_width', 'activation'])
def test_device_admission_rejects_different_policy_and_tampered_effects(monkeypatch,corruption):
    text, _, _ = source()
    api = provider()
    exact = copy.deepcopy(POLICY)
    exact['numerical_semantics']['readout']['strategy'] = 'exact_rational_host'
    proof = api.admit_numeric_schedule(text,package=PACKAGE,numeric_policy=exact,readout_options=options())
    assert proof['status'] == 'declined'
    backend = api._core_aten_module(PACKAGE,'catalog')
    original = backend.build
    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            if corruption == 'seed':
                stream.consts['bias_c'] = stream.consts['bias_c'] + 1
            else:
                kind,field = ('mvout','local') if corruption == 'store_width' else (
                    'config_st','acc_scale' if corruption == 'multiplier' else 'acc_act')
                value = flags()['c_acc'] if corruption == 'store_width' else .25 if corruption == 'multiplier' else 0
                stream.ins = [i._replace(attrs=tuple((k,value if k==field else v) for k,v in i.attrs))
                    if i.kind == kind else i for i in stream.ins]
        return module,manifest,matches
    monkeypatch.setattr(backend,'build',altered)
    proof = api.admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY,readout_options=options())
    assert proof['status'] == 'declined'
    assert any(word in proof['reason'] for word in ('multiplier bits','source-proved bias','store width'))


@pytest.mark.parametrize('corruption', [
    'truncated', 'offset', 'rows', 'cols', 'shrunk', 'stride', 'block_stride',
    'pixel_repeats', 'duplicate', 'late', 'local', 'accumulate', 'truncated_reuse', 'seed_storage',
])
def test_bias_seed_requires_complete_slices_before_compute(monkeypatch, corruption):
    text, _, _ = source(m=129, k=1, n=129) if corruption == 'truncated_reuse' else source(m=17, k=1, n=35)
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            seeds = [i for i in stream.ins if i.kind == 'mvin' and i.a('local') & flags()['acc_i8']]
            assert len(seeds) > 1
            if corruption == 'truncated_reuse':
                stream.ins.remove(seeds[-1])
            elif corruption == 'truncated':
                stream.ins = [i for i in stream.ins if i not in seeds[1:]]
            elif corruption == 'duplicate':
                position = stream.ins.index(seeds[0])
                stream.ins.insert(position, seeds[0])
            elif corruption == 'late':
                stream.ins.remove(seeds[0])
                position = next(j for j, i in enumerate(stream.ins) if i.kind == 'compute')
                stream.ins.insert(position + 1, seeds[0])
            elif corruption == 'seed_storage':
                stream.consts['bias_c'] = stream.consts['bias_c'].astype(np.int8)
            elif corruption == 'offset':
                stream.ins = [i._replace(off=i.off + 4) if i in seeds else i for i in stream.ins]
            else:
                config = corruption in ('shrunk', 'stride', 'block_stride', 'pixel_repeats')
                field = corruption
                value = dict(rows=1, cols=1, shrunk=1, stride=4, block_stride=1,
                             pixel_repeats=2, local=flags()['acc_i8'] | 1,
                             accumulate=flags()['c_acc'])[corruption]
                if corruption == 'accumulate':
                    field = 'local'
                stream.ins = [
                    i._replace(attrs=tuple((k, value if k == field else v) for k, v in i.attrs))
                    if ((i.kind == 'config_ld' and i.a('load_id') == seeds[0].a('load_id'))
                        if config else i in seeds) else i for i in stream.ins]
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', proof
    assert any(word in proof['reason'] for word in ('seed', 'MVIN3'))


def mutate_config_at_use(stream, kind, field, value, use, load_id=None):
    """Change only the state around each use; restore it immediately afterward."""
    current = None
    instructions = []
    for ins in stream.ins:
        if ins.kind == kind and (kind != 'config_ld' or ins.a('load_id') == load_id):
            current = ins
        if ins.kind == use:
            assert current is not None
            attrs = dict(current.attrs)
            attrs[field] = value
            if kind == 'config_st' and field == 'acc_scale':
                attrs.pop('multiplier_bits', None)
            instructions.extend([current._replace(attrs=tuple(sorted(attrs.items()))), ins, current])
        else:
            instructions.append(ins)
    stream.ins = instructions


def test_c_stride_change_after_preload_rejects_wrong_readout(monkeypatch):
    text, scales, offsets = source(m=17, k=1, n=35)
    inputs = dict(a=np.full((17, 1), 11, np.int8), b=np.full((1, 35), 7, np.int8))
    expected = f32_readout(inputs['a'].astype(np.int64) @ inputs['b'].astype(np.int64) + offsets,
                           form_multiplier(scales*.25, 7), 'i8', relu=True)
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            np.testing.assert_array_equal(simulate(stream, inputs)['out'], expected)
            mutate_config_at_use(stream, 'config_ex', 'c_stride', 2, 'compute')
            actual = simulate(stream, inputs)['out']
            assert np.count_nonzero(actual != expected) == 435
            assert (int(actual[0, 16]), int(expected[0, 16])) == (3, 2)
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', proof
    assert 'effective CONFIG_EX' in proof['reason']


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('use', ['preload', 'compute', 'mvout'])
@pytest.mark.parametrize('kind,field,value', [
    ('config_ex', 'dataflow', 0), ('config_ex', 'act', 1),
    ('config_ex', 'acc_scale', .5), ('config_ex', 'a_stride', 2),
    ('config_ex', 'c_stride', 2), ('config_ex', 'a_transpose', 1),
    ('config_ex', 'b_transpose', 1),
    ('config_ld', 'stride', 4), ('config_ld', 'scale', .5),
    ('config_ld', 'shrunk', 1), ('config_ld', 'block_stride', 1),
    ('config_ld', 'pixel_repeats', 2),
])
def test_effective_config_mutations_decline(monkeypatch, raw, use, kind, field, value):
    text, _, _ = source(m=17, k=1, n=35, raw=raw)
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            # Exercise a live operand load state on both seeded and zero-seed paths.
            mutate_config_at_use(match.kernel.stream, kind, field, value, use, load_id=0)
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', proof


@pytest.mark.parametrize('load_id,use', [(0, 'mvin'), (1, 'mvin'), (2, 'mvin'),
    (1, 'compute'), (2, 'compute'), (1, 'mvout'), (2, 'mvout')])
@pytest.mark.parametrize('field,value', [('stride', 4), ('scale', .5), ('shrunk', 1),
                                       ('block_stride', 1), ('pixel_repeats', 2)])
def test_operand_and_seed_load_config_mutations_decline(monkeypatch, load_id, use, field, value):
    text, _, _ = source(m=17, k=1, n=35)
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            mutate_config_at_use(match.kernel.stream, 'config_ld', field, value, use, load_id)
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', proof


@pytest.mark.parametrize('raw', [False, True])
@pytest.mark.parametrize('field,value', [('stride', 1), ('acc_act', 0), ('acc_scale', .5)])
def test_effective_store_config_mutations_decline(monkeypatch, raw, field, value):
    text, _, _ = source(m=17, k=1, n=35, raw=raw)
    if raw and field == 'acc_act':
        value = 1
    api = provider()
    backend = api._core_aten_module(PACKAGE, 'catalog')
    original = backend.build

    def altered(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        for match in matches:
            mutate_config_at_use(match.kernel.stream, 'config_st', field, value, 'mvout')
        return module, manifest, matches

    monkeypatch.setattr(backend, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'declined', proof


@pytest.mark.parametrize('fixture', ['neg', 'relu', 'permute'])
def test_f32_selection_preserves_existing_noninteger_admission(fixture):
    spec = importlib.util.spec_from_file_location('fixture_source', PACKAGE / 'tests/core_aten/fixture_source.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    text = fixtures.read(PACKAGE / 'tests/fixtures/core_aten' / fixture / 'model.mlir')
    api = provider()
    previous = copy.deepcopy(POLICY)
    previous['numerical_semantics']['readout']['strategy'] = 'exact_rational_host'
    baseline = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=previous)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert baseline['status'] == proof['status'] == 'proven', proof
    assert proof == baseline


def test_full_matrix_seed_uses_each_output_rows_bias_slice():
    m, k, n = 17, 1, 35
    text, scales, offsets = source(m, k, n)
    seed = np.arange(m*n, dtype=np.int32).reshape(m, n) - 200
    literal = '[' + ','.join('[' + ','.join(map(str, row)) + ']' for row in seed) + ']'
    text = text.replace(f'%init=linalg.fill ins(%z:i32) outs(%e:tensor<{m}x{n}xi32>)->tensor<{m}x{n}xi32>',
                        f'%init=arith.constant dense<{literal}>:tensor<{m}x{n}xi32>')
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY, readout_options=options())
    assert proof['status'] == 'proven', proof
    _, _, matches = build(text, readout_options=options())
    match, = [m for m in matches if m.kernel.meta.get('composition_plan')]
    inputs = dict(a=np.ones((m,k), np.int8), b=np.ones((k,n), np.int8))
    expected = f32_readout(seed.astype(np.int64) + 1 + offsets, form_multiplier(scales*.25, 7), 'i8', relu=True)
    np.testing.assert_array_equal(simulate(match.kernel.stream, inputs)['out'], expected)
