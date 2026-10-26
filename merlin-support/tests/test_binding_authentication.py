"""Every admission path authenticates the manifest bindings that offload rewrites."""

import importlib.util
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def fixtures(name):
    spec = importlib.util.spec_from_file_location(f'binding_{name}', TESTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def two_raw_contractions(m=4, k=6, n=5):
    def contraction(i):
        return f'''%e{i}=tensor.empty():tensor<{m}x{n}xi32>
    %z{i}=arith.constant 0:i32
    %init{i}=linalg.fill ins(%z{i}:i32) outs(%e{i}:tensor<{m}x{n}xi32>)->tensor<{m}x{n}xi32>
    %mm{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,
      affine_map<(m,n,k)->(m,n)>], iterator_types=["parallel","parallel","reduction"]}}
    ins(%a{i},%b{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init{i}:tensor<{m}x{n}xi32>) {{
    ^bb0(%x:i8,%w:i8,%c:i32):
      %xx=arith.extsi %x:i8 to i32
      %ww=arith.extsi %w:i8 to i32
      %p=arith.muli %xx,%ww:i32
      %v=arith.addi %p,%c:i32
      linalg.yield %v:i32 }} -> tensor<{m}x{n}xi32>'''
    result = f'tensor<{m}x{n}xi32>'
    return f'''builtin.module {{ func.func @forward(%a0:tensor<{m}x{k}xi8>,%b0:tensor<{k}x{n}xi8>,
      %a1:tensor<{m}x{k}xi8>,%b1:tensor<{k}x{n}xi8>) -> ({result},{result}) {{
{contraction(0)}
{contraction(1)}
func.return %mm0,%mm1:{result},{result}
}} }}'''


def _swap(mapping, left, right):
    mapping[left], mapping[right] = mapping[right], mapping[left]


def contraction_case(mode):
    readout = fixtures('test_composition_readout')

    def edit(manifest, matches):
        producers = [m for m in matches if m.kernel.meta.get('integer_producer')]
        assert len(producers) == 2
        rows = manifest['bindings']
        if mode == 'match_and_binding_results':
            producers[0].result, producers[1].result = producers[1].result, producers[0].result
        if mode in ('binding_results', 'match_and_binding_results'):
            rows[0]['result'], rows[1]['result'] = rows[1]['result'], rows[0]['result']
        if mode == 'binding_operands':
            rows[0]['operands']['a'], rows[1]['operands']['a'] = rows[1]['operands']['a'], rows[0]['operands']['a']
    return readout, two_raw_contractions(), edit


def shared_case():
    shared = fixtures('test_shared_input_fusion')
    text, _, _ = shared.source(5, 16, (16, 16), constant=False)
    return shared, text, lambda manifest, _: _swap(manifest['bindings'][0]['operands'], 'w0', 'w1')


def pooled_case():
    pooled = fixtures('test_pooled_store')
    text = pooled.fused_source(n=1, h=2, w=2, k=4, c=4)[0]
    return pooled, text, lambda manifest, _: _swap(manifest['bindings'][0]['operands'], 'a', 'b')


def operand_sum_case():
    opsum = fixtures('test_scaled_operand_sum')
    text, _ = opsum.staged(17, 20, 'per_operand', opsum.GRID[:2], opsum.GRID[2:])
    return opsum, text, lambda manifest, _: _swap(manifest['bindings'][0]['operands'], 'a', 'b')


CASES = {
    'contraction_binding_results': lambda: contraction_case('binding_results'),
    'contraction_match_and_binding_results': lambda: contraction_case('match_and_binding_results'),
    'contraction_binding_operands': lambda: contraction_case('binding_operands'),
    'shared_input_weights': shared_case,
    'pooled_operands': pooled_case,
    'operand_sum_operands': operand_sum_case,
}


def admit(module, text, edit=None, monkeypatch=None):
    api = module.provider()
    if edit is not None:
        backend = api._core_aten_module(module.PACKAGE, 'catalog')
        original = backend.build

        def altered(*args, **kwargs):
            device, manifest, matches = original(*args, **kwargs)
            edit(manifest, matches)
            return device, manifest, matches
        monkeypatch.setattr(backend, 'build', altered)
    return api.admit_numeric_schedule(text, package=module.PACKAGE, numeric_policy=module.POLICY,
                                      readout_options=module.options())


@pytest.mark.parametrize('case', sorted(CASES))
def test_unaltered_bindings_are_admitted(case):
    module, text, _ = CASES[case]()
    proof = admit(module, text)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('case', sorted(CASES))
def test_altered_bindings_are_declined(monkeypatch, case):
    module, text, edit = CASES[case]()
    proof = admit(module, text, edit, monkeypatch)
    assert proof['status'] == 'declined', proof
    assert 'binding' in proof['reason'] or 'bound results' in proof['reason'], proof['reason']


@pytest.mark.parametrize('case', sorted(CASES))
def test_binding_symbol_must_name_its_own_stream(monkeypatch, case):
    module, text, _ = CASES[case]()

    def rename(manifest, matches):
        for row in (*manifest['bindings'], *manifest['kernels']):
            row['symbol'] = 'gemmini_ca_' + '0' * 16

    proof = admit(module, text, rename, monkeypatch)
    assert proof['status'] == 'declined', proof
    assert 'binding symbol' in proof['reason']


@pytest.mark.parametrize('corruption', [
    'none', 'proof_digest', 'missing_digest', 'source', 'binding', 'stream', 'rebuilt_stream', 'manifest_digest',
])
def test_compile_authenticates_full_admitted_digests(tmp_path, monkeypatch, corruption):
    import copy
    import json
    import numpy as np

    module = fixtures('test_composition_readout')
    api = module.provider()
    text = module.source()[0]
    proof = api.admit_numeric_schedule(text, package=module.PACKAGE, numeric_policy=module.POLICY,
                                       readout_options=module.options())
    assert proof['status'] == 'proven', proof
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    _, manifest, _ = catalog.build(text, readout_options=module.options())
    assert {k['symbol']: k['stream_sha256'] for k in proof['kernels']} == {
        k['symbol']: k['stream_sha256'] for k in manifest['kernels']}
    proof = copy.deepcopy(proof)
    if corruption == 'proof_digest':
        digest = proof['kernels'][0]['stream_sha256']
        # A suffix change preserves the 16-digit symbol prefix.
        proof['kernels'][0]['stream_sha256'] = digest[:16] + '0' * 48
    elif corruption == 'missing_digest':
        del proof['kernels'][0]['stream_sha256']
    elif corruption == 'binding':
        proof['bindings'][0]['operands']['a']['func_arg'] = 1
    original = catalog.build

    def altered(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        if corruption in ('stream', 'rebuilt_stream'):
            for match in matches:
                stream = match.kernel.stream
                stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs),
                    'multiplier_bits': int(np.float32(.25).view(np.uint32)), 'acc_scale': .25}.items())))
                    if i.kind == 'config_st' else i for i in stream.ins]
            if corruption == 'rebuilt_stream':
                digest = catalog._stream_digest(matches[0].kernel)
                symbol = 'gemmini_ca_' + digest[:16]
                manifest['kernels'][0].update(symbol=symbol, stream_sha256=digest)
                manifest['bindings'][0]['symbol'] = symbol
        elif corruption == 'manifest_digest':
            manifest['kernels'][0]['stream_sha256'] = '0' * 64
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', altered)
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text + ('\n' if corruption == 'source' else ''))
    manifest_path, obj = api.core_aten_catalog(prepared, tmp_path / 'work', package=module.PACKAGE,
        numeric_policy=module.POLICY, readout_options=module.options(), numeric_schedule_proof=proof)
    actual = json.loads(manifest_path.read_text())
    if corruption == 'none':
        assert obj is not None and compiled == [True]
        assert actual['status'] == 'compiled'
        assert actual['numeric_schedule_proof'] == proof
    else:
        assert obj is None and not compiled
        assert actual['status'] == 'declined'
        assert not actual['kernels'] and not actual['bindings']
        assert 'compilation' not in actual


@pytest.mark.parametrize('entry', ['core_aten_catalog', 'core_aten_offload'])
@pytest.mark.parametrize('selection', ['argument', 'environment'])
def test_public_entries_admit_numeric_kernels_without_json(tmp_path, monkeypatch, entry, selection):
    import json

    module = fixtures('test_composition_readout')
    api = module.provider()
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: {})
    kwargs = dict(package=module.PACKAGE, readout_options=module.options())
    if selection == 'argument':
        kwargs['numeric_policy'] = module.POLICY
    else:
        from merlin.targetgen import software_spec
        monkeypatch.setenv('MERLIN_CORE_ATEN_SOFTWARE_SPEC', str(tmp_path / 'selected.yaml'))
        monkeypatch.setattr(software_spec, 'load_software_spec', lambda *args, **kwargs: module.POLICY)
    text = module.source()[0]
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    result = getattr(api, entry)(prepared, tmp_path / 'work', **kwargs)
    manifest_path = result[0 if entry == 'core_aten_catalog' else 3]
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'compiled'
    proof = manifest['numeric_schedule_proof']
    assert proof['status'] == 'proven'
    assert {k['symbol']: k['stream_sha256'] for k in proof['kernels']} == {
        k['symbol']: k['stream_sha256'] for k in manifest['kernels']}
    if entry == 'core_aten_offload':
        assert result[0].read_text().count('func.call @ca_') == 1


def test_offload_declines_source_changed_during_compilation(tmp_path, monkeypatch):
    import json

    module = fixtures('test_composition_readout')
    api = module.provider()
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(module.source()[0])

    def compile_and_change(*args, **kwargs):
        prepared.write_text(prepared.read_text() + '\n')
        return {}

    monkeypatch.setattr(catalog, 'compile_to_object', compile_and_change)
    result = api.core_aten_offload(prepared, tmp_path / 'work', package=module.PACKAGE,
        numeric_policy=module.POLICY, readout_options=module.options())
    assert result[:3] == (None, None, None)
    manifest = json.loads(result[3].read_text())
    assert manifest['status'] == 'declined' and not manifest['kernels']
    assert 'prepared source changed' in manifest['reason']


@pytest.mark.parametrize('tampered', [False, True])
def test_partial_catalog_proves_linked_kernels_and_retains_host_work(tmp_path, monkeypatch, tampered):
    import json

    module = fixtures('test_composition_readout')
    text = (module.PACKAGE / 'tests/core_aten/fixtures/scalar_mean/minimal.mlir').read_text()
    host = '''%host_empty = tensor.empty() : tensor<2x3xf32>
    %host = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2x3xf32>) outs(%host_empty : tensor<2x3xf32>) {
      ^bb(%x:f32,%out:f32):
        %squared = arith.mulf %x,%x : f32
        linalg.yield %squared : f32
    } -> tensor<2x3xf32>
    '''
    text = text.replace('func.return %c0_35, %arg0', host + 'func.return %c0_35, %host')
    api = module.provider()
    strict = api.admit_numeric_schedule(text, package=module.PACKAGE, numeric_policy=module.POLICY)
    assert strict['status'] == 'declined' and 'every live' in strict['reason']
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    original = catalog.build

    def altered(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        if tampered:
            result = next(iter(manifest['bindings'][0]['result'].values()))
            result['op_ordinal'] += 1
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', altered)
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    result = api.core_aten_offload(prepared, tmp_path / 'work', package=module.PACKAGE,
                                   numeric_policy=module.POLICY)
    manifest = json.loads(result[3].read_text())
    if tampered:
        assert result[:3] == (None, None, None) and not compiled
        assert manifest['status'] == 'declined' and 'binding' in manifest['reason']
    else:
        assert manifest['status'] == 'compiled' and compiled == [True]
        assert manifest['numeric_schedule_proof']['coverage'] == 'selected_catalog_regions'
        rewritten = result[0].read_text()
        assert 'arith.mulf' in rewritten and rewritten.count('func.call @ca_') == 1
        assert len(manifest['kernels']) == len(manifest['numeric_schedule_proof']['kernels']) == 1


@pytest.mark.parametrize('corruption', ['none', 'stream', 'binding', 'match_and_binding'])
@pytest.mark.parametrize('numeric_policy', [None, 'selected'])
def test_public_byte_movement_is_source_proved(tmp_path, monkeypatch, corruption, numeric_policy):
    import json

    module = fixtures('test_composition_readout')
    api = module.provider()
    text = '''builtin.module {
      func.func @forward(%a:tensor<3xf32>,%b:tensor<2xf32>) -> tensor<5xf32> {
        %out = tensor.concat dim(0) %a,%b : (tensor<3xf32>,tensor<2xf32>) -> tensor<5xf32>
        func.return %out : tensor<5xf32>
      }
    }'''
    proof = api.admit_numeric_schedule(text, package=module.PACKAGE,
        numeric_policy=module.POLICY if numeric_policy else None)
    assert proof['status'] == 'proven', proof
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    original = catalog.build

    def altered(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        if corruption == 'stream':
            stream = matches[0].kernel.stream
            stream.ins = [i._replace(attrs=tuple((k, 0 if k == 'stride' else v) for k,v in i.attrs))
                          if i.kind == 'config_ld' else i for i in stream.ins]
        elif corruption in ('binding', 'match_and_binding'):
            operands = manifest['bindings'][0]['operands']
            operands['x0'], operands['x1'] = operands['x1'], operands['x0']
            if corruption == 'match_and_binding':
                match = matches[0]
                match.operands = [(name, value) for (name, _), (_, value) in
                                  zip(match.operands, reversed(match.operands))]
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', altered)
    monkeypatch.delenv('MERLIN_CORE_ATEN_SOFTWARE_SPEC', raising=False)
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    result = api.core_aten_offload(prepared, tmp_path / 'work', package=module.PACKAGE,
        numeric_policy=module.POLICY if numeric_policy else None)
    manifest = json.loads(result[3].read_text())
    if corruption == 'none':
        assert compiled == [True] and manifest['status'] == 'compiled'
        assert manifest['numeric_schedule_proof']['kernels'][0]['source_proof'] == 'closed_static_element_selectors'
    else:
        assert not compiled and result[:3] == (None, None, None)
        assert manifest['status'] == 'declined'


@pytest.mark.parametrize('tampered', [False, True])
def test_mixed_numeric_and_byte_kernels_each_require_a_proof(tmp_path, monkeypatch, tampered):
    import json

    module = fixtures('test_composition_readout')
    text = module.source(raw=True)[0]
    text = text.replace('-> tensor<3x5xi32> {', '-> (tensor<3x5xi32>,tensor<3x17xi8>) {', 1)
    text = text.replace('func.return %mm:tensor<3x5xi32>',
        '%copy = tensor.cast %a : tensor<3x17xi8> to tensor<3x17xi8>\n'
        'func.return %mm,%copy:tensor<3x5xi32>,tensor<3x17xi8>')
    api = module.provider()
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    original = catalog.build

    def altered(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        if tampered:
            for binding, match in zip(manifest['bindings'], matches):
                if match.kernel.stream.bitwise:
                    binding['operands']['x0'] = {'func_arg': 1}
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', altered)
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    result = api.core_aten_offload(prepared, tmp_path / 'work', package=module.PACKAGE,
        numeric_policy=module.POLICY, readout_options=module.options())
    manifest = json.loads(result[3].read_text())
    if tampered:
        assert not compiled and result[:3] == (None, None, None)
        assert manifest['status'] == 'declined' and 'binding' in manifest['reason']
    else:
        assert compiled == [True] and manifest['status'] == 'compiled'
        kernels = manifest['numeric_schedule_proof']['kernels']
        assert len(kernels) == len(manifest['kernels']) == 2
        assert {k['device_readout'] for k in kernels} == {'full_i32', 'scratchpad_bytes'}
        assert result[0].read_text().count('func.call @ca_') == 2
