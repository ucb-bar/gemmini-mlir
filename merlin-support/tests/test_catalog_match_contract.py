"""Malformed catalog match items decline before provider attribute access."""

import hashlib
from types import MappingProxyType, SimpleNamespace

import pytest

import test_composition_readout as readout


CONTRACT = ('catalog.build()[2] items must expose .kernel.meta (mapping) '
            'and .kernel.stream with .ins')
ENTRIES = ['routing', 'admission', 'integer', 'linear', 'limb']


def invoke(api, entry, text, tmp_path):
    options = dict(package=readout.PACKAGE, numeric_policy=readout.POLICY)
    if entry == 'routing':
        (tmp_path / 'model.mlir').write_text(text)
        return api.routing(tmp_path, target='gemmini', package=readout.PACKAGE,
                           facts={}, eligible=[], readout_options=readout.options())
    if entry == 'admission':
        return api.admit_numeric_schedule(text, **options, readout_options=readout.options())
    if entry == 'integer':
        return api._admit_integer_regions(text, **options, readout_options=readout.options())
    if entry == 'linear':
        return api._admit_quantized_linear_regions(text, **options)
    return api._admit_limb_regions(text, **options)


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('matches', [
    ['symbol'],
    [{'kernel': {'meta': {}, 'stream': {'ins': []}}}],
    [SimpleNamespace(kernel=SimpleNamespace(meta=[], stream=SimpleNamespace(ins=[])))],
    [SimpleNamespace(kernel=SimpleNamespace(meta={}))],
    [SimpleNamespace(kernel=SimpleNamespace(meta={}, stream=SimpleNamespace()))],
    None, {}, '', iter(()),
], ids=['str-item', 'dict-item', 'nonmapping-meta', 'missing-stream', 'missing-ins',
        'none', 'mapping', 'string', 'iterator'])
def test_malformed_catalog_matches_are_refused(monkeypatch, tmp_path, caplog, entry, matches):
    api = readout.provider()
    text, _, _ = readout.source()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    monkeypatch.setattr(catalog, 'build', lambda *args, **kwargs: (None, {}, matches))

    result = invoke(api, entry, text, tmp_path)

    if entry == 'routing':
        assert result is None
        assert CONTRACT in caplog.text
    else:
        assert result['status'] == 'declined'
        assert CONTRACT in result['reason']
        assert result['source_sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert result['kernels'] == result['bindings'] == []


@pytest.mark.parametrize('entry', ['routing', 'admission'])
@pytest.mark.parametrize('sequence', [list, tuple], ids=['list', 'tuple'])
def test_valid_catalog_matches_remain_accepted(monkeypatch, tmp_path, entry, sequence):
    api = readout.provider()
    text, _, _ = readout.source()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    original = catalog.build

    def build(*args, **kwargs):
        module, manifest, matches = original(*args, **kwargs)
        assert matches
        for match in matches:
            match.kernel.meta = MappingProxyType(match.kernel.meta)
        return module, manifest, sequence(matches)

    monkeypatch.setattr(catalog, 'build', build)
    result = invoke(api, entry, text, tmp_path)
    if entry == 'routing':
        assert result.region_offload is not None
    else:
        assert result['status'] == 'proven', result
