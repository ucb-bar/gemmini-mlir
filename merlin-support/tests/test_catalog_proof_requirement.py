"""Direct catalog compilation and public adapters cannot link unadmitted code."""

import json

import numpy as np
import pytest

import test_composition_readout as readout


@pytest.mark.parametrize('entry', ['direct', 'builder', 'cli', 'offload_cli'])
def test_no_entry_links_an_unproved_fused_readout(monkeypatch, tmp_path, entry):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    original = catalog.match_all

    def altered(*args, **kwargs):
        matches, declines = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs),
                'multiplier_bits': int(np.float32(.25).view(np.uint32)), 'acc_scale': .25}.items())))
                if i.kind == 'config_st' else i for i in stream.ins]
        return matches, declines

    monkeypatch.setattr(catalog, 'match_all', altered)
    monkeypatch.setattr(catalog, '_execution_provider', lambda: api)
    monkeypatch.setattr(api, '_selected_numeric_policy', lambda *args: readout.POLICY)
    monkeypatch.setattr(api, '_readout_options', lambda *args: readout.options())
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    source, work = tmp_path / 'model.mlir', tmp_path / 'work'
    source.write_text(readout.source()[0])
    if entry == 'direct':
        manifest = catalog.compile_catalog(source, work, readout_options=readout.options())
    elif entry == 'builder':
        path, obj = catalog.merlin_builder(tmp_path, numeric_policy=readout.POLICY,
                                           readout_options=readout.options())(source, work)
        assert obj is None
        manifest = json.loads(path.read_text())
    else:
        monkeypatch.setattr('sys.argv', ['catalog', str(source), '--workdir', str(work)])
        main = catalog.main if entry == 'cli' else api._core_aten_module(readout.PACKAGE, 'offload').main
        assert main() == 1
        manifest = json.loads((work / 'core_aten_catalog.json').read_text())
    assert manifest['status'] == 'declined' and not manifest['kernels'] and not compiled


def test_unchanged_nonlegacy_kernel_also_requires_a_proof(monkeypatch, tmp_path):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    source = tmp_path / 'model.mlir'
    source.write_text(readout.source()[0])
    manifest = catalog.compile_catalog(source, tmp_path / 'work', readout_options=readout.options())
    assert manifest['status'] == 'declined' and 'requires' in manifest['reason'] and not compiled


def test_offload_cli_admits_and_preserves_its_output_path(monkeypatch, tmp_path):
    api = readout.provider()
    catalog = api._core_aten_module(readout.PACKAGE, 'catalog')
    monkeypatch.setattr(catalog, '_execution_provider', lambda: api)
    monkeypatch.setattr(api, '_selected_numeric_policy', lambda *args: None)
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: {})
    source, work = tmp_path / 'model.mlir', tmp_path / 'work'
    source.write_text('''builtin.module { func.func @forward(%x: tensor<17xi8>) -> tensor<17xi8> {
      %y = tensor.cast %x : tensor<17xi8> to tensor<17xi8>
      func.return %y : tensor<17xi8>
    } }''')
    monkeypatch.setattr('sys.argv', ['offload', str(source), '--workdir', str(work)])
    assert api._core_aten_module(readout.PACKAGE, 'offload').main() == 0
    manifest = json.loads((work / 'core_aten_catalog.json').read_text())
    assert manifest['numeric_schedule_proof']['status'] == 'proven'
    assert 'func.call @ca_' in (work / 'offloaded.mlir').read_text()
    assert (work / 'core_aten_shim.c').is_file()
