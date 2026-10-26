"""Every quantized movement element must select its source storage code."""

import json
from fractions import Fraction

import pytest

from conftest import PACKAGE, POLICY, provider
from test_quantized_movement import source


@pytest.mark.parametrize('output', [None, (Fraction(1,16), -128, 127, 7, 'int8'),
                                    (Fraction(1,16), 0, 255, 128, 'uint8')])
@pytest.mark.parametrize('corruption', ['none', 'load_stride', 'store_stride', 'load_offset', 'readout'])
def test_movement_selector_admission(monkeypatch, tmp_path, output, corruption):
    api = provider()
    catalog = api._core_aten_module(PACKAGE, 'catalog')
    original = catalog.match_all
    text = source(output=output)

    def altered(*args, **kwargs):
        matches, declines = original(*args, **kwargs)
        for match in matches:
            stream = match.kernel.stream
            if corruption in ('load_stride', 'store_stride'):
                kind = 'config_ld' if corruption == 'load_stride' else 'config_st'
                stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs), 'stride': 0}.items())))
                              if i.kind == kind else i for i in stream.ins]
            elif corruption == 'load_offset':
                stream.ins = [i._replace(off=i.off + 1) if i.kind == 'mvin' else i for i in stream.ins]
            elif corruption == 'readout':
                if stream.readouts:
                    stream.readouts[0].constant += 10
                else:
                    from mlir_oot.core_aten.requant import Readout
                    stream.readouts.append(Readout('y', [('y', 1)], constant=10))
        return matches, declines

    monkeypatch.setattr(catalog, 'match_all', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    if corruption == 'none':
        assert proof['status'] == 'proven', proof
        assert proof['kernels'][0]['movement']['method'] == 'decoded_source_selectors'
        return
    assert proof['status'] == 'declined', proof
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    prepared = tmp_path / 'model.mlir'
    prepared.write_text(text)
    result = api.core_aten_offload(prepared, tmp_path / 'work', package=PACKAGE, numeric_policy=POLICY)
    assert result[:3] == (None, None, None) and not compiled
    assert json.loads(result[3].read_text())['status'] == 'declined'


@pytest.mark.parametrize('shape', [(1, 6, 3, 5), (2, 4, 1, 17)])
@pytest.mark.parametrize('kind', ['cat', 'transpose', 'permute'])
@pytest.mark.parametrize('scheme', ['same', 'requant'])
def test_independent_layout_shapes_and_tails_are_admitted(shape, kind, scheme):
    from test_layout_scalability import source as layout_source
    proof = provider().admit_numeric_schedule(layout_source(kind, 'i8', shape, scheme),
                                               package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('axis', [1, 3])
def test_distinct_input_contracts_and_unsigned_codes_are_admitted(axis):
    from test_layout_scalability import asymmetric_cat_source
    proof = provider().admit_numeric_schedule(asymmetric_cat_source(axis), package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('low,high,zp,storage', [(-127,127,-7,'int8'), (1,254,129,'uint8')])
@pytest.mark.parametrize('literal', [None, .015, -1.5])
def test_quantized_domains_zero_points_and_literals(low, high, zp, storage, literal):
    text = source(low=low, high=high, zp=zp, storage=storage)
    if literal is not None:
        text = text.replace('linalg.yield %a : f32',
                            f'%c = arith.constant {literal} : f32\n linalg.yield %c : f32')
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    # A raw copy cannot implement the quantizer's clamp outside the declared range.
    assert proof['status'] == ('declined' if literal is None else 'proven'), proof
    if literal is None:
        assert 'semantic gate' in proof['reason']


def test_movement_provenance_capacity_refuses_before_interpreting_large_maps():
    text = source().replace('tensor<256x', 'tensor<600000x')
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'declined' and 'capacity' in proof['reason'], proof


def test_decoded_differential_backstop_is_independent(monkeypatch):
    api = provider()
    module = api._core_aten_module(PACKAGE, 'movement_proof')
    original = module.execute

    def wrong_execution(stream, buffers, **kwargs):
        original(stream, buffers, **kwargs)
        out = next(i for i, arg in enumerate(stream.args) if arg.name == 'y')
        buffers[out].flat[0] ^= 1

    monkeypatch.setattr(module, 'execute', wrong_execution)
    proof = api.admit_numeric_schedule(source(), package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'declined' and 'differential' in proof['reason'], proof


@pytest.mark.parametrize('fixture', ['select_scatter', 'slice_scatter'])
@pytest.mark.parametrize('corrupt', [False, True])
def test_scatter_source_numbering_and_stride_proof(monkeypatch, fixture, corrupt):
    from fixture_source import read
    api = provider()
    catalog = api._core_aten_module(PACKAGE, 'catalog')
    original = catalog.match_all

    def matches(*args, **kwargs):
        found, declines = original(*args, **kwargs)
        if corrupt:
            for match in found:
                stream = match.kernel.stream
                stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs), 'stride': 0}.items())))
                              if i.kind == 'config_ld' else i for i in stream.ins]
        return found, declines

    monkeypatch.setattr(catalog, 'match_all', matches)
    text = read(PACKAGE / 'tests/fixtures/core_aten' / fixture / 'model.mlir')
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == ('declined' if corrupt else 'proven'), proof
    if not corrupt:
        assert proof['kernels'][0]['movement']['code_domain'] == 'every_legal_storage_code'
