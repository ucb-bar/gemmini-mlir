"""Limb admission re-derives constants and schedules from the integer source."""

import json

import pytest

from conftest import PACKAGE, POLICY, provider
from test_limbs import annotate, linear_source


@pytest.mark.parametrize('reduction', [False, True])
@pytest.mark.parametrize('corruption', ['all_constants', 'one_constant', 'load_stride'])
def test_corrupted_limb_stream_is_declined_before_compilation(monkeypatch, tmp_path, reduction, corruption):
    api = provider()
    catalog = api._core_aten_module(PACKAGE, 'catalog')
    original = catalog.match_all
    text = annotate(linear_source(reduction=reduction), indices=(1,), exponents=(-8,))
    assert api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)['status'] == 'proven'

    def altered(*args, **kwargs):
        matches, declines = original(*args, **kwargs)
        for match in matches:
            if 'limb_plan' not in match.numeric:
                continue
            stream = match.kernel.stream
            if corruption == 'all_constants':
                for data in stream.consts.values():
                    data[...] = 0
            elif corruption == 'one_constant':
                next(iter(stream.consts.values())).flat[0] += 1
            else:
                stream.ins = [i._replace(attrs=tuple(sorted({**dict(i.attrs), 'stride': 0}.items())))
                              if i.kind == 'config_ld' else i for i in stream.ins]
        return matches, declines

    monkeypatch.setattr(catalog, 'match_all', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'declined', proof
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    source = tmp_path / 'model.mlir'
    source.write_text(text)
    result = api.core_aten_offload(source, tmp_path / 'work', package=PACKAGE, numeric_policy=POLICY)
    assert result[:3] == (None, None, None) and not compiled
    assert json.loads(result[3].read_text())['status'] == 'declined'
