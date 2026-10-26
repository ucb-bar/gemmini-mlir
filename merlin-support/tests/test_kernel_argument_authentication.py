"""The linked call ABI must equal each admitted source-bound physical ABI."""

import json

import pytest

from test_binding_authentication import CASES, admit


@pytest.mark.parametrize('case', ['pooled_operands', 'shared_input_weights', 'operand_sum_operands'])
@pytest.mark.parametrize('field', ['order', 'name', 'role', 'shape', 'dtype', 'missing'])
def test_manifest_argument_edits_decline_before_linking(monkeypatch, tmp_path, case, field):
    module, text, _ = CASES[case]()
    api = module.provider()
    options = module.options()
    proof = api.admit_numeric_schedule(text, package=module.PACKAGE, numeric_policy=module.POLICY,
                                       readout_options=options)
    assert proof['status'] == 'proven', proof

    def edit(manifest, _):
        args = manifest['kernels'][0]['args']
        if field == 'order':
            args[0], args[1] = args[1], args[0]
        elif field == 'missing':
            args.pop()
        else:
            args[0][field] = dict(name='other', role='scratch', shape=[1], dtype='i32')[field]

    altered = admit(module, text, edit, monkeypatch)
    assert altered['status'] == 'declined' and 'arguments' in altered['reason'], altered
    # Also authenticate on compilation with an otherwise genuine proof.
    catalog = api._core_aten_module(module.PACKAGE, 'catalog')
    compiled = []
    monkeypatch.setattr(catalog, 'compile_to_object', lambda *args, **kwargs: compiled.append(True) or {})
    source = tmp_path / 'model.mlir'
    source.write_text(text)
    manifest_path, obj = api.core_aten_catalog(source, tmp_path / 'work', package=module.PACKAGE,
        numeric_policy=module.POLICY, readout_options=options, numeric_schedule_proof=proof)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'declined' and obj is None and not compiled


@pytest.mark.parametrize('field', ['order', 'element_bytes'])
def test_byte_selector_abi_is_authenticated_without_numeric_policy(monkeypatch, field):
    from test_composition_readout import PACKAGE, provider
    api = provider()
    catalog = api._core_aten_module(PACKAGE, 'catalog')
    original = catalog.build
    text = '''builtin.module { func.func @forward(%x: tensor<17xi16>) -> tensor<17xi16> {
      %y = tensor.cast %x : tensor<17xi16> to tensor<17xi16>
      func.return %y : tensor<17xi16>
    } }'''
    assert api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=None)['status'] == 'proven'

    def altered(*args, **kwargs):
        device, manifest, matches = original(*args, **kwargs)
        arguments = manifest['kernels'][0]['args']
        if field == 'order':
            arguments.reverse()
        else:
            arguments[0]['element_bytes'] = 1
        return device, manifest, matches

    monkeypatch.setattr(catalog, 'build', altered)
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=None)
    assert proof['status'] == 'declined' and 'arguments' in proof['reason'], proof
