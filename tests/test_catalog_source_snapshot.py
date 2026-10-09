import hashlib

from mlir_oot import golden_device_catalog as catalog


def test_catalog_retains_exact_consumed_bytes_when_prepared_source_is_overwritten(tmp_path, monkeypatch):
    source = tmp_path/'prepared.mlir'
    original = b'builtin.module {\r\n}\r\n'
    source.write_bytes(original)
    work = tmp_path/'catalog'
    seen = []

    def build(text, **kwargs):
        seen.append(text.encode('utf-8'))
        return object(), dict(covered_contractions=1, source_sha256=hashlib.sha256(text.encode()).hexdigest())

    def compile(module, llvm_bin, output):
        output.mkdir()
        # Mirrors the shared prepared-path overwrite at the offload boundary.
        source.write_text('builtin.module {}')
        return {'object_sha256': 'mock compiler receipt'}

    monkeypatch.setattr(catalog, 'build_catalog', build)
    monkeypatch.setattr(catalog, 'compile_module', compile)
    result = catalog.compile_catalog(source, tmp_path/'llvm', work)
    assert seen == [original]
    assert (work/'catalog_source.mlir').read_bytes() == original
    assert result['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert result['source_snapshot'] == str((work/'catalog_source.mlir').resolve())
