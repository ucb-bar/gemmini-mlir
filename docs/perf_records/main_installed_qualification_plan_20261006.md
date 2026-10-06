# External installed qualification plan

Not executed. Wait for pipeline conflicts to close and root to commit the coherent integration. Qualification must use the final immutable commit, not the dirty checkout.

## Existing runner

From the private main checkout, resolve `git rev-parse HEAD`, then run:

```sh
PYTHONPATH=src python build_tools/scripts/qualify_installed.py --ref <FINAL_COMMIT> --suite codegen-declaration --label golden-core-installed-20261006 --timeout 600
PYTHONPATH=src python build_tools/scripts/qualify_installed.py --ref <FINAL_COMMIT> --suite reviewed-corpus --label golden-reviewed-installed-20261006 --timeout 600
```

Use the existing qualified Python environment for the runner. The first suite installs core without experiments; the second validates the newer main reviewed-corpus handoff and required release entrypoint. Distinct unused labels prevent overwriting evidence. The runner archives the selected commit, builds sdists with `uv build --sdist`, builds wheels FROM those sdists, checks distribution layout/source parity, installs in a fresh venv, runs payload/module/entrypoint probes outside checkout, copies only selected committed tests/support inputs, and freezes dependencies. It clears source-path leakage for child execution. Retain report, subprocess logs, archive and artifact hashes.

## Golden runtime payload extension

Use the produced core wheel in a second fresh external venv and cwd with PYTHONPATH unset. Assert imported merlin paths lie under that venv, not any checkout. Resolve every packaged runtime dependency using `merlin.common.paths.data_path` and compare bytes to final committed package_resources entries. Confirm new certificate headers/template plus current-main baremetal support resources all exist. Generate default and fully admitted source-attention helpers using installed API only, compile in the external directory against installed headers, and execute the existing small original-consumer fixtures. Keep the tests copied from the selected commit, without its conftest injecting checkout paths. Record generated source/header/compiler/.d/command/object/SO hashes and default behavior separately from optional policies. No external provider or optional research package should be needed merely to emit the portable helper.

A test passing from the checkout does not establish this installed gate. Missing headers, provider imports, test fixtures or tools must fail visibly; do not add symlinks back into source. Existing full1600 numerical and target receipts remain separate from packaging validation.

## Inherited gate fixes proposed (read-only)

- qualify_installed reviewed-corpus source_inputs is a named reference-fixture edge. A co-located target-ok explanation accurately scopes the existing explicit reference descriptor; no global ledger expansion.
- isa_census field offsets affect mask construction. Prefer explicit selected format/field-layout proof with validated widths and source provenance, threading through the parser. Do not hide assumptions by renaming variables or claiming unsupported derivation. A dedicated standard-format parser is an alternative only with an explicit supported ISA contract.
