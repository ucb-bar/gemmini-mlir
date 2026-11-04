# Source identity transport gap

Status: external compiler invocation API gap, not a numerical failure. No compiler
or checker workaround is applied, and no unchanged check is requested.

## Reproduced boundary

Request `6_1793065488225486124_e9c59e7b23b9bb798c40bdc1da548102`
reports ACT_ACT L0/L1 passes and an L2 preflight tool failure:
`plan source_sha256 differs from exact source bytes`.

The compiler's `params.global_program_plan.source_sha256` equals the exact bytes
it receives. Those bytes are the public ABI enrichment's rendered MLIR, whereas
preflight inventories the original source. All three cases reproduce this:

| Public case | Original SHA256 | Enriched SHA256 = emitted plan SHA256 |
|---|---|---|
| SY_act_act_i8_aligned | `bb2c61090acfb0b08b7900b9063ea46d9764e8e82098b7e13710d729d7c35c0e` | `d2b859e9c3d090037758ed17ece57a8dc4b2081dcbe85ac8ad5383750fc7d4cb` |
| SY_act_act_i8_k_tail | `ba3ea1634fdb37f64f6431e88c7b95297c03317a4eb07337bdcf39c04526ba21` | `5ac57e4aee2c0490bb9999e9a3cfec70415cea166dd4c989e7c6941b27c3a3e5` |
| SY_act_act_i8_decode | `ad6bef52e805fa163cb912427881a91b190979715df763cad41141a8b82060b2` | `6cd86d763ffc8da39cce0154d1d13acccc21bb495ff1cbf858a44c0f38d13b73` |

These are evidence, not a compiler lookup table. No implementation may choose
source identity by case name, public roster, shapes, or a guessed original text.

## Public mechanisms checked

* `logical_contract.yaml` and the public `mlir_oot_backend_contract.yaml`,
  `argv_templates`, specify only `{tool}`, `{input_mlir}`, and `{output_json}`.
  `submission/manifest.yaml` uses exactly these placeholders. No original-source
  argument, receipt sidecar, stdin payload, or source-identity environment variable
  is declared. General tooling variables do not declare an original source receipt.
* `merlin.targetgen.contract.linalg_iface.bind_logical_io_metadata(text, inputs,
  outputs)` parses the original, adds `merlin.logical_io`, `merlin.logical_inputs`,
  `merlin.logical_outputs`, and `merlin.logical_input_roles`, then returns
  `(str(module), receipt)`. Its receipt includes original `source_sha256` and
  rendered `normalized_sha256`; the returned MLIR contains neither digest nor
  receipt. The binder verifies unchanged body/type/arity structurally.
* Rendering also changes whitespace, singleton return-type parentheses and index
  attribute syntax. Stripping the four added attributes cannot reconstruct the
  original exact bytes.
* `merlin.targetgen.oot_starterkit.plan.source_operation_inventory` hashes its
  exact UTF-8 argument. `_source_plan_problems` compares that hash directly to the
  plan, with no enrichment-receipt transport or identity-attribute interpretation.
  The public preflight contract expressly requires exact source bytes.

## Reproduction without evaluator internals

Run from the workspace with `CASE_DIR` pointing to the public aligned case:

```python
import hashlib, json, os, pathlib, sys, yaml
sys.path.insert(0, 'submission/compiler_library')
from merlin.targetgen.contract.linalg_iface import bind_logical_io_metadata
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
p = pathlib.Path(os.environ['CASE_DIR'])
raw = (p / 'capsule.interface.mlir').read_bytes()
decl = yaml.safe_load((p / 'capsule.yaml').read_text())
enriched, receipt = bind_logical_io_metadata(
    raw.decode('utf-8'), decl['inputs'],
    [{'name': 'Y0', 'shape': [32, 16], 'dtype': 'i32'}])
assert receipt['source_sha256'] == hashlib.sha256(raw).hexdigest()
assert receipt['normalized_sha256'] == source_operation_inventory(enriched)['source_sha256']
assert receipt['source_sha256'] != receipt['normalized_sha256']
print(json.dumps(receipt, indent=2))
```

The failed authored command buffer is under
`selfcheck_out/requests/6_1793065488225486124_e9c59e7b23b9bb798c40bdc1da548102/SY_act_act_i8_aligned/command_buffer.json`;
its plan hash matches this `normalized_sha256`.

## Smallest upstream correction

For the current enrichment path, transport the binder's existing receipt to the
compiler through a declared identity attribute or receipt-sidecar argument. The
owner must bind the receipt to both original and enriched exact bytes; the
compiler can then use its supplied original identity for the plan while inventory
and task indices still come from the structurally equivalent enriched source.
Preflight must retain this receipt binding rather than accept arbitrary asserted
hashes. Alternatively, pass unchanged original source to the compiler with logical
I/O metadata in a separate declared sidecar. Neither transport exists in the
current supplied API, so an isolated compiler adapter cannot implement it honestly.
