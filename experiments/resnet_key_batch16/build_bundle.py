"""Explicit experiment selection using the ordinary source-bound catalog."""
from pathlib import Path
import json,hashlib
from mlir_oot.rectifier_residual_catalog import build
from mlir_oot.golden_rectified_resadd import Capabilities
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/key_batch16_normal';old=Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_normal_binding_v1/key_bundle/joint.json');r=json.loads(old.read_text());assert len(r['routes'])==1;route=r['routes'][0]
for item in r['certificates']:
 assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
result=build(Path(r['original_manifest_path']).parent,[Path(x['path'])for x in r['certificates']],Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin'),W/'key_bundle',capabilities=Capabilities(**route['capabilities']),coalesce_internal_spad=route['coalesce_internal_spad'],ordering_source=route['ordering_source'],panel_batch=16,identity_i32_acc_dma_accumulate=route['identity_i32_acc_dma_accumulate'])
print('ORDINARY_KEY_BUNDLE_DONE',flush=True)
