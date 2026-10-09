"""Fresh scalar stand-in gate of unchanged 2101 host/source objects."""
import hashlib
import importlib.util
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent
BASE=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal')
OWN=WORK.parents[1]
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('whole_probe', OWN/'tests/fused_whole_model_probe.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build=BASE/'controlled2095/qualification/build_view'
capture=BASE/'normal/capture'
catalog=json.loads((build/'device_catalog/device_catalog.json').read_text())
inputs=[build/'host_llvm/model.native.ll', build/'device/device_catalog_shim.c', build/'device_catalog/device_catalog.json', capture/'golden.npy', *map(Path,catalog['native_oracle_sources'])]
before={str(p):sha(p) for p in inputs}
report=module.native_validate(capture, build, WORK/'native', LLVM, atol=0.,rtol=0.)
assert before=={str(p):sha(p) for p in inputs}
report.update(input_pins=before, frozen_source_unchanged=True, profile_wrapper_native_execution=False, native_scope='unchanged semantic source using same current selected scalar stand-ins; profiling wrapper target-only')
(WORK/'native_qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
