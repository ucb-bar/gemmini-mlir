"""Final-link diagnostic of accepted2076; semantic objects are never rebuilt."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np

from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_device_profile import emit, parse_profile
from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'current2076_profile_v2'
BASE = Path('/scratch/agustin/tmp/gemmini-tiny-finite-domain-20261007/out/artifacts/probes/finite-scale-normal-2070-v2-20261007')
CATALOG_ROOT = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
RUNTIME = Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    (WORK / name).write_text(json.dumps(value, indent=2)+'\n')

def run(argv, label, timeout=1200):
    argv = list(map(str, argv))
    save(label+'.argv.json', argv)
    with (WORK / (label+'.log')).open('w') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    assert result.returncode == 0, label

WORK.mkdir(exist_ok=False)
build_path = BASE / 'whole/build.json'
build = json.loads(build_path.read_text())
argv = build['candidate_link_argv']
elf = Path(argv[-1])
assert sha(elf) == build['candidate_elf_sha256'] == '3c8a3c01f56b7a99e67e4b5da49a8211b20283a3d71b5dba83a499a38afe1a23'
objects = {path:digest for path, digest in build['candidate_objects'].items()}
for path, digest in objects.items():
    assert sha(path) == digest, path
control = ROOT / 'current2076_profile/reproduced_control.elf'
old_reproduce = json.loads((control.parent/'reproduce2076.argv.json').read_text())
assert old_reproduce == [*argv[:-1], str(control)]
assert sha(control) == sha(elf), 'Original link failed byte-exact reproduction'
catalog_path = CATALOG_ROOT / 'device_catalog/device_catalog.json'
catalog = json.loads(catalog_path.read_text())
source = Path(catalog['source_snapshot'])
assert sha(source) == catalog['source_sha256']
operations = list(parse_module(source.read_text()).walk())
symbols = [kernel['symbol'] for kernel in catalog['kernels']]
bindings = catalog['bindings']
assert len(bindings) == 155
assert [r['source_operation_ordinal'] for r in bindings] == sorted(r['source_operation_ordinal'] for r in bindings)
for route in bindings:
    operation = operations[route['source_operation_ordinal']]
    assert operation.name == 'linalg.generic'
    assert [str(value.type) for value in operation.operands] == route['tensor_types']
    assert operation.attributes['prov.region_id'].data == route['region']
calls = [r['symbol'] for r in bindings]
assert set(calls) == set(symbols) and len(symbols) == 5
shim = CATALOG_ROOT / 'device/device_catalog_shim.c'
shim_object = shim.with_suffix('.o')
assert str(shim_object) in objects
for symbol in symbols:
    assert shim.read_text().count(f'extern void {symbol}(void *lhs, void *rhs, void *out);') == 1
nm = subprocess.check_output([LLVM/'llvm-nm', '--undefined-only', shim_object], text=True)
assert {line.split()[-1] for line in nm.splitlines()} == set(symbols)
host_llvm = BASE / 'selected/target/host_llvm/expanded.ll'
binding_path = BASE / 'selected/target/host_llvm/source_binding.json'
binding = json.loads(binding_path.read_text())
assert binding['all_original_device155_references_conserved']
boundaries = [dict(id=i, symbol=k['symbol'], pointer_arity=3, dimensions=k['dimensions'], schedule=k['schedule'], category='primitive_contraction_call') for i, k in enumerate(catalog['kernels'])]
ids = {r['symbol']:r['id'] for r in boundaries}
counts = Counter(calls)
manifest = dict(schema='current2076_primitive_boundary_profile_v1', diagnostic_only=True, accepted_stock_job=2076, baseline_unprofiled_cycles=380396343, accepted_elf=str(elf), accepted_elf_sha256=sha(elf), baseline_reproduced_sha256=sha(control), boundaries=boundaries, expected_device_calls=155, expected_symbol_calls={str(ids[s]):n for s,n in counts.items()}, expected_execution_id_order=[ids[s] for s in calls], source_bound_calls=[dict(ordinal=i, id=ids[r['symbol']], **r) for i,r in enumerate(bindings)], catalog_path=str(catalog_path), catalog_sha256=sha(catalog_path), source_path=str(source), source_sha256=sha(source), host_llvm_path=str(host_llvm), host_llvm_sha256=sha(host_llvm), source_binding_path=str(binding_path), source_binding_sha256=sha(binding_path), shim_source_path=str(shim), shim_source_sha256=sha(shim), semantic_object_sha256=objects, scope='Actual155 primitive kernel calls inside unmodified accepted2076. Callback includes CPU command issue,array service,DMA and fences. Gaps include physical descriptor dispatch,host observers,attention,allocation and profiler overhead. Neither interval is pure utilization; no historic callback subtraction or new performance champion.')
save('profile_manifest.json', manifest)
c = WORK / 'profile.c'
c.write_text(emit([(s,3) for s in symbols]))
obj = WORK / 'profile.o'
run([*argv[:7], '-I', RUNTIME/'baremetal/spike', '-I', RUNTIME/'c', '-c', c, '-o', obj], 'compile_profile')
end = argv.index('-lm')
profile_elf = WORK / 'model.elf'
link = [*argv[:end], str(obj), '-Wl,--wrap=merlin_run_multi', '-Wl,--wrap=htif_exit', *['-Wl,--wrap='+s for s in symbols], *argv[end:-1], str(profile_elf)]
run(link, 'link_profile')
for path,digest in objects.items():
    assert sha(path) == digest, 'Semantic object drifted'
audit = audit_elf(profile_elf.read_bytes())
assert audit['status'] == 'pass'
save('nofsm_audit.json', audit)
run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike', '-g', '--extension=gemmini', '--isa=rv64gc', '-m0x80000000:0x400000000', profile_elf], 'spike')
text = (WORK/'spike.log').read_text().replace('\r','')
adapter_path = BASE / 'whole/target_validation_m16g.json'
adapter = json.loads(adapter_path.read_text())
reference,golden = Path(adapter['reference_path']),Path(adapter['torch_golden_path'])
assert sha(reference) == adapter['reference_sha256'] and sha(golden) == adapter['torch_golden_sha256']
r,g = np.load(reference,allow_pickle=False),np.load(golden,allow_pickle=False)
assert r.shape == g.shape and r.size == 256000 and np.allclose(r,g,atol=.03125,rtol=.02,equal_nan=False)
raw = hashlib.sha256(r.astype('<f4',copy=False).tobytes()).hexdigest()
assert raw == adapter['spike_output_sha256']
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',text,re.M) == [('256000','1024000',raw)]
assert text.splitlines().count('DONE') == 1 and text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
profile = parse_profile(text,manifest)
assert [row[1] for row in profile['events']] == manifest['expected_execution_id_order']
save('spike_profile.json',profile)
qualification = dict(schema='current2076_profile_original_output_qualification_v1',status='pass',diagnostic_only=True,semantic_objects_unchanged=True,baseline_link_byte_exact=True,source_bound_155_calls=True,profile_conservation=True,original_output_gate=dict(count=256000,raw_sha256=raw,source_bitexact=True,torch_atol=.03125,torch_rtol=.02,torch_allclose=True),elf_path=str(profile_elf),elf_sha256=sha(profile_elf),nofsm_audit=audit,baseline_unprofiled_cycles=380396343,spike_is_functional_retired_instructions=True,firesim_cycles=None,scope=manifest['scope'],pins={str(path):sha(path) for path in [build_path,catalog_path,source,shim,host_llvm,binding_path,adapter_path,reference,golden,WORK/'spike.log',WORK/'profile_manifest.json',Path(__file__)]})
save('qualification.json',qualification)
print(json.dumps(dict(status='pass',elf_sha256=qualification['elf_sha256'],bound_calls=155,original_output_bitexact=True,zero_fsm=True)),flush=True)
