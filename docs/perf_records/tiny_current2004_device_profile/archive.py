"""Close immutable current-profile and rejected stationary experiment artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
O = P.parents[2]

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def check(pins):
    for p, digest in pins.items():
        assert sha(p) == digest, p

def paths_in(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from paths_in(k)
            yield from paths_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from paths_in(v)
    elif isinstance(value, str) and value.startswith('/'):
        p = Path(value)
        if p.is_file():
            yield p

def close(case, label, record, extra=()):
    files = {p for p in case.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    files.update(map(Path, extra))
    files.add(Path(__file__))
    for p in tuple(files):
        if p.suffix == '.json':
            files.update(paths_in(read(p)))
    record.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                  token_usage_available=False, pins={str(p): sha(p) for p in sorted(files)})
    check(record['pins'])
    dest = O / 'docs/perf_records' / label
    dest.mkdir(exist_ok=False)
    # Preserve exact observation/recipe text in Git, keep binary and large IR externally pinned.
    for p in sorted(case.rglob('*')):
        if p.is_file() and p.suffix in {'.json', '.py', '.c', '.S', '.log', '.stdout', '.txt', '.d'} and p.stat().st_size < 400000:
            target = dest / p.relative_to(case)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    shutil.copyfile(__file__, dest / 'archive.py')
    receipt = O / 'docs/perf_records' / (label + '_qualification.json')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(label, len(files), receipt, flush=True)

stationary = P / 'tiny-transposed-stationary-current-20261006'
pair, qualified, independent = [read(stationary / p) for p in ['pair_result.json', 'qualification.json', 'independent/qualification.json']]
assert pair['status'] == qualified['status'] == independent['status'] == 'pass'
assert pair['warm_change_percent'] > 0 and qualified['all45056_original_i32_outputs_exact']
assert qualified['same_executable_bytes_and_all_symbol_addresses']
for arm in ['dense', 'transposed']:
    assert sha(stationary / arm / 'build/layer.elf') == pair['arms'][arm]['elf_sha256']
check(read(stationary / 'wave_census.json')['pins'])
close(stationary, 'tiny_transposed_stationary_complete_negative', {
    'schema': 'compiler_optimization_journey_v1',
    'hypothesis': 'Algebraically transpose an exact i8/i8/i32 contraction to make the small activation operand stationary, with reversible offline readonly parameter permutation and complete runtime input/output copies.',
    'ownership': 'Generic tensor/parameter permutations, exact integer contraction semantics and copy/ownership infrastructure Merlin; primitive layout/resource/schedule/ISA implementation OOT.',
    'emitted_change': 'Original M8,N5632,K2048 becomes primitive M5632,N8,K2048 with cached small B and ordinary CPU primitive loops. Runtime generic upstream copies form A-transpose and restore C. Both arms contain the same executable functions at the same addresses; offline parameter bytes and the selected arm differ.',
    'before': pair['arms']['dense'], 'after': pair['arms']['transposed'],
    'warm_change_percent': pair['warm_change_percent'],
    'cost_scope': pair['ROI'],
    'gate': qualified,
    'independent_cases': independent,
    'source_command_census': read(stationary / 'command_census.json'),
    'execute_geometry_census': read(stationary / 'wave_census.json'),
    'result': 'REJECTED: +45.3076608615% complete matched GSIM warm cycles; no production routing, whole model build or stock admission.',
    'dynamic_BD_abstraction': 'Planned separately, unimplemented here. This negative remains immutable and does not establish profitability of a counted dynamic-BD alternative.',
    'actual_hardware_cycles': None, 'whole_cycle_projection': None,
}, [O/'docs/perf_records/tiny_packed_rhs_current_completed_negative.json'])

profile = P / 'tiny-current2004-profile-20261006'
strict = read(profile / 'profile/spike_validation.json')
overlay = read(profile / 'overlay_receipt.json')
build = read(profile / 'profile/profile_build.json')
prior = read(O / 'docs/perf_records/tiny_rectangular_contraction_whole_qualification.json')
assert strict['status'] == 'pass' and strict['spike_full_output_match']
assert strict['outputs'] == 256000 and strict['torch_allclose']
assert strict['complete155_exact_original_callorder_counts_conservation']
assert strict['nofsm_audit']['status'] == 'pass'
assert len(strict['boundary_profile']['events']) == build['expected_device_calls'] == 155
assert sha(profile/'profile/model.elf') == strict['elf_sha256'] == build['elf_sha256']
check(overlay['pins']); check(build['objects']); check(prior['pins'])
close(profile, 'tiny_current2004_device_profile', {
    'schema': 'current_whole_device_boundary_profile_qualification_v1',
    'source_stock_job': 2004, 'source_stock_cycles': 422018733,
    'source_elf_sha256': overlay['source_elf_sha256'],
    'profile_elf_sha256': strict['elf_sha256'],
    'hypothesis': 'Measure the current whole-model inside155 device-boundary spans and outside gaps using integer-only final-link wrappers around the immutable current arithmetic/runtime/device objects.',
    'ownership': 'Generic whole boundary instrumentation, count/order/conservation methodology Merlin; target compiler/ABI/audit and actual stock execution OOT.',
    'emitted_change': 'Only profiler wrapper object and link --wrap options added. Current2004 model.o and every runtime/device/startup/main/weights object are identical. Original input/source/ABI bindings and155 calls retained.',
    'gate': strict,
    'native_gate_scope': 'Accepted current2004 native all256000 original bits and originalTorch gate retained via identical model arithmetic and complete ABI/runtime object identity. No claim of a separately compiled native timestamp wrapper. Fresh target whole digest,155 order/count/conservation and final noFSM passed.',
    'physical_added_bytes': overlay['physical_added_elf_bytes'],
    'overlay_object_added_bytes': overlay['overlay_added_object_bytes'],
    'measurement_scope': 'Inside-boundary spans include CPU address/control/config/command issue/fences/waits and accelerator/DMA. Outside includes host work and profiler bookkeeping. No physical accelerator-only percentage or overlap decomposition.',
    'retired_instructions_not_hardware_cycles': strict['functional_instructions_not_hardware_cycles'],
    'hardware_forward_cycles': None, 'hardware_device_boundary_cycles': None,
    'hardware_outside_cycles': None,
    'marker': 'Inherited37bdf9be0856 nonunique; exactELF SHA and complete object/source identity authoritative.',
    'result': 'QUALIFIED instrumentation arm; root review/release required before sole queue owner admission. Current2004 actual split remains UNKNOWN until stock run closes.',
}, list(prior['pins']) + list(overlay['pins']) + list(build['objects']))
