"""Close exact checker qualification and retain both diagnosed measurement errors."""
from pathlib import Path
import hashlib
import json
import shutil
from mlir_oot.no_fsm_audit import audit_elf

root = Path.cwd()
work = root / 'out/artifacts/probes/benchmark-buffer-20261006'
core = Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
records = root / 'docs/perf_records'
snapshot = records / 'source_snapshots/benchmark_buffer_exact'
snapshot.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins = {}
def add(path, expected=None):
    path = Path(path).resolve(); digest = sha(path)
    if expected is not None: assert digest == expected, str(path)
    key = str(path)
    if key in pins: assert pins[key] == digest, key
    pins[key] = digest

audits = {}
for arm in (work, work/'machine_counter', work/'aligned_matched'):
    build = json.loads((arm/'build.json').read_text())
    for path, digest in build['pins'].items(): add(path, digest)
    receipt = json.loads((arm/'spike_receipt.json').read_text())
    add(arm/'check.elf', receipt['elf_sha256'])
    add(arm/'spike.log', receipt['log_sha256'])
    audits[str(arm.relative_to(work))] = audit_elf((arm/'check.elf').read_bytes())
    assert audits[str(arm.relative_to(work))]['status'] == 'pass'
    if arm == work:
        assert receipt['status'] == 'fail'
    else:
        assert receipt['status'] == 'pass' and receipt['returncode'] == 0

selected = work/'aligned_matched'
assert sha(core/'merlin/runtime/c/benchmark_buffer.h') == sha(selected/'benchmark_buffer.h')
assert sha(core/'merlin/tests/runtime/test_benchmark_buffer.py') == sha(selected/'native_fixture.py')
text = (selected/'spike.log').read_text()
assert 'BENCHMARK_BUFFER EXACT PASS' in text
counts = {}
for line in text.splitlines():
    if line.startswith('BENCH_'):
        kind, offset, value = line.split()
        counts.setdefault(offset, {})[kind] = int(value.removeprefix('instructions'))
assert counts == {'offset0': {'BENCH_BYTE':917684,'BENCH_WORD':131425},
                  'offset1': {'BENCH_BYTE':917572,'BENCH_WORD':917681}}
for path in work.rglob('*'):
    if path.is_file() and path.suffix in ('.py','.c','.h','.json','.log'):
        add(path)
        target = snapshot/path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target); add(target)
add(core/'merlin/runtime/c/benchmark_buffer.h')
add(core/'merlin/tests/runtime/test_benchmark_buffer.py')
(records/'benchmark_buffer_exact_qualification.json').write_text(json.dumps({
    'schema': 'portable_exact_benchmark_buffer_qualification_v1',
    'ownership': 'Merlin exact comparison/fill; OOT actual target timer/compiler/ELF qualification.',
    'scope': 'Validation and poisoning outside model ROI. No kernel/model performance claim.',
    'property_gate': {'native': 'PASS', 'actual_RV64GC': 'PASS',
        'input_offsets':16,'output_offsets':16,'lengths':'0..257',
        'mismatch_positions':'every position, reverse and two-mismatch ordering',
        'fill_values':256,'dirty_guards_and_source_immutability':True,
        'checks_every_byte':True,'checksum_substitution':False},
    'counts': counts, 'metric': 'Functional Spike mcycle instruction proxy; not hardware cycles.',
    'read_checks_per_arm':16,
    'aligned_reduction_percent':100*(917684-131425)/917684,
    'unaligned_policy':'Exact byte fallback, no assumed target alignment.',
    'initial_counter_failure':'User instret CSR unavailable; aligned/unaligned minimal check PASS before fault. Original ELF/log retained.',
    'rejected_preliminary_cost':'15 byte checks hoisted away; outlined word helper lost alignment. Functional properties passed, comparative cost invalid.',
    'corrected_cost':'OOT compiler memory barrier prevents each pure check from being hoisted; alignment established inside aligned helper.',
    'audits':audits, 'pins':pins,
    'hardware_gain':None, 'token_usage_available':False,
},indent=2)+'\n')
print(json.dumps({'status':'closed','pins':len(pins),'target_properties':'pass'}))
