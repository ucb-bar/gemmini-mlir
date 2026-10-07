"""Close current champion stock boundary timings and geometry-only reference."""
from collections import defaultdict
from pathlib import Path
import hashlib, importlib.util, json, re, sqlite3
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'current2101_stock_profile'
REPO = Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

a_path = WORK / 'root_admission.json'
a = json.loads(a_path.read_text())
packet = Path(a['source_packet'])
assert sha(packet) == a['source_packet_sha256']
p = json.loads(packet.read_text())
assert isinstance(p['pins'], list) and len(p['pins']) == a['source_pins_revalidated'] == 358
for row in p['pins']:
    path = Path(row['path'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
job = json.loads((WORK / 'job.json').read_text())
jid = job['job_id']
assert jid == 2104
jobdir = Path(f'/scratch/firesim_queue/jobs/{jid}')
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
assert (terminal['state'], terminal['phase'], terminal['exit_code']) == ('DONE', 'DONE', 0)
paths = [jobdir / 'runworkload-full.json', WORK / 'preflight/firesim_preflight.json',
         WORK / 'actual_staged_identity.json', WORK / 'queue_submission.json']
definition, flight, staged, submission = [json.loads(x.read_text()) for x in paths]
assert definition['job_id'] == staged['job_id'] == jid
assert definition['user'] == 'agustin' and definition['hw_config'] == flight['hardware_key'] == ALIAS
assert definition['stage_from'] == flight['elf'] == p['profile_elf'] == job['elf']
assert sha(job['elf']) == job['elf_sha256'] == p['profile_elf_sha256'] == flight['elf_sha256']
assert definition['hwdb_config_artifact_sha256'] == flight['hwdb_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
assert staged['phase'] == 'RUNNING' and staged['observed_before_teardown']
assert staged['objects']['elf']['sha256'] == job['elf_sha256']
assert staged['objects']['bitstream']['sha256'] == BIT
assert submission['root_admission_sha256'] == sha(a_path)
audit = audit_elf(Path(job['elf']).read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
for key in ['profile_parser', 'profile_manifest']:
    assert sha(a[key]['path']) == a[key]['sha256']
manifest = json.loads(Path(a['profile_manifest']['path']).read_text())
spec = importlib.util.spec_from_file_location('mlir_oot.qualified_current_profile', a['profile_parser']['path'])
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
raw = (jobdir / 'simulation/sim_slot_0/uartlog').read_bytes()
text = raw.decode().replace('\r', '')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert text.splitlines().count('DONE') == 1
assert text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', text, re.M) == [('1000', '4000', a['original_gate']['raw_output_sha256'])]
metric, = re.findall(r'^METRIC cycles (\d+)$', text, re.M)
profile = parser.parse_profile(text, manifest)
assert [row[1] for row in profile['events']] == list(range(71))
categories = defaultdict(lambda: dict(calls=0, callback_cycles=0, preceding_gap_cycles=0))
events = []
boundaries = {row['id']: row for row in manifest['boundaries']}
for ordinal, idx, gap, callback in profile['events']:
    bound = boundaries[idx]
    category = categories[bound['category']]
    category['calls'] += 1
    category['callback_cycles'] += callback
    category['preceding_gap_cycles'] += gap
    events.append(dict(id=idx, ordinal=ordinal, symbol=bound['symbol'], category=bound['category'],
                       source_symbol=bound.get('source_symbol'), shape=bound.get('shape'),
                       preceding_gap_cycles=gap, callback_cycles=callback))

# Prior source/geometry joins authenticate reference association only. No source
# numeric or timer equivalence is established by these geometry comparisons.
old_path = WORK.parent / 'current2071_profile/stock2074_terminal.json'
alignment_path = REPO / 'docs/perf_records/q1013_exact1988_stock1999_profile_alignment.json'
reference_path = REPO / 'docs/perf_records/q1013_diagnostic_reference_firesim.json'
old = json.loads(old_path.read_text())
alignment = json.loads(alignment_path.read_text())
reference = json.loads(reference_path.read_text())
reference_cycles = {row['index']: row['cycles'] for row in reference['layer_cycles']}
current = {row['symbol']: row for row in events}
assert len(current) == 71
paired = []
comparison = defaultdict(lambda: dict(reference_cycles=0, current_callback_cycles=0))
for association, geometry_join in zip(old['paired_layers'], alignment['paired_layers'], strict=True):
    assert association['reference_index'] == geometry_join['reference_index']
    assert association['reference_cycles'] == reference_cycles[association['reference_index']]
    event = current[association['current_symbol']]
    category = association['category']
    if category in ('pointwise', 'direct'):
        keys = ('m', 'n', 'k') if category == 'pointwise' else ('h', 'w', 'cin', 'cout', 'stride')
        assert all(event['shape'][key] == geometry_join['current1988_shape'][key] for key in keys)
    pair = dict(reference_index=association['reference_index'], reference_layer=association['reference_layer'],
                current_symbol=event['symbol'], current_boundary_id=event['id'], category=category,
                geometry=association['geometry'], reference_cycles=association['reference_cycles'],
                current_callback_cycles=event['callback_cycles'],
                difference_cycles=event['callback_cycles'] - association['reference_cycles'],
                source_numeric_equivalence=False, timer_boundary_equivalence=False)
    paired.append(pair)
    comparison[category]['reference_cycles'] += pair['reference_cycles']
    comparison[category]['current_callback_cycles'] += pair['current_callback_cycles']
comparison['residual'] = dict(reference_cycles=2192393, current_callback_cycles=categories['residual_adapter']['callback_cycles'])
for row in comparison.values():
    row['difference_cycles'] = row['current_callback_cycles'] - row['reference_cycles']
archive = WORK / 'stock2104_uart.txt'
archive.write_bytes(raw)
pins = {str(path): sha(path) for path in paths + [archive, packet, a_path, WORK / 'job.json',
            Path(__file__), WORK.parent / 'submit_current2101_stock_profile.py',
            old_path, alignment_path, reference_path, Path(a['profile_manifest']['path'])]}
record = dict(schema='root_current2101_stock2104_boundary_diagnostic_v1', status='complete_diagnostic_only',
              job_id=jid, terminal=terminal, source_packet_sha256=sha(packet), source_pins_revalidated=358,
              actual_staged_identity=staged, final_nofsm=audit,
              baseline_job=2101, baseline_cycles=28728702,
              profile_metric_cycles=int(metric), profile_forward_cycles=profile['forward_counter'],
              callback_cycles=profile['device_counter'], outside_callback_cycles=profile['host_gap_counter'],
              tail_cycles=profile['tail_counter'],
              profile_metric_minus_baseline=int(metric) - 28728702,
              profile_forward_minus_baseline=profile['forward_counter'] - 28728702,
              instrumentation_delta_scope='Single instrumented versus prior uninstrumented observation; includes layout/timer perturbations and run variability, not a separately isolated timer calibration.',
              original_gate=a['original_gate'], category_measurements=dict(categories), events=events,
              reference_geometry_comparison=dict(comparison), paired_layers=paired,
              reference_job=1876, reference_kernel_cycles=reference['kernel_cycles'],
              hardware_alias=ALIAS, hwdb_sha256=HWDB, bitstream_archive_sha256=TAR,
              actual_staged_bit_sha256=BIT, elapsed_queue_seconds=terminal['ended_at'] - terminal['started_at'],
              scope='All71 current source-bound boundaries, no new champion. Callbacks include CPU adapters/readout/issue/DMA/waits; outside includes CPU/profiler. Pure accelerator/host utilization UNKNOWN. Permitted ZIP reference compares geometry with differing numerical computation and timer boundaries; differences locate work, not causal attainable savings.',
              token_usage_available=False, pins=pins)
save(WORK / 'stock2104_terminal.json', record)
print(json.dumps({key: record[key] for key in ['status', 'profile_forward_cycles', 'callback_cycles', 'outside_callback_cycles', 'profile_forward_minus_baseline', 'category_measurements', 'reference_geometry_comparison']}, indent=2))
