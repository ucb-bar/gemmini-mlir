"""Close the actual staged stock2074 diagnostic and its original output gate."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from mlir_oot.golden_device_profile import parse_profile
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
JOB = Path('/scratch/firesim_queue/jobs/2074')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        terminal = dict(db.execute('select id,state,phase,exit_code,submitted_at,started_at,ended_at from jobs where id=?', (2074,)).fetchone())
    assert (terminal['state'], terminal['phase'], terminal['exit_code']) == ('DONE', 'DONE', 0)
    spec = json.loads((JOB / 'runworkload-full.json').read_text())
    preflight = json.loads((WORK / 'preflight/firesim_preflight.json').read_text())
    manifest = json.loads((WORK / 'profile_manifest.json').read_text())
    qualification = json.loads((WORK / 'qualification.json').read_text())
    staged = json.loads((WORK / 'actual_staged_identity.json').read_text())
    assert spec['job_id'] == staged['job_id'] == 2074
    assert spec['user'] == 'agustin' and spec['hw_config'] == preflight['hardware_key'] == 'alveo_u250_firesim_gemmini_rocket_stock'
    assert spec['stage_from'] == preflight['elf'] == str(WORK / 'model.elf')
    assert sha(WORK / 'model.elf') == preflight['elf_sha256'] == qualification['elf_sha256'] == staged['objects']['elf']['sha256']
    assert spec['hwdb_config_artifact_sha256'] == preflight['hwdb_artifact_sha256'] == sha(spec['hwdb_config_artifact']) == '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
    assert staged['objects']['bitstream']['sha256'] == '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
    assert sha(preflight['bitstream_path']) == preflight['bitstream_sha256'] == 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
    for obj in staged['objects'].values():
        p = Path(obj['path'])
        if p.is_file():
            assert sha(p) == obj['sha256']
    for path, digest in manifest['semantic_object_sha256'].items():
        assert sha(path) == digest
    assert sha(manifest['catalog_path']) == manifest['catalog_sha256']
    assert sha(manifest['source_path']) == manifest['source_sha256']
    assert sha(manifest['host_llvm_path']) == manifest['host_llvm_sha256']
    assert sha(WORK / 'spike.log') == qualification['spike_log_sha256']
    audit = audit_elf((WORK / 'model.elf').read_bytes())
    assert audit['status'] == 'pass'
    uart = JOB / 'simulation/sim_slot_0/uartlog'
    text = uart.read_text().replace('\r', '')
    assert text.splitlines().count('DONE') == 1
    assert text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
    assert '*** PASSED ***' in text
    assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', text, re.M) == [('1000', '4000', qualification['original_accuracy_gate']['raw_output_sha256'])]
    metric = re.findall(r'^METRIC cycles (\d+)$', text, re.M)
    assert len(metric) == 1
    profile = parse_profile(text, manifest)
    assert [x[1] for x in profile['events']] == list(range(71))
    boundaries = {row['id']: row for row in manifest['boundaries']}
    categories = defaultdict(lambda: dict(calls=0, callback_cycles=0, preceding_gap_cycles=0))
    events = []
    for ordinal, idx, gap, callback in profile['events']:
        bound = boundaries[idx]
        row = categories[bound['category']]
        row['calls'] += 1
        row['callback_cycles'] += callback
        row['preceding_gap_cycles'] += gap
        events.append(dict(bound, ordinal=ordinal, preceding_gap_cycles=gap, callback_cycles=callback))
    alignment_path = REPO / 'docs/perf_records/q1013_exact1988_stock1999_profile_alignment.json'
    reference_path = REPO / 'docs/perf_records/q1013_diagnostic_reference_firesim.json'
    alignment = json.loads(alignment_path.read_text())
    reference = json.loads(reference_path.read_text())
    ref_ticks = {r['index']: r['cycles'] for r in reference['layer_cycles']}
    region_events = {r['source_region']: r for r in events if r.get('source_region')}
    stem = next(r for r in events if r['category'] == 'stem_pool_adapter')
    classifier = next(r for r in events if r['category'] == 'classifier_primitive')
    paired = []
    aggregate_comparison = defaultdict(lambda: dict(reference_cycles=0, current_callback_cycles=0))
    for old in alignment['paired_layers']:
        assert ref_ticks[old['reference_index']] == old['reference1876_cycles']
        category = old['category']
        current = stem if category == 'pooled_stem' else classifier if category == 'classifier' else region_events[old['current_region']]
        if category in ('pointwise', 'direct'):
            keys = ('m', 'n', 'k') if category == 'pointwise' else ('h', 'w', 'cin', 'cout', 'stride')
            assert all(old['current1988_shape'][key] == current['schedule'][key] for key in keys)
        paired.append(dict(reference_index=old['reference_index'], reference_layer=old['reference_layer'], current_source_region=old['current_region'], current_symbol=current['symbol'], category=category, geometry=old['geometry'], reference_cycles=old['reference1876_cycles'], current_callback_cycles=current['callback_cycles'], difference_cycles=current['callback_cycles'] - old['reference1876_cycles'], source_numeric_equivalence=False, timer_boundary_equivalence=False))
        aggregate_comparison[category]['reference_cycles'] += old['reference1876_cycles']
        aggregate_comparison[category]['current_callback_cycles'] += current['callback_cycles']
    aggregate_comparison['residual'] = dict(reference_cycles=2192393, current_callback_cycles=categories['residual_adapter']['callback_cycles'])
    for row in aggregate_comparison.values():
        row['difference_cycles'] = row['current_callback_cycles'] - row['reference_cycles']
    archive = WORK / 'stock2074_uart.txt'
    archive.write_bytes(uart.read_bytes())
    record = dict(schema='current2071_stock2074_boundary_profile_terminal_v1', status='complete_diagnostic_only', job_id=2074, terminal=terminal, baseline_job=2071, baseline_cycles=29698347, profile_metric_cycles=int(metric[0]), profile_forward_cycles=profile['forward_counter'], callback_cycles=profile['device_counter'], outside_callback_cycles=profile['host_gap_counter'], tail_cycles=profile['tail_counter'], profile_metric_minus_baseline=int(metric[0]) - 29698347, profile_forward_minus_baseline=profile['forward_counter'] - 29698347, expected_source_bound_calls=71, original_output_gate=qualification['original_accuracy_gate'], nofsm_audit=audit, actual_staged_identity=staged, hardware_alias=spec['hw_config'], hwdb_artifact_sha256=preflight['hwdb_artifact_sha256'], bitstream_tar_sha256=preflight['bitstream_sha256'], elapsed_queue_seconds=terminal['ended_at'] - terminal['started_at'], scope='Callbacks include CPU adapters, readout, command issue, transfers and waits; outside includes CPU/profiler overhead. Pure accelerator/host utilization UNKNOWN. Profile is not an optimized champion. Reference compares permitted ZIP geometry, different numeric workload and timer boundaries; differences locate work, not causal attainable savings.', category_measurements=dict(categories), events=events, reference_job=1876, reference_cycles=reference['kernel_cycles'], reference_geometry_comparison=dict(aggregate_comparison), paired_layers=paired, token_usage_available=False, token_usage=None, pins={str(p):sha(p) for p in [archive, WORK / 'profile_manifest.json', WORK / 'qualification.json', WORK / 'actual_staged_identity.json', WORK / 'queue_submission.json', WORK / 'preflight/firesim_preflight.json', WORK / 'build_profile.py', Path(__file__), WORK / 'model.elf', alignment_path, reference_path]})
    (WORK / 'stock2074_terminal.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k:record[k] for k in ['status','profile_metric_cycles','profile_forward_cycles','callback_cycles','outside_callback_cycles','elapsed_queue_seconds','reference_geometry_comparison']}, indent=2))


if __name__ == '__main__':
    main()
