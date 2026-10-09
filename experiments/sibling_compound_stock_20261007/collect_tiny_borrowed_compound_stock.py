"""Close actual stock complete-cost labels against immutable original source."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3

from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'tiny_borrowed_compound_stock'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


admission_path = WORK / 'root_admission.json'
admission = json.loads(admission_path.read_text())
packet_path = Path(admission['source_packet'])
assert sha(packet_path) == admission['source_packet_sha256']
packet = json.loads(packet_path.read_text())
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
closure_path = packet_path.with_name('complete_link_closure.json')
closure = json.loads(closure_path.read_text())
for path, digest in closure['pins'].items():
    assert sha(path) == digest, path
job = json.loads((WORK / 'job.json').read_text())
jid = job['job_id']
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
assert (terminal['state'], terminal['phase'], terminal['exit_code']) == ('DONE', 'DONE', 0)
definition_path = Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json')
definition = json.loads(definition_path.read_text())
flight = json.loads((WORK / 'preflight/firesim_preflight.json').read_text())
stage = json.loads((WORK / 'actual_staged_identity.json').read_text())
assert definition['hw_config'] == flight['hardware_key'] == 'alveo_u250_firesim_gemmini_rocket_stock'
assert stage['observed_before_teardown'] and stage['phase'] == 'RUNNING'
assert definition['stage_from'] == flight['elf'] == job['elf']
assert sha(job['elf']) == flight['elf_sha256'] == stage['objects']['elf']['sha256'] == job['elf_sha256']
assert stage['objects']['bitstream']['sha256'] == '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
assert sha(flight['hwdb_artifact']) == flight['hwdb_artifact_sha256'] == definition['hwdb_config_artifact_sha256']
audit = audit_elf(Path(job['elf']).read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
raw = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0/uartlog').read_bytes()
text = raw.decode()
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
uart = WORK / 'uart.txt'; uart.write_bytes(raw)
expected_path = Path(admission['expected'])
expected = json.loads(expected_path.read_text())
parser = Path(admission['parser'])
assert sha(expected_path) == admission['expected_sha256'] and sha(parser) == admission['parser_sha256']
spec = importlib.util.spec_from_file_location('sealed_complete_paired_protocol', parser)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
report = module.parse_report(text, expected)
save(WORK / 'parsed_report.json', report)
original = report['mean_cycles']['original_full']
assert all(report['mean_cycles'][name] > original for name in ('serial_tiles', 'overlap_tiles'))
paths = [packet_path, closure_path, admission_path, definition_path, expected_path, parser,
    uart, Path(job['elf']), Path(__file__), WORK.parent / 'submit_tiny_borrowed_compound_stock.py',
    WORK / 'parsed_report.json', WORK / 'job.json', WORK / 'queue_submission.json',
    WORK / 'queue_terminal.json', WORK / 'actual_staged_identity.json',
    WORK / 'preflight/firesim_preflight.json']
record = dict(schema='root_tiny_borrowed_compound_stock2112_complete_cost_v1',
    status='REJECTED_PERFORMANCE', job_id=jid, terminal=terminal,
    complete_source_report=report, original_source_pins_revalidated=184,
    actual_link_closure_pins_revalidated=len(closure['pins']),
    all90112_i32_and45056_i8_exact=True, all_inputs_and_dirty_guards_exact=True,
    all_target_rounding_and_sticky_presets_exact=True, final_nofsm=audit,
    actual_staged_identity=stage, queue_elapsed_seconds=terminal['ended_at'] - terminal['started_at'],
    complete_cost=expected['complete_cost'], whole_model_route_installed=False,
    current_whole_cycles=378946263, current_whole_job=2085,
    scope='Six complete same-ELF windows, original/serial/overlap/overlap/serial/original; intermediate layout and scalar consumer schedule differ. No isolated copy or pure accelerator utilization attribution. Negative hardware result receives no whole-model credit.',
    driver_execution_snapshot='UNKNOWN: ELF/bitstream captured before teardown; executable snapshot attempt arrived during teardown and is not accepted as live-driver evidence',
    pins={str(path): sha(path) for path in paths})
save(WORK / 'qualification.json', record)
print(json.dumps(dict(job_id=jid, status=record['status'], mean_cycles=report['mean_cycles'],
    delta_percent=report['relative_to_original_percent'], original_source_exact=True)))
