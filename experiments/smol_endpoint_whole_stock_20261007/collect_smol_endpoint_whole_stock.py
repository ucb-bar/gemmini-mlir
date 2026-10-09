"""Verify complete original Smol output and whole hardware timing, without section sums."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'smol_endpoint_whole_stock'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


root_path = WORK / 'root_admission.json'
root = json.loads(root_path.read_text())
source_path = WORK / 'source_admission.json'
assert sha(source_path) == root['source_admission_sha256']
source = json.loads(source_path.read_text())
for path, digest in source['pins'].items():
    assert sha(path) == digest, path
job = json.loads((WORK / 'job.json').read_text())
jid = job['job_id']
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
if terminal['state'] in ('QUEUED', 'RUNNING'):
    print(json.dumps(dict(job_id=jid, status=terminal['state'], phase=terminal['phase'], whole_cycles='UNKNOWN')))
    raise SystemExit(0)
definition_path = Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json')
definition = json.loads(definition_path.read_text())
flight = json.loads((WORK / 'preflight/firesim_preflight.json').read_text())
stage_path = WORK / 'actual_staged_identity.json'
stage = json.loads(stage_path.read_text())
assert definition['stage_from'] == flight['elf'] == job['elf'] == root['elf']
assert sha(job['elf']) == root['elf_sha256'] == flight['elf_sha256'] == stage['objects']['elf']['sha256']
assert stage['phase'] == 'RUNNING' and stage['observed_before_teardown']
assert definition['hw_config'] == flight['hardware_key'] == 'alveo_u250_firesim_gemmini_rocket_stock'
assert definition['timeout'] == root['timeout_seconds'] == 14400
assert stage['objects']['bitstream']['sha256'] == '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
assert sha(flight['hwdb_artifact']) == flight['hwdb_artifact_sha256'] == definition['hwdb_config_artifact_sha256']
for path, digest in stage['driver_snapshot_pins'].items():
    assert sha(path) == digest, path
driver = WORK / 'actual_driver/FireSim-xilinx_alveo_u250'
assert sha(driver) == stage['objects']['actual_driver']['sha256']
assert './FireSim-xilinx_alveo_u250 ' in (WORK / 'actual_driver/sim-run.sh').read_text()
audit = audit_elf(Path(job['elf']).read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
uart_source = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0/uartlog')
raw = uart_source.read_bytes()
text = raw.decode().replace('\r', '')
uart = WORK / 'uart.txt'; uart.write_bytes(raw)
pins = {str(path): sha(path) for path in [root_path, source_path, definition_path, stage_path,
    uart, Path(job['elf']), Path(__file__), WORK.parent / 'submit_smol_endpoint_whole_stock.py',
    WORK / 'job.json', WORK / 'queue_submission.json', WORK / 'queue_terminal.json',
    WORK / 'preflight/firesim_preflight.json']}
pins.update(stage['driver_snapshot_pins'])
base = dict(schema='root_smol_endpoint_stock2113_whole_v1', job_id=jid,
    terminal=terminal, actual_staged_identity=stage, final_nofsm=audit,
    source_pins_revalidated=len(source['pins']), original_source_pins_revalidated=191,
    source_native1600_bitwise_exact=True, original_gate=root['original_gate'],
    numerical_policy=source['numeric_policy'], layout=source['layout'],
    queue_elapsed_seconds=terminal['ended_at'] - terminal['started_at'], pins=pins)
try:
    if (terminal['state'], terminal['phase'], terminal['exit_code']) != ('DONE', 'DONE', 0):
        raise ValueError('bounded hardware lifecycle did not complete successfully')
    if '*** PASSED ***' not in text or 'COMMAND_EXIT_CODE="0"' not in text:
        raise ValueError('simulator/command completion marker absent')
    parser = Path(source['protocol']['parser_file'])
    spec = importlib.util.spec_from_file_location('sealed_smol_whole_protocol', parser)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.verify_protocol(text, np.load(source['protocol']['golden'], allow_pickle=False))
except Exception as error:
    record = dict(**base, status='EVALUATION_REFUSED', error_type=type(error).__name__,
        reason=str(error), whole_cycles='UNKNOWN',
        whole_elementwise_accuracy='UNKNOWN if original full digest differs; OUT1 alone cannot establish the1600-element tolerance gate',
        scope='Retain actual bounded trial refusal; no predecessor or native target result transfer.')
    save(WORK / 'qualification.json', record)
    print(json.dumps(dict(job_id=jid, status=record['status'], reason=record['reason'])))
    raise SystemExit(1)
cycles = report['cycles']
control = 258621872969
record = dict(**base, status='verified_complete_original_whole',
    report=report, candidate_cycles=cycles, control_job=1906, control_cycles=control,
    saving_cycles=control - cycles, reduction_percent=100 * (control - cycles) / control,
    target_cycles=5000000000, remaining_target_gap=cycles - 5000000000,
    bitwise_exact_all1600=True,
    scope='Actual complete new48-source normal ELF on stock, unchanged original1600 output gate; old1906 comparison is historical, not an object-matched single-transform experiment. No group×48 price, summed section saving, pure host/accelerator fraction or general physical memory-capacity claim.')
save(WORK / 'qualification.json', record)
print(json.dumps(dict(job_id=jid, status=record['status'], whole_cycles=cycles,
    original1600_bitwise_exact=True, historical1906_reduction_percent=record['reduction_percent'])))
