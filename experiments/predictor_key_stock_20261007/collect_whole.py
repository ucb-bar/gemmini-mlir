"""Require the complete original whole protocol, immutable source and staging."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

import numpy as np

WORK = Path(__file__).resolve().parent / 'predictor_key_current2071_whole_stock'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

admission_path = WORK / 'root_admission.json'
admission = json.loads(admission_path.read_text())
packet_path = Path(admission['source_packet'])
assert sha(packet_path) == admission['source_packet_sha256']
packet = json.loads(packet_path.read_text())
for path, digest in packet['flat_file_pins'].items():
    assert sha(path) == digest, path
jid = json.loads((WORK / 'queue_job.json').read_text())['job_id']
assert jid == 2081
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
assert (terminal['state'], terminal['phase'], terminal['exit_code']) == ('DONE', 'DONE', 0)
job = Path(f'/scratch/firesim_queue/jobs/{jid}')
definition_path = job / 'runworkload-full.json'
definition = json.loads(definition_path.read_text())
flight_path = WORK / 'preflight/firesim_preflight.json'
flight = json.loads(flight_path.read_text())
staged_path = WORK / 'actual_staged_identity.json'
staged = json.loads(staged_path.read_text())
submit_path = WORK / 'queue_submission.json'
submit = json.loads(submit_path.read_text())
assert definition['user'] == 'agustin' and definition['hw_config'] == flight['hardware_key'] == ALIAS
assert definition['stage_from'] == packet['candidate_elf'] == flight['elf']
assert submit['source_packet_sha256'] == sha(packet_path)
assert submit['root_admission_sha256'] == sha(admission_path)
assert flight['elf_sha256'] == sha(packet['candidate_elf']) == packet['candidate_elf_sha256']
assert staged['objects']['elf']['sha256'] == flight['elf_sha256']
assert staged['job_id'] == jid and staged['phase'] == 'RUNNING' and staged['observed_before_teardown']
assert staged['objects']['bitstream']['sha256'] == BIT
assert definition['hwdb_config_artifact_sha256'] == flight['hwdb_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
assert flight['elf_nofsm_audit'] == packet['controlled_whole_gate']['nofsm']
assert flight['elf_nofsm_audit']['status'] == 'pass'
assert not flight['elf_nofsm_audit']['forbidden'] and not flight['elf_nofsm_audit']['unknown']
raw = (job / 'simulation/sim_slot_0/uartlog').read_bytes()
text = raw.decode().replace('\r', '')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert text.splitlines().count('DONE') == 1
assert text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
metrics = re.findall(r'^METRIC cycles (\d+)$', text, re.M)
assert len(metrics) == 1
golden_path = Path(packet['controlled_whole_gate']['native']['original_golden'])
assert sha(golden_path) == packet['controlled_whole_gate']['native']['original_golden_sha256']
golden = np.load(golden_path, allow_pickle=False)
assert golden.size == 1000 and golden.dtype == np.float32
digest = hashlib.sha256(golden.astype('<f4', copy=False).tobytes()).hexdigest()
assert digest == admission['original_output_digest']
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', text, re.M) == [('1000', '4000', digest)]
cycles = int(metrics[0])
archive = WORK / 'stock2081_uart.txt'
archive.write_bytes(raw)
result = dict(schema='root_predictor_key_resnet_stock2081_whole_terminal_v1',
    status='verified_complete_original_whole', job_id=jid, terminal=terminal,
    control_job=2071, control_cycles=29698347, candidate_cycles=cycles,
    saving_cycles=29698347-cycles, saving_fraction=(29698347-cycles)/29698347,
    zip_reference_job=1876, zip_reference_cycles=22387449, remaining_zip_gap=cycles-22387449,
    original_output_gate=dict(count=1000, raw_sha256=digest, original_golden=str(golden_path),
        original_golden_sha256=sha(golden_path), source_bitexact=True, atol=0, rtol=0),
    source_packet=str(packet_path), source_packet_sha256=sha(packet_path),
    flat_source_pins_revalidated=679, executed_entry_closure=packet['executed_entry_closure'],
    actual_staged_identity=staged, final_nofsm=flight['elf_nofsm_audit'],
    hardware_alias=ALIAS, hwdb_sha256=HWDB, bitstream_archive_sha256=TAR,
    actual_staged_bit_sha256=BIT, elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],
    scope='Complete uninstrumented original ResNet whole METRIC window. Nine-product source-key residual is the sole selected change; original host8/runtime/weights/other target objects remain2071. The final simulator PASSED counter includes validation/setup and is not the model metric. One stock observation, no additive section projection or pure utilization claim. ZIP numeric workload/timers differ and remain a reference.',
    pins={str(p):sha(p) for p in [admission_path,packet_path,definition_path,flight_path,
        staged_path,submit_path,archive,Path(__file__)]}, token_usage_available=False,token_usage=None)
(WORK / 'stock2081_terminal.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key:result[key] for key in ['status','candidate_cycles','saving_cycles',
    'saving_fraction','remaining_zip_gap','elapsed_queue_seconds']}), flush=True)
