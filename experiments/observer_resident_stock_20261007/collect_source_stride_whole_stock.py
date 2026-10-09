"""Close the original ResNet whole protocol and stock execution identities."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

import numpy as np

WORK = Path(__file__).resolve().parent / 'source_stride_whole_stock'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


admission_path = WORK / 'root_admission.json'
a = json.loads(admission_path.read_text())
packet_path = Path(a['source_packet'])
assert sha(packet_path) == a['source_packet_sha256']
packet = json.loads(packet_path.read_text())
assert len(packet['pins']) == a['pins_revalidated'] == 1959
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
jid = json.loads((WORK / 'queue_job.json').read_text())['job_id']
assert jid == 2086
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
    db.row_factory = sqlite3.Row
    terminal = dict(db.execute(
        'SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',
        (jid,),
    ).fetchone())
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
assert definition['user'] == 'agustin'
assert definition['hw_config'] == flight['hardware_key'] == ALIAS
assert definition['stage_from'] == flight['elf'] == a['candidate_elf'] == packet['candidate_elf']['path']
assert flight['elf_sha256'] == sha(a['candidate_elf']) == a['candidate_elf_sha256'] == packet['candidate_elf']['sha256']
assert submit['source_packet_sha256'] == sha(packet_path)
assert submit['root_admission_sha256'] == sha(admission_path)
assert staged['job_id'] == jid and staged['phase'] == 'RUNNING' and staged['observed_before_teardown']
assert staged['objects']['elf']['sha256'] == flight['elf_sha256']
assert staged['objects']['bitstream']['sha256'] == BIT
assert definition['hwdb_config_artifact_sha256'] == flight['hwdb_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
audit = flight['elf_nofsm_audit']
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
entry = a['actual_selected_entry_closure']
assert entry['status'] == 'PASS' and entry['elf_sha256'] == flight['elf_sha256']
assert entry['selected_symbol_entry_count'] == 1 and entry['selected_primitive_link_relocation_equivalent']
assert len(entry['branch_relocation_witness']) == 10
validation_path = Path(packet['arms']['controlled']['strict_validation']['path'])
assert sha(validation_path) == packet['arms']['controlled']['strict_validation']['sha256']
validation = json.loads(validation_path.read_text())
assert validation['elf_sha256'] == flight['elf_sha256']
native_path = Path(packet['arms']['controlled']['native_output']['path'])
assert sha(native_path) == packet['arms']['controlled']['native_output']['sha256']
golden_path = Path(a['original_golden'])
assert sha(golden_path) == a['original_golden_sha256']
output = np.load(native_path, allow_pickle=False)
golden = np.load(golden_path, allow_pickle=False)
assert output.dtype == golden.dtype == np.float32 and output.size == golden.size == a['original_count'] == 1000
assert a['original_atol'] == a['original_rtol'] == 0
assert np.array_equal(output.reshape(-1).view(np.uint32), golden.reshape(-1).view(np.uint32))
digest = hashlib.sha256(output.astype('<f4', copy=False).tobytes()).hexdigest()
assert digest == a['original_output_digest']
raw = (job / 'simulation/sim_slot_0/uartlog').read_bytes()
text = raw.decode().replace('\r', '')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert text.splitlines().count('DONE') == 1
assert text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
values = re.findall(r'^METRIC cycles (\d+)$', text, re.M)
assert len(values) == 1 and int(values[0]) > 0
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', text, re.M) == [('1000', '4000', digest)]
cycles = int(values[0])
archive = WORK / 'stock2086_uart.txt'
archive.write_bytes(raw)
result = dict(
    schema='root_source_stride_ResNet_stock2086_whole_terminal_v1',
    status='verified_complete_original_whole', job_id=jid, terminal=terminal,
    control_job=2081, control_cycles=29618198, candidate_cycles=cycles,
    saving_cycles=29618198-cycles, saving_fraction=1-cycles/29618198,
    target_cycles=22000000, remaining_target_gap=cycles-22000000,
    original_output_gate=dict(count=1000, raw_sha256=digest, compiled_source_bitexact=True,
        original_atol=0, original_rtol=0, original_pass=True,
        golden=str(golden_path), golden_sha256=sha(golden_path)),
    source_packet=str(packet_path), source_packet_sha256=sha(packet_path), source_pins_revalidated=1959,
    actual_staged_identity=staged, final_nofsm=audit,
    actual_selected_entry_closure=entry, hardware_alias=ALIAS, hwdb_sha256=HWDB,
    bitstream_archive_sha256=TAR, actual_staged_bit_sha256=BIT,
    elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],
    scope='One uninstrumented original ResNet whole METRIC observation. Fresh normal build reproduces all52 default kernel/adapter pairs and aggregate. Sole selected primitive source-stride command retention; all52 adapters and51 other primitives unchanged. Control2081 ELF reproduced byte-exact. Capsule gains are not added or projected to whole cycles; simulator PASSED count includes setup/validation and is not the performance metric.',
    pins={str(p): sha(p) for p in [admission_path, packet_path, validation_path, native_path,
        golden_path, definition_path, flight_path, staged_path, submit_path, archive, Path(__file__)]},
    token_usage_available=False, token_usage=None,
)
(WORK / 'stock2086_terminal.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({key: result[key] for key in ['status', 'candidate_cycles', 'saving_cycles',
    'saving_fraction', 'remaining_target_gap', 'elapsed_queue_seconds']}), flush=True)
