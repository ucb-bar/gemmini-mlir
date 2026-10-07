"""Close stock complete M8 observation without projecting a whole-model win."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3

WORK = Path(__file__).resolve().parent / 'rne_zero_product_stock'
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
assert len(packet['pins']) == a['pins_revalidated'] == 152
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
for path_key, digest_key in [('candidate_elf', 'candidate_elf_sha256'),
        ('strict_timing_stdout', 'strict_timing_stdout_sha256'),
        ('protocol_parser', 'protocol_parser_sha256'), ('expected', 'expected_sha256')]:
    assert sha(a[path_key]) == a[digest_key], path_key
jid = json.loads((WORK / 'queue_job.json').read_text())['job_id']
assert jid == 2087
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
assert definition['user'] == 'agustin'
assert definition['hw_config'] == flight['hardware_key'] == ALIAS
assert definition['stage_from'] == flight['elf'] == a['candidate_elf']
assert submit['source_packet_sha256'] == sha(packet_path)
assert submit['root_admission_sha256'] == sha(admission_path)
assert submit['declaration_sha256'] == sha(WORK / 'declaration.json')
assert flight['elf_sha256'] == a['candidate_elf_sha256']
assert staged['objects']['elf']['sha256'] == flight['elf_sha256']
assert staged['job_id'] == jid and staged['phase'] == 'RUNNING' and staged['observed_before_teardown']
assert staged['objects']['bitstream']['sha256'] == BIT
assert definition['hwdb_config_artifact_sha256'] == flight['hwdb_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
audit = flight['elf_nofsm_audit']
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
raw = (job / 'simulation/sim_slot_0/uartlog').read_bytes()
text = raw.decode().replace('\r', '')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
declaration = json.loads((WORK / 'declaration.json').read_text())
expected = Path(a['expected']).read_bytes()
spec = importlib.util.spec_from_file_location('sealed_i8_parser', a['protocol_parser'])
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
strict = parser.parse_report(Path(a['strict_timing_stdout']).read_text(), declaration, expected)
observation = parser.parse_report(text, declaration, expected)
archive = WORK / 'stock2087_uart.txt'
archive.write_bytes(raw)
result = dict(schema='root_rne_zero_stock2087_complete_terminal_v1',
    status='verified_complete_original_M8_pair', job_id=jid, terminal=terminal,
    observation=observation, strict_source_observation=strict,
    saving_cycles=observation['control_mean_cycles']-observation['candidate_mean_cycles'],
    saving_fraction=-observation['fraction_change'],
    source_packet=str(packet_path), source_packet_sha256=sha(packet_path),
    source_pins_revalidated=152, original_gate='45056 original i8, guards and immutable inputs; separate all-five-FRM/seven-sticky target gate remains source packet qualification',
    actual_staged_identity=staged, final_nofsm=audit,
    hardware_alias=ALIAS, hwdb_sha256=HWDB, bitstream_archive_sha256=TAR,
    actual_staged_bit_sha256=BIT, elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],
    prelabel_winner='UNKNOWN', whole_cycles='UNKNOWN', hardware_whole_promotion=False,
    historical_word_only_unresolved_source_pin=True,
    scope='One stock complete M8 ABBA observation. Both arms include preparation, table loads, certificate, source fallback, finishing, stores and frame. It replaces sufficient post-product zero admission with an explicit source-factor early zero threshold. The rounded magnitude bound certifies both original separate products only under original typed RNE/effect permissions; fallback retains original arithmetic. Control is the previous zero-bin M8 helper, not2076 whole. Mixed FP/branch/load issue costs remain unpriced before this observation. No all22 compiled or whole model claim. Word-only historical source drift remains unpromoted.',
    pins={str(p):sha(p) for p in [admission_path,packet_path,definition_path,flight_path,
        staged_path,submit_path,WORK/'declaration.json',archive,Path(a['protocol_parser']),
        Path(a['expected']),Path(__file__)]}, token_usage_available=False, token_usage=None)
(WORK / 'stock2087_terminal.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key:result[key] for key in ['status','saving_cycles','saving_fraction','elapsed_queue_seconds']}), flush=True)
