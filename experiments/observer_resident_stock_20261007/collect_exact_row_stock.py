"""Root complete matched source-observer stock cycle closure."""
from pathlib import Path
import hashlib
import json
import sqlite3
from parse_exact_row_group import parse

WORK = Path(__file__).resolve().parent/'exact_row_stock'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

admission_path = WORK/'root_admission.json'
admission = json.loads(admission_path.read_text())
for row in admission['source_packets']:
    path = Path(row['path'])
    assert sha(path) == row['sha256']
    packet = json.loads(path.read_text())
    assert len(packet['pins']) == row['revalidated_pins']
    for pin, digest in packet['pins'].items():
        assert sha(pin) == digest, pin
assert sha(admission['parser']['path']) == admission['parser']['sha256']
jobs = json.loads((WORK/'jobs.json').read_text())
assert [(x['name'],x['job_id']) for x in jobs] == [('control_aligned',2091), ('candidate_aligned',2092)]
rows = []
pins = {str(path):sha(path) for path in [admission_path, WORK/'jobs.json', Path(__file__), Path(admission['parser']['path'])]}
for row in jobs:
    jid = row['job_id']; folder = Path(row['folder'])
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code']) == ('DONE','DONE',0)
    job = Path(f'/scratch/firesim_queue/jobs/{jid}')
    definition_path = job/'runworkload-full.json'; definition = json.loads(definition_path.read_text())
    flight_path = folder/'preflight/firesim_preflight.json'; flight = json.loads(flight_path.read_text())
    stage_path = folder/'actual_staged_identity.json'; stage = json.loads(stage_path.read_text())
    submission_path = folder/'queue_submission.json'; submission = json.loads(submission_path.read_text())
    assert definition['user'] == 'agustin' and definition['hw_config'] == flight['hardware_key'] == ALIAS
    assert definition['stage_from'] == flight['elf'] == row['elf']
    assert flight['elf_sha256'] == sha(row['elf']) == row['elf_sha256']
    assert stage['job_id'] == jid and stage['phase'] == 'RUNNING' and stage['observed_before_teardown']
    assert stage['objects']['elf']['sha256'] == row['elf_sha256']
    assert stage['objects']['bitstream']['sha256'] == BIT
    assert definition['hwdb_config_artifact_sha256'] == flight['hwdb_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
    assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
    assert submission['source_packets'] == admission['source_packets']
    audit = flight['elf_nofsm_audit']
    assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
    raw = (job/'simulation/sim_slot_0/uartlog').read_bytes()
    text = raw.decode().replace('\r','')
    assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    observation = parse(text)
    strict = parse((Path(row['elf']).parent/'stdout').read_text())
    for key in ['stats', 'carrier_differences', 'original_compiled_consumer_and_guards_pass']:
        assert observation[key] == strict[key]
    archive = folder/f'stock{jid}_uart.txt'; archive.write_bytes(raw)
    for path in [definition_path, flight_path, stage_path, submission_path, archive]:
        pins[str(path)] = sha(path)
    rows.append(dict(name=row['name'], job_id=jid, cycles=observation['cycles'], observation=observation,
        terminal=terminal, elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'], actual_staged_identity=stage, final_nofsm=audit))
control, candidate = rows
receipt = dict(schema='root_exact_row_complete_stock2091_2092_terminal_v1', status='verified_complete_original_pair',
    rows=rows, control_cycles=control['cycles'], candidate_cycles=candidate['cycles'],
    saving_cycles=control['cycles']-candidate['cycles'], saving_fraction=1-candidate['cycles']/control['cycles'],
    source_packets=admission['source_packets'], revalidated_source_pins=361,
    common_data_symbols=admission['common_data_symbols'], original_observer=admission['source_observer'],
    complete_roi=admission['complete_roi'], hardware_alias=ALIAS, hwdb_sha256=HWDB,
    bitstream_archive_sha256=TAR, actual_staged_bit_sha256=BIT,
    source_approximate_policy=admission['source_approximate_policy'],
    default_enabled=False, whole_cycles='UNKNOWN', pins=pins, token_usage_available=False,
    scope='Two matched complete12-head source-consumer capsules. One stock observation per arm; no all48-group multiplier, whole-model prediction, instruction/cycle equivalence or causal utilization claim. Fresh full48 native1600-bit exact source successor and target whole gate are separately qualified.')
(WORK/'stock2091_2092_terminal.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:receipt[key] for key in ['status','control_cycles','candidate_cycles','saving_cycles','saving_fraction']}),flush=True)
