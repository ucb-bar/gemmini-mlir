"""Require both original ranked stock arms, source pins and actual staging."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'predictor_key_original_stock'
PACKET = Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_original_ranked_v1/stock_packet.json')
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

packet = json.loads(PACKET.read_text())
assert sha(PACKET) == '80a2dac306d9f96ff94011c263182355adb1954f012671a34c8f8fac7ee753d8'
for path,digest in packet['flat_file_pins'].items():
    assert sha(path) == digest, path
parser_path = Path(packet['protocol']['path'])
assert sha(parser_path) == packet['protocol']['sha256']
spec = importlib.util.spec_from_file_location('stock_ranked_protocol',parser_path)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
jobs = json.loads((WORK/'jobs.json').read_text())
assert [row['arm'] for row in jobs] == [0,1]
records = []
pins = {str(PACKET):sha(PACKET),str(Path(__file__)):sha(__file__),str(WORK/'jobs.json'):sha(WORK/'jobs.json')}
for row in jobs:
    folder = Path(row['folder'])
    job = Path(f"/scratch/firesim_queue/jobs/{row['job_id']}")
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(row['job_id'],)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code']) == ('DONE','DONE',0)
    definition_path = job/'runworkload-full.json'
    definition = json.loads(definition_path.read_text())
    flight_path = folder/'preflight/firesim_preflight.json'
    flight = json.loads(flight_path.read_text())
    staged_path = folder/'actual_staged_identity.json'
    staged = json.loads(staged_path.read_text())
    submitted_path = folder/'queue_submission.json'
    submitted = json.loads(submitted_path.read_text())
    assert definition['job_id'] == staged['job_id'] == row['job_id'] and definition['user'] == 'agustin'
    assert definition['hw_config'] == flight['hardware_key'] == ALIAS
    assert definition['stage_from'] == flight['elf'] == row['elf']
    assert submitted['packet_sha256'] == sha(PACKET) and submitted['pins_revalidated'] == 112
    assert staged['observed_before_teardown'] and staged['phase'] == 'RUNNING'
    assert staged['objects']['elf']['sha256'] == row['elf_sha256'] == flight['elf_sha256'] == sha(row['elf'])
    assert staged['objects']['bitstream']['sha256'] == BIT
    assert flight['hwdb_artifact'] == definition['hwdb_config_artifact']
    assert flight['hwdb_artifact_sha256'] == definition['hwdb_config_artifact_sha256'] == sha(flight['hwdb_artifact']) == HWDB
    assert flight['bitstream_sha256'] == sha(flight['bitstream_path']) == TAR
    assert flight['elf_nofsm_audit']['status'] == 'pass' and not flight['elf_nofsm_audit']['forbidden'] and not flight['elf_nofsm_audit']['unknown']
    uart = job/'simulation/sim_slot_0/uartlog'
    raw = uart.read_bytes()
    text = raw.decode().replace('\r','')
    assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    report = parser.parse(text,arm=row['arm'],elements=802816)
    archive = folder/f"stock{row['job_id']}_uart.txt"
    archive.write_bytes(raw)
    for path in [archive,staged_path,flight_path,submitted_path,definition_path]:
        pins[str(path)] = sha(path)
    records.append(dict(arm=row['arm'],job=terminal,report=report,staged_identity=staged,nofsm_audit=flight['elf_nofsm_audit'],elapsed_queue_seconds=terminal['ended_at']-terminal['started_at']))
control,candidate = [row['report']['counters']['cycles'] for row in records]
result = dict(schema='predictor_key_original_ranked_stock2078_2079_terminal_v1',status='verified_complete_original_scope_pair',source_packet=str(PACKET),source_packet_sha256=sha(PACKET),source_pins_revalidated=112,original_elements=802816,control_cycles=control,candidate_cycles=candidate,saving_cycles=control-candidate,saving_fraction=(control-candidate)/control,records=records,hardware_alias=ALIAS,hwdb_sha256=HWDB,bitstream_tar_sha256=TAR,actual_staged_bit_sha256=BIT,scope=packet['scope'],same_addresses_warmups=packet['same_addresses_and_warmups'],source_numeric_contract=packet['source_numeric_contract'],complete_original_source_outputs=True,private_workspace=packet['private_workspace'],large_GSIM_complete_timing='Not qualified: budget reached after timedcounter without final fullmarker; engine/logs retained separately. Stock complete marker closes measured pair independently.',whole_model_cycles=None,whole_accuracy=None,pins=pins,token_usage_available=False,token_usage=None)
(WORK/'stock2078_2079_terminal.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['status','control_cycles','candidate_cycles','saving_cycles','saving_fraction']}),flush=True)
