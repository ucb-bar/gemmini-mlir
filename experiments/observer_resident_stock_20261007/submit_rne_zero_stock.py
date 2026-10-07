"""Submit the complete original M8 pair; preserve actual staged identities."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight

WORK = Path(__file__).resolve().parent / 'rne_zero_observer_stock'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

ap = WORK / 'root_admission.json'
a = json.loads(ap.read_text())
packet_path = Path(a['source_packet'])
assert sha(packet_path) == a['source_packet_sha256']
packet = json.loads(packet_path.read_text())
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
assert sha(a['protocol_parser']) == a['protocol_parser_sha256']
assert sha(a['expected']) == a['expected_sha256']
assert sha(a['strict_timing_stdout']) == a['strict_timing_stdout_sha256']
elf = Path(a['candidate_elf'])
assert sha(elf) == a['candidate_elf_sha256']
flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'),
    'merlin-golden-nofsm-probe', elf, ALIAS, TAR, WORK / 'preflight')
assert flight['elf_nofsm_audit']['status'] == 'pass'
environment = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background',
    '--chipyard', '/scratch2/agustin/wt/chipyard-stock', '--workload',
    'merlin-golden-nofsm-probe', '--bootbinary', 'probe.elf', '--stage-from', str(elf),
    '--hw-config', ALIAS, '--hwdb-config-artifact', flight['hwdb_artifact'],
    '--priority', '0', '--timeout', '600', '--project', 'gemmini-golden-nofsm']
result = subprocess.run(argv, env=environment, capture_output=True, text=True)
save(WORK / 'queue_submission.json', dict(argv=argv,
    environment_keys=sorted(environment), returncode=result.returncode,
    stdout=result.stdout, stderr=result.stderr, source_packet=str(packet_path),
    source_packet_sha256=sha(packet_path), source_pins_revalidated=164,
    root_admission_sha256=sha(ap), declaration_sha256=sha(WORK/'declaration.json'),
    complete_cost_scope=True, no_whole_forecast=True, prelabel_winner='UNKNOWN'))
assert result.returncode == 0
match = re.search(r'job_id=(\d+)', result.stdout)
assert match
jid = int(match.group(1))
save(WORK / 'queue_job.json', dict(job_id=jid))
print(json.dumps(dict(submitted_complete_M8_pair=jid, source_pins=164)), flush=True)
seen = False
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        state = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
    stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    staged_elf = stage / 'merlin-golden-nofsm-probe0-probe.elf'
    bit = stage / 'xilinx_alveo_u250/firesim.bit'
    if not seen and state['phase'] == 'RUNNING' and staged_elf.is_file() and bit.is_file():
        objects = {name: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            for name, path in [('elf', staged_elf), ('bitstream', bit)]}
        assert objects['elf']['sha256'] == a['candidate_elf_sha256']
        assert objects['bitstream']['sha256'] == BIT
        save(WORK / 'actual_staged_identity.json', dict(job_id=jid, phase=state['phase'],
            observed_before_teardown=True, objects=objects))
        seen = True
        print(json.dumps(dict(staged_identity_preserved=jid)), flush=True)
    if state['state'] in ('DONE', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
        save(WORK / 'queue_terminal.json', state)
        print(json.dumps(dict(terminal=state, staged_identity_preserved=seen)), flush=True)
        assert seen
        break
    time.sleep(3)
else:
    raise RuntimeError('Identity observer deadline exceeded; job submission remains recorded')
