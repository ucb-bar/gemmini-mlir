"""Root-only stock pair admission and automatic pre-teardown identity capture."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'predictor_key_original_stock'
PACKET = Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_original_ranked_v1/stock_packet.json')
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def terminal(job_id):
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (job_id,)).fetchone())

WORK.mkdir(exist_ok=False)
packet = json.loads(PACKET.read_text())
assert sha(PACKET) == '80a2dac306d9f96ff94011c263182355adb1954f012671a34c8f8fac7ee753d8'
assert packet['status'] == 'FUNCTIONALLY_QUALIFIED_FOR_STOCK_COST_DECISION' and packet['original_elements'] == 802816
assert len(packet['flat_file_pins']) == 112
for path, digest in packet['flat_file_pins'].items():
    assert sha(path) == digest, path
bytes0, bytes1 = [Path(arm['elf']['path']).read_bytes() for arm in packet['elfs']]
assert len(bytes0) == len(bytes1)
different = [i for i, (a, b) in enumerate(zip(bytes0, bytes1, strict=True)) if a != b]
assert len(different) == 1 and (bytes0[different[0]], bytes1[different[0]]) == (0, 1)
parser_path = Path(packet['protocol']['path'])
assert sha(parser_path) == packet['protocol']['sha256']
spec = importlib.util.spec_from_file_location('qualified_predictor_key_protocol', parser_path)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
for arm in packet['elfs']:
    report = parser.parse(Path(arm['stdout']['path']).read_text(), arm=arm['arm'], elements=802816)
    assert report == arm['report']
environment = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
jobs = []
for arm in packet['elfs']:
    folder = WORK / f"arm{arm['arm']}"
    folder.mkdir()
    elf = Path(arm['elf']['path'])
    flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'), 'merlin-golden-nofsm-probe', elf, ALIAS, TAR, folder / 'preflight')
    assert flight['elf_sha256'] == arm['elf']['sha256']
    argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background', '--chipyard', '/scratch2/agustin/wt/chipyard-stock', '--workload', 'merlin-golden-nofsm-probe', '--bootbinary', 'probe.elf', '--stage-from', str(elf), '--hw-config', ALIAS, '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0', '--timeout', '600', '--project', 'gemmini-golden-nofsm']
    submitted = subprocess.run(argv, env=environment, capture_output=True, text=True)
    receipt = dict(argv=argv, environment_keys=sorted(environment), returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr, packet=str(PACKET), packet_sha256=sha(PACKET), pins_revalidated=112, original_elements=802816, selector_byte_difference=different[0], cycles_prediction=None)
    save(folder / 'queue_submission.json', receipt)
    assert submitted.returncode == 0
    match = re.search(r'job_id=(\d+)', submitted.stdout)
    assert match
    job_id = int(match.group(1))
    jobs.append(dict(arm=arm['arm'], job_id=job_id, folder=str(folder), elf=str(elf), elf_sha256=arm['elf']['sha256']))
    save(WORK / 'jobs.json', jobs)
    print(json.dumps(dict(submitted=job_id, arm=arm['arm'], pins=112)), flush=True)

seen, completed = set(), set()
deadline = time.monotonic()+1800
while len(completed) != len(jobs):
    assert time.monotonic() < deadline, 'Identity observer timeout; submissions remain recorded'
    for row in jobs:
        jid = row['job_id']
        if jid in completed:
            continue
        state = terminal(jid)
        folder = Path(row['folder'])
        stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
        elf, bit = stage / 'merlin-golden-nofsm-probe0-probe.elf', stage / 'xilinx_alveo_u250/firesim.bit'
        if jid not in seen and state['phase'] == 'RUNNING' and elf.is_file() and bit.is_file():
            objects = {name:dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size) for name, path in [('elf', elf), ('bitstream', bit)]}
            assert objects['elf']['sha256'] == row['elf_sha256'] and objects['bitstream']['sha256'] == BIT
            save(folder / 'actual_staged_identity.json', dict(job_id=jid, phase=state['phase'], observed_before_teardown=True, objects=objects))
            seen.add(jid)
            print(json.dumps(dict(staged_identity_preserved=jid)), flush=True)
        if state['state'] in ('DONE', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            save(folder / 'queue_terminal.json', state)
            completed.add(jid)
            print(json.dumps(dict(terminal=jid, state=state, staged_identity_preserved=jid in seen)), flush=True)
    if len(completed) != len(jobs):
        time.sleep(3)
assert seen == completed, 'A terminal pair is missing actual staged identity; retain explicit failure'
