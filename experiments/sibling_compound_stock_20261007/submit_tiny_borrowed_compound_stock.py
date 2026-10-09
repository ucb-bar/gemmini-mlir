"""Root-owned empirical pricing of the complete, source-closed three-arm packet."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.no_fsm_audit import audit_elf

BASE = Path(__file__).resolve().parent
WORK = BASE / 'tiny_borrowed_compound_stock'
OWNER = Path('/scratch/agustin/tmp/gemmini-sibling-producer-consumer-20261007')
PACKET = OWNER / 'out/artifacts/probes/paired-pointwise-borrowed-release-20261007/qualification.json'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


WORK.mkdir(exist_ok=False)
assert sha(PACKET) == 'c377187a5e6d915deb93f98f94a96ae7ea1d33c2eee8a4a999fc3a9dda383d1b'
packet = json.loads(PACKET.read_text())
assert len(packet['pins']) == 184
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
expected_path = Path(packet['expected']['path'])
expected = json.loads(expected_path.read_text())
parser = Path(packet['parser']['path'])
spec = importlib.util.spec_from_file_location('sealed_paired_protocol', parser)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
strict = module.parse_report(Path(expected['strict_stdout_path']).read_text(), expected)
assert strict['status'] == 'pass'
elf = Path(packet['elf']['path'])
assert sha(elf) == packet['elf']['sha256'] == expected['elf_sha256']
audit = audit_elf(elf.read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
admission = dict(schema='root_complete_tiny_borrowed_compound_stock_admission_v1',
    status='source_closed_empirical_hardware_trial', source_packet=str(PACKET),
    source_packet_sha256=sha(PACKET), source_pins_revalidated=184,
    expected=str(expected_path), expected_sha256=sha(expected_path),
    parser=str(parser), parser_sha256=sha(parser), elf=str(elf), elf_sha256=sha(elf),
    source_scope=packet['scope'], original_source_outputs=packet['source_and_output'],
    complete_cost=expected['complete_cost'], source_helper_schedule=packet['source_helper_schedule'],
    resource_and_lifetime=packet['resource_and_lifetime'], overlap_limit=packet['overlap_limit'],
    memory_layout=packet['memory_layout'], native_and_strict_source_gate=True,
    same_elf_six_windows=['original', 'serial', 'overlap', 'overlap', 'serial', 'original'],
    whole_route_installed=False, whole_cycles='UNKNOWN', final_nofsm=audit)
save(WORK / 'root_admission.json', admission)
flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'),
    'merlin-golden-nofsm-probe', elf, ALIAS, TAR, WORK / 'preflight')
env = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background',
    '--chipyard', '/scratch2/agustin/wt/chipyard-stock', '--workload', 'merlin-golden-nofsm-probe',
    '--bootbinary', 'probe.elf', '--stage-from', str(elf), '--hw-config', ALIAS,
    '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0', '--timeout', '600',
    '--project', 'gemmini-golden-nofsm']
result = subprocess.run(argv, env=env, capture_output=True, text=True)
save(WORK / 'queue_submission.json', dict(argv=argv, returncode=result.returncode,
    stdout=result.stdout, stderr=result.stderr, root_admission_sha256=sha(WORK / 'root_admission.json')))
assert result.returncode == 0
jid = int(re.search(r'job_id=(\d+)', result.stdout).group(1))
save(WORK / 'job.json', dict(job_id=jid, elf=str(elf), elf_sha256=sha(elf)))
print(json.dumps(dict(submitted=jid, scope=packet['scope'])), flush=True)
seen = False
deadline = time.monotonic() + 1800
while True:
    assert time.monotonic() < deadline, 'observer bound; job and evidence retained'
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        state = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
    stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    paths = {'elf': stage / 'merlin-golden-nofsm-probe0-probe.elf',
             'bitstream': stage / 'xilinx_alveo_u250/firesim.bit'}
    if not seen and state['phase'] == 'RUNNING' and all(path.is_file() for path in paths.values()):
        driver_files = [p for p in (stage / 'xilinx_alveo_u250').iterdir() if p.is_file() and p.name != 'firesim.bit']
        paths.update({f'driver_file:{p.name}': p for p in driver_files})
        objects = {key: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                   for key, path in paths.items()}
        assert objects['elf']['sha256'] == admission['elf_sha256'] and objects['bitstream']['sha256'] == BIT
        capture = WORK / 'actual_driver'; capture.mkdir()
        for p in driver_files:
            shutil.copyfile(p, capture / p.name)
        save(WORK / 'actual_staged_identity.json', dict(job_id=jid, phase='RUNNING',
            observed_before_teardown=True, objects=objects,
            driver_snapshot_pins={str(p): sha(p) for p in capture.iterdir()}))
        seen = True
        print(json.dumps(dict(staged=jid)), flush=True)
    if state['state'] not in ('QUEUED', 'RUNNING'):
        save(WORK / 'queue_terminal.json', state)
        print(json.dumps(dict(terminal=state)), flush=True)
        break
    time.sleep(3)
assert seen
