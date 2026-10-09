"""Root-owned bounded full-model evaluation; source closure is not hardware evidence."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.no_fsm_audit import audit_elf

BASE = Path(__file__).resolve().parent
WORK = BASE / 'smol_endpoint_whole_stock'
ADMISSION = Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007/out/artifacts/probes/smol-endpoint-full-evaluation/admission.json')
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ELF_SHA = 'f90b8c7e8d6c1ed668a0bb4eaf7918caaa2dc8e3fe4f93b6c14143f995889f07'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


WORK.mkdir(exist_ok=False)
packet = json.loads(ADMISSION.read_text())
assert packet['status'] == 'READY_FOR_PARENT_AUTHORIZED_EMPIRICAL_STOCK_TRIAL_NOT_SUBMITTED'
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
assert packet['original_source_pins_reclosed'] == 191
assert packet['normal_native']['elements'] == 1600 and packet['normal_native']['bitwise_mismatches'] == 0
assert packet['layout']['regions_disjoint'] and packet['layout']['empirical_evaluation_authorized']
assert packet['execution_bound']['seconds'] == 14400
assert packet['protocol']['expected_sha256'] == '1e5b274d4c99cb6710994542f509769ba61ed37f435dc6ee1ed05ed1b8cc03de'
elf = Path(packet['elf']['path'])
assert sha(elf) == packet['elf']['sha256'] == ELF_SHA
audit = audit_elf(elf.read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
shutil.copyfile(ADMISSION, WORK / 'source_admission.json')
root_admission = dict(schema='root_smol_endpoint_whole_stock_admission_v1',
    status='authorized_empirical_full_model_evaluation', source_admission=str(ADMISSION),
    source_admission_sha256=sha(ADMISSION), source_pins_revalidated=len(packet['pins']),
    original_gate=dict(elements=1600, atol=0.03125, rtol=0.02,
        raw_output_sha256=packet['protocol']['expected_sha256']),
    elf=str(elf), elf_sha256=ELF_SHA, timeout_seconds=14400,
    timeout_basis=packet['execution_bound'], layout=packet['layout'],
    capacity_status='UNKNOWN before empirical execution; address extents are source-proved',
    no_predecessor_target_or_hardware_pass_transfer=True,
    no_group_times_48_prediction=True, whole_cycles='UNKNOWN', final_nofsm=audit)
save(WORK / 'root_admission.json', root_admission)
flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'),
    'merlin-golden-nofsm-probe', elf, ALIAS, TAR, WORK / 'preflight')
env = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background',
    '--chipyard', '/scratch2/agustin/wt/chipyard-stock', '--workload', 'merlin-golden-nofsm-probe',
    '--bootbinary', 'probe.elf', '--stage-from', str(elf), '--hw-config', ALIAS,
    '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0', '--timeout', '14400',
    '--project', 'gemmini-golden-nofsm']
result = subprocess.run(argv, env=env, capture_output=True, text=True)
save(WORK / 'queue_submission.json', dict(argv=argv, returncode=result.returncode,
    stdout=result.stdout, stderr=result.stderr, root_admission_sha256=sha(WORK / 'root_admission.json')))
assert result.returncode == 0
jid = int(re.search(r'job_id=(\d+)', result.stdout).group(1))
save(WORK / 'job.json', dict(job_id=jid, elf=str(elf), elf_sha256=ELF_SHA))
print(json.dumps(dict(submitted=jid, scope='complete SmolVLA original1600', timeout_seconds=14400)), flush=True)
seen = False
deadline = time.monotonic() + 18000
while True:
    assert time.monotonic() < deadline, 'observer bound; job and evidence retained'
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        state = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
    stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    paths = {'elf': stage / 'merlin-golden-nofsm-probe0-probe.elf',
             'bitstream': stage / 'xilinx_alveo_u250/firesim.bit',
             'sim_run': stage / 'sim-run.sh',
             'actual_driver': stage / 'FireSim-xilinx_alveo_u250',
             'actual_driver_bundle': stage / 'driver-bundle.tar.gz'}
    if not seen and state['phase'] == 'RUNNING' and all(path.is_file() for path in paths.values()):
        driver_files = [p for p in (stage / 'xilinx_alveo_u250').iterdir() if p.is_file() and p.name != 'firesim.bit']
        paths.update({f'driver_file:{p.name}': p for p in driver_files})
        objects = {key: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                   for key, path in paths.items()}
        assert objects['elf']['sha256'] == ELF_SHA and objects['bitstream']['sha256'] == BIT
        capture = WORK / 'actual_driver'; capture.mkdir()
        assert './FireSim-xilinx_alveo_u250 ' in paths['sim_run'].read_text()
        for p in [*driver_files, paths['sim_run'], paths['actual_driver'], paths['actual_driver_bundle']]:
            shutil.copyfile(p, capture / p.name)
        definition = Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json')
        shutil.copyfile(definition, WORK / 'job_definition.json')
        config = Path(f'/scratch/firesim_queue/jobs/{jid}/config_runtime.yaml')
        if config.is_file():
            shutil.copyfile(config, WORK / 'actual_runtime.yaml')
        save(WORK / 'actual_staged_identity.json', dict(job_id=jid, phase='RUNNING',
            observed_before_teardown=True, objects=objects,
            driver_snapshot_pins={str(p): sha(p) for p in capture.iterdir()},
            loaded_header_bit_build_equivalence='UNKNOWN; actual executed artifacts retained'))
        seen = True
        print(json.dumps(dict(staged=jid, driver_bundle_files=len(driver_files))), flush=True)
    if state['state'] not in ('QUEUED', 'RUNNING'):
        save(WORK / 'queue_terminal.json', state)
        print(json.dumps(dict(terminal=state)), flush=True)
        break
    time.sleep(3)
assert seen
