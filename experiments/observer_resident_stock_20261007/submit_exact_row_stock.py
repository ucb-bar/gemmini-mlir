"""Root admission and stock mcycle measurement of the exact row-packing pair."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight
from parse_exact_row_group import parse

ROOT = Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007')
WORK = Path(__file__).resolve().parent/'exact_row_stock'
PACKET = ROOT/'docs/perf_records/exact_row_mcycle_common_address_pair.json'
GROUP = ROOT/'docs/perf_records/exact_row_radix_complete_group_qualification.json'
NORMAL = ROOT/'docs/perf_records/exact_row_normal_source_qualification.json'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

WORK.mkdir(exist_ok=False)
packets = []
for path, expected in [(PACKET, 65), (GROUP, 140), (NORMAL, 156)]:
    packet = json.loads(path.read_text())
    assert len(packet['pins']) == expected, (path, len(packet['pins']))
    for pin, digest in packet['pins'].items():
        assert sha(pin) == digest, pin
    packets.append(dict(path=str(path), sha256=sha(path), revalidated_pins=expected))
packet = json.loads(PACKET.read_text())
group = json.loads(GROUP.read_text())
assert group['native'] == dict(groups=48, preparations=12, products=23040, fallback=0, original1600_bit_mismatches=0)
assert group['callback_count'] == 480
disassembly = (ROOT/'out/exact_row_clock_pair/driver.disassembly').read_text()
assert len(re.findall(r'\bmcycle\b', disassembly)) == 2 and 'minstret' not in disassembly
assert packet['data_symbols_equal']['consumer_input'][0] == '00000000809b08c0'
original = packet['records']
for name, row in original.items():
    elf = ROOT/'out/exact_row_clock_pair'/name/'model.elf'
    assert sha(elf) == row['elf_sha256']
    assert row['returncode'] == 0
    audit = json.loads((elf.parent/'nofsm.json').read_text())
    assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
    assert audit['elf_sha256'] == row['elf_sha256']
    observation = parse((elf.parent/'stdout').read_text())
    assert [value for _, value in observation['stats']] == row['stats']
save(WORK/'root_admission.json', dict(schema='root_exact_row_stock_pair_admission_v1', status='pass',
    source_packets=packets, common_data_symbols=packet['data_symbols_equal'],
    source_observer='Original compiled quantization consumer plus source scales, all guards/immutable inputs and unchanged eight provider counters; three unchanged internal BF16 differences are unobserved by this consumer',
    complete_roi='mcycle around allocation, alignment, workspace guard preparation, complete480-product12-head source provider, and all output stores; validation outside ROI',
    default_enabled=False, hardware_cycles='UNKNOWN', whole_cycles='UNKNOWN',
    source_approximate_policy='Existing explicit RMS4 approximation unchanged; new row-packing optimization is source-exact',
    parser=dict(path=str(Path(__file__).parent/'parse_exact_row_group.py'), sha256=sha(Path(__file__).parent/'parse_exact_row_group.py'))))
environment = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
jobs = []
for name, row in original.items():
    folder = WORK/name
    folder.mkdir()
    elf = ROOT/'out/exact_row_clock_pair'/name/'model.elf'
    flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'), 'merlin-golden-nofsm-probe', elf, ALIAS, TAR, folder/'preflight')
    assert flight['elf_sha256'] == row['elf_sha256'] and flight['elf_nofsm_audit']['status'] == 'pass'
    argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background', '--chipyard', '/scratch2/agustin/wt/chipyard-stock',
        '--workload', 'merlin-golden-nofsm-probe', '--bootbinary', 'probe.elf', '--stage-from', str(elf), '--hw-config', ALIAS,
        '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0', '--timeout', '1800', '--project', 'gemmini-golden-nofsm']
    submitted = subprocess.run(argv, env=environment, capture_output=True, text=True)
    save(folder/'queue_submission.json', dict(argv=argv, environment_keys=sorted(environment), returncode=submitted.returncode,
        stdout=submitted.stdout, stderr=submitted.stderr, source_packets=packets, prelabel_winner='UNKNOWN'))
    assert submitted.returncode == 0
    match = re.search(r'job_id=(\d+)', submitted.stdout)
    assert match
    jid = int(match.group(1))
    jobs.append(dict(name=name, job_id=jid, folder=str(folder), elf=str(elf), elf_sha256=row['elf_sha256']))
    save(WORK/'jobs.json', jobs)
    print(json.dumps(dict(submitted=jid, arm=name, source_pins=361)), flush=True)
seen, completed = set(), set()
deadline = time.monotonic()+5400
while len(completed) != len(jobs):
    assert time.monotonic() < deadline, 'Observer deadline exceeded; retain submissions'
    for row in jobs:
        jid = row['job_id']
        if jid in completed:
            continue
        with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            state = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
        folder = Path(row['folder'])
        stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
        elf = stage/'merlin-golden-nofsm-probe0-probe.elf'
        bit = stage/'xilinx_alveo_u250/firesim.bit'
        if jid not in seen and state['phase'] == 'RUNNING' and elf.is_file() and bit.is_file():
            objects = {name:dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                for name, path in [('elf', elf), ('bitstream', bit)]}
            assert objects['elf']['sha256'] == row['elf_sha256'] and objects['bitstream']['sha256'] == BIT
            save(folder/'actual_staged_identity.json', dict(job_id=jid, phase=state['phase'], observed_before_teardown=True, objects=objects))
            seen.add(jid)
            print(json.dumps(dict(staged_identity_preserved=jid)), flush=True)
        if state['state'] in ('DONE', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            save(folder/'queue_terminal.json', state)
            completed.add(jid)
            print(json.dumps(dict(terminal=state, staged_identity_preserved=jid in seen)), flush=True)
    if len(completed) != len(jobs):
        time.sleep(3)
assert seen == completed, 'Actual staged identity missing; retain explicit failure'
