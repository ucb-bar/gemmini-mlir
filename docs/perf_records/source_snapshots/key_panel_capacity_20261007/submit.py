"""Root-owned stock decision for capacity-derived complete residual batches."""
from pathlib import Path
import hashlib, importlib.util, json, re, sqlite3, subprocess, time
from mlir_oot.golden_firesim_preflight import preflight
ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'stock'
PACKET = ROOT.parents[1] / 'docs/perf_records/key_panel_capacity_complete_qualification_20261007.json'
EXPECTED = '75947dde636c45959f495c52ab493d23ceebec5590f1c38cd55519604e56e04f'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2) + '\n')

WORK.mkdir(exist_ok=False)
assert sha(PACKET) == EXPECTED
p = json.loads(PACKET.read_text())
assert p['status'] == 'PASS' and p['control_actual2101_kernel_and_adapter_byte_identical']
assert len(p['pins']) == p['source_pins_revalidated'] == 98
for path, digest in p['pins'].items():
    assert sha(path) == digest, path
independent = p['cases'][1]
assert independent['all65536_pairs_covered'] and independent['elements'] == 133120
case = p['cases'][0]
assert case['original_capture'] and case['same_elf_except_selector'] and case['elements'] == 802816
elfs = [Path(a['elf']) for a in case['arms']]
raw = [elf.read_bytes() for elf in elfs]
assert len(raw[0]) == len(raw[1])
assert [i for i, (x, y) in enumerate(zip(*raw, strict=True)) if x != y] == [case['selector_offset']]
assert (raw[0][case['selector_offset']], raw[1][case['selector_offset']]) == (0, 1)
parser_path = ROOT.parents[1] / 'tests/predictor_key_capsule_protocol.py'
spec = importlib.util.spec_from_file_location('key_original_protocol', parser_path)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
for arm in case['arms']:
    assert sha(arm['elf']) == arm['elf_sha256'] and arm['nofsm_audit']['status'] == 'pass'
    parser.parse(Path(arm['elf']).with_name('spike.stdout').read_text(), arm=arm['arm'], elements=802816)
save(WORK / 'root_admission.json', dict(source_packet=str(PACKET), source_packet_sha256=EXPECTED,
     pins_revalidated=98, original_elements=802816, all_source_pairs_independently_pass=True,
     same_elf_selector_only=True, actual_current2101_control_reproduced=True,
     parser=dict(path=str(parser_path), sha256=sha(parser_path)),
     scope='Matched complete ranked residual cost including scratch/guards/setup/seed/readback/reload/finalfence; no pre-labelled winner or whole forecast'))
env = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
jobs = []
for arm in case['arms']:
    folder = WORK / f"arm{arm['arm']}"
    folder.mkdir()
    elf = Path(arm['elf'])
    flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'), 'merlin-golden-nofsm-probe', elf, ALIAS, TAR, folder / 'preflight')
    assert flight['elf_nofsm_audit']['status'] == 'pass'
    argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background', '--chipyard',
            '/scratch2/agustin/wt/chipyard-stock', '--workload', 'merlin-golden-nofsm-probe',
            '--bootbinary', 'probe.elf', '--stage-from', str(elf), '--hw-config', ALIAS,
            '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0', '--timeout', '600',
            '--project', 'gemmini-golden-nofsm']
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    save(folder / 'queue_submission.json', dict(argv=argv, returncode=result.returncode, stdout=result.stdout,
         stderr=result.stderr, source_packet_sha256=EXPECTED, root_admission_sha256=sha(WORK / 'root_admission.json')))
    assert result.returncode == 0
    match = re.search(r'job_id=(\d+)', result.stdout)
    assert match
    jid = int(match.group(1))
    jobs.append(dict(arm=arm['arm'], factor=arm['factor'], job_id=jid, folder=str(folder), elf=str(elf), elf_sha256=sha(elf)))
    save(WORK / 'jobs.json', jobs)
    print(json.dumps(dict(submitted=jid, factor=arm['factor'])), flush=True)
seen, done = set(), set()
deadline = time.monotonic() + 3600
while len(done) < len(jobs):
    assert time.monotonic() < deadline, 'Observer expired; keep submissions'
    for row in jobs:
        jid = row['job_id']
        if jid in done:
            continue
        with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
        stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
        elf, bit = stage / 'merlin-golden-nofsm-probe0-probe.elf', stage / 'xilinx_alveo_u250/firesim.bit'
        folder = Path(row['folder'])
        if jid not in seen and terminal['phase'] == 'RUNNING' and elf.is_file() and bit.is_file():
            objects = {key: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                       for key, path in [('elf', elf), ('bitstream', bit)]}
            assert objects['elf']['sha256'] == row['elf_sha256'] and objects['bitstream']['sha256'] == BIT
            save(folder / 'actual_staged_identity.json', dict(job_id=jid, phase='RUNNING', observed_before_teardown=True, objects=objects))
            seen.add(jid)
            print(json.dumps(dict(staged=jid)), flush=True)
        if terminal['state'] in ('DONE', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            save(folder / 'queue_terminal.json', terminal)
            done.add(jid)
            print(json.dumps(dict(terminal=terminal, stage_captured=jid in seen)), flush=True)
    if len(done) < len(jobs):
        time.sleep(3)
assert seen == done
