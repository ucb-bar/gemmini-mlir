"""Admit the source-bound current champion diagnostic before queue submission."""
from pathlib import Path
import hashlib, importlib.util, json, re, sqlite3, subprocess, time
from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'current2101_stock_profile'
OWNER = Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007')
PACKET = OWNER / 'docs/perf_records/current2101_boundary_profile_qualification.json'
PACKET_SHA = '9ee3b15d5324bbfad9f4429f97020316222f4e5fe68e300e887da81eebd763f2'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

WORK.mkdir(exist_ok=True)
assert not (WORK / 'job.json').exists(), 'Do not repeat a submitted diagnostic'
assert sha(PACKET) == PACKET_SHA
p = json.loads(PACKET.read_text())
assert p['status'] == 'qualified_diagnostic_ready_for_root_review'
assert isinstance(p['pins'], list) and len(p['pins']) == 358
for row in p['pins']:
    path = Path(row['path'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
for key in ['semantic_object_identity', 'baseline_link_byte_exact',
            'all19_current_dense_and_both_flat_winners_retained',
            'fresh_native_original1000_exact', 'strict_spike_original1000_exact',
            'final_elf_zero_fsm', 'typed_source_host_undefined_actual_entry_join',
            'profile_conserved']:
    assert p[key] is True, key
gate = p['original_accuracy_gate']
assert gate['pass_gate'] and gate['elements'] == 1000 and gate['atol'] == gate['rtol'] == 0
assert gate['raw_output_sha256'] == '0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787'
elf = Path(p['profile_elf'])
assert sha(elf) == p['profile_elf_sha256']
assert sha(p['baseline_elf']) == p['baseline_elf_sha256']
assert sha(p['profile_manifest']) == p['profile_manifest_sha256']
assert sha(p['profile_parser']) == p['profile_parser_sha256']
manifest = json.loads(Path(p['profile_manifest']).read_text())
assert manifest['accepted_elf'] == p['baseline_elf']
assert manifest['accepted_elf_sha256'] == manifest['baseline_reproduced_sha256'] == p['baseline_elf_sha256']
assert manifest['baseline_unprofiled_stock_job'] == p['baseline_job'] == 2101
assert manifest['baseline_unprofiled_firesim_cycles'] == p['baseline_stock_cycles'] == 28728702
assert len(manifest['semantic_object_sha256']) == 12
for path, digest in manifest['semantic_object_sha256'].items():
    assert sha(path) == digest
join_path = OWNER / 'out/current2101_profile/actual_entry_join.json'
join = json.loads(join_path.read_text())
assert join['source_order_exact'] and join['all71_wrap_to_real_calls_exact']
assert len(join['boundaries']) == len(manifest['boundaries']) == 71
for index, (actual, declared) in enumerate(zip(join['boundaries'], manifest['boundaries'], strict=True)):
    assert actual['id'] == declared['id'] == index
    if declared['category'] == 'classifier_primitive':
        assert actual['host_undefined'] is None and actual['source_callee'] is None
        shim = Path(manifest['classifier_shim_source_path'])
        assert sha(shim) == manifest['classifier_shim_source_sha256']
        assert declared['symbol'] in shim.read_text()
        disassembly = (OWNER / 'out/current2101_profile/wrapper_disassembly.txt').read_text()
        wrapper = disassembly.split('<__wrap_' + declared['symbol'] + '>:', 1)[1]
        assert '<' + declared['symbol'] + '>' in wrapper.split('\n\n', 1)[0]
    else:
        assert actual['host_undefined'] == declared['symbol']
    assert actual['declared_void_pointer_arity'] == declared['pointer_arity']
    assert actual['wrapper_calls_actual_selected_boundary'] and actual['exact_executed_boundary_count'] == 1
spec = importlib.util.spec_from_file_location('mlir_oot.qualified_current_profile', p['profile_parser'])
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
spike_path = OWNER / 'out/current2101_profile/spike.log'
spike = spike_path.read_text().replace('\r', '')
assert spike.splitlines().count('DONE') == 1
assert spike.splitlines().count('METRIC memref_rank_mismatch 0') == 1
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', spike, re.M) == [('1000', '4000', gate['raw_output_sha256'])]
assert parser.parse_profile(spike, manifest) == p['spike_profile']
audit = audit_elf(elf.read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
admission = dict(schema='root_current2101_profile_admission_v1', status='PASS',
                 source_packet=str(PACKET), source_packet_sha256=PACKET_SHA,
                 source_pins_revalidated=358, elf=str(elf), elf_sha256=sha(elf),
                 baseline_job=2101, baseline_cycles=28728702,
                 original_gate=gate, source_boundaries=71,
                 actual_entry_join=dict(path=str(join_path), sha256=sha(join_path)),
                 profile_parser=dict(path=p['profile_parser'], sha256=p['profile_parser_sha256']),
                 profile_manifest=dict(path=p['profile_manifest'], sha256=p['profile_manifest_sha256']),
                 nofsm_audit=audit, manifest_semantic_object_count=12,
                 preliminary_refusal='Admission metadata corrections before any queue submission: baseline accepted_elf is separate from profile ELF; 12 original manifest objects plus new timer; classifier primitive joins via separately pinned shim and actual wrapper call, not host undefined callee.',
                 scope='Boundary diagnostic only; callback includes CPU issue/adapters/readout/DMA/waits. Pure accelerator utilization and causal optimization savings UNKNOWN.')
save(WORK / 'root_admission.json', admission)
flight = preflight(Path('/scratch2/agustin/wt/chipyard-stock'),
                   'merlin-golden-nofsm-probe', elf, ALIAS, TAR, WORK / 'preflight')
assert flight['hwdb_artifact_sha256'] == HWDB
assert flight['elf_nofsm_audit']['status'] == 'pass'
env = json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env) == {'PATH', 'SHELL', 'TERM', 'SSH_AUTH_SOCK'}
argv = ['/usr/local/bin/firesim-queue', 'runworkload-full', '--background',
        '--chipyard', '/scratch2/agustin/wt/chipyard-stock',
        '--workload', 'merlin-golden-nofsm-probe', '--bootbinary', 'probe.elf',
        '--stage-from', str(elf), '--hw-config', ALIAS,
        '--hwdb-config-artifact', flight['hwdb_artifact'], '--priority', '0',
        '--timeout', '1800', '--project', 'gemmini-golden-nofsm']
result = subprocess.run(argv, env=env, capture_output=True, text=True)
save(WORK / 'queue_submission.json', dict(argv=argv, returncode=result.returncode,
     stdout=result.stdout, stderr=result.stderr, root_admission_sha256=sha(WORK / 'root_admission.json')))
assert result.returncode == 0
match = re.search(r'job_id=(\d+)', result.stdout)
assert match
jid = int(match.group(1))
save(WORK / 'job.json', dict(job_id=jid, elf=str(elf), elf_sha256=sha(elf)))
print(json.dumps(dict(submitted=jid, source_pins=358, diagnostic_only=True)), flush=True)
seen = False
deadline = time.monotonic() + 5400
while True:
    assert time.monotonic() < deadline, 'Observer expired; do not cancel or infer terminal'
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        state = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (jid,)).fetchone())
    stage = Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    staged_elf = stage / 'merlin-golden-nofsm-probe0-probe.elf'
    bit = stage / 'xilinx_alveo_u250/firesim.bit'
    if not seen and state['phase'] == 'RUNNING' and staged_elf.is_file() and bit.is_file():
        objects = {key: dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
                   for key, path in [('elf', staged_elf), ('bitstream', bit)]}
        assert objects['elf']['sha256'] == sha(elf) and objects['bitstream']['sha256'] == BIT
        save(WORK / 'actual_staged_identity.json', dict(job_id=jid, phase='RUNNING', observed_before_teardown=True, objects=objects))
        seen = True
        print(json.dumps(dict(staged=jid)), flush=True)
    if state['state'] in ('DONE', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
        save(WORK / 'queue_terminal.json', state)
        print(json.dumps(dict(terminal=state, actual_stage_captured=seen)), flush=True)
        assert seen
        break
    time.sleep(3)
