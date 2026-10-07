"""Root stock complete paired convolution cost admission and identity capture."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight

WORK = Path(__file__).resolve().parent/'resident_tail_stock'
PACKET = Path('/scratch/agustin/tmp/gemmini-resident-tail-before-last-20261007/docs/perf_records/current_stationary_B_tail_complete_capsules.json')
PACKET_SHA = '9fa6cea3048351b3d1c072b9df488e9f1081831b1feedf0d8fae24caf5987a6d'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

WORK.mkdir(exist_ok=True)
assert not any(WORK.iterdir()), 'Do not mutate existing admission/submission'
assert sha(PACKET)==PACKET_SHA
packet=json.loads(PACKET.read_text())
assert packet['functional_status']=='PASS' and len(packet['flat_file_pins'])==350
for row in packet['flat_file_pins']:
    assert sha(row['path'])==row['sha256'],row['path']
parser_row, = [x for x in packet['flat_file_pins'] if x['path'].endswith('/parse_stock.py')]
parser_path=Path(parser_row['path'])
assert sha(parser_path)==parser_row['sha256']
spec=importlib.util.spec_from_file_location('resident_tail_parser',parser_path)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
original={name:packet['cases'][name] for name in ['control','tail']}
assert original['control']['common_addresses']==original['tail']['common_addresses']
for name,row in packet['cases'].items():
    assert sha(row['elf']['path'])==row['elf']['sha256']
    assert sha(row['result']['path'])==row['result']['sha256']
    result=json.loads(Path(row['result']['path']).read_text())
    assert result['status']=='pass' and result['nofsm_status']=='pass'
    assert result['run_completed'] and result['run_returncode']==0
    assert result['elf_sha256']==row['elf']['sha256']
    assert result['outputs_checked']==row['outputs_checked']
    assert result['guards_checked']==row['guards_checked']==8192
    assert result['immutable_input_bytes_checked']==row['immutable_input_bytes_checked']
    observed=parser.parse(result['stdout'], outputs=row['outputs_checked']//2)
    assert observed['cycles']==row['cycles']
    assert result['strict_spike']['status']=='pass'
proof=json.loads(Path(packet['source_proof']['path']).read_text())
save(WORK/'root_admission.json',dict(schema='root_stationary_B_tail_pair_admission_v1',status='pass',
    source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,pins_revalidated=350,
    common_operand_addresses=original['control']['common_addresses'],
    original_full_cost=True,source_outputs=50176,guards=8192,immutable_inputs=2384384,
    original_source_K_order=True,final_nofsm=True,all_setup_fences_compute_and_stores_decoder_timed=True,
    prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN',
    typed_route='default-off typed retained flat spatial planes with one shorttile and at least two full tiles; source per-output K fixed',
    stock_parser=parser_row,
    checks='Every original two-store source output, guard,immutableinput; mandatory positive exact single mcycle/PASS protocol; separate independent non-square spatial/channel tails'))
environment=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
jobs=[]
for name,row in original.items():
    folder=WORK/name
    folder.mkdir()
    elf=Path(row['elf']['path'])
    flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',elf,ALIAS,TAR,folder/'preflight')
    assert flight['elf_sha256']==row['elf']['sha256'] and flight['elf_nofsm_audit']['status']=='pass'
    argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock',
        '--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',str(elf),'--hw-config',ALIAS,
        '--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','600','--project','gemmini-golden-nofsm']
    submitted=subprocess.run(argv,env=environment,capture_output=True,text=True)
    save(folder/'queue_submission.json',dict(argv=argv,environment_keys=sorted(environment),returncode=submitted.returncode,
        stdout=submitted.stdout,stderr=submitted.stderr,source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,
        source_pins_revalidated=350,prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN'))
    assert submitted.returncode==0
    match=re.search(r'job_id=(\d+)',submitted.stdout)
    assert match
    jid=int(match.group(1))
    jobs.append(dict(name=name,job_id=jid,folder=str(folder),elf=str(elf),elf_sha256=row['elf']['sha256']))
    save(WORK/'jobs.json',jobs)
    print(json.dumps(dict(submitted=jid,arm=name,pins=350)),flush=True)
seen,completed=set(),set()
deadline=time.monotonic()+5400
while len(completed)!=len(jobs):
    assert time.monotonic()<deadline,'Observer deadline exceeded; retain submissions'
    for row in jobs:
        jid=row['job_id']
        if jid in completed:
            continue
        with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            state=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
        folder=Path(row['folder'])
        stage=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
        elf=stage/'merlin-golden-nofsm-probe0-probe.elf'
        bit=stage/'xilinx_alveo_u250/firesim.bit'
        if jid not in seen and state['phase']=='RUNNING' and elf.is_file() and bit.is_file():
            objects={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
                for name,path in [('elf',elf),('bitstream',bit)]}
            assert objects['elf']['sha256']==row['elf_sha256'] and objects['bitstream']['sha256']==BIT
            save(folder/'actual_staged_identity.json',dict(job_id=jid,phase=state['phase'],observed_before_teardown=True,objects=objects))
            seen.add(jid)
            print(json.dumps(dict(staged_identity_preserved=jid)),flush=True)
        if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
            save(folder/'queue_terminal.json',state)
            completed.add(jid)
            print(json.dumps(dict(terminal=state,staged_identity_preserved=jid in seen)),flush=True)
    if len(completed)!=len(jobs):
        time.sleep(3)
assert seen==completed,'Actual staged identity missing; retain explicit failure'
