"""Root stock complete paired convolution cost admission and identity capture."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight

WORK = Path(__file__).resolve().parent/'source_stride_retained_stock'
PACKET = Path('/scratch/agustin/tmp/gemmini-source-stride-retained-20261007/docs/perf_records/paired_source_stride_retained_complete_capsules.json')
PACKET_SHA = 'e23a2b259e2b72bf76584d5d8bb2a8cfbb3a40e779a664f600e8e025d648535c'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

WORK.mkdir(exist_ok=False)
assert sha(PACKET)==PACKET_SHA
packet=json.loads(PACKET.read_text())
assert packet['status']=='PASS' and len(packet['pins'])==292
for path,digest in packet['pins'].items():
    assert sha(path)==digest,path
contract=packet['stock_contract']
assert contract['counter_rows']==1 and contract['required_pass_line']=='PAIRED_CONV_PASS N=50176'
assert contract['checks']==dict(final_source_output_bytes=50176,second_store_output_bytes=50176,immutable_input_bytes=790528,guard_bytes=8192)
original=packet['rows']['original']
assert original['control']['common_operand_addresses']==original['compact_source_stride']['common_operand_addresses']
for row in original.values():
    assert sha(row['elf']['path'])==row['elf']['sha256']
    strict=Path(row['strict_stdout']['path']).read_text().replace('\r','')
    assert strict.splitlines().count(contract['required_pass_line'])==1
    assert len(re.findall(contract['cycle_pattern'],strict,re.M))==1 and 'PAIRED_CONV_FAIL' not in strict
save(WORK/'root_admission.json',dict(status='pass',source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,
    pins_revalidated=292,common_operand_addresses=original['control']['common_operand_addresses'],
    source_contract=contract,prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN',original_full_cost=True))
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
        source_pins_revalidated=292,prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN'))
    assert submitted.returncode==0
    match=re.search(r'job_id=(\d+)',submitted.stdout)
    assert match
    jid=int(match.group(1))
    jobs.append(dict(name=name,job_id=jid,folder=str(folder),elf=str(elf),elf_sha256=row['elf']['sha256']))
    save(WORK/'jobs.json',jobs)
    print(json.dumps(dict(submitted=jid,arm=name,pins=292)),flush=True)
seen,completed=set(),set()
deadline=time.monotonic()+1800
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
