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

WORK = Path(__file__).resolve().parent/'source_residue_stock'
PACKET = Path('/scratch/agustin/tmp/gemmini-current-source-residue-20261007/docs/perf_records/current_source_residue_complete_capsules.json')
PACKET_SHA = '753e57e4c76df35524b1bce56ef1b5a7ec1a12cab357e5f0716f7619d2449b6b'
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
assert packet['status']=='PASS' and len(packet['pins'])==281
for path,digest in packet['pins'].items():
    assert sha(path)==digest,path
parser_path=Path(packet['stock_parser']['path'])
assert sha(parser_path)==packet['stock_parser']['sha256']
spec=importlib.util.spec_from_file_location('source_residue_parser',parser_path)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
original={name:packet['arms'][name] for name in packet['primary_stock_pair']}
assert list(original)==['control','stride_residue_compact']
assert original['control']['common_operand_addresses']==original['stride_residue_compact']['common_operand_addresses']
for name,row in original.items():
    assert sha(row['elf']['path'])==row['elf']['sha256']
    assert row['outputs_checked']==25088 and row['guard_bytes']==4096 and row['immutable_input_bytes']==2459648
    audit=json.loads(Path(row['audit']['path']).read_text())
    assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    assert audit['elf_sha256']==row['elf']['sha256']
    parser.parse(Path(row['strict_stdout']['path']).read_text(),schedule=name,outputs=25088)
    observed=parser.parse(Path(row['gsim_stdout']['path']).read_text(),schedule=name,outputs=25088)
    assert observed['kernel_cycles']==row['gsim_cycles']
proof=json.loads(Path(packet['prelabel']['path']).read_text())
assert proof['default_source_object_text_identical'] and proof['retention_commands_and_dma_pointers_identical']
assert proof['results']['compact']['command_stream_sha256']==proof['results']['residue']['command_stream_sha256']
for arm in ['default','residue','compact']:
    assert proof['results'][arm]['counts']['gemmini.compute']==36864
    assert proof['results'][arm]['counts']['gemmini.preload']==36864
assert proof['results']['default']['requested_dma_bytes']['A']==409600
assert proof['results']['compact']['requested_dma_bytes']['A']==100352
save(WORK/'root_admission.json',dict(schema='root_current_source_residue_pair_admission_v1',status='pass',
    source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,pins_revalidated=281,
    common_operand_addresses=original['control']['common_operand_addresses'],
    original_full_cost=True,source_outputs=25088,guards=4096,immutable_inputs=2459648,
    original_source_K_order=True,final_nofsm=True,all_setup_fences_compute_and_stores_timed=True,
    prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN',
    source_route='layout/resource-derived row-residue plus command retention, existing explicit default-off options;43 is audit identity only',
    checks='Every original source output, guard,immutableinput; mandatory positive exact single mcycle/PASS protocol; separate independent non-square spatial/channel tails'))
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
        source_pins_revalidated=281,prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN'))
    assert submitted.returncode==0
    match=re.search(r'job_id=(\d+)',submitted.stdout)
    assert match
    jid=int(match.group(1))
    jobs.append(dict(name=name,job_id=jid,folder=str(folder),elf=str(elf),elf_sha256=row['elf']['sha256']))
    save(WORK/'jobs.json',jobs)
    print(json.dumps(dict(submitted=jid,arm=name,pins=281)),flush=True)
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
