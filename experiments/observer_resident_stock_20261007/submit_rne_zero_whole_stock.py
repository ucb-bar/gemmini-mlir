"""Root-only original Tiny whole-model stock admission with immutable staging."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

import numpy as np
from mlir_oot.golden_firesim_preflight import preflight

WORK=Path(__file__).resolve().parent/'rne_zero_whole_stock'
PACKET=Path('/scratch/agustin/tmp/gemmini-rne-observer-cells-20261007/docs/perf_records/rne_zero_observer_whole_2076_20261007/receipt.json')
PACKET_SHA='f61d89e0d3662df9e402700ae164a01c1c42d023fbf9af67db120b3bdcc8525f'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

WORK.mkdir(exist_ok=False)
assert sha(PACKET)==PACKET_SHA
packet=json.loads(PACKET.read_text())
assert packet['status']=='pass' and len(packet['pins'])==260
for path,digest in packet['pins'].items():
    assert sha(path)==digest,path
assert packet['baseline2076_ELF_byteidentical'] and packet['only_changed_linked_object']=='model.o'
assert packet['unchanged_nonmodel_leaves']==11 and packet['all155devicebindings']
assert packet['original_compiled_output_words']==256000 and packet['native_original22context_i8_words']==991232
validation_path=Path(packet['standard_reference_validation'])
assert sha(validation_path)==packet['standard_reference_validation_sha256']
validation=json.loads(validation_path.read_text())
elf=Path(packet['candidate_ELF_path'])
assert sha(elf)==packet['candidate_ELF_sha256']==validation['elf_sha256']
assert validation['status']=='pass' and validation['exit_code']==0 and validation['spike_full_output_match']
assert validation['torch_atol']==0.03125 and validation['torch_rtol']==0.02 and validation['torch_allclose']
reference=Path(validation['reference_path']);golden_path=Path(validation['torch_golden_path'])
assert sha(reference)==validation['reference_sha256'] and sha(golden_path)==validation['torch_golden_sha256']
output=np.load(reference,allow_pickle=False);golden=np.load(golden_path,allow_pickle=False)
assert output.dtype==np.float32 and output.size==golden.size==256000
assert np.allclose(output.reshape(-1),golden.reshape(-1),atol=0.03125,rtol=0.02)
digest=hashlib.sha256(output.astype('<f4',copy=False).tobytes()).hexdigest()
assert digest==validation['spike_output_sha256']=='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
spike_path=Path(validation['spike_console_path'])
assert sha(spike_path)==validation['spike_console_sha256']
console=spike_path.read_text().replace('\r','')
assert console.splitlines().count('DONE')==1 and console.splitlines().count('METRIC memref_rank_mismatch 0')==1
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',console,re.M)==[('256000','1024000',digest)]
admission=dict(schema='root_rne_zero_original_Tiny_whole_admission_v1',status='pass',
    source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,pins_revalidated=260,
    standard_reference_validation=str(validation_path),standard_reference_validation_sha256=sha(validation_path),
    candidate_elf=str(elf),candidate_elf_sha256=sha(elf),original_output_digest=digest,
    original_count=256000,original_torch_atol=0.03125,original_torch_rtol=0.02,
    independent_original_torch_allclose=True,strict_full_original_source_output=True,
    whole_cycles='UNKNOWN',prelabel_winner='UNKNOWN',no_section_gain_sum=True)
save(WORK/'root_admission.json',admission)
flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',elf,ALIAS,TAR,WORK/'preflight')
assert flight['elf_nofsm_audit']['status']=='pass'
environment=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock',
    '--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',str(elf),'--hw-config',ALIAS,
    '--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','1800','--project','gemmini-golden-nofsm']
submitted=subprocess.run(argv,env=environment,capture_output=True,text=True)
save(WORK/'queue_submission.json',dict(argv=argv,environment_keys=sorted(environment),returncode=submitted.returncode,
    stdout=submitted.stdout,stderr=submitted.stderr,source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,
    source_pins_revalidated=260,root_admission_sha256=sha(WORK/'root_admission.json'),
    prospective_winner='UNKNOWN',whole_cycles='UNKNOWN'))
assert submitted.returncode==0
match=re.search(r'job_id=(\d+)',submitted.stdout)
assert match
jid=int(match.group(1));save(WORK/'queue_job.json',dict(job_id=jid))
print(json.dumps(dict(submitted_original_Tiny_whole=jid,source_pins=260)),flush=True)
seen=False;deadline=time.monotonic()+2200
while time.monotonic()<deadline:
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        state=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    stage=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    staged_elf=stage/'merlin-golden-nofsm-probe0-probe.elf';bit=stage/'xilinx_alveo_u250/firesim.bit'
    if not seen and state['phase']=='RUNNING' and staged_elf.is_file() and bit.is_file():
        objects={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
            for name,path in [('elf',staged_elf),('bitstream',bit)]}
        assert objects['elf']['sha256']==admission['candidate_elf_sha256'] and objects['bitstream']['sha256']==BIT
        save(WORK/'actual_staged_identity.json',dict(job_id=jid,phase=state['phase'],observed_before_teardown=True,objects=objects))
        seen=True;print(json.dumps(dict(staged_identity_preserved=jid)),flush=True)
    if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
        save(WORK/'queue_terminal.json',state)
        print(json.dumps(dict(terminal=state,staged_identity_preserved=seen)),flush=True)
        assert seen
        break
    time.sleep(3)
else:
    raise RuntimeError('Observer timeout; submission is retained')
