"""Submit qualified current2076 diagnostic; preserve actual staged identities."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.golden_device_profile import parse_profile

ROOT = Path(__file__).resolve().parent
WORK = ROOT/'current2076_profile_v2'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

qpath = WORK/'qualification.json'
q = json.loads(qpath.read_text())
assert q['status']=='pass' and q['diagnostic_only'] and q['source_bound_155_calls'] and q['semantic_objects_unchanged']
assert q['original_output_gate']['count']==256000 and q['original_output_gate']['source_bitexact'] and q['original_output_gate']['torch_allclose']
for path,digest in q['pins'].items():
    assert sha(path)==digest,path
manifest=json.loads((WORK/'profile_manifest.json').read_text())
for path,digest in manifest['semantic_object_sha256'].items():
    assert sha(path)==digest,path
profile=parse_profile((WORK/'spike.log').read_text(),manifest)
assert [row[1] for row in profile['events']]==manifest['expected_execution_id_order']
elf=Path(q['elf_path'])
assert sha(elf)==q['elf_sha256']=='b394d167ae97fca18cb86e72c6a6f44d3fca1f815ae5f777c6b4804329c560a8'
flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',elf,ALIAS,TAR,WORK/'preflight')
environment=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(environment)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock','--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',str(elf),'--hw-config',ALIAS,'--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','1800','--project','gemmini-golden-nofsm']
result=subprocess.run(argv,env=environment,capture_output=True,text=True)
receipt=dict(argv=argv,environment_keys=sorted(environment),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,qualification=str(qpath),qualification_sha256=sha(qpath),source_pins_revalidated=len(q['pins']),semantic_objects_revalidated=len(manifest['semantic_object_sha256']),scope=manifest['scope'],diagnostic_not_new_champion=True)
save(WORK/'queue_submission.json',receipt)
assert result.returncode==0
match=re.search(r'job_id=(\d+)',result.stdout)
assert match
jid=int(match.group(1))
save(WORK/'queue_job.json',dict(job_id=jid))
print(json.dumps(dict(submitted_profile=jid,source_bound_calls=155)),flush=True)
seen=False
deadline=time.monotonic()+1900
while time.monotonic()<deadline:
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        state=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    stage=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
    staged_elf,bit=stage/'merlin-golden-nofsm-probe0-probe.elf',stage/'xilinx_alveo_u250/firesim.bit'
    if not seen and state['phase']=='RUNNING' and staged_elf.is_file() and bit.is_file():
        objects={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for name,path in [('elf',staged_elf),('bitstream',bit)]}
        assert objects['elf']['sha256']==q['elf_sha256'] and objects['bitstream']['sha256']==BIT
        save(WORK/'actual_staged_identity.json',dict(job_id=jid,phase=state['phase'],observed_before_teardown=True,objects=objects))
        seen=True
        print(json.dumps(dict(staged_identity_preserved=jid)),flush=True)
    if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
        save(WORK/'queue_terminal.json',state)
        print(json.dumps(dict(terminal=state,staged_identity_preserved=seen)),flush=True)
        assert seen
        break
    time.sleep(3)
else:
    raise RuntimeError('Profile identity observer deadline exceeded; job submission remains recorded')
