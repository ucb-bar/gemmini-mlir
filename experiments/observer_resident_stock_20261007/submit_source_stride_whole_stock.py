"""Admit the fresh-reproduced original ResNet whole and capture stock staging."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time

import numpy as np
from mlir_oot.golden_firesim_preflight import preflight

WORK=Path(__file__).resolve().parent/'source_stride_whole_stock'
PACKET=Path('/scratch/agustin/tmp/gemmini-source-stride-retained-20261007/docs/perf_records/paired_source_stride_retained_current2081_whole_qualification.json')
PACKET_SHA='b80143c744d508b98a2673579fea5dcaa0bb928fae6c07c0d0ba1d88714c146a'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
CANON=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
FRESH=Path('/scratch/agustin/tmp/gemmini-source-stride-retained-restored-20261007')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def bound_json(row):
    assert sha(row['path'])==row['sha256'],row['path']
    return json.loads(Path(row['path']).read_text())

WORK.mkdir(exist_ok=False)
assert sha(PACKET)==PACKET_SHA
packet=json.loads(PACKET.read_text());assert packet['status']=='PASS' and len(packet['pins'])==1959
for path,digest in packet['pins'].items():assert sha(path)==digest,path
assert packet['control_reproduction_byte_identical'] and packet['all52_adapter_bytes_identical']
assert packet['all51_unselected_primitive_objects_identical'] and packet['original_host_runtime_weights_and_other_objects_unchanged']
assert sha(packet['control_elf']['path'])==packet['control_elf']['sha256']=='335bb2bc7688e835fa3e5cfeeac0eaf1ee91ecf6e02de58f0d8080f82471a4ee'
repro=bound_json(packet['fresh_normal_api_reproduction']);assert repro['status']=='PASS'
old=Path(repro['original_control_manifest']).parent
control=Path(repro['results']['control']['manifest']).parent
selected=Path(repro['results']['candidate']['manifest']).parent
folders=sorted(p.parent.name for p in old.glob('gemmini_exact_requant_*/kernel.o'))
assert len(folders)==52
changed=[]
for name in folders:
    assert (old/name/'adapter.o').read_bytes()==(control/name/'adapter.o').read_bytes()==(selected/name/'adapter.o').read_bytes()
    assert (old/name/'kernel.o').read_bytes()==(control/name/'kernel.o').read_bytes()
    if (control/name/'kernel.o').read_bytes()!=(selected/name/'kernel.o').read_bytes():changed.append(name)
assert changed==['gemmini_exact_requant_24']
assert sha(control/'requant.o')==repro['results']['control']['object_sha256']=='9faece1a5e4f78f2ba7d2ce1ab277e7c5719101bf582484e67a92e9612735abc'
assert sha(selected/'requant.o')==repro['results']['candidate']['object_sha256']=='d3f4bf830c07c4166a8fae87911197da9b3ab0541baf0c38b60b52b8bb482a1e'
source_identity={}
for name in ['captured_requant_bundle.py','golden_resident_conv.py','golden_resident_stripe_conv.py',
    'golden_device_lower.py','golden_gemm.py','execute_wave_estimator.py','ir/gemmini_dialect.py']:
    path=FRESH/'mlir_oot'/name
    assert path.read_bytes()==(CANON/'mlir_oot'/name).read_bytes(),name
    source_identity[name]=sha(path)
entry=bound_json(packet['actual_selected_entry_and_relocation_closure'])
assert entry['status']=='PASS' and entry['selected_symbol_entry_count']==1 and entry['selected_primitive_link_relocation_equivalent']
assert entry['original1000_full_digest_match'] and len(entry['branch_relocation_witness'])==10
native=Path(packet['arms']['controlled']['native_output']['path']);assert sha(native)==packet['arms']['controlled']['native_output']['sha256']
previous=Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/docs/perf_records/predictor_key_current2071_whole_qualification.json')
assert sha(previous)=='49e7ef0e3360a8409eb0322d4062611e9fb1b60277f15e2120cfbad66512a6fa'
old_gate=json.loads(previous.read_text())['controlled_whole_gate']['native']
golden_path=Path(old_gate['original_golden']);assert sha(golden_path)==old_gate['original_golden_sha256']
output=np.load(native,allow_pickle=False);golden=np.load(golden_path,allow_pickle=False)
assert output.dtype==golden.dtype==np.float32 and output.size==golden.size==1000
assert output.astype('<f4',copy=False).tobytes()==golden.astype('<f4',copy=False).tobytes()
digest=hashlib.sha256(output.astype('<f4',copy=False).tobytes()).hexdigest()
assert digest=='0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787'
for arm in packet['arms'].values():
    assert arm['original_all1000_words_exact'] and arm['nofsm_status']=='pass'
    console=Path(arm['strict_console']['path']).read_text().replace('\r','')
    assert console.splitlines().count('DONE')==1 and console.splitlines().count('METRIC memref_rank_mismatch 0')==1
    assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',console,re.M)==[('1000','4000',digest)]
elf=Path(packet['candidate_elf']['path']);assert sha(elf)==packet['candidate_elf']['sha256']==entry['elf_sha256']
admission=dict(schema='root_source_stride_original_ResNet_whole_admission_v1',status='pass',source_packet=str(PACKET),
    source_packet_sha256=PACKET_SHA,pins_revalidated=1959,candidate_elf=str(elf),candidate_elf_sha256=sha(elf),
    original_output_digest=digest,original_count=1000,original_golden=str(golden_path),original_golden_sha256=sha(golden_path),
    original_atol=0,original_rtol=0,independent_all1000_native_source_exact=True,
    fresh52_kernel_adapter_byte_identity=True,changed_fresh_kernel=changed[0],fresh_source_canonical_identity=source_identity,
    actual_selected_entry_closure=entry,whole_cycles='UNKNOWN',prelabel_winner='UNKNOWN',no_section_gain_sum=True)
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
    source_pins_revalidated=1959,root_admission_sha256=sha(WORK/'root_admission.json'),whole_cycles='UNKNOWN'))
assert submitted.returncode==0
match=re.search(r'job_id=(\d+)',submitted.stdout);assert match
jid=int(match.group(1));save(WORK/'queue_job.json',dict(job_id=jid))
print(json.dumps(dict(submitted_original_ResNet_whole=jid,source_pins=1959,fresh52_independently_closed=True)),flush=True)
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
        print(json.dumps(dict(terminal=state,staged_identity_preserved=seen)),flush=True);assert seen;break
    time.sleep(3)
else:raise RuntimeError('Observer timeout; submission retained')
