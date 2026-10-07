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

WORK=Path(__file__).resolve().parent/'resident_tail_whole_stock'
PACKET=Path('/scratch/agustin/tmp/gemmini-resident-tail-normal-20261007/docs/perf_records/current_stationary_B_tail_2090_whole_qualification.json')
PACKET_SHA='d383055858257223f8ff3b8b0e1399193a37badaab5ef0d24c83e178ede9882f'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
CANON=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
FRESH=Path('/scratch/agustin/tmp/gemmini-resident-tail-normal-20261007')

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def bound_json(row):
    assert sha(row['path'])==row['sha256'],row['path']
    return json.loads(Path(row['path']).read_text())
WORK.mkdir(exist_ok=False)
assert sha(PACKET)==PACKET_SHA
packet=json.loads(PACKET.read_text());assert packet['status']=='PASS' and len(packet['pins'])==2619
for path,digest in packet['pins'].items():assert sha(path)==digest,path
assert packet['control_reproduction_byte_identical'] and packet['all14_control_partial_link_stages_byte_identical']
assert packet['all_host_runtime_weights_and_other_primitives_unchanged_in_controlled_arm']
assert sha(packet['control_elf']['path'])==packet['control_elf']['sha256']=='8c964c8d437adbec08c9f508a78d425cc76c0d1dd7b57be303ca815e3e154fbc'
repro=bound_json(packet['bundle_closure']);assert repro['status']=='PASS'
assert repro['default_all52_and_aggregate_object_bytes_identical'] and repro['all52_source_numeric_effect_ABI_bindings_and_adapters_identical']
control=Path(repro['control_manifest']).parent;selected=Path(packet['selected_bundle']['path']).parent
folders=sorted(p.parent.name for p in control.glob('gemmini_exact_requant_*/kernel.o'));assert len(folders)==52
changed=[]
for name in folders:
    assert (control/name/'adapter.o').read_bytes()==(selected/name/'adapter.o').read_bytes()
    if (control/name/'kernel.o').read_bytes()!=(selected/name/'kernel.o').read_bytes():changed.append(name)
assert changed==['gemmini_exact_requant_47','gemmini_exact_requant_50']
for row in packet['selected']:
    assert sha(selected/row['symbol_for_audit']/'kernel.o')==row['new_object']
independent=Path(__file__).resolve().parent/'resident_tail_whole_link_review.json'
link_review=json.loads(independent.read_text());assert link_review['status']=='PASS' and link_review['elf_sha256']==packet['candidate_elf']['sha256']
assert [row['body_bytes'] for row in link_review['selected']]==[13570,13262]
assert all(row['all_other_bytes_equal'] and row['actual_entry_count']==1 and len(row['relocation_targets_and_registers'])==11 for row in link_review['selected'])
source_identity={}
for name in ['captured_requant_bundle.py','golden_resident_conv.py','golden_resident_stripe_conv.py','golden_device_lower.py','golden_gemm.py','execute_wave_estimator.py','paired_readout_binding.py','ir/gemmini_dialect.py']:
    path=FRESH/'mlir_oot'/name;assert path.read_bytes()==(CANON/'mlir_oot'/name).read_bytes(),name
    source_identity[name]=sha(path)
entry=bound_json(packet['actual_selected_entry_and_relocation_closure']);assert entry['status']=='PASS'
assert [row['actual_entry_count'] for row in entry['selected']]==[1,1]
native=Path(packet['arms']['controlled']['native_output']['path']);assert sha(native)==packet['arms']['controlled']['native_output']['sha256']
golden_path=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/perf-studies/golden-resnet/closed-recipe/capture/golden.npy')
assert sha(golden_path)=='3ada92201aedf3da3a99441bc43c3b24b18063cbf91ea1173288d9534494f946'
output=np.load(native,allow_pickle=False);golden=np.load(golden_path,allow_pickle=False)
assert output.dtype==golden.dtype==np.float32 and output.size==golden.size==1000
assert output.astype('<f4',copy=False).tobytes()==golden.astype('<f4',copy=False).tobytes()
digest=hashlib.sha256(output.astype('<f4',copy=False).tobytes()).hexdigest()
assert digest=='0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787'
for arm in packet['arms'].values():
    assert arm['all1000_original_words_exact'] and arm['nofsm_status']=='pass'
    console=Path(arm['strict_console']['path']).read_text().replace('\r','')
    assert console.splitlines().count('DONE')==1 and console.splitlines().count('METRIC memref_rank_mismatch 0')==1
    assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',console,re.M)==[('1000','4000',digest)]
elf=Path(packet['candidate_elf']['path']);assert sha(elf)==packet['candidate_elf']['sha256']==entry['elf_sha256']
admission=dict(schema='root_stationary_tail_original_ResNet_whole_admission_v1',status='pass',source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,pins_revalidated=2619,candidate_elf=str(elf),candidate_elf_sha256=sha(elf),
 original_output_digest=digest,original_count=1000,original_golden=str(golden_path),original_golden_sha256=sha(golden_path),original_atol=0,original_rtol=0,independent_all1000_native_source_exact=True,
 fresh52_kernel_adapter_byte_identity=True,changed_fresh_kernels=changed,fresh_source_canonical_identity=source_identity,actual_selected_entry_closure=entry,
 independent_selected_link_review=str(independent),independent_selected_link_review_sha256=sha(independent),whole_cycles='UNKNOWN',prelabel_winner='UNKNOWN',no_section_gain_sum=True)
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
    source_pins_revalidated=2619,root_admission_sha256=sha(WORK/'root_admission.json'),whole_cycles='UNKNOWN'))
assert submitted.returncode==0
match=re.search(r'job_id=(\d+)',submitted.stdout);assert match
jid=int(match.group(1));save(WORK/'queue_job.json',dict(job_id=jid))
print(json.dumps(dict(submitted_original_ResNet_whole=jid,source_pins=2619,fresh52_independently_closed=True)),flush=True)
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
