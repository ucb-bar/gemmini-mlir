"""Close the original Tiny whole protocol and stock identity, without a fit."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

import numpy as np

WORK=Path(__file__).resolve().parent/'rne_zero_whole_stock'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
HWDB='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

admission_path=WORK/'root_admission.json';a=json.loads(admission_path.read_text())
packet_path=Path(a['source_packet']);assert sha(packet_path)==a['source_packet_sha256']
packet=json.loads(packet_path.read_text());assert len(packet['pins'])==260
for path,digest in packet['pins'].items():assert sha(path)==digest,path
validation_path=Path(a['standard_reference_validation'])
assert sha(validation_path)==a['standard_reference_validation_sha256']
validation=json.loads(validation_path.read_text())
jid=json.loads((WORK/'queue_job.json').read_text())['job_id'];assert jid==2085
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
job=Path(f'/scratch/firesim_queue/jobs/{jid}')
definition_path=job/'runworkload-full.json';definition=json.loads(definition_path.read_text())
flight_path=WORK/'preflight/firesim_preflight.json';flight=json.loads(flight_path.read_text())
staged_path=WORK/'actual_staged_identity.json';staged=json.loads(staged_path.read_text())
submit_path=WORK/'queue_submission.json';submit=json.loads(submit_path.read_text())
assert definition['user']=='agustin' and definition['hw_config']==flight['hardware_key']==ALIAS
assert definition['stage_from']==flight['elf']==a['candidate_elf']==packet['candidate_ELF_path']
assert flight['elf_sha256']==sha(a['candidate_elf'])==a['candidate_elf_sha256']==validation['elf_sha256']
assert submit['source_packet_sha256']==sha(packet_path) and submit['root_admission_sha256']==sha(admission_path)
assert staged['job_id']==jid and staged['phase']=='RUNNING' and staged['observed_before_teardown']
assert staged['objects']['elf']['sha256']==flight['elf_sha256'] and staged['objects']['bitstream']['sha256']==BIT
assert definition['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])==HWDB
assert flight['bitstream_sha256']==sha(flight['bitstream_path'])==TAR
audit=flight['elf_nofsm_audit'];assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
raw=(job/'simulation/sim_slot_0/uartlog').read_bytes();text=raw.decode().replace('\r','')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert text.splitlines().count('DONE')==1 and text.splitlines().count('METRIC memref_rank_mismatch 0')==1
values=re.findall(r'^METRIC cycles (\d+)$',text,re.M);assert len(values)==1 and int(values[0])>0
reference_path=Path(validation['reference_path']);golden_path=Path(validation['torch_golden_path'])
assert sha(reference_path)==validation['reference_sha256'] and sha(golden_path)==validation['torch_golden_sha256']
output=np.load(reference_path,allow_pickle=False);golden=np.load(golden_path,allow_pickle=False)
assert output.dtype==np.float32 and output.size==golden.size==256000
assert np.allclose(output.reshape(-1),golden.reshape(-1),atol=0.03125,rtol=0.02)
digest=hashlib.sha256(output.astype('<f4',copy=False).tobytes()).hexdigest()
assert digest==a['original_output_digest']==validation['spike_output_sha256']
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',text,re.M)==[('256000','1024000',digest)]
cycles=int(values[0]);archive=WORK/'stock2085_uart.txt';archive.write_bytes(raw)
result=dict(schema='root_rne_zero_Tiny_stock2085_whole_terminal_v1',status='verified_complete_original_whole',
    job_id=jid,terminal=terminal,control_job=2076,control_cycles=380396343,candidate_cycles=cycles,
    saving_cycles=380396343-cycles,saving_fraction=1-cycles/380396343,
    target_cycles=300000000,remaining_target_gap=cycles-300000000,
    original_output_gate=dict(count=256000,raw_sha256=digest,compiled_source_bitexact=True,
        original_torch_atol=0.03125,original_torch_rtol=0.02,original_torch_pass=True,
        golden=str(golden_path),golden_sha256=sha(golden_path)),
    source_packet=str(packet_path),source_packet_sha256=sha(packet_path),source_pins_revalidated=260,
    actual_staged_identity=staged,final_nofsm=audit,hardware_alias=ALIAS,hwdb_sha256=HWDB,
    bitstream_archive_sha256=TAR,actual_staged_bit_sha256=BIT,
    elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],
    scope='One uninstrumented original Tiny whole METRIC observation. Explicit source-bound zero observer is the sole model.o change; eleven other objects and all155 target bindings unchanged. Current2076 control ELF was reproduced exactly. Instructions rose while section cycles fell; no additive section estimate, pure utilization claim or fitted whole model. Final simulator PASSED counter includes validation/setup and is not this performance metric.',
    pins={str(p):sha(p) for p in [admission_path,packet_path,validation_path,definition_path,
        flight_path,staged_path,submit_path,archive,Path(__file__)]},token_usage_available=False,token_usage=None)
(WORK/'stock2085_terminal.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['status','candidate_cycles','saving_cycles','saving_fraction','remaining_target_gap','elapsed_queue_seconds']}),flush=True)
