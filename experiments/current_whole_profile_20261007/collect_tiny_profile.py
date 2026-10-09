"""Close actual current2076 source-bound stock profile without utilization claims."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re
import sqlite3

from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_device_profile import parse_profile

WORK=Path(__file__).resolve().parent/'current2076_profile_v2'
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
qpath=WORK/'qualification.json'
q=json.loads(qpath.read_text())
manifest_path=WORK/'profile_manifest.json'
manifest=json.loads(manifest_path.read_text())
for path,digest in {**q['pins'],**manifest['semantic_object_sha256']}.items():
    assert sha(path)==digest,path
jid=json.loads((WORK/'queue_job.json').read_text())['job_id']
assert jid==2080
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
job=Path(f'/scratch/firesim_queue/jobs/{jid}')
definition=json.loads((job/'runworkload-full.json').read_text())
flight=json.loads((WORK/'preflight/firesim_preflight.json').read_text())
staged=json.loads((WORK/'actual_staged_identity.json').read_text())
submit=json.loads((WORK/'queue_submission.json').read_text())
assert definition['user']=='agustin' and definition['hw_config']==flight['hardware_key']=='alveo_u250_firesim_gemmini_rocket_stock'
assert definition['stage_from']==q['elf_path']==flight['elf']
assert submit['qualification_sha256']==sha(qpath)
assert q['elf_sha256']==flight['elf_sha256']==sha(q['elf_path'])==staged['objects']['elf']['sha256']
assert definition['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
assert flight['bitstream_sha256']==sha(flight['bitstream_path'])=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
assert staged['job_id']==jid and staged['phase']=='RUNNING' and staged['observed_before_teardown']
assert staged['objects']['bitstream']['sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
assert flight['elf_nofsm_audit']==q['nofsm_audit'] and q['nofsm_audit']['status']=='pass'
uart=job/'simulation/sim_slot_0/uartlog'
raw=uart.read_bytes();text=raw.decode().replace('\r','')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',text,re.M)==[('256000','1024000',q['original_output_gate']['raw_sha256'])]
assert text.splitlines().count('DONE')==1 and text.splitlines().count('METRIC memref_rank_mismatch 0')==1
metrics=re.findall(r'^METRIC cycles (\d+)$',text,re.M);assert len(metrics)==1
profile=parse_profile(text,manifest)
assert [row[1] for row in profile['events']]==manifest['expected_execution_id_order']
ops=list(parse_module(Path(manifest['source_path']).read_text()).walk())
by_id={row['id']:row for row in manifest['boundaries']}
categories=defaultdict(lambda:dict(calls=0,callback_cycles=0,preceding_gap_cycles=0))
events=[]
for event,route in zip(profile['events'],manifest['source_bound_calls'],strict=True):
    ordinal,idx,gap,callback=event
    assert ordinal==route['ordinal'] and idx==route['id']
    operation=ops[route['source_operation_ordinal']]
    attrs=operation.attributes
    fqn=attrs['prov.fqn'].data if 'prov.fqn' in attrs else 'UNKNOWN'
    role=fqn.rsplit('.',1)[-1]
    record=categories[role];record['calls']+=1;record['callback_cycles']+=callback;record['preceding_gap_cycles']+=gap
    events.append(dict(ordinal=ordinal,id=idx,source_region=route['region'],source_fqn=fqn,dimensions=by_id[idx]['dimensions'],preceding_gap_cycles=gap,callback_cycles=callback))
archive=WORK/'stock2080_uart.txt';archive.write_bytes(raw)
result=dict(schema='current2076_stock2080_primitive_boundary_terminal_v1',status='complete_diagnostic_only',job_id=jid,terminal=terminal,baseline_stock_job=2076,baseline_unprofiled_cycles=380396343,profile_metric_cycles=int(metrics[0]),profile_forward_cycles=profile['forward_counter'],callback_cycles=profile['device_counter'],outside_callback_cycles=profile['host_gap_counter'],tail_cycles=profile['tail_counter'],profile_metric_minus_baseline=int(metrics[0])-380396343,callbacks=155,source_categories=dict(categories),events=events,original_output_gate=q['original_output_gate'],nofsm_audit=q['nofsm_audit'],actual_staged_identity=staged,elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],scope=manifest['scope']+' Preceding gap grouping is a source execution location,not a causal kernel attribution. This profile does not change the380396343 champion.',pins={str(path):sha(path) for path in [archive,qpath,manifest_path,WORK/'queue_submission.json',WORK/'preflight/firesim_preflight.json',WORK/'actual_staged_identity.json',Path(__file__)]},token_usage_available=False,token_usage=None)
(WORK/'stock2080_terminal.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['status','profile_metric_cycles','profile_forward_cycles','callback_cycles','outside_callback_cycles','tail_cycles','source_categories','elapsed_queue_seconds']},indent=2),flush=True)
