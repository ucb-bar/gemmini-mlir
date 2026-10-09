"""Close complete group2098 and explicit byte-identical prior2092 control reuse."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sqlite3
WORK=Path(__file__).resolve().parent/'polynomial_constants_stock'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text())
a_path=WORK/'root_admission.json';a=load(a_path)
assert a['status']=='pass' and a['source_pins_revalidated']==469
for path,h in a['pins'].items():assert sha(path)==h,path
p=load(a['source_packet']);assert sha(a['source_packet'])==a['source_packet_sha256'] and len(p['pins'])==469
for path,h in p['pins'].items():assert sha(path)==h,path
adj=load(a['protocol_adjunct']);assert sha(a['protocol_adjunct'])==a['protocol_adjunct_sha256']
for path,h in adj['pins'].items():assert sha(path)==h,path
ctrl=load(a['historical_control_receipt']);assert sha(a['historical_control_receipt'])==a['historical_control_receipt_sha256']
for path,h in ctrl['pins'].items():assert sha(path)==h,path
for row in ctrl['source_packets']:
 assert sha(row['path'])==row['sha256'];q=load(row['path'])
 assert len(q['pins'])==row['revalidated_pins']
 for path,h in q['pins'].items():assert sha(path)==h,path
old,=[x for x in ctrl['rows'] if x['job_id']==2092]
assert old==a['historical_control'] and old['cycles']==3857281394
assert old['actual_staged_identity']['objects']['elf']['sha256']==p['clock_pair']['control']['sha256']==sha(p['clock_pair']['control']['elf'])
jid=load(WORK/'queue_job.json')['job_id'];assert jid==2098
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
 db.row_factory=sqlite3.Row
 terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
job=Path(f'/scratch/firesim_queue/jobs/{jid}');d_path=job/'runworkload-full.json';d=load(d_path)
f_path=WORK/'preflight/firesim_preflight.json';f=load(f_path);s_path=WORK/'actual_staged_identity.json';s=load(s_path)
submission_path=WORK/'queue_submission.json';submission=load(submission_path)
assert submission['root_admission_sha256']==sha(a_path) and submission['returncode']==0
assert d['user']=='agustin' and d['hw_config']==f['hardware_key']==ctrl['hardware_alias']=='alveo_u250_firesim_gemmini_rocket_stock'
assert d['stage_from']==f['elf']==a['candidate_elf']
assert f['elf_sha256']==a['candidate_elf_sha256']==sha(a['candidate_elf'])==p['clock_pair']['candidate']['sha256']
assert s['job_id']==jid and s['phase']=='RUNNING' and s['observed_before_teardown']
assert s['objects']['elf']['sha256']==f['elf_sha256']
assert s['objects']['bitstream']['sha256']==ctrl['actual_staged_bit_sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
assert d['hwdb_config_artifact_sha256']==f['hwdb_artifact_sha256']==sha(f['hwdb_artifact'])==ctrl['hwdb_sha256']=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
assert f['bitstream_sha256']==sha(f['bitstream_path'])==ctrl['bitstream_archive_sha256']=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
audit=f['elf_nofsm_audit'];assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
parser_path=Path(a['parser']);assert sha(parser_path)==a['parser_sha256']
spec=importlib.util.spec_from_file_location('frozen_polynomial_protocol',parser_path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
raw=(job/'simulation/sim_slot_0/uartlog').read_bytes();text=raw.decode().replace('\r','')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
assert 'UNEXPECTED SOURCE REFUSAL' not in text and 'CONSUMER_TRAP' not in text
actual=parser.parse_report(text)
assert {k:v for k,v in actual.items() if k!='cycles'}=={k:v for k,v in a['strict_parsed'].items() if k!='cycles'}
archive=WORK/'stock2098_uart.txt';archive.write_bytes(raw)
cycles=actual['cycles'];assert cycles==3611264318
result=dict(schema='root_prepared_polynomial_constants_stock2098_complete_group_v1',status='verified_complete_original_consumer_group',job_id=jid,terminal=terminal,
 candidate_cycles=cycles,control_job=2092,control_cycles=old['cycles'],saving_cycles=old['cycles']-cycles,saving_fraction=1-cycles/old['cycles'],
 observation=actual,original_i8_count=786432,original_bf16_scale_count=1024,callbacks=480,original_compiled_consumer_and_guards=True,
 source_packet=a['source_packet'],source_packet_sha256=a['source_packet_sha256'],source_pins_revalidated=469,protocol_adjunct_sha256=a['protocol_adjunct_sha256'],protocol_pins_revalidated=4,
 actual_staged_identity=s,final_nofsm=audit,historical_control_receipt=a['historical_control_receipt'],historical_control_receipt_sha256=a['historical_control_receipt_sha256'],historical_control=old,control_source_pins_revalidated=361,
 control_reuse=a['control_reuse'],common62symbols=a['common62symbols'],independently_checked_actual_symbols=True,
 numerical_policy=a['numerical_policy'],complete_roi=a['complete_roi'],default_enabled=False,whole_cycles='UNKNOWN',no_group_multiplier=True,
 hardware_alias=f['hardware_key'],hwdb_sha256=f['hwdb_artifact_sha256'],bitstream_archive_sha256=f['bitstream_sha256'],actual_staged_bit_sha256=s['objects']['bitstream']['sha256'],
 elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],scope='One fresh complete first12-head group measured against byteexact historical2092 stock control, same62namedmemorysymbols/sourceconsumer/clock/config. Two internal carrier differences remain unobserved by original consumer. Native all1600wholewords exact; targetwhole remains a separate gate. No repeat-stability/wholeforecast or causal cycle attribution. SimulatorPASSED includesvalidation.',
 pins={str(x):sha(x) for x in [a_path,Path(a['source_packet']),Path(a['protocol_adjunct']),Path(a['historical_control_receipt']),parser_path,d_path,f_path,s_path,submission_path,archive,Path(__file__)]},token_usage_available=False,token_usage=None)
(WORK/'stock2098_terminal.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','candidate_cycles','saving_cycles','saving_fraction','elapsed_queue_seconds']}))
