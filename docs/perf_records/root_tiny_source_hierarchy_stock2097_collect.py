"""Close original Tiny M8 ABBA outputs and actual stock identities."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sqlite3
from types import SimpleNamespace

WORK=Path(__file__).resolve().parent/'source_hierarchy_stock_v2'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
a_path=WORK/'root_admission.json';a=json.loads(a_path.read_text())
p_path=Path(a['source_packet']);assert sha(p_path)==a['source_packet_sha256']
p=json.loads(p_path.read_text());assert len(p['pins'])==a['pins_revalidated']==322
for path,digest in p['pins'].items():assert sha(path)==digest,path
original_path=Path(p['original_300_pin_packet']);assert sha(original_path)==p['original_packet_sha256']
original=json.loads(original_path.read_text());assert len(original['pins'])==300
for path,digest in original['pins'].items():assert sha(path)==digest,path
adj_path=p_path.parent/'validation_scope_adjunct.json'
assert sha(adj_path)=='a642429996e1dfdde468b8e6ed6f12e27e5cc78550bfabc167fef7f03b5a1448'
adj=json.loads(adj_path.read_text());assert adj['original_receipt_sha256']==sha(p_path)
assert sha(adj['actual_main_source'])==adj['actual_main_sha256']
assert adj['full_table_sweeps_before_or_between_windows']==0 and adj['full_table_sweeps_after_four_windows']==1
jid=json.loads((WORK/'queue_job.json').read_text())['job_id'];assert jid==2097
with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
job=Path(f'/scratch/firesim_queue/jobs/{jid}');definition_path=job/'runworkload-full.json'
d=json.loads(definition_path.read_text());f_path=WORK/'preflight/firesim_preflight.json';f=json.loads(f_path.read_text())
s_path=WORK/'actual_staged_identity.json';s=json.loads(s_path.read_text())
assert d['user']=='agustin' and d['hw_config']==f['hardware_key']=='alveo_u250_firesim_gemmini_rocket_stock'
assert d['stage_from']==f['elf']==a['candidate_elf'] and sha(f['elf'])==f['elf_sha256']==a['candidate_elf_sha256']==adj['actual_elf_sha256']
assert s['job_id']==jid and s['observed_before_teardown'] and s['phase']=='RUNNING'
assert s['objects']['elf']['sha256']==f['elf_sha256']
assert s['objects']['bitstream']['sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
assert d['hwdb_config_artifact_sha256']==f['hwdb_artifact_sha256']==sha(f['hwdb_artifact'])=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
assert f['bitstream_sha256']==sha(f['bitstream_path'])=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
audit=f['elf_nofsm_audit'];assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
def parse_function(path,namespace):
    tree=ast.parse(path.read_text());fn,=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='parse_report']
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['parse_report']
base=Path(a['base_parser_path']);parser=Path(a['parser_path'])
assert sha(base)==a['base_parser_sha256'] and sha(parser)==a['parser_sha256']
parse=parse_function(parser,{'first':SimpleNamespace(parse_report=parse_function(base,{}))})
raw=(job/'simulation/sim_slot_0/uartlog').read_bytes();text=raw.decode().replace('\r','')
assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
parsed=parse(text);assert [int(x['id']) for x in parsed['rows']]==[0,1,1,0]
for row in parsed['named_storage']:assert {k:v for k,v in row.items() if k not in ('id','sample')}==a['expected_storage']
rows=parsed['rows'];cycles={i:[int(x['cycles']) for x in rows if int(x['id'])==i] for i in (0,1)}
parsed['actual_hardware_cycles']=[int(row['cycles']) for row in rows]
parsed['scope']='Complete firstM8 helper; initial actualcandidate sourcegate warms firstM8 requests; fulltable validation only afterall4windows; no independent coldcache observation'
means={i:sum(v)/len(v) for i,v in cycles.items()};assert means[0]==3513995.5 and means[1]==3514995.5
archive=WORK/'stock2097_uart.txt';archive.write_bytes(raw)
result=dict(schema='root_source_hierarchy_M8_stock2097_terminal_v1',status='complete_original_gate_pass_no_measured_win',job_id=jid,terminal=terminal,
 source_packet=str(p_path),source_packet_sha256=sha(p_path),source_pins_revalidated=322,original_source_pins_revalidated=300,
 validation_scope_adjunct=adj,validation_scope_adjunct_sha256=sha(adj_path),actual_staged_identity=s,final_nofsm=audit,
 original_i8_count=45056,original_gate='All original source i8 bytes, dirtyguards and inputhashes pass; source fiveFRM/sevensticky qualified, original final policy unchanged',
 ABBA=parsed,cycles_by_arm=cycles,control_mean_cycles=means[0],candidate_mean_cycles=means[1],mean_regression_cycles=means[1]-means[0],mean_regression_fraction=means[1]/means[0]-1,
 source_instructions={'control':1726434,'candidate':1764719},hardware_alias=f['hardware_key'],hwdb_sha256=f['hwdb_artifact_sha256'],bitstream_archive_sha256=f['bitstream_sha256'],
 actual_staged_bit_sha256=s['objects']['bitstream']['sha256'],scope='Complete firstM8 helper ABBA. Initial actualcandidate sourcegate warmup; fulltable hashes afterall4ROIs only. Matched9named addresses/2immutable tablehashes. Singlepair marginal result does not establish stable regression or hardware cache causality. SimulatorPASSED cycles include setup/validation; no whole extrapolation.',
 default_enabled=False,whole_candidate_promoted=False,whole_cycles='UNKNOWN',
 pins={str(x):sha(x) for x in [a_path,p_path,original_path,adj_path,definition_path,f_path,s_path,parser,base,archive,Path(__file__)]},token_usage_available=False,token_usage=None)
save(WORK/'stock2097_terminal.json',result)
print(json.dumps({k:result[k] for k in ['status','control_mean_cycles','candidate_mean_cycles','mean_regression_cycles','mean_regression_fraction']}))
