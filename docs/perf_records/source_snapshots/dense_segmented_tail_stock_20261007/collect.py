"""Close actual borrowed-input control/candidate complete stock observations."""
from pathlib import Path
import hashlib,importlib.util,json,sqlite3
WORK=Path(__file__).resolve().parent/'dense_segmented_tail_stock'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
HWDB='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a_path=WORK/'root_admission.json';a=json.loads(a_path.read_text());p_path=Path(a['source_packet']);assert sha(p_path)==a['source_packet_sha256'];p=json.loads(p_path.read_text());assert len(p['pins'])==516
for path,d in p['pins'].items():assert sha(path)==d,path
parserpath=Path(a['stock_parser']['path']);assert sha(parserpath)==a['stock_parser']['sha256'];spec=importlib.util.spec_from_file_location('dense_parser',parserpath);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
jobs=json.loads((WORK/'jobs.json').read_text());assert [(r['name'],r['job_id']) for r in jobs]==[('control',2102),('tail',2103)]
pins={str(x):sha(x) for x in [a_path,p_path,Path(a['whole_packet']),WORK/'jobs.json',Path(__file__)]};rows=[]
for row in jobs:
 jid=row['job_id'];folder=Path(row['folder']);job=Path(f'/scratch/firesim_queue/jobs/{jid}')
 with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
  db.row_factory=sqlite3.Row;terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
 assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
 paths=[job/'runworkload-full.json',folder/'preflight/firesim_preflight.json',folder/'actual_staged_identity.json',folder/'queue_submission.json'];definition,flight,staged,submit=[json.loads(x.read_text()) for x in paths]
 assert definition['user']=='agustin' and definition['hw_config']==flight['hardware_key']==ALIAS
 assert definition['stage_from']==flight['elf']==row['elf'] and sha(row['elf'])==row['elf_sha256']==flight['elf_sha256']
 assert definition['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])==HWDB
 assert flight['bitstream_sha256']==sha(flight['bitstream_path'])==TAR
 assert flight['elf_nofsm_audit']['status']=='pass' and not flight['elf_nofsm_audit']['forbidden'] and not flight['elf_nofsm_audit']['unknown']
 assert staged['job_id']==jid and staged['phase']=='RUNNING' and staged['observed_before_teardown']
 assert staged['objects']['elf']['sha256']==row['elf_sha256'] and staged['objects']['bitstream']['sha256']==BIT
 assert submit['source_packet_sha256']==sha(p_path) and submit['source_pins_revalidated']==516 and submit['root_admission_sha256']==sha(a_path)
 raw=(job/'simulation/sim_slot_0/uartlog').read_bytes();text=raw.decode().replace('\r','');assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
 parsed=parser.parse(text,outputs=a['outputs'],inputs=a['immutable_inputs']);archive=folder/f'stock{jid}_uart.txt';archive.write_bytes(raw)
 for x in paths+[archive]:pins[str(x)]=sha(x)
 rows.append(dict(arm=row['name'],job_id=jid,cycles=parsed['cycles'],observation=parsed,terminal=terminal,actual_staged_identity=staged,final_nofsm=flight['elf_nofsm_audit']))
control,candidate=rows
record=dict(schema='root_dense_borrowed_input_stationary_tail_stock2102_2103_v1',status='verified_complete_original_pair',rows=rows,control_cycles=control['cycles'],candidate_cycles=candidate['cycles'],saving_cycles=control['cycles']-candidate['cycles'],saving_fraction=1-candidate['cycles']/control['cycles'],source_packet=str(p_path),source_packet_sha256=sha(p_path),source_pins_revalidated=516,actual_selected_whole_binding=a['actual_selected_binding'],original_gate=dict(outputs=a['outputs'],guards=a['guards'],immutable_inputs=a['immutable_inputs'],original_borrowed_owner_bytes=a['original_owner_bytes']),same4addresses=a['common_addresses'],hardware_alias=ALIAS,hwdb_sha256=HWDB,bitstream_archive_sha256=TAR,actual_staged_bit_sha256=BIT,complete_scope=a['complete_scope'],whole_forecast='UNKNOWN',automatic_policy=False,scope='One original borrowed196x512x1024 complete capsule per arm. Actual current2095/2101 objects; original401408byte segmented owner, no packed copy. Allconfigs/flush/loads/preloads/computes/stores/finalfence insideclock; originaloutputs/immutableinputs/guards checked outside. Paired hardware differs from separate whole observation; no added section gains or causal array/cache/CPU price.',pins=pins,token_usage_available=False)
(WORK/'stock2102_2103_terminal.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['status','control_cycles','candidate_cycles','saving_cycles','saving_fraction']}))
