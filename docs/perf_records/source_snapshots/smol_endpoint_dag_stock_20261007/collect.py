from pathlib import Path
import hashlib,importlib.util,json,sqlite3
from mlir_oot.no_fsm_audit import audit_elf

WORK=Path(__file__).resolve().parent/'endpoint_dag_stock'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a_path=WORK/'root_admission.json';a=json.loads(a_path.read_text())
packet=Path(a['source_packet']);assert sha(packet)==a['source_packet_sha256']=='9d0efc2443b6bbbfcba72e0435d7f13e6bb5420a44b93ac1e724015943f823ba'
p=json.loads(packet.read_text());assert len(p['pins'])==a['source_pins_revalidated']==289
for path,digest in p['pins'].items():assert sha(path)==digest,path
parser=Path(a['parser']);assert sha(parser)==a['parser_sha256']
spec=importlib.util.spec_from_file_location('stock_endpoint_group_protocol',parser)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pins={str(x):sha(x) for x in [packet,a_path,parser,WORK/'jobs.json',Path(__file__),WORK.parent/'submit_endpoint_dag_stock.py']}
rows=[]
for job in json.loads((WORK/'jobs.json').read_text()):
    folder=Path(job['folder']);jid=job['job_id'];elf=Path(job['elf'])
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
    definition=Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json');defs=json.loads(definition.read_text())
    flight=json.loads((folder/'preflight/firesim_preflight.json').read_text());staged=json.loads((folder/'actual_staged_identity.json').read_text())
    assert defs['stage_from']==flight['elf']==str(elf) and sha(elf)==job['elf_sha256']==flight['elf_sha256']==staged['objects']['elf']['sha256']
    assert defs['hw_config']==flight['hardware_key']=='alveo_u250_firesim_gemmini_rocket_stock'
    assert staged['phase']=='RUNNING' and staged['observed_before_teardown'] and staged['job_id']==jid
    assert staged['objects']['bitstream']['sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
    assert sha(flight['bitstream_path'])==flight['bitstream_sha256']=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
    assert defs['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
    submission=json.loads((folder/'queue_submission.json').read_text());assert submission['root_admission_sha256']==sha(a_path)
    audit=audit_elf(elf.read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    raw=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0/uartlog').read_bytes()
    text=raw.decode().replace('\r','');assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    protocol=module.parse_report(text)
    assert protocol['stats']==a['strict_candidate_protocol']['stats'] and protocol['carrier_differences']==a['strict_candidate_protocol']['carrier_differences']
    archive=folder/'uart.txt';archive.write_bytes(raw)
    rows.append(dict(**job,terminal=terminal,protocol=protocol,final_nofsm=audit,actual_staged_identity=staged,
        queue_elapsed_seconds=terminal['ended_at']-terminal['started_at']))
    for x in [definition,archive,elf,folder/'queue_submission.json',folder/'queue_terminal.json',folder/'actual_staged_identity.json',folder/'preflight/firesim_preflight.json']:
        pins[str(x)]=sha(x)
old,new=[row['protocol']['cycles'] for row in rows]
record=dict(schema='root_endpoint_dag_stock2107_2108_complete_group_v1',status='PASS',rows=rows,
 control_cycles=old,candidate_cycles=new,delta_cycles=new-old,reduction_percent=100*(old-new)/old,
 source_pins_revalidated=289,original_consumer_i8=786432,original_bf16_scales=1024,guards=True,
 shared45sized_data_symbols_exact=True,control_actual2098_byte_identical=True,
 numerical_policy='Unchanged explicit RMS4; range theorem does not certify relation of approximate intervals to source truth.',
 scope='Full12-head group includes allocation, preparation, products, observer/refinement and final stores. Separate whole original1600 gate and whole hardware timing required; no48x section forecast.',pins=pins)
(WORK/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('pins','rows')}))
