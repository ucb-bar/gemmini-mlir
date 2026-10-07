from pathlib import Path
import hashlib, importlib.util, json, sqlite3
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'stock'
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p,v): p.write_text(json.dumps(v, indent=2)+'\n')
admission = WORK / 'root_admission.json'
a = json.loads(admission.read_text())
packet = Path(a['source_packet'])
assert sha(packet) == a['source_packet_sha256'] == '75947dde636c45959f495c52ab493d23ceebec5590f1c38cd55519604e56e04f'
p = json.loads(packet.read_text())
assert len(p['pins']) == a['pins_revalidated'] == 98
for path,digest in p['pins'].items(): assert sha(path) == digest, path
parser_path = Path(a['parser']['path'])
assert sha(parser_path) == a['parser']['sha256']
spec = importlib.util.spec_from_file_location('capacity_source_protocol',parser_path)
parser = importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
jobs = json.loads((WORK/'jobs.json').read_text())
pins = {str(x):sha(x) for x in [packet,admission,parser_path,WORK/'jobs.json',Path(__file__),WORK.parent/'submit_stock.py']}
rows=[]
for job in jobs:
    folder=Path(job['folder']); jid=job['job_id']
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code']) == ('DONE','DONE',0)
    definition_path=Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json')
    defs=json.loads(definition_path.read_text())
    flight=json.loads((folder/'preflight/firesim_preflight.json').read_text())
    stage=json.loads((folder/'actual_staged_identity.json').read_text())
    assert defs['hw_config']==flight['hardware_key']=='alveo_u250_firesim_gemmini_rocket_stock'
    assert defs['stage_from']==flight['elf']==job['elf']
    assert sha(job['elf'])==job['elf_sha256']==flight['elf_sha256']==stage['objects']['elf']['sha256']
    assert stage['job_id']==jid and stage['phase']=='RUNNING' and stage['observed_before_teardown']
    assert stage['objects']['bitstream']['sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
    assert sha(flight['bitstream_path'])==flight['bitstream_sha256']=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
    assert defs['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
    audit=audit_elf(Path(job['elf']).read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    raw=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0/uartlog').read_bytes()
    text=raw.decode().replace('\r','')
    assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    result=parser.parse(text,arm=job['arm'],elements=802816)
    archive=folder/'uart.txt';archive.write_bytes(raw)
    submission=json.loads((folder/'queue_submission.json').read_text())
    assert submission['root_admission_sha256']==sha(admission)
    rows.append(dict(**job,terminal=terminal,protocol=result,final_nofsm=audit,actual_staged_identity=stage,
        queue_elapsed_seconds=terminal['ended_at']-terminal['started_at']))
    for x in [definition_path,archive,folder/'queue_submission.json',folder/'queue_terminal.json',folder/'actual_staged_identity.json',folder/'preflight/firesim_preflight.json']:
        pins[str(x)]=sha(x)
old,new=(row['protocol']['counters']['cycles'] for row in rows)
record=dict(schema='root_stock2105_2106_capacity_batch_complete_v1',status='PASS',rows=rows,
    control_cycles=old,candidate_cycles=new,delta_cycles=new-old,reduction_percent=100*(old-new)/old,
    complete_ranked_original_elements=802816,source_pins_revalidated=98,same_elf_selector_only=True,
    control_current2101_objects_byte_identical=True,independent_all65536_pairs_and_tails_exact=True,
    scope='Complete section includes ranked adapter, scratch, setup, commands, DDR readback/reload and final fence. Whole-model integration and cycles remain unmeasured. No new numeric policy.',pins=pins)
save(WORK/'qualification.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('pins','rows')}))
