from pathlib import Path
import hashlib,importlib.util,json,re,sqlite3
from mlir_oot.no_fsm_audit import audit_elf

BASE=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def transport(job,admission):
    folder=Path(job['folder']);jid=job['job_id'];elf=Path(job['elf'])
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
    definition=Path(f'/scratch/firesim_queue/jobs/{jid}/runworkload-full.json');defs=json.loads(definition.read_text())
    flight=json.loads((folder/'preflight/firesim_preflight.json').read_text());stage=json.loads((folder/'actual_staged_identity.json').read_text())
    assert defs['stage_from']==flight['elf']==str(elf)
    assert sha(elf)==job['elf_sha256']==flight['elf_sha256']==stage['objects']['elf']['sha256']
    assert defs['hw_config']==flight['hardware_key']=='alveo_u250_firesim_gemmini_rocket_stock'
    assert stage['job_id']==jid and stage['phase']=='RUNNING' and stage['observed_before_teardown']
    assert stage['objects']['bitstream']['sha256']=='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
    assert sha(flight['bitstream_path'])==flight['bitstream_sha256']=='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
    assert defs['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])=='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
    submission=json.loads((folder/'queue_submission.json').read_text());assert submission['root_admission_sha256']==sha(admission)
    audit=audit_elf(elf.read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    raw=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0/uartlog').read_bytes()
    text=raw.decode().replace('\r','');assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    archive=folder/'uart.txt';archive.write_bytes(raw)
    pins={str(x):sha(x) for x in [definition,archive,elf,admission,folder/'queue_submission.json',folder/'queue_terminal.json',folder/'actual_staged_identity.json',folder/'preflight/firesim_preflight.json']}
    return dict(**job,terminal=terminal,final_nofsm=audit,actual_staged_identity=stage,queue_elapsed_seconds=terminal['ended_at']-terminal['started_at']),text,pins

work=BASE/'key_batch16_whole_stock';a_path=work/'root_admission.json';a=json.loads(a_path.read_text())
packet=Path(a['source_packet']);assert sha(packet)==a['source_packet_sha256']=='aed77043dc6ed9d5364e3ffc8c57ed22d27b69fa6f37dd456bd3cd354447d26d'
p=json.loads(packet.read_text());assert len(p['pins'])==a['source_pins_revalidated']==810
for path,digest in p['pins'].items():assert sha(path)==digest,path
job,=json.loads((work/'jobs.json').read_text());row,text,pins=transport(job,a_path)
assert text.splitlines().count('DONE')==1 and text.splitlines().count('METRIC memref_rank_mismatch 0')==1
gate=a['original_gate'];assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$',text,re.M)==[('1000','4000',gate['raw_output_sha256'])]
cycles,=re.findall(r'^METRIC cycles (\d+)$',text,re.M)
previous=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/root_resnet_dense_tail_whole_stock2101_terminal.json')
old=json.loads(previous.read_text());assert old['candidate_cycles']==28728702
assert old['actual_staged_identity']['objects']['elf']['sha256']==p['control_sha256']
assert old['actual_staged_identity']['objects']['bitstream']['sha256']==row['actual_staged_identity']['objects']['bitstream']['sha256']
for x in [packet,previous,work/'jobs.json',Path(__file__),BASE/'submit_resnet_whole_and_prefix.py']:pins[str(x)]=sha(x)
record=dict(schema='root_resnet_key_batch16_stock2109_whole_v1',status='verified_complete_original_whole',job_id=job['job_id'],
 row=row,control_job=2101,control_cycles=28728702,candidate_cycles=int(cycles),saving_cycles=28728702-int(cycles),
 reduction_percent=100*(28728702-int(cycles))/28728702,target_cycles=22000000,remaining_target_gap=int(cycles)-22000000,
 original_output_gate=gate,bitwise_exact=True,source_pins_revalidated=810,control_reproduction_byte_exact=True,
 scope='Actual uninstrumented whole-model comparison; only source-proven selected key residual kernel/adapter change, other68 routes/stem/classifier/host/runtime/weights retained. No sum of section savings; compiler catalog default4 and primitive Plan default1 unchanged.',pins=pins)
save(work/'qualification.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('pins','row')}))

work=BASE/'quant_prefix_stock';a_path=work/'root_admission.json';a=json.loads(a_path.read_text())
packet=Path(a['source_packet']);assert sha(packet)==a['source_packet_sha256']=='1bf3258b7ff9f3fc19080e1c15f8f8b828125bacf23b4f0b3822440546d9b88d'
p=json.loads(packet.read_text());assert len(p['file_pins'])==a['source_pins_revalidated']==195
for pin in p['file_pins']:assert sha(pin['path'])==pin['sha256'] and Path(pin['path']).stat().st_size==pin['bytes'],pin['path']
parser=Path(a['parser']);assert sha(parser)==a['parser_sha256']
spec=importlib.util.spec_from_file_location('frozen_quant_prefix_protocol',parser)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pins={str(x):sha(x) for x in [packet,parser,work/'jobs.json',Path(__file__),BASE/'submit_resnet_whole_and_prefix.py']}
rows=[]
for job in json.loads((work/'jobs.json').read_text()):
    row,text,newpins=transport(job,a_path)
    protocol=module.parse(text,arm=job['protocol_arm'],**a['protocol']['kwargs']);row['protocol']=protocol
    assert [(r['input'],r['output']) for r in protocol['rows']]==[(r['input'],r['output']) for r in p['strict_results'][job['arm']]['rows']]
    row['mean_two_consecutive_windows_cycles']=sum(r['cycles'] for r in protocol['rows'])/2
    rows.append(row);pins.update(newpins)
old,new=[r['mean_two_consecutive_windows_cycles'] for r in rows]
record=dict(schema='root_quant_prefix_stock2110_2111_complete_cost_v1',status='REJECTED_PERFORMANCE',rows=rows,
 control_mean_cycles=old,candidate_mean_cycles=new,delta_percent=100*(new-old)/old,
 source_pins_revalidated=195,source_integer_outputs_exact=True,original_inputs_and_guards=True,
 whole_model_built=False,core_policy_promoted=False,fixed_prefix_bits=18,requested_table_bytes=524288,
 scope='Two consecutive complete windows per arm, no ABBA or stability claim. Isolated allocation/layout/copy graph differs from whole-model bufferization. Exact arithmetic/source-hit coverage and retirement are not proof of profitability; no alternate partition search or whole credit.',pins=pins)
assert new>old
save(work/'qualification.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('pins','rows')}))
