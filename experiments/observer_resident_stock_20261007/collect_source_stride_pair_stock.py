"""Collect only complete original paired convolution stock cost observations."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

WORK=Path(__file__).resolve().parent/'source_stride_retained_stock'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
HWDB='5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

admission_path=WORK/'root_admission.json'
admission=json.loads(admission_path.read_text())
packet_path=Path(admission['source_packet'])
assert sha(packet_path)==admission['source_packet_sha256']
packet=json.loads(packet_path.read_text())
assert len(packet['pins'])==292
for path,digest in packet['pins'].items():
    assert sha(path)==digest,path
jobs=json.loads((WORK/'jobs.json').read_text())
assert [(x['name'],x['job_id']) for x in jobs]==[('control',2083),('compact_source_stride',2084)]
original=packet['rows']['original']
assert original['control']['common_operand_addresses']==original['compact_source_stride']['common_operand_addresses']==admission['common_operand_addresses']
rows=[]
pins={str(p):sha(p) for p in [admission_path,packet_path,WORK/'jobs.json',Path(__file__)]}
for row in jobs:
    jid=row['job_id'];name=row['name'];folder=Path(row['folder'])
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        terminal=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
    assert (terminal['state'],terminal['phase'],terminal['exit_code'])==('DONE','DONE',0)
    job=Path(f'/scratch/firesim_queue/jobs/{jid}')
    definition_path=job/'runworkload-full.json';definition=json.loads(definition_path.read_text())
    flight_path=folder/'preflight/firesim_preflight.json';flight=json.loads(flight_path.read_text())
    stage_path=folder/'actual_staged_identity.json';stage=json.loads(stage_path.read_text())
    submit_path=folder/'queue_submission.json';submit=json.loads(submit_path.read_text())
    assert definition['user']=='agustin' and definition['hw_config']==flight['hardware_key']==ALIAS
    assert definition['stage_from']==flight['elf']==row['elf']==original[name]['elf']['path']
    assert flight['elf_sha256']==sha(row['elf'])==row['elf_sha256']==original[name]['elf']['sha256']
    assert submit['source_packet_sha256']==sha(packet_path) and submit['source_pins_revalidated']==292
    assert stage['job_id']==jid and stage['phase']=='RUNNING' and stage['observed_before_teardown']
    assert stage['objects']['elf']['sha256']==flight['elf_sha256']
    assert stage['objects']['bitstream']['sha256']==BIT
    assert definition['hwdb_config_artifact_sha256']==flight['hwdb_artifact_sha256']==sha(flight['hwdb_artifact'])==HWDB
    assert flight['bitstream_sha256']==sha(flight['bitstream_path'])==TAR
    audit=flight['elf_nofsm_audit']
    assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    raw=(job/'simulation/sim_slot_0/uartlog').read_bytes();text=raw.decode().replace('\r','')
    contract=packet['stock_contract']
    assert text.splitlines().count(contract['required_pass_line'])==1
    assert 'PAIRED_CONV_FAIL' not in text and '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    values=re.findall(contract['cycle_pattern'],text,re.M)
    assert len(values)==1 and int(values[0])>0
    archive=folder/f'stock{jid}_uart.txt';archive.write_bytes(raw)
    for path in [definition_path,flight_path,stage_path,submit_path,archive]:pins[str(path)]=sha(path)
    rows.append(dict(name=name,job_id=jid,cycles=int(values[0]),terminal=terminal,
        elapsed_queue_seconds=terminal['ended_at']-terminal['started_at'],
        actual_staged_identity=stage,final_nofsm=audit))
control,candidate=rows
result=dict(schema='root_source_stride_stock2083_2084_complete_terminal_v1',status='verified_complete_original_pair',
    rows=rows,control_cycles=control['cycles'],candidate_cycles=candidate['cycles'],
    saving_cycles=control['cycles']-candidate['cycles'],saving_fraction=1-candidate['cycles']/control['cycles'],
    source_packet=str(packet_path),source_packet_sha256=sha(packet_path),source_pins_revalidated=292,
    original_gate=packet['stock_contract'],common_operand_addresses=admission['common_operand_addresses'],
    hardware_alias=ALIAS,hwdb_sha256=HWDB,bitstream_archive_sha256=TAR,actual_staged_bit_sha256=BIT,
    prelabel_winner='UNKNOWN',whole_cycles='UNKNOWN',default_enabled=False,
    scope='One complete original source-stride convolution observation per arm on stock RTL. Commands, pointer stream and immutable operands are identical; ordinary CPU loops change code footprint/issue/control. Full setup/input/weight DMA, both stores, final fence and decoder are timed. No attribution to pure issue, no summed whole forecast; independent tiny tails slightly regress in GSIM. Fresh whole-model qualification and timing remain separate.',
    pins=pins,token_usage_available=False,token_usage=None)
(WORK/'stock2083_2084_terminal.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key] for key in ['status','control_cycles','candidate_cycles','saving_cycles','saving_fraction']}),flush=True)
