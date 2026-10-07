"""Measure the actual borrowed-input dense alternative with original complete cost."""
from pathlib import Path
import hashlib, importlib.util, json, re, sqlite3, subprocess, time
from mlir_oot.golden_firesim_preflight import preflight
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'dense_segmented_tail_stock'
OWNER=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007')
PACKET=OWNER/'docs/perf_records/current_dense_stationary_tail_segmented_capsule.json'
PACKET_SHA='1a71e7104e2cb3aa5f2c931bf5b99949847dd505277330f183e1f776dda36866'
WHOLE=OWNER/'docs/perf_records/current_dense_stationary_tail_2095_whole_qualification.json'
WHOLE_SHA='765ebdf9db0728b9e18922135579f669a1d85028f6dd295cdef2b1809c82f26f'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
WORK.mkdir(exist_ok=False)
assert sha(PACKET)==PACKET_SHA and sha(WHOLE)==WHOLE_SHA
p=json.loads(PACKET.read_text());whole=json.loads(WHOLE.read_text())
assert p['status']=='PASS' and len(p['pins'])==516
for path,d in p['pins'].items():assert sha(path)==d,path
assert p['source26']['actual_current_control_byte_identical'] and p['source26']['complete_command_multiset_identical'] and p['source26']['all_DMA_configuration_fence_store_order_identical']
selected,=[r for r in whole['selected_actual19'] if r['kernel']==p['source26']['kernel']]
assert selected['segmented'] and sha(selected['original'])==selected['original_sha256'] and sha(selected['selected'])==selected['selected_sha256']
parser_path,=[Path(x) for x in p['pins'] if x.endswith('/parse_stock.py')]
spec=importlib.util.spec_from_file_location('dense_stock_parser',parser_path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
outputs=p['source26']['shape']['m']*p['source26']['shape']['n'];assert outputs==200704
arms={}
for name,key in [('control','control'),('tail','candidate')]:
    row=p[key];assert sha(row['path'])==row['sha256'];v=json.loads(Path(row['path']).read_text())
    assert v['status']=='PASS' and v['nofsm_status']=='pass' and v['strict']['status']=='PASS' and v['complete_finish_roi']
    assert v['outputs_checked']==outputs and v['guards_checked']==4096 and v['immutable_input_bytes_checked']==925696
    assert v['common_addresses']==p['common_addresses'] and sha(v['elf'])==v['elf_sha256']
    assert v['compiled_object_sha256']==selected['original_sha256' if name=='control' else 'selected_sha256']
    stdout=Path(v['elf']).parent/'spike.stdout';assert sha(stdout)==v['strict']['stdout_sha256'];parser.parse(stdout.read_text(),outputs=outputs,inputs=925696)
    assert v['GSIM']['complete'] and v['GSIM']['returncode']==0
    assert parser.parse(v['GSIM']['stdout'],outputs=outputs,inputs=925696)['cycles']==p['control_gsim_cycles' if name=='control' else 'candidate_gsim_cycles']
    arms[name]=v
admission=dict(schema='root_actual_segmented_dense_tail_complete_admission_v1',status='PASS',source_packet=str(PACKET),source_packet_sha256=PACKET_SHA,source_pins_revalidated=516,whole_packet=str(WHOLE),whole_packet_sha256=WHOLE_SHA,actual_selected_binding=selected,stock_parser=dict(path=str(parser_path),sha256=sha(parser_path)),common_addresses=p['common_addresses'],outputs=outputs,guards=4096,immutable_inputs=925696,original_owner_bytes=p['original_owner_bytes'],complete_scope=p['complete_scope'],prelabel_winner='UNKNOWN',whole_forecast='UNKNOWN',purpose='Calibrate complete-cost alternative for actual borrowed196x512x1024 shape; no section sum, physical cache claim or automatic policy')
save(WORK/'root_admission.json',admission)
env=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066'];assert set(env)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
jobs=[]
for name,v in arms.items():
    folder=WORK/name;folder.mkdir();elf=Path(v['elf'])
    flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',elf,ALIAS,TAR,folder/'preflight');assert flight['elf_nofsm_audit']['status']=='pass'
    argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock','--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',str(elf),'--hw-config',ALIAS,'--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','600','--project','gemmini-golden-nofsm']
    result=subprocess.run(argv,env=env,capture_output=True,text=True);save(folder/'queue_submission.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,source_packet_sha256=PACKET_SHA,source_pins_revalidated=516,root_admission_sha256=sha(WORK/'root_admission.json')));assert result.returncode==0
    match=re.search(r'job_id=(\d+)',result.stdout);assert match;jid=int(match.group(1));jobs.append(dict(name=name,job_id=jid,folder=str(folder),elf=str(elf),elf_sha256=sha(elf)));save(WORK/'jobs.json',jobs);print(json.dumps(dict(submitted=jid,arm=name)),flush=True)
seen=set();done=set();deadline=time.monotonic()+5400
while len(done)<len(jobs):
    assert time.monotonic()<deadline,'Observer timeout; retain live submissions'
    for row in jobs:
        jid=row['job_id']
        if jid in done:continue
        with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row;state=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
        stage=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0');elf=stage/'merlin-golden-nofsm-probe0-probe.elf';bit=stage/'xilinx_alveo_u250/firesim.bit';folder=Path(row['folder'])
        if jid not in seen and state['phase']=='RUNNING' and elf.is_file() and bit.is_file():
            objects={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for name,path in [('elf',elf),('bitstream',bit)]};assert objects['elf']['sha256']==row['elf_sha256'] and objects['bitstream']['sha256']==BIT;save(folder/'actual_staged_identity.json',dict(job_id=jid,phase='RUNNING',observed_before_teardown=True,objects=objects));seen.add(jid);print(json.dumps(dict(staged=jid)),flush=True)
        if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
            save(folder/'queue_terminal.json',state);done.add(jid);print(json.dumps(dict(terminal=state,staged=jid in seen)),flush=True)
    if len(done)<len(jobs):time.sleep(3)
assert seen==done
