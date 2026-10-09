"""Fresh matched complete-group hardware decision, root queue owner only."""
from pathlib import Path
import hashlib, importlib.util, json, re, sqlite3, subprocess, time
from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.no_fsm_audit import audit_elf

ROOT=Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007')
WORK=Path(__file__).resolve().parent/'endpoint_dag_stock'
PACKET=ROOT/'docs/perf_records/prepared_endpoint_dag_complete_group_qualification.json'
EXPECTED='9d0efc2443b6bbbfcba72e0435d7f13e6bb5420a44b93ac1e724015943f823ba'
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
WORK.mkdir(exist_ok=False)
assert sha(PACKET)==EXPECTED
p=json.loads(PACKET.read_text());assert len(p['pins'])==289
for path,digest in p['pins'].items():assert sha(path)==digest,path
assert p['native']['elements']==1600 and p['native']['bitwise_mismatches']==0
assert p['resources']['exclusive_stack_increase_bytes']==16448
parser=ROOT/'experiments/attention_projection_frontier/polynomial_constants_protocol.py'
assert sha(parser)==p['pins'][str(parser)]=='c02daca82450cf387debfa1be4b73d1f26117790a33dbca2b5baf872bb0a6b85'
spec=importlib.util.spec_from_file_location('endpoint_original_group_protocol',parser)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
clock=ROOT/'out/artifacts/probes/prepared-endpoint-dag-v3/clock_pair'
strict=module.parse_report((clock/'candidate/stdout').read_text())
assert strict==json.loads((clock/'protocol.json').read_text())
rows=[]
for arm in ['control','candidate']:
    elf=clock/arm/'model.elf'
    assert sha(elf)==p['clock_'+arm+'_sha256']
    audit=audit_elf(elf.read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    rows.append(dict(arm=arm,elf=str(elf),elf_sha256=sha(elf),nofsm=audit))
assert rows[0]['elf_sha256']=='e46721f7b0dae400dcaf4750a053c59b4222749bacf05cb45261f0cac955e916'
nm='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm'
def data(elf):
    text=subprocess.check_output([nm,'-S','--defined-only',elf],text=True)
    return {v[3]:v[:3] for line in text.splitlines() if len(v:=line.split())==4 and v[2].upper() in ('B','D','R')}
symbols=[data(row['elf']) for row in rows]
assert symbols[0]==symbols[1] and len(symbols[0])==45
a=dict(schema='root_endpoint_dag_fresh_stock_pair_admission_v1',source_packet=str(PACKET),source_packet_sha256=EXPECTED,
 source_pins_revalidated=289,parser=str(parser),parser_sha256=sha(parser),strict_candidate_protocol=strict,
 native1600_original_exact=True,control_actual2098_byte_exact=True,shared45sized_data_symbols=symbols[0],
 preparation_stack_and_workspace_in_complete_roi=True,source_numeric_policy_unchanged=True,
 scope='Fresh matched full12-head group, including setup/preparation/products/observer/refinement/publication/guards. Section timing only, no whole forecast or stock full memory capacity inference.')
save(WORK/'root_admission.json',a)
env=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
jobs=[]
for row in rows:
    folder=WORK/row['arm'];folder.mkdir()
    flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',Path(row['elf']),ALIAS,TAR,folder/'preflight')
    argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock','--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',row['elf'],'--hw-config',ALIAS,'--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','1800','--project','gemmini-golden-nofsm']
    result=subprocess.run(argv,env=env,capture_output=True,text=True)
    save(folder/'queue_submission.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,root_admission_sha256=sha(WORK/'root_admission.json')))
    assert result.returncode==0
    jid=int(re.search(r'job_id=(\d+)',result.stdout).group(1))
    jobs.append(dict(**row,job_id=jid,folder=str(folder)))
    save(WORK/'jobs.json',jobs);print(json.dumps(dict(submitted=jid,arm=row['arm'])),flush=True)
seen=set();done=set();deadline=time.monotonic()+4000
while len(done)<len(jobs):
    assert time.monotonic()<deadline,'Observer bound reached; submissions retained'
    for row in jobs:
        jid=row['job_id']
        if jid in done:continue
        with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            state=dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?',(jid,)).fetchone())
        stage=Path(f'/scratch/firesim_queue/jobs/{jid}/simulation/sim_slot_0')
        paths={'elf':stage/'merlin-golden-nofsm-probe0-probe.elf','bitstream':stage/'xilinx_alveo_u250/firesim.bit'}
        folder=Path(row['folder'])
        if jid not in seen and state['phase']=='RUNNING' and all(x.is_file() for x in paths.values()):
            for name in ['FireSim-xilinx_alveo_u250','FireSim-generated.const.h']:
                path=stage/name
                if path.is_file():paths[name]=path
            objects={k:dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size) for k,v in paths.items()}
            assert objects['elf']['sha256']==row['elf_sha256'] and objects['bitstream']['sha256']==BIT
            save(folder/'actual_staged_identity.json',dict(job_id=jid,phase='RUNNING',observed_before_teardown=True,objects=objects))
            seen.add(jid);print(json.dumps(dict(staged=jid)),flush=True)
        if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
            save(folder/'queue_terminal.json',state);done.add(jid);print(json.dumps(dict(terminal=state)),flush=True)
    if len(done)<len(jobs):time.sleep(3)
assert seen==done
