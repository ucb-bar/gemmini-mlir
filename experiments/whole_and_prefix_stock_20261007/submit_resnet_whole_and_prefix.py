"""Root queue decisions: normal whole batch16 and one fixed prefix map pair."""
from pathlib import Path
import hashlib,importlib.util,json,re,sqlite3,subprocess,time
import numpy as np
from mlir_oot.golden_firesim_preflight import preflight
from mlir_oot.no_fsm_audit import audit_elf

BASE=Path(__file__).resolve().parent
WHOLE=BASE/'key_batch16_whole_stock';PREFIX=BASE/'quant_prefix_stock'
CANON=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
ALIAS='alveo_u250_firesim_gemmini_rocket_stock'
TAR='a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'
BIT='6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
WHOLE.mkdir(exist_ok=False);PREFIX.mkdir(exist_ok=False)
packet=Path('/scratch/agustin/tmp/gemmini-key-batch16-normal-20261007/docs/perf_records/resnet_key_batch16_current2101_whole_qualification.json')
assert sha(packet)=='aed77043dc6ed9d5364e3ffc8c57ed22d27b69fa6f37dd456bd3cd354447d26d'
p=json.loads(packet.read_text());assert len(p['pins'])==810
for path,digest in p['pins'].items():assert sha(path)==digest,path
assert p['control_job']==2101 and p['control_sha256']=='1aa4fa5a921153d10d951d3acea89b3e2d6ba1bba1f3f97fc02718ab749bf3d3'
assert p['other_catalog_bindings']['count']==68 and p['stem_classifier_binding_unchanged']
assert not p['global_default_changed'] and len(p['changed_objects'])==2
for name in ['golden_key_rectified_resadd.py','rectifier_residual_catalog.py']:
    assert (Path(p['candidate_elf']).parents[5]/'mlir_oot'/name).read_bytes()==(CANON/'mlir_oot'/name).read_bytes(),name
g=p['whole_gates']['controlled'];assert g['status']=='PASS' and g['native']['exact_equal'] and g['spike']['target_native_exact']
native=Path(g['spike']['reference_path']);golden=Path(g['spike']['torch_golden_path'])
assert sha(native)==sha(golden)==g['spike']['original_golden_sha256']
value=np.load(native,allow_pickle=False);assert value.size==1000 and value.dtype==np.float32
digest=hashlib.sha256(value.astype('<f4').tobytes()).hexdigest();assert digest==g['spike']['spike_output_sha256']
entry=p['entry_closure'];assert entry['status']=='PASS' and entry['elf_sha256']==p['candidate_sha256']
selected,=entry['selected'];assert selected['actual_entry_count']==1 and selected['all_nonrelocation_bytes_equal'] and selected['branch_opcode_registers_and_targets_exact']
assert len(p['public_adapter_calls_selected_kernel_once'])==1
save(WHOLE/'root_admission.json',dict(schema='root_key_batch16_normal_whole_stock_admission_v1',status='PASS',
 source_packet=str(packet),source_packet_sha256=sha(packet),source_pins_revalidated=810,
 control_job=2101,control_reproduction_byte_exact=True,other68_routes_and_stem_classifier_unchanged=True,
 exact_current_canonical_lowering_sources=True,selected_linked_body_and_executed_entry_closed=True,
 original_gate=dict(elements=1000,atol=0,rtol=0,raw_output_sha256=digest),native_and_actual_strict_exact=True,
 candidate_elf=p['candidate_elf'],candidate_elf_sha256=p['candidate_sha256'],whole_cycles='UNKNOWN',no_section_gain_sum=True))
rows=[dict(arm='whole',folder=str(WHOLE),elf=p['candidate_elf'],elf_sha256=p['candidate_sha256'])]
packet=Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007/out/current_quant_prefix_seal_v2/qualification.json')
assert sha(packet)=='1bf3258b7ff9f3fc19080e1c15f8f8b828125bacf23b4f0b3822440546d9b88d'
p=json.loads(packet.read_text());assert len(p['file_pins'])==195
for pin in p['file_pins']:
    assert Path(pin['path']).stat().st_size==pin['bytes'] and sha(pin['path'])==pin['sha256'],pin['path']
assert p['source_proof_rederived'] and p['interval_table_rederived_byte_identical'] and p['hot_lookup_calls']==0
assert p['linked_original_scalar_llvm_diff']=='PASS' and p['partition']['bits']==18
parser=Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007/mlir_oot/quant_prefix_capsule.py')
assert any(pin['path']==str(parser) and pin['sha256']==sha(parser) for pin in p['file_pins'])
spec=importlib.util.spec_from_file_location('qualified_quant_prefix_protocol',parser)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
elves=[Path(p['arms'][arm]['elf']['path']) for arm in ['control','candidate']]
raw=[x.read_bytes() for x in elves]
assert len(raw[0])==len(raw[1]) and [i for i,(x,y) in enumerate(zip(*raw,strict=True)) if x!=y]==[p['selector_only_byte_offset']]
assert [x[p['selector_only_byte_offset']] for x in raw]==[0,1]
for arm,elf in enumerate(elves):
    name=['control','candidate'][arm]
    assert sha(elf)==p['arms'][name]['elf']['sha256']
    folder=PREFIX/name;folder.mkdir()
    rows.append(dict(arm=name,protocol_arm=arm,folder=str(folder),elf=str(elf),elf_sha256=sha(elf)))
save(PREFIX/'root_admission.json',dict(schema='root_closed_quant_prefix_stock_admission_v1',status='PASS',source_packet=str(packet),
 source_packet_sha256=sha(packet),source_pins_revalidated=195,source_and_table_rederived=True,source_fallback_llvm_diff='PASS',
 hot_lookup_calls=0,same_elf_selector_only=True,protocol=p['protocol'],parser=str(parser),parser_sha256=sha(parser),
 original_quant150528_padded158700_inputs602112_guards16384=True,compiled_finite_executor_values=4177920,
 numeric_contract=p['numeric_contract'],scope=p['scope'],control_scope=p['control_scope'],retirement_delta_percent=p['retired_delta_percent'],whole_cycles='UNKNOWN'))
env=json.loads(Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2066_submission_environment.json').read_text())['replacement2066']
assert set(env)=={'PATH','SHELL','TERM','SSH_AUTH_SOCK'}
jobs=[]
for row in rows:
    folder=Path(row['folder']);elf=Path(row['elf'])
    audit=audit_elf(elf.read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
    flight=preflight(Path('/scratch2/agustin/wt/chipyard-stock'),'merlin-golden-nofsm-probe',elf,ALIAS,TAR,folder/'preflight')
    admission=WHOLE/'root_admission.json' if row['arm']=='whole' else PREFIX/'root_admission.json'
    argv=['/usr/local/bin/firesim-queue','runworkload-full','--background','--chipyard','/scratch2/agustin/wt/chipyard-stock','--workload','merlin-golden-nofsm-probe','--bootbinary','probe.elf','--stage-from',str(elf),'--hw-config',ALIAS,'--hwdb-config-artifact',flight['hwdb_artifact'],'--priority','0','--timeout','600','--project','gemmini-golden-nofsm']
    result=subprocess.run(argv,env=env,capture_output=True,text=True)
    save(folder/'queue_submission.json',dict(argv=argv,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,root_admission=str(admission),root_admission_sha256=sha(admission)))
    assert result.returncode==0
    jid=int(re.search(r'job_id=(\d+)',result.stdout).group(1));jobs.append(dict(**row,job_id=jid))
    save(BASE/'resnet_whole_and_prefix_stock_jobs.json',jobs)
    save(WHOLE/'jobs.json',[x for x in jobs if x['arm']=='whole'])
    save(PREFIX/'jobs.json',[x for x in jobs if x['arm']!='whole'])
    print(json.dumps(dict(submitted=jid,arm=row['arm'])),flush=True)
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
            objects={k:dict(path=str(v),sha256=sha(v),bytes=v.stat().st_size) for k,v in paths.items()}
            assert objects['elf']['sha256']==row['elf_sha256'] and objects['bitstream']['sha256']==BIT
            save(folder/'actual_staged_identity.json',dict(job_id=jid,phase='RUNNING',observed_before_teardown=True,objects=objects))
            seen.add(jid);print(json.dumps(dict(staged=jid)),flush=True)
        if state['state'] in ('DONE','FAILED','CANCELLED','TIMED_OUT'):
            save(folder/'queue_terminal.json',state);done.add(jid);print(json.dumps(dict(terminal=state)),flush=True)
    if len(done)<len(jobs):time.sleep(3)
assert seen==done
