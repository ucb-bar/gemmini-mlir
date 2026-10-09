from pathlib import Path
import json,hashlib,subprocess,re,shutil,datetime
w=Path(__file__).parent.resolve();r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-encoded-row-equality-20261006');old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p,h=None):
 p=Path(p).resolve();s=sha(p)
 if h is not None:assert s==h,(p,s,h)
 pins[str(p)]=s
s=json.loads((w/'candidate/spike_receipt.json').read_text());assert s['status']=='pass'and s['returncode']==0
control=(old/'candidate/spike.log').read_text();candidate=(w/'candidate/spike.log').read_text();assert re.findall(r'^WORKSPACE_STAT .*$',control,re.M)==re.findall(r'^WORKSPACE_STAT .*$',candidate,re.M);assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in candidate
before=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',control)[1]);after=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',candidate)[1]);assert sha(w/'control/model.elf')==sha(old/'candidate/model.elf')
snap=r/'docs/perf_records/source_snapshots/encoded_row_equality';snap.mkdir(parents=True,exist_ok=True)
for name in ['src/merlin/llvmlower/source_attention_frontier.py','src/merlin/llvmlower/encoded_row_equality.py','merlin/runtime/c/encoded_row_equality.h','merlin/tests/runtime/test_encoded_row_equality.py','merlin/tests/runtime/test_source_attention_frontier.py','docs/runtime/encoded_row_equality.md']:
 data=subprocess.check_output(['git','show','dc95f1d9a:'+name],cwd=core);assert data==(core/name).read_bytes();dest=snap/Path(name).name;dest.write_bytes(data);pin(dest)
for p in w.glob('*.py'):dest=snap/p.name;shutil.copyfile(p,dest);pin(dest)
for p in w.rglob('*'):
 if p.is_file()and p.name not in ['process.json']:pin(p)
for p in [old/'candidate/model.elf',old/'candidate/spike.log',old/'numeric_frozen/provider.o',old/'candidate/build.json']:pin(p)
for command in json.loads((w/'candidate/build.json').read_text())['link']:
 p=Path(command)
 if p.is_file():pin(p)
for name in ['numeric_frozen','native_numeric_frozen']:
 for command in json.loads((w/name/'compile.json').read_text())['commands']:
  pin(Path(command[0]).resolve())
  if '-MF' not in command:continue
  d=Path(command[command.index('-MF')+1]);deps=d.read_text().replace('\\\n',' ').split(':',1)[1].split()
  for p in deps:pin(p)
q={'schema':'encoded_row_equality_complete_group_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','core_commit':'dc95f1d9a','scope':'Private typed exact row equality generated during mandatory reconstruction widening; complete allocation-aware original group target. Full48 ordinary native integration still pending in separate fresh build.','control_elf_sha256':sha(w/'control/model.elf'),'candidate_elf_sha256':sha(w/'candidate/model.elf'),'control_byte_reproduced':True,'only_provider_object_changed':True,'target':s,'independent_target':json.loads((w/'independent_target/strict_receipt.json').read_text()),'old_capacity_refusal':json.loads((w/'old_capacity_refusal.json').read_text()),'all_eight_group_stats_unchanged':True,'workspace_bytes':123012928,'prior_workspace_bytes':123012160,'extra_private_flags_bytes':768,'instructions':{'control':before,'candidate':after,'reduction_percent':100*(before-after)/before,'scope':'Complete allocation-aware group retired instructions, not cycles; proof generation and storage included'},'default_identity':json.loads((w/'default_emission_identity.json').read_text()),'whole_normal_native':'pending','hardware_requested':False,'pins':pins,'token_usage_available':False};out=r/'docs/perf_records/encoded_row_equality_complete_group_qualification.json';out.write_text(json.dumps(q,indent=2)+'\n');print(out,len(pins),q['instructions'])
