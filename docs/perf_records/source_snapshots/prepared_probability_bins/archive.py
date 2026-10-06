from pathlib import Path
import json,hashlib,subprocess,re,shutil,datetime,ctypes
w=Path(__file__).parent.resolve();r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-probability-bins-20261006');old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p,h=None):
 p=Path(p).resolve();s=sha(p)
 if h is not None:assert s==h,(p,s,h)
 pins[str(p)]=s
v=json.loads((w/'native/validation.json').read_text());assert v['allclose']and v['bitwise_mismatches']==0 and v['calls'][:3]==[48,0,0]and v['product_calls']==23040 and not v['callback_errors']
a=json.loads((w/'candidate/spike_receipt.json').read_text());assert a['status']=='pass'and a['returncode']==0
control=(old/'candidate/spike.log').read_text();candidate=(w/'candidate/spike.log').read_text();assert re.findall(r'^WORKSPACE_STAT .*$',control,re.M)==re.findall(r'^WORKSPACE_STAT .*$',candidate,re.M);assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in candidate
before=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',control)[1]);after=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',candidate)[1]);assert sha(w/'control/model.elf')==sha(old/'candidate/model.elf')
lib=ctypes.CDLL(str(w/'native_numeric_frozen/provider.so'));lib.group_provider_workspace_bytes.restype=ctypes.c_size_t;assert lib.group_provider_workspace_bytes()==123012160
snap=r/'docs/perf_records/source_snapshots/prepared_probability_bins';snap.mkdir(parents=True,exist_ok=True)
for name in ['src/merlin/llvmlower/source_attention_frontier.py','merlin/runtime/c/prepared_bf16_interval.h','merlin/tests/runtime/test_prepared_bf16_interval.py','merlin/tests/runtime/test_source_attention_frontier.py','docs/runtime/prepared_probability_bins.md']:
 data=subprocess.check_output(['git','show','4566c5877:'+name],cwd=core);assert data==(core/name).read_bytes();dest=snap/Path(name).name;dest.write_bytes(data);pin(dest)
for p in w.glob('*.py'):dest=snap/p.name;shutil.copyfile(p,dest);pin(dest)
for p in w.rglob('*'):
 if p.is_file()and p.name not in ['process.json','native_process.json']:pin(p)
for p in [old/'candidate/model.elf',old/'candidate/spike.log',old/'numeric_frozen/provider.o',old/'candidate/build.json']:pin(p)
for command in json.loads((w/'candidate/build.json').read_text())['link']:
 p=Path(command)
 if p.is_file():pin(p)
for command in v['commands']:
 for arg in command:
  p=Path(arg)
  if p.is_file():pin(p)
q={'schema':'prepared_probability_bins_qualification_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','core_commit':'4566c5877','scope':'Explicit default-off exact endpoint bin value reuse; original replay/refinement/intervals unchanged. Complete group target and full48 native numeric adapter only; not normal new implementation-seal or whole target qualification.','control_elf_sha256':sha(w/'control/model.elf'),'candidate_elf_sha256':sha(w/'candidate/model.elf'),'control_byte_reproduced':True,'only_provider_object_changed':True,'native':v,'target':a,'independent_target':json.loads((w/'independent_target/strict_receipt.json').read_text()),'target_independent_modes':5,'raw_check_calls':3376815,'all_eight_group_stats_unchanged':True,'workspace_bytes':123012160,'instructions':{'control':before,'candidate':after,'reduction_percent':100*(before-after)/before,'scope':'Allocation-aware complete group retired instructions, not cycles'},'hardware_requested':False,'limitations':['Earlier source-site total does not predict realized improvement; actual reduction is only0.685%.','Native full48 retains prior accepted ordinary host and changes numeric library; fresh production implementation reseal remains separate.','Independent first proof log lacked observed rc; retained and superseded by strict_rc.log with actual rc0.'],'pins':pins,'token_usage_available':False};out=r/'docs/perf_records/prepared_probability_bins_qualification.json';out.write_text(json.dumps(q,indent=2)+'\n');print(out,len(pins),q['instructions'])
