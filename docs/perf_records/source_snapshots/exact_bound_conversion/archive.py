from pathlib import Path
import json,hashlib,re,subprocess,shutil,datetime
w=Path(__file__).parent.resolve();r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-exact-bound-conversion-20261006');old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p)
snap=r/'docs/perf_records/source_snapshots/exact_bound_conversion';snap.mkdir(parents=True,exist_ok=True)
for repo,commit,names in [(core,'e0c524854',['src/merlin/llvmlower/exact_bound_conversion.py','src/merlin/llvmlower/source_attention_frontier.py','merlin/runtime/c/ordered_fma_bounds.h','merlin/runtime/c/prepared_fma_product_bounds.h','merlin/runtime/c/separable_fma_radius.h','merlin/tests/runtime/test_exact_bound_conversion.py','merlin/tests/runtime/test_source_attention_frontier.py','docs/runtime/exact_bound_conversion.md']),(r,'037aa23',['mlir_oot/host_outward_fp.py','tests/test_host_outward_fp.py'])]:
 for name in names:
  data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo);assert data==(repo/name).read_bytes();dest=snap/Path(name).name;dest.write_bytes(data);pin(dest)
for p in w.glob('*.py'):shutil.copyfile(p,snap/p.name);pin(snap/p.name)
for p in w.rglob('*'):
 if p.is_file()and p.name not in ('native_process.json','process.json'):pin(p)
for directory in ('default_numeric','numeric_frozen','native_numeric_frozen'):
 for command in json.loads((w/directory/'compile.json').read_text())['commands']:
  pin(Path(command[0]).resolve())
  if '-MF' in command:
   dep=Path(command[command.index('-MF')+1]);pin(dep)
   for path in dep.read_text().replace('\\\n',' ').split(':',1)[1].split():pin(path)
for arm in ('control','candidate'):
 for path in json.loads((w/arm/'build.json').read_text())['link']:
  if Path(path).is_file():pin(path)
v=json.loads((w/'native/validation.json').read_text());assert v['bitwise_mismatches']==0 and v['allclose'] and v['calls'][:3]==[48,0,0]and v['product_calls']==23040
for command in v['commands']:
 for path in command:
  if Path(path).is_file():pin(path)
target=json.loads((w/'candidate/spike_receipt.json').read_text());assert target['returncode']==0 and target['status']=='pass';assert sha(w/'default_numeric/provider.o')==sha(old/'numeric_frozen/provider.o');assert sha(w/'control/model.elf')==sha(old/'candidate/model.elf')
a=(old/'candidate/spike.log').read_text();b=(w/'candidate/spike.log').read_text();assert re.findall(r'^WORKSPACE_STAT .*$',a,re.M)==re.findall(r'^WORKSPACE_STAT .*$',b,re.M);assert re.findall(r'^UNOBSERVED_CARRIER.*$',a,re.M)==re.findall(r'^UNOBSERVED_CARRIER.*$',b,re.M);before=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',a)[1]);after=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',b)[1]);pin(old/'candidate/spike.log');pin(old/'candidate/model.elf');pin(old/'numeric_frozen/provider.o')
q={'schema':'exact_finite_bound_conversion_qualification_v1','status':'pass','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_commit':'e0c524854','target_provider_commit':'037aa23','scope':'Only exact finite-domain f64 to f32 bound floor/ceil conversion. Source operations, intervals, replay and observations unchanged; polynomial conversion hooks not enabled. Complete current71193 control, allocation-aware group ROI.','instructions':{'control':before,'candidate':after,'reduction_percent':100*(before-after)/before,'hardware_cycles':None},'target':target,'native48':v,'independent_proof':json.loads((w/'independent_target/receipt.json').read_text()),'default_provider_object_and_elf_byteidentical':True,'all_eight_stats_and_carrier_diagnostics_unchanged':True,'core_focused_tests':35,'provider_tests':17,'normal_binding':'fresh normalized build running separately; numeric adapter qualification does not replace it','hardware_requested':False,'token_usage_available':False,'pins':pins};p=r/'docs/perf_records/exact_bound_conversion_qualification.json';p.write_text(json.dumps(q,indent=2)+'\n');print(p,len(pins),q['instructions'])
