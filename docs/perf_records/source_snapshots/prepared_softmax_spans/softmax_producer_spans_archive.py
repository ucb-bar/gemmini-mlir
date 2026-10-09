from pathlib import Path
import json,hashlib,subprocess,re,shutil,datetime
w=Path(__file__).parent.resolve();r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-softmax-producer-span-20261006');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p):
 p=Path(p).resolve();pins[str(p)]=sha(p)
snap=r/'docs/perf_records/source_snapshots/prepared_softmax_spans';snap.mkdir(parents=True,exist_ok=True)
for name in ['src/merlin/llvmlower/source_attention_frontier.py','src/merlin/llvmlower/prepared_softmax_spans.py','merlin/runtime/c/prepared_softmax_spans.h','merlin/tests/runtime/test_prepared_softmax_spans.py','merlin/tests/runtime/test_source_attention_frontier.py','docs/runtime/prepared_softmax_spans.md']:
 data=subprocess.check_output(['git','show','df669e10c:'+name],cwd=core);assert data==(core/name).read_bytes();dest=snap/Path(name).name;dest.write_bytes(data);pin(dest)
results=[]
for case,old in [(w,Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group')),(r/'out/softmax_producer_spans_encoded',r/'out/encoded_row_equality')]:
 target=json.loads((case/'candidate/spike_receipt.json').read_text());native=json.loads((case/'native/validation.json').read_text());assert target['status']=='pass'and target['returncode']==0;assert native['bitwise_mismatches']==0 and native['allclose'] and native['calls'][:3]==[48,0,0] and native['product_calls']==23040
 a=(old/'candidate/spike.log').read_text();b=(case/'candidate/spike.log').read_text();assert re.findall(r'^WORKSPACE_STAT .*$',a,re.M)==re.findall(r'^WORKSPACE_STAT .*$',b,re.M);assert re.findall(r'^UNOBSERVED_CARRIER.*$',a,re.M)==re.findall(r'^UNOBSERVED_CARRIER.*$',b,re.M);assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in b
 before=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',a)[1]);after=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',b)[1]);assert sha(case/'control/model.elf')==sha(old/'candidate/model.elf')
 for p in case.rglob('*'):
  if p.is_file()and p.name not in ('native_process.json','process.json'):pin(p)
 for p in case.glob('*.py'):
  dest=snap/(case.name+'_'+p.name);shutil.copyfile(p,dest);pin(dest)
 for name in ('numeric_frozen','native_numeric_frozen'):
  for command in json.loads((case/name/'compile.json').read_text())['commands']:
   pin(Path(command[0]).resolve())
   if '-MF' in command:
    dep=Path(command[command.index('-MF')+1]);pin(dep)
    for path in dep.read_text().replace('\\\n',' ').split(':',1)[1].split():pin(path)
 for command in native['commands']:
  for path in command:
   if Path(path).is_file():pin(path)
 for path in json.loads((case/'candidate/build.json').read_text())['link']:
  if Path(path).is_file():pin(path)
 for p in [old/'candidate/model.elf',old/'candidate/spike.log',old/'candidate/build.json']:pin(p)
 results.append({'name':case.name,'control_elf_sha256':sha(case/'control/model.elf'),'candidate_elf_sha256':sha(case/'candidate/model.elf'),'control_byte_reproduced':True,'target':target,'native':native,'instructions':{'control':before,'candidate':after,'reduction_percent':100*(before-after)/before},'original_consumer_and_stats_unchanged':True,'only_provider_object_changed':True})
q={'schema':'private_softmax_producer_spans_qualification_v1','status':'pass','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_commit':'df669e10c','scope':'Optional private producer coverage removes repeated softmax finite/order/mask scan; source rounding, intervals, replay and observations unchanged. Two independent complete allocation-aware target comparisons, not summed gains. Full48 native numeric adapters reuse separately accepted actual workspace/host controls; not fresh normal implementation reseals or whole target qualification.','results':results,'independent_target':json.loads((w/'independent_target/strict_receipt.json').read_text()),'focused_tests':42,'default_identity':json.loads((w/'default_identity.json').read_text()),'hardware_requested':False,'token_usage_available':False,'pins':pins};p=r/'docs/perf_records/private_softmax_producer_spans_qualification.json';p.write_text(json.dumps(q,indent=2)+'\n');print(p,len(pins));print([x['instructions']for x in results])
