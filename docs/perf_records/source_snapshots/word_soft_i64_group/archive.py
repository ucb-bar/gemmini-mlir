from pathlib import Path
import json,hashlib,re,datetime,shutil
w=Path(__file__).resolve().parent;root=w.parents[1];h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());pins={}
def pin(p,expected=None):
 p=Path(p).resolve();digest=h(p)
 if expected is not None:assert digest==expected,(p,digest,expected)
 pins[str(p)]=digest
v=read(w/'native/validation.json');assert v['allclose'] and v['bitwise_mismatches']==0 and v['calls'][:3]==[48,0,0] and v['product_calls']==23040 and not v['callback_errors']
for p,d in [('native/model.o',v['model_object_sha256']),('native/model.so',v['library_sha256']),('native_numeric_frozen/provider.so',v['numeric_library_sha256'])]:pin(w/p,d)
old=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_i64');pin(old/'build/lower/model.ll',v['source_llvm_sha256']);pin(old/'native/model.o',v['model_object_sha256']);pin(old/'bridge/bridge.c');pin(old/'workspace_llvm_closure.json');pin(old/'declared_contracts.json')
prior=w.parent/'frontier_i64_allocated_pair_v2/candidate';assert h(prior/'model.elf')==h(w/'control/model.elf');stats={};metrics={}
for arm,d in [('control',prior),('candidate',w/'candidate')]:
 receipt=read(d/'spike_receipt.json');assert receipt['status']=='pass' and receipt['returncode']==0;pin(d/'model.elf',receipt['elf_sha256']);pin(d/'spike.log',receipt['log_sha256']);pin(d/'spike_receipt.json');audit=read(d/'model.nofsm_audit.json');assert audit['status']=='pass'and not audit['forbidden']and not audit['unknown'];pin(d/'model.nofsm_audit.json')
 raw=(d/'spike.log').read_text();assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in raw;metrics[arm]=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',raw)[1]);stats[arm]=[int(x)for _,x in re.findall(r'WORKSPACE_STAT (\d+) (\d+)',raw)];assert len(stats[arm])==8
 for arg in read(w/arm/'build.json')['link']:
  p=Path(arg)
  if p.is_absolute()and p.is_file():pin(p)
for directory in ['numeric_frozen','native_numeric_frozen','native','control','candidate']:
 for p in (w/directory).iterdir():
  if p.is_file():pin(p)
for name in ['build.py','validate_native.py','run_native.sh','native.log','archive.py']:pin(w/name)
snap=root/'docs/perf_records/source_snapshots/word_soft_i64_group';snap.mkdir(parents=True,exist_ok=True)
for name in ['build.py','validate_native.py','run_native.sh','archive.py']:shutil.copyfile(w/name,snap/name);pin(snap/name)
manifest=read(w/'native_numeric_frozen/manifest.json')
for name,digest in manifest['local_pins'].items():pin(w/'native_numeric_frozen'/name,digest)
r={'schema':'word_soft_i64_complete_group_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','scope':'Explicit prepared word-domain composition atop qualified private-soft/builtin-floor/i64 complete group; target original compiled consumer and frozen normal-host full48 native numerical gates. This is not a new normal provider binding seal or whole target execution.','core_commit':'b3f09def1','control_elf_sha256':h(prior/'model.elf'),'candidate_elf_sha256':h(w/'candidate/model.elf'),'control_reproduced_byte_exact':True,'only_target_object_changed':'provider.o','workspace_bytes':123012160,'allocation_and_initialization_in_roi':True,'same_device_calls':480,'same_i32_readback_bytes':86507520,'original_compiled_consumer':{'i8_words':786432,'bf16_scales':1024,'exact':True,'guards_and_inputs':True},'unobserved_carrier_differences':{'control':3,'candidate':4},'retired_roi_instructions':metrics,'instruction_reduction_percent':100*(1-metrics['candidate']/metrics['control']),'stats':stats,'full48_native':v,'fresh_implementation_seal_required_before_normal_provider_promotion':True,'hardware_cycles':None,'stock_admitted':False,'whole_target_executed':False,'token_usage_available':False,'pins':pins}
out=root/'docs/perf_records/word_soft_i64_complete_group_qualification.json';out.write_text(json.dumps(r,indent=2)+'\n');print(out,len(pins),r['instruction_reduction_percent'])
