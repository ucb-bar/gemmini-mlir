from pathlib import Path
import hashlib,json,shutil
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();w=root/'out/artifacts/probes/smol-classification-20261006';target=w/'classification';proof=w/'representation_proof'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((target/'build.json').read_text());s=json.loads((target/'spike_receipt.json').read_text());c=json.loads((target/'compile.json').read_text());default=json.loads((w/'default_control/compile.json').read_text());p=json.loads((proof/'build.json').read_text())
assert default['default_object_byte_exact']is True
assert s['status']=='pass'and s['returncode']==0
assert sha(target/'model.elf')==s['elf_sha256']and sha(target/'spike.log')==s['log_sha256']
text=(target/'spike.log').read_text();counter=int(next(line.split()[1]for line in text.splitlines()if line.startswith('WORKSPACE_GROUP_CYCLES ')))
control=Path(b['control_build']).parent;controltext=(control/'spike.log').read_text()
statistics=lambda t:[line for line in t.splitlines()if line.startswith('WORKSPACE_STAT ')]
assert statistics(text)==statistics(controltext)
assert 'WORKSPACE_GROUP ALL196608 AND GUARDS PASS'in text
assert (proof/'spike.log').read_text().strip()=='CLASSIFICATION PASS 565925'
# Snapshot the original fixture instead of making a future mutable core test
# file an immutable runtime dependency of this historical target proof.
fixture=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/merlin/tests/runtime/test_source_numeric_capability.py')
snapshot=proof/'original_fixture_test_source_numeric_capability.py';shutil.copy2(fixture,snapshot)
assert p['pins'][str(fixture)]==sha(snapshot)
p['source_fixture_original']={'path':str(fixture),'sha256':sha(snapshot),'snapshot':str(snapshot)}
p['pins'].pop(str(fixture));p['pins'][str(snapshot)]=sha(snapshot)
(proof/'build.json').write_text(json.dumps(p,indent=2)+'\n')
pins={}
for r in (b,c,default,p):
 for name,h in r['pins'].items():
  assert sha(name)==h,name
  if name in pins:assert pins[name]==h
  pins[name]=h
for name in [*w.glob('*.py'),target/'build.json',target/'compile.json',target/'spike_receipt.json',target/'spike.log',target/'watch.py',w/'default_control/compile.json',proof/'build.json',proof/'spike.log',control/'spike_receipt.json',control/'spike.log',w/'native_frozen/manifest.json']:
 pins[str(name)]=sha(name)
for d in (target,proof):
 a=audit_elf((d/('model.elf'if d==target else'classification.elf')).read_bytes());assert a['status']=='pass'
original=json.loads((control/'build.json').read_text())
assert [arg for arg in b['link']if arg.endswith('.o')and 'smol-classification-20261006/classification/provider.o'not in arg]==[arg for arg in original['link']if arg.endswith('.o')and str(control/'provider.o')!=arg]
native=json.loads((w/'native_frozen/manifest.json').read_text())
for name,h in native['local_pins'].items():assert sha(w/'native_frozen'/name)==h; pins[str(w/'native_frozen'/name)]=h
for name,h in native['transitive_compile_dependencies'].items():assert sha(name)==h;pins[name]=h
receipt={'schema':'standard_classification_production_attention_v1','scope':'Complete original12-head/256-query/1024-key group; actual normal compiler provider, signed-radix xDSL product callbacks and ranked workspace bridge. Retired instructions in functional Spike, not FireSim cycles or a whole-model prediction.','core_commit':'05e119b95','optimization':'Explicit standard finite classification compiler capability only; all original math and observations unchanged. Generic compiler capability in Merlin; actual target ABI/products/ISA proof in OOT.','default_object_byte_exact':True,'default_control_object_sha256':'3e7618d68769e8966aa07110fa56010e2e9a29c8f72af93b2163801d04a60570','control_instructions':b['control_instructions'],'candidate_instructions':counter,'reduction_percent':100*(b['control_instructions']-counter)/b['control_instructions'],'unchanged_objects':'All15product kernels, immutable input/source data, driver, bridge, startup/linker remain byteidentical. Only provider.o changed.','unchanged_stats':statistics(text),'complete_original_accepted_carriers_and_guards':196608,'original_quantized_i8':196608,'original_escaping_bf16_scales':256,'original_observation_reason':'Accepted carrier bytes match the prior complete source-quant qualified row/floor control; no new carrier approximation.','source_replay_fmas':4461440,'product_calls':480,'logical_readback_bytes':86507520,'workspace_bytes':121963584,'classification_contract':b['classification_contract'],'independent_target_representation_checks':565925,'target_rounding_modes':5,'remaining_classification_imports':'Five __fpclassifyd callsites for preexisting NaN tests remain. Static imports are not dynamic costs.','native_compiled_so_sha256':native['local_pins']['provider.so'],'full48_native_status':'Same actual SO as accepted full48row/floor; separate byteidentity/source/input/consumer witness reclosed by Tiny; no rerun or new native timing claim.','candidate_elf_sha256':s['elf_sha256'],'candidate_nofsm':b['nofsm'],'representation_nofsm':p['nofsm'],'spike_receipt':s,'pins':pins,'default_policy_changed':False,'source_accuracy_gate_changed':False,'whole_hardware_admitted':False,'token_usage_available':False}
(root/'docs/perf_records/smol_standard_classification_complete_group.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'instructions':counter,'reduction_percent':receipt['reduction_percent'],'live_pins':len(pins),'whole_hardware':'held'}))
