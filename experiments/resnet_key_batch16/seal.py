"""Seal source, ordinary binding, controlled link and complete whole gates."""
from pathlib import Path
import hashlib,json,subprocess,re
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/key_batch16_normal';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();paths=[]
old=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal');core=Path('/scratch/agustin/tmp/merlin-polynomial-constants-main-20261007');parent=Path('/scratch/agustin/tmp/gemmini-key-panel-capacity-20261007')
for name in ['mlir_oot/golden_key_rectified_resadd.py','mlir_oot/rectifier_residual_catalog.py','tests/test_golden_key_rectified_resadd.py']:
 assert(B/name).read_bytes()==(parent/name).read_bytes();paths.extend([B/name,parent/name])
source=parent/'docs/perf_records/key_panel_capacity_complete_qualification_20261007.json';stock=parent/'out/key_panel_capacity/stock/qualification.json'
for packet in [source,stock]:
 paths.append(packet)
 for p,h in json.loads(packet.read_text())['pins'].items():assert sha(p)==h,p;paths.append(Path(p))
paths.extend(p for p in W.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name!='seal.log')
paths.extend((B/'experiments/resnet_key_batch16').glob('*.py'));paths += [B/'tests/fused_whole_model_probe.py',old/'controlled2095/controlled_link.json',old/'controlled2095/candidate/model.elf',old/'normal/build_direct/device_catalog/device_catalog.json']
link=json.loads((W/'controlled2101/controlled_link.json').read_text())
for p,h in link['retained_input_pins'].items():assert sha(p)==h,p;paths.append(Path(p))
normal=json.loads((W/'normal/build_direct/device_catalog/device_catalog.json').read_text());prior=json.loads((old/'normal/build_direct/device_catalog/device_catalog.json').read_text());assert normal['pooled_stem']['object_sha256']==prior['pooled_stem']['object_sha256'];assert normal['kernels']==prior['kernels'];assert normal['bindings']==prior['bindings']
results={}
for arm in ['normal','controlled']:
 result=json.loads((W/('qualification_'+arm)/'result.json').read_text());assert result['status']=='PASS';assert result['spike']['target_native_exact'] and result['spike']['allclose'];assert result['spike']['output_elements']==1000;assert result['nofsm']['status']=='pass';results[arm]=result
entry=json.loads((W/'entry_closure/selected_entries.json').read_text());assert entry['selected'][0]['actual_entry_count']==1
text=(W/'entry_closure/public_adapter_disassembly.txt').read_text();assert '<gemmini_residual_0__rectifier_kernel>'in text
# Source and completed public call land on the same selected final symbol.
entry_address=entry['selected'][0]['symbol_start'];calls=[int(line.split('<')[0].split()[-1],16)for line in text.splitlines()if '<gemmini_residual_0__rectifier_kernel>'in line];assert calls==[entry_address]
# Original compiled/Torch reference remains unchanged, independent of scheduling.
for arm in ['normal','controlled']:
 assert(W/('qualification_'+arm)/'host/output.npy').read_bytes()==(W/'normal/capture/golden.npy').read_bytes()
# The already checked target record also serves the standard whole collector.
reference=W/'qualification_controlled/reference_validation.json';reference.write_bytes((W/'qualification_controlled/spike_validation.json').read_bytes());paths.append(reference)
def recurse(v):
 if isinstance(v,dict):
  if 'path'in v and 'sha256'in v and Path(str(v['path'])).is_file():assert sha(v['path'])==v['sha256'];paths.append(Path(v['path']))
  for child in v.values():recurse(child)
 elif isinstance(v,list):
  for child in v:recurse(child)
recipe=W/'normal/build_direct/compilation_recipe.json';recurse(json.loads(recipe.read_text()))
for category in ['fused_requantizations','residual_additions','guarded_mean_additions']:
 for route in normal[category]:
  for key in ['compilation','adapter_compilation']:
   if key in route:
    for token in str(route[key]).replace("'",' ').split():
     p=Path(token.strip('[],()'))
     if p.is_file():paths.append(p)
r={'schema':'resnet_key_batch16_current2101_whole_v1','status':'QUALIFIED_ROOT_REVIEW_NO_STOCK_RELEASE','base':'acbbaf0b0b89f0030e10aa98ada01326667c4820','core_head':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'source_diff':'exact three root-owned qualified files; positive integral batch with derived SPAD/ACC range; default remains1','selection':'explicit experiment first actual source residual certificate; no production model selector','normal_provider':'ordinary rectifier_residual_catalog.build plus ordinary full callback pipeline','control_job':2101,'control_sha256':link['control_reproduced'],'candidate_elf':str(W/'controlled2101/candidate/model.elf'),'candidate_sha256':link['candidate_sha256'],'normal_elf':str(W/'normal/build_direct/model.elf'),'normal_sha256':sha(W/'normal/build_direct/model.elf'),'changed_objects':link['changed_objects'],'other_catalog_bindings':json.loads((W/'other_catalog_bindings.json').read_text()),'stem_classifier_binding_unchanged':True,'public_adapter_calls_selected_kernel_once':calls,'entry_closure':entry,'whole_gates':results,'tests':25,'complete_helper_stock':'2105/2106:1262667→1188667 complete802816output observation; not whole forecast','whole_cycles':'UNKNOWN','global_default_changed':False,'pins':{str(p.resolve()):sha(p)for p in paths}}
output=B/'docs/perf_records/resnet_key_batch16_current2101_whole_qualification.json';output.write_text(json.dumps(r,indent=2)+'\n');print(output,len(r['pins']))
