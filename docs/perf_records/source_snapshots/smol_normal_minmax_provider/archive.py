from pathlib import Path
import json,hashlib,datetime,shutil
w=Path(__file__).resolve().parent;root=w.parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={}
def pin(p):
 p=Path(p);pins[str(p)]=sha(p)
def read(p):return json.loads(Path(p).read_text())
hp=read(w/'build/host_provider/host_provider.json');assert hp['status']=='completed' and len(hp['objects'])==17
for p,h in hp['compilation_pins'].items():assert sha(p)==h;pin(p)
for record in hp['source_model']+[o[k]for o in hp['objects']for k in ('object','llvm')]:assert sha(record['path'])==record['sha256'];pin(record['path'])
v=read(w/'native/validation.json');assert v['allclose'] and v['bitwise_mismatches']==0 and v['calls'][:3]==[48,0,0] and v['product_calls']==23040 and not v['callback_errors']
f=read(w/'native/source_fallback_validation.json');assert f['mismatches']==0 and f['output_descriptor_unchanged'] and f['all_inputs_unchanged'] and f['fallback_calls']==1
for p,h in f['input_pins'].items():assert sha(p)==h;pin(p)
for k,h in [('native/model.so',v['library_sha256']),('native/model.o',v['model_object_sha256'])]:assert sha(w/k)==h;pin(w/k)
old=w.parent/'normal_attention_provider';assert sha(w/'build/model.o')==sha(old/'build/model.o');pin(old/'build/model.o')
a=(old/'build/lower/model.ll').read_bytes().splitlines(keepends=True);b=(w/'build/lower/model.ll').read_bytes().splitlines(keepends=True);assert a[0].startswith(b'; ModuleID = ') and b[0].startswith(b'; ModuleID = ') and a[1:]==b[1:]
audit=read(w/'build/model.nofsm_audit.json');assert audit['status']=='pass'
for rel in ['build.py','run.sh','validate_native.py','run_native.sh','check_source_fallback.py','declared_contracts.json','preparation.json','build_result.json','numeric_frozen/reclosure.json','build/compilation_recipe.json','build/host_provider/host_provider.json','build/model.elf','build/model.nofsm_audit.json','native/validation.json','native/source_fallback_validation.json','native/compiled_model_reuse.json','native/output.npy','native.log']:pin(w/rel)
proof=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/numeric_minmax_native_screen')
for rel in ['registered_numeric_witness.json','complete_observation_pair.json','native_validation.json']:pin(proof/rel)
for p,h in read(proof/'registered_numeric_witness.json')['numeric_dependency_pins'].items():assert sha(p)==h;pin(p)
result=read(w/'build_result.json');r={'scope':'Ordinary source-bound minmax numerical provider,17explicit imports/ABI/compiler/object/final-link closure and actual normal private-workspace/retained-source native gates. New whole target execution held until prior target terminal.','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elf_sha256':sha(w/'build/model.elf'),'build_hash':result['build_hash'],'provider_objects':17,'implementation_contract':read(w/'declared_contracts.json'),'native':v,'source_fallback':f,'host_model_object_byte_identical_control':True,'host_llvm_difference':'Only first nonsemantic ModuleID path comment; all remaining bytes identical. Native host object reuse explicitly proved and recorded.','native_integer_products':'Exact functional stand-ins, not device performance. Target imports are actual15xDSLproductobjects.','nofsm':audit,'original_gate_unchanged':True,'whole_target_status':'not launched; prior row/floor whole target is live','hardware_admitted':False,'token_usage_available':False,'pins':pins}
p=root/'docs/perf_records/smol_normal_minmax_provider_build_native.json';p.write_text(json.dumps(r,indent=2)+'\n')
snap=root/'docs/perf_records/source_snapshots/smol_normal_minmax_provider';snap.mkdir(parents=True,exist_ok=True)
for name in ['build.py','run.sh','validate_native.py','run_native.sh','check_source_fallback.py','archive.py']:shutil.copyfile(w/name,snap/name)
print(p,len(pins),r['build_hash'])
