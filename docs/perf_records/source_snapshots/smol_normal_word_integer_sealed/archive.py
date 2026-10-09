from pathlib import Path
import json, hashlib, datetime, shutil
w=Path(__file__).resolve().parent
root=w.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
pins={}
def pin(p,expected=None):
 p=Path(p).resolve();h=sha(p)
 if expected is not None:assert h==expected,(str(p),h,expected)
 pins[str(p)]=h
hp=read(w/'build/host_provider/host_provider.json')
assert hp['status']=='completed' and len(hp['objects'])==17
for p,h in hp['compilation_pins'].items():pin(p,h)
for rec in hp['source_model']+[o[k] for o in hp['objects'] for k in ('object','llvm')]:pin(rec['path'],rec['sha256'])
v=read(w/'native/validation.json')
assert v['allclose'] and v['bitwise_mismatches']==0 and v['calls'][:3]==[48,0,0] and v['product_calls']==23040 and not v['callback_errors']
f=read(w/'native/source_fallback_validation.json')
assert f['mismatches']==0 and f['output_descriptor_unchanged'] and f['all_inputs_unchanged'] and f['fallback_calls']==1
for p,h in f['input_pins'].items():pin(p,h)
for rel,h in [('native/model.so',v['library_sha256']),('native/model.o',v['model_object_sha256']),('native_numeric_frozen/provider.so',v['numeric_library_sha256'])]:pin(w/rel,h)
closure=read(w/'workspace_llvm_closure.json')
assert closure['allocation_count']==1 and closure['call_count']==48 and closure['required_bytes']==123012160 and closure['public_forward_signature_equal_prior_normal']
for rec in closure['files']:pin(rec['path'],rec['sha256'])
audit=read(w/'build/model.nofsm_audit.json');assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown'];pin(w/'build/model.elf',audit['elf_sha256'])
# Freeze local scripts/headers before later edits; target compiler dependency pins above are independently reclosed.
snap=root/'docs/perf_records/source_snapshots/smol_normal_word_integer_sealed'
snap.mkdir(parents=True,exist_ok=True)
for name in ['prepare.py','build.py','run.sh','validate_native.py','run_native.sh','check_source_fallback.py','watch_normal.py','archive.py']:
 shutil.copyfile(w/name,snap/name);pin(snap/name)
for directory in ['numeric_frozen','native_numeric_frozen','bridge']:
 for p in (w/directory).iterdir():
  if p.is_file():pin(p)
for rel in ['numeric_identity_preflight.json','numeric_identity_validation.json','numeric_identity_collision_reclosure.json','historical_identity_refusals.json','validator_frozen/9bbedd45b.py','validator_frozen/ccbade851.py','declared_contracts.json','preparation.json','workspace_contract.json','workspace_llvm_closure.json','numeric_witness.json','build_result.json','build/compilation_recipe.json','build/lower/lowering_recipe.json','build/host_provider/host_provider.json','build/model.nofsm_audit.json','native/validation.json','native/source_fallback_validation.json','native/compiled_model_reuse.json','native/output.npy','native.log']:pin(w/rel)
prior=root/'out/normal_attention_provider';pin(prior/'build/lower/model.ll');pin(prior/'build/model.o')
caps=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/docs/perf_records/word_soft_i64_complete_group_qualification.json');pin(caps)
r={'schema':'normal_word_integer_sealed_qualification_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','scope':'Ordinary full48 source-bound integer reconstruction provider, refreshed private workspace ABI and actual normal native execution with exact integer product stand-ins. Not a whole target execution or hardware result.','elf_sha256':sha(w/'build/model.elf'),'build_hash':read(w/'build_result.json')['build_hash'],'source_contract':read(w/'declared_contracts.json'),'workspace_llvm':closure,'native':v,'retained_source_fallback':f,'nofsm':audit,'normal_native_model_recompiled_in_fresh_directory':False,'normal_native_model_reuse':read(w/'native/compiled_model_reuse.json'),'private_workspace_bytes_increased':1048576,'public_abi_unchanged':True,'same_workspace_pointer_all48_calls':True,'actual_xdsl_product_objects':15,'imported_provider_objects':17,'prior_old_workspace_capsule_refusal_preserved':True,'matched_capsule_retired_instructions':{'control':2726638089,'candidate':2537394919,'scope':'Complete allocation-aware group ROI, not cycles; extra scratch allocation and initialization included.'},'numeric_identity_validation':read(w/'numeric_identity_validation.json'),'collision_validation':read(w/'numeric_identity_collision_reclosure.json'),'historical_seal_defects_preserved':read(w/'historical_identity_refusals.json'),'historical_native_compile_command_is_inherited_evidence_not_reexecuted':True,'whole_target_execution':'not launched','hardware_admitted':False,'original_accuracy_gate':{'atol':0.03125,'rtol':0.02,'unchanged':True},'token_usage_available':False,'pins':pins}
out=root/'docs/perf_records/smol_normal_word_integer_sealed_qualification.json';out.write_text(json.dumps(r,indent=2)+'\n');print(out,len(pins),r['elf_sha256'])
