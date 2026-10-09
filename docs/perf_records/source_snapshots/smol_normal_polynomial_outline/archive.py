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
for command in v['commands']:
 for arg in command:
  if Path(arg).is_file():pin(arg)
for p in (w/'native').iterdir():
 if p.is_file():pin(p)
pin('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle/model.mlir','4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc')
pin('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy',v['original_golden_sha256'])
closure=read(w/'workspace_llvm_closure.json')
assert closure['allocation_count']==1 and closure['call_count']==48 and closure['required_bytes']==123012928 and closure['public_forward_signature_equal_prior_normal']
for rec in closure['files']:pin(rec['path'],rec['sha256'])
audit=read(w/'build/model.nofsm_audit.json');assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown'];pin(w/'build/model.elf',audit['elf_sha256'])
# Freeze local scripts/headers before later edits; target compiler dependency pins above are independently reclosed.
snap=root/'docs/perf_records/source_snapshots/smol_normal_polynomial_outline'
snap.mkdir(parents=True,exist_ok=True)
for name in ['close_normal.py','prepare.py','build.py','run.sh','validate_native.py','run_native.sh','check_source_fallback.py','watch_normal.py','archive.py']:
 shutil.copyfile(w/name,snap/name);pin(snap/name)
for directory in ['numeric_frozen','native_numeric_frozen','bridge']:
 for p in (w/directory).iterdir():
  if p.is_file():pin(p)
for rel in ['numeric_identity_validation.json','declared_contracts.json','preparation.json','workspace_contract.json','workspace_llvm_closure.json','numeric_witness.json','build_result.json','build/compilation_recipe.json','build/lower/lowering_recipe.json','build/host_provider/host_provider.json','build/model.nofsm_audit.json','native/validation.json','native/source_fallback_validation.json','native/compiled_model_reuse.json','native/output.npy','native.log']:pin(w/rel)
prior=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_frontier_composed');pin(prior/'build/lower/model.ll');pin(prior/'build/model.o')
caps=Path('/scratch/agustin/tmp/gemmini-polynomial-outline-20261006/docs/perf_records/polynomial_outline_complete_group_qualification.json');pin(caps)
r={'schema':'normal_polynomial_outline_qualification_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','scope':'Ordinary full48 explicitly outlined polynomial provider, unchanged private workspace ABI and actual normal native execution with exact integer product stand-ins. Not a whole target execution or hardware result.','elf_sha256':sha(w/'build/model.elf'),'build_hash':read(w/'build_result.json')['build_hash'],'source_contract':read(w/'declared_contracts.json'),'workspace_llvm':closure,'native':v,'retained_source_fallback':f,'nofsm':audit,'normal_native_model_recompiled_in_fresh_directory':bool(read(w/'native/compiled_model_reuse.json').get('compile_command')),'private_workspace_bytes_increased':0,'public_abi_unchanged':True,'same_workspace_pointer_all48_calls':True,'actual_xdsl_product_objects':15,'imported_provider_objects':17,'matched_capsule_retired_instructions':{'control':2165148233,'candidate':2143755020,'scope':'Complete allocation-aware group ROI, not cycles; allocation and initialization included.'},'numeric_identity_validation':read(w/'numeric_identity_validation.json'),'core_commit':'f3b2103ad','target_provider_commit':'037aa23','probability_point_spans_enabled':True,'outline_polynomial_batch':True,'whole_target_execution':'not launched','hardware_admitted':False,'original_accuracy_gate':{'atol':0.03125,'rtol':0.02,'unchanged':True},'token_usage_available':False,'inherited_numeric_witness_core_commit_scope':'The composed_features core_commit records base point feature0b8; actual outlined implementation f3b2103ad and frozen C/header/compiler identities are authoritative in this receipt and linked group qualification.','pins':pins}
out=root/'docs/perf_records/smol_normal_polynomial_outline_qualification.json';out.write_text(json.dumps(r,indent=2)+'\n');print(out,len(pins),r['elf_sha256'])
