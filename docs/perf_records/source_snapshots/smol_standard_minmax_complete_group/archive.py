"""Archive closed source capability, byte-identical emission, and rejected guard."""
from pathlib import Path
import hashlib,json,shutil
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/smol-minmax-20261006'
records=root/'docs/perf_records';snapshot=records/'source_snapshots/smol_standard_minmax_complete_group'
snapshot.mkdir(exist_ok=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text())
pins={}
def add(path,expected=None):
 path=str(Path(path).resolve());digest=sha(path)
 if expected is not None:assert digest==expected,path
 if path in pins:assert pins[path]==digest,path
 pins[path]=digest
def counter(path):
 text=Path(path).read_text();assert'WORKSPACE_GROUP ALL196608 AND GUARDS PASS'in text
 value=int(next(line.split()[1]for line in text.splitlines()if line.startswith('WORKSPACE_GROUP_CYCLES ')))
 return value,[line for line in text.splitlines()if line.startswith('WORKSPACE_STAT ')]
default=read(work/'core_default/compile.json');selected=read(work/'core_selected/compile.json')
build=read(work/'builtin/build.json');guard=read(work/'guarded/build.json')
assert default['object_byte_exact']and selected['object_byte_exact']
add(default['comparison_object'],default['object_sha256']);add(selected['comparison_object'],selected['object_sha256'])
add(work/'core_selected/model.elf',selected['elf_sha256']);add(work/'builtin/model.elf',selected['elf_sha256'])
control=Path(build['control_build']).parent
value,stats=counter(work/'builtin/spike.log');baseline,oldstats=counter(control/'spike.log');negative,negative_stats=counter(work/'guarded/spike.log')
assert value==3243485384 and baseline==3370349620 and negative==3588927896
assert stats==oldstats==negative_stats
audits={}
for name in('builtin','guarded'):
 receipt=read(work/name/'spike_receipt.json');assert receipt['status']=='pass'and receipt['returncode']==0
 add(work/name/'model.elf',receipt['elf_sha256']);add(work/name/'spike.log',receipt['log_sha256'])
 audits[name]=audit_elf((work/name/'model.elf').read_bytes());assert audits[name]['status']=='pass'
for name,marker in [('representation','GUARDED_MINMAX PASS 859360'),('representation_builtin','BUILTIN_MINMAX PASS 859360')]:
 assert(work/(name+'.log')).read_text().strip()==marker
 proof=read(work/(name+'_build.json'))
 for path,digest in proof['pins'].items():add(path,digest)
 audits[name]=audit_elf((work/(name+'.elf')).read_bytes());assert audits[name]['status']=='pass'
for receipt in(default,selected,build,guard):
 for path,digest in receipt['pins'].items():add(path,digest)
same=lambda command,provider:[arg for arg in command if arg.endswith('.o')and arg!=str(provider)]
assert same(build['link'],work/'builtin/provider.o')==same(read(control/'build.json')['link'],control/'provider.o')
manifest=read(work/'native_frozen/manifest.json');journey=read(records/'smol_numeric_minmax_full48_journey.json')
add(work/'native_frozen/manifest.json',journey['frozen_manifest_sha256'])
for path,digest in journey['independently_rehashed_frozen_pins'].items():add(path,digest)
for path,digest in manifest['local_pins'].items():add(work/'native_frozen'/path,digest)
for path,digest in manifest['transitive_compile_dependencies'].items():add(path,digest)
add(manifest['compile_commands'][0][0],manifest['compiler_sha256'])
validation=read(records/'smol_numeric_minmax_full48_validation.json');gate=journey['gate']
assert validation['allclose']and validation['bitwise_mismatches']==0and validation['elements']==1600
assert validation['actual_runtime_calls']==48and validation['full_group_source_fallback_calls']==0
assert validation['original_atol']==.03125and validation['original_rtol']==.02
assert validation['evaluator_shared_sha256']==sha(work/'native_frozen/provider.so')
assert gate['original_input_equal_groups']==48and gate['compiled_source_integer_words']==9437184and gate['compiled_source_scale_words']==12288
assert gate['integer_mismatches']==gate['bf16_scale_mismatches']==0
files=[*work.glob('*.py'),*work.glob('*.h'),*work.glob('representation*.c'),*work.glob('representation*.json'),*work.glob('representation*.log')]
for arm in('builtin','guarded','core_default','core_selected'):
 for path in(work/arm).iterdir():
  if path.suffix in('.json','.log')or path.name=='numeric_capability.h':files.append(path)
for path in files:
 add(path);relative=path.relative_to(work);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination);add(destination)
for name in('smol_numeric_minmax_full48_validation.json','smol_numeric_minmax_full48_observations.json','smol_numeric_minmax_full48_journey.json'):add(records/name)
for name in('build.json','spike.log','spike_receipt.json'):add(control/name)
receipt={'schema':'standard_minmax_production_attention_v1','scope':'Complete original12head/256query/1024key group. Functional Spike retired instructions, notFireSimcycles orwholemodelestimate.','core_commit':'1c5b1d1dc','ownership':'General numericcapability/defaultlibraryhooks inMerlin; actualISA/products/resources/rankedABI inOOT.','default_object_byte_exact':True,'general_emitter_object_and_elf_byte_exact_to_measured_experiment':True,'control_instructions':baseline,'candidate_instructions':value,'reduction_percent':100*(baseline-value)/baseline,'unchanged_device_driver_bridge_input_objects':True,'accepted_carriers_and_guards':196608,'unchanged_stats':stats,'source_replay_fmas':4461440,'product_calls':480,'logical_readback_bytes':86507520,'workspace_bytes':121963584,'contract':build['contract'],'independent_target_pair_checks':859360,'target_rounding_modes':5,'observed_scope':'Allotherresultbits,infinitysigns,NaNpairedwithnumber,operandonce,roundmodepreserved. Min/maxsignedzero+NaNpayload distinctions independentlyunobserved; wholeoriginalconsumer observationsstill exact.','full48_native':gate,'original_gate':{'elements':1600,'bitwise_mismatches':0,'atol':.03125,'rtol':.02,'fallback_calls':0},'native_so_sha256':sha(work/'native_frozen/provider.so'),'elf_sha256':selected['elf_sha256'],'rejected_guarded_variant':{'source_zero_nan_library_paths_preserved':True,'instructions':negative,'increase_percent':100*(negative-baseline)/baseline,'all_original_carriers_guards_and_stats_exact':True,'production_enabled':False},'nofsm':audits,'pins':pins,'default_policy_changed':False,'source_accuracy_gate_changed':False,'hardware_admitted':False,'token_usage_available':False}
(records/'smol_standard_minmax_complete_group.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'closed','pins':len(pins),'instructions':value,'reduction_percent':receipt['reduction_percent'],'negative_guard_percent':receipt['rejected_guarded_variant']['increase_percent'],'full48':'pass','hardware':'unknown'}))
