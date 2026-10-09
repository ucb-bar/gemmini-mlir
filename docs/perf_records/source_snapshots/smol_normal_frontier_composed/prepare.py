from pathlib import Path
import json,hashlib,shutil,subprocess,ctypes as C
w=Path(__file__).resolve().parent;source=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed');prior=w.parent/'normal_attention_provider_word_soft_i64_sealed';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((source/'native/validation.json').read_text())['bitwise_mismatches']==0
for name in ['numeric_frozen','native_numeric_frozen']:
 old=source/name;d=w/name;d.mkdir()
 for p in old.iterdir():
  if p.suffix in ('.h','.c'):shutil.copyfile(p,d/p.name)
 commands=json.loads((old/'compile.json').read_text())['commands'];actual=[]
 for cmd in commands:
  cmd=[x.replace(str(old),str(d))for x in cmd];subprocess.run(cmd,check=True);actual.append(cmd)
 deps=(d/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split();pins={str(Path(p).resolve()):sha(p)for p in deps};pins[str(Path(actual[0][0]).resolve())]=sha(Path(actual[0][0]).resolve())
 if name=='numeric_frozen':
  assert sha(d/'provider.o')==sha(old/'provider.o');(d/'reclosure.json').write_text(json.dumps({'compile_commands':actual,'compilation_pins':pins,'compile_cwd':str(Path.cwd()),'object_sha256':sha(d/'provider.o'),'same_qualified_provider_object':True},indent=2)+'\n')
 else:
  manifest={'compile':actual[0],'compile_commands':actual,'compile_cwd':str(Path.cwd()),'compiler_sha256':sha(Path(actual[0][0]).resolve()),'dependency_file_sha256':sha(d/'provider.d'),'transitive_compile_dependencies':{str(Path(p).resolve()):sha(p)for p in deps},'local_pins':{p.name:sha(p)for p in d.iterdir()if p.suffix in('.h','.c','.so')},'scope':'Fresh actual encoded rows, producer spans, exact casts, probability bins and four-cell scheduling implementation, not inherited identity'}
  (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
lib=C.CDLL(str(w/'native_numeric_frozen/provider.so'));lib.group_provider_workspace_bytes.restype=C.c_size_t;lib.group_provider_workspace_alignment.restype=C.c_size_t;size=lib.group_provider_workspace_bytes();assert size==123012928
bridge=w/'bridge';bridge.mkdir();shutil.copyfile(prior/'bridge/bridge.c',bridge/'bridge.c');base=json.loads((w/'numeric_frozen/reclosure.json').read_text())['compile_commands'][0];common=base[:base.index('-c')]
for ext,flags in [('o',['-c']),('ll',['-S','-emit-llvm'])]:subprocess.run([*common,*flags,str(bridge/'bridge.c'),'-o',str(bridge/('bridge.'+ext))],check=True)
(w/'workspace_contract.json').write_text(json.dumps({'required_bytes':size,'alignment':64,'query':'actual compiled group_provider_workspace_bytes()','normal_gate':'pending','native_provider_sha256':sha(w/'native_numeric_frozen/provider.so')},indent=2)+'\n')
old=json.loads((prior/'numeric_witness.json').read_text());keys=['source_contract','quant_contract','workspace_effect_contract','compiled_source_quant_proof_sha256','full_live_consumer_closure_sha256','complete_consumer_semantic_sha256','complete_source_dag_semantic_sha256','numerical_policy'];proof={k:old[k]for k in keys};native=w/'native_numeric_frozen';m=json.loads((native/'manifest.json').read_text());pins=dict(m['transitive_compile_dependencies']);pins.update({str(native/p):h for p,h in m['local_pins'].items()});pins[str(Path(m['compile'][0]).resolve())]=m['compiler_sha256'];pins[str(native/'provider.d')]=sha(native/'provider.d');pins[str(native/'provider.so')]=sha(native/'provider.so')
proof.update(workspace_bytes=size,workspace_alignment=64,native_shared_sha256=sha(native/'provider.so'),complete_compile_manifest_sha256=sha(native/'manifest.json'),numeric_dependency_pins=pins,qualification_scope='Encoded rows, producer spans, exact casts, probability bins and four-cell scheduling independent target and complete original group/full48 native numeric gate; fresh normal integration pending',prior_source_observation_proof={'path':str(prior/'numeric_witness.json'),'sha256':sha(prior/'numeric_witness.json'),'scope':'Source/consumer semantic proof only; old implementation identity not authoritative'},exact_bound_conversion={'core_commit':'e0c524854','target_provider_commit':'037aa23','selected_native':str(source/'native/validation.json'),'selected_native_sha256':sha(source/'native/validation.json'),'target_group':str(source/'candidate/spike_receipt.json'),'target_group_sha256':sha(source/'candidate/spike_receipt.json')});(w/'numeric_witness.json').write_text(json.dumps(proof,indent=2)+'\n')
print('fresh identities and actual ABI ready',size)

proof["composed_features"]={"word_interval_enclosure":True,"integer_reconstruction":True,"prepare_encoded_rows":True,"prepare_softmax_spans":True,"exact_bound_conversion":True,"prepare_probability_bins":True,"polynomial_batch_four":True,"core_commit":"973a944f9"}
(w/"numeric_witness.json").write_text(json.dumps(proof,indent=2)+"\n")
