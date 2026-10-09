"""Reclose owned evidence and archive positive and negative observations."""
import ctypes,hashlib,json,re,shutil,subprocess
from pathlib import Path
from mlir_oot.no_fsm_audit import audit_elf
from merlin.runtime.numeric_provider_identity import validate_numeric_provider_identity
root=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
docs=root/'docs/perf_records';snapshot=docs/'source_snapshots/root_checkpoint_20261006T1430'
snapshot.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,r):
 (docs/name).write_text(json.dumps(r,indent=2)+'\n')
def copy(path,name):
 out=snapshot/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out);return out
names=['stock1988_paired_resident_packets_terminal.json','stock1989_tiny_two_products_outline_terminal.json',
 'stock1990_exact1974_profile_terminal.json','prepared_probability_bins_qualification.json',
 'encoded_row_equality_complete_group_qualification.json','smol_normal_encoded_row_equality_qualification.json']
closures=[]
for name in names:
 r=json.loads((docs/name).read_text());pins=r['pins']
 for p,h in pins.items():assert sha(p)==h,(name,p)
 closures.append({'receipt':str(docs/name),'receipt_sha256':sha(docs/name),'pins_reclosed':len(pins)})
normal=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/normal_attention_provider_encoded_rows')
witness=json.loads((normal/'numeric_witness.json').read_text());native=normal/'native_numeric_frozen'
lib=ctypes.CDLL(str(native/'provider.so'));lib.group_provider_workspace_bytes.restype=ctypes.c_size_t;lib.group_provider_workspace_alignment.restype=ctypes.c_size_t
identity=validate_numeric_provider_identity(witness,manifest_path=native/'manifest.json',native_library_path=native/'provider.so',queried_workspace_bytes=lib.group_provider_workspace_bytes(),queried_workspace_alignment=lib.group_provider_workspace_alignment())
assert identity['workspace_bytes']==123012928
audit=audit_elf((normal/'build/model.elf').read_bytes());assert audit['status']=='pass'
save('root_encoded_row_normal_and_terminals_reclosure.json',{'schema':'golden_root_evidence_reclosure_v1',
 'status':'pass','closures':closures,'normal_encoded_row_identity':identity,'fresh_all_executable_nofsm':audit,
 'root_selected_tests':{'passed':66,'log':str(copy('/tmp/root_encoded_rows_integration_20261006.log','encoded_row_selected_tests.log'))},
 'scope':'Current typed compiler integration, fresh normal workspace/native1600/fallback identity and actual stock receipts. Native exact products are not whole target execution. No additive gains.'})

core=Path('/scratch/agustin/tmp/merlin-one-endpoint-word-20261006')
group=root/'out/artifacts/probes/root-one-endpoint-complete-group-20261006'
control=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group/candidate/spike.log')
files=[p for p in group.rglob('*')if p.is_file()and p.suffix in ('.py','.c','.h','.json','.log','.ll','.so','.o','.elf','.npy')]
files += [core/'merlin/runtime/c/one_endpoint_word_polynomial.h',core/'src/merlin/llvmlower/source_attention_frontier.py',
 core/'merlin/tests/runtime/test_one_endpoint_word_polynomial.py',core/'merlin/tests/runtime/test_source_attention_frontier.py',core/'docs/runtime/one_endpoint_word_polynomial.md',control]
group_build=json.loads((group/'build.json').read_text())
files += [Path(p)for p in group_build['original_link_input_pins']]
for p,h in group_build['original_link_input_pins'].items():assert sha(p)==h
sections=[]
for name in ['root-one-endpoint-word-20261006','root-one-endpoint-word-v2-20261006']:
 w=root/'out/artifacts/probes'/name;receipt=json.loads((w/'gsim_receipt.json').read_text());assert receipt['pass']
 arms=[int(m.rsplit(' ',1)[-1])for m in receipt['markers']if re.fullmatch(r'ONE_ENDPOINT_WORD [0-3] [0-9]+',m)]
 assert len(arms)==4
 sections.append({'probe':name,'elf_sha256':receipt['elf_sha256'],'engine_sha256':receipt['engine_sha256'],
  'ABBA_cycles':arms,'control_mean':(arms[0]+arms[3])/2,'candidate_mean':(arms[1]+arms[2])/2,
  'relative_change':(arms[1]+arms[2])/(arms[0]+arms[3])-1,'ambiguous_bf16_bins':[17,36]})
 files += [p for p in w.rglob('*')if p.is_file()and p.suffix in ('.py','.c','.h','.json','.log','.ll','.so','.o','.elf','.npy')]
for p in [core/'merlin/runtime/c/one_endpoint_word_polynomial.h',core/'src/merlin/llvmlower/source_attention_frontier.py',core/'merlin/tests/runtime/test_one_endpoint_word_polynomial.py',core/'merlin/tests/runtime/test_source_attention_frontier.py',core/'docs/runtime/one_endpoint_word_polynomial.md',group/'build.py',group/'validate_native.py']:
 copy(p,'one_endpoint/'+p.name)
for name in ['all_tests','structure_v2','docs']:
 p=Path('/tmp')/f'root_one_endpoint_{name}_20261006.log';files.append(copy(p,'one_endpoint/'+p.name))
files.append(Path('/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/build_receipt.json'))
pins={str(p.resolve()):sha(p)for p in files}
logs=[control.read_text(),(group/'candidate/spike.log').read_text()]
instructions=[int(re.search(r'^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$',s,re.M)[1])for s in logs]
stats=[[int(v)for _,v in re.findall(r'^WORKSPACE_STAT (\d+) (\d+)$',s,re.M)]for s in logs]
assert instructions==[2537394919,2727948979]
validation=json.loads((group/'native/validation.json').read_text());assert validation['bitwise_mismatches']==0 and validation['calls'][:3]==[48,0,0]
save('smol_one_endpoint_word_negative_journey.json',{'schema':'golden_complete_cost_negative_journey_v1',
 'status':'held_negative_complete_cost','core_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=core,text=True).strip(),
 'sections':sections,'complete_group_instructions':instructions,'relative_instruction_change':instructions[1]/instructions[0]-1,
 'all_eight_stats':stats,'unobserved_carrier_differences':[4,1346],'native':validation,
 'normal_numeric_identity':json.loads((group/'native_numeric_frozen/identity_validation.json').read_text()),
 'default_unchanged':True,'normal_core_imported':False,'hardware_admitted':False,
 'scope':'Exact interval enclosure widens BF16 ambiguity/replay. V2 isolated7.857% GSIM saving does not establish complete profitability; complete group regresses7.5098% instructions. Original786432i8/1024scales/guards/inputs and full1600 gate pass. Frozen accepted host reused, not a new normal binding/whole target run. Initial test and negative V1 evidence retained.',
 'pins':pins})

p=root/'out/artifacts/probes/root-resnet1988-profile-20261006'
profile=json.loads((p/'profile_build.json').read_text());validation=json.loads((p/'reference_validation.json').read_text())
assert validation['exact_equal']and validation['elements']==1000 and profile['control_reproduced_byteidentical']
pins=dict(profile['pins'])
for name in ['profile_build.json','reference_validation.json','model.elf','model.nofsm_audit.json','device_profile.c','device_profile.o','spike.log','build.log','build.py','validate_retained_console.py','host/output.npy']:
 f=p/name;pins[str(f)]=sha(f)
for name in ['build.py','validate_retained_console.py']:copy(p/name,'resnet1988_profile/'+name)
for path,h in pins.items():assert sha(path)==h
save('root_resnet1988_current_profile_qualification.json',{'schema':'golden_current_best_diagnostic_qualification_v1',
 'status':'pass','control_job':1988,'control_cycles':33633109,'control_elf_sha256':profile['base_elf_sha256'],
 'elf':str(p/'model.elf'),'elf_sha256':profile['elf_sha256'],'strict_original1000':validation,
 'profile_boundaries':70,'current_control_byte_reproduced':True,'pins':pins,
 'probe_harness_fix':'Target completed before first native-reference parser failed. Retained saved-console independently verified; build.py subsequently includes reference staging. Arithmetic/link/wrapper objects unchanged.',
 'scope':'Exact current1988 primitive device boundaries plus conserved host gaps; stock timing pending. Older1974/1903 profiles cannot repartitioncurrent1988.'})
copy(Path(__file__),'archive.py')
print(json.dumps({'root_closures':closures,'negative_pins':len(files),'profile_pins':len(pins),'status':'pass'}),flush=True)
