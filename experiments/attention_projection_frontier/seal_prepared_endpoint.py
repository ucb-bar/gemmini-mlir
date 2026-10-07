"""Seal the prepared endpoint source, complete cost and clock qualification."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-endpoint-dag-v3';C=B/'out/artifacts/probes/prepared-polynomial-constants';core=Path('/scratch/agustin/tmp/merlin-prepared-endpoint-dag-20261007')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
control=(C/'strict/stdout').read_text();candidate=(W/'strict/stdout').read_text();clock=(W/'clock_pair/candidate/stdout').read_text()
get=lambda s,k:next(int(t.split()[1])for t in s.splitlines()if t.startswith(k+' '))
a=get(control,'WORKSPACE_GROUP_INSTRUCTIONS');b=get(candidate,'WORKSPACE_GROUP_INSTRUCTIONS');assert(a,b)==(1560212849,1515040109)
for s in [candidate,clock]:
 assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in s
 assert [t for t in s.splitlines()if t.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]==[t for t in control.splitlines()if t.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]
native=json.loads((W/'native/validation.json').read_text());assert native['bitwise_mismatches']==0 and native['calls'][:3]==[48,0,0]and native['product_calls']==23040
assert(W/'clock_pair/control/model.elf').read_bytes()==(C/'clock_pair/candidate/model.elf').read_bytes()
paths=[p for p in W.rglob('*')if p.is_file()and'installed_venv'not in p.parts and'__pycache__'not in p.parts and'.pytest_cache'not in p.parts]
paths += [p for p in (core/'out/artifacts/endpoint-delivery').rglob('*')if p.is_file()]
paths += [core/'src/merlin/llvmlower/prepared_endpoint_dag.py',core/'merlin/tests/runtime/test_prepared_endpoint_dag.py',core/'docs/reference/prepared_endpoint_dag.md']
for name in ['build_prepared_endpoint_dag.py','qualify_prepared_endpoint_group.py','check_prepared_endpoint_target.py','attribute_prepared_endpoint.py','build_prepared_endpoint_clock_pair.py','seal_prepared_endpoint.py','polynomial_constants_protocol.py']:
 paths.append(B/'experiments/attention_projection_frontier'/name)
paths += [C/'candidate/model.elf',C/'candidate/target_numeric/provider.o',C/'candidate/native_numeric/provider.so',C/'strict/stdout',C/'attribution/attribution.json',C/'clock_pair/candidate/model.elf',C/'clock_pair/candidate/stdout',C/'clock_pair/candidate/build.json']
for f in [W/'target_build.json',W/'clock_pair/control/build.json',W/'clock_pair/candidate/build.json']:
 for item in json.loads(f.read_text())['link']:
  p=Path(item)
  if p.is_file():paths.append(p)
for f in [W/'native_numeric/provider.d',W/'target_numeric/provider.d']:
 if f.exists():
  for item in f.read_text().replace('\\\n',' ').split()[1:]:
   p=Path(item)
   if p.is_file():paths.append(p)
r={'schema':'prepared_endpoint_dag_complete_group_v1','status':'QUALIFIED_FOR_ROOT_REVIEW_NOT_HARDWARE_RELEASED','core_head':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'base':'eb15a85ce550ec63c27530d56af66893fe629766','source_policy':'unchanged explicit RMS4; endpoint range proof does not certify source truth of heuristic intervals','contract':'producer copy validates membership/finite spans and derives maxima; immutable complete epoch; invalidate row before every refinement; positive magnitude admission follows original scalar order; checked fallback retained; quantizer validators unchanged','native':native,'instructions':{'control':a,'candidate':b,'saved':a-b,'reduction_percent':(a-b)/a*100},'stock_cycles':'UNKNOWN','whole_cycle_forecast':'NONE','clock_control_sha256':sha(W/'clock_pair/control/model.elf'),'clock_candidate_sha256':sha(W/'clock_pair/candidate/model.elf'),'clock_control_identity':'byte-exact actual2098 e467; historical stock3611264318 cycles; no fresh control timing claimed','consumer':'original786432i8/1024BF16scales/guards/8stats/carrier2','resources':json.loads((W/'resources.json').read_text()),'address_comparison':json.loads((W/'clock_pair/address_comparison.json').read_text()),'test_counts':{'source':20,'installed':12,'target_shapes':3,'target_scenarios':579,'target_modes':5},'delivery':json.loads((core/'out/artifacts/endpoint-delivery/identity.json').read_text()),'limitations':['additional private stack metadata; complete group includes preparation','no whole target launched','histogram is whole executable; provider-exclusive frames alone scoped, shared callees unscoped','initial target puts fixture empty stdout retained; fresh printf fixture passed','initial package invocations missing pip/setuptools retained; isolated build succeeded'],'pins':{str(p.resolve()):sha(p)for p in paths}}
out=B/'docs/perf_records/prepared_endpoint_dag_complete_group_qualification.json';out.write_text(json.dumps(r,indent=2)+'\n');print(out,len(r['pins']),r['clock_candidate_sha256'])
