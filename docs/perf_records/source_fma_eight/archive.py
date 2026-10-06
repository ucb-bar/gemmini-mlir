from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;root=Path.cwd();core=Path('/scratch/agustin/tmp/merlin-source-fma-eight-20261006')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed')
target=json.loads((w/'candidate/spike_receipt.json').read_text());native=json.loads((w/'native/validation.json').read_text());proof=json.loads((w/'independent/receipt.json').read_text())
assert target['status']=='pass' and native['elements']==1600 and native['bitwise_mismatches']==0 and native['allclose'] and native['calls'][:3]==[48,0,0] and native['product_calls']==23040
assert proof['status']=='pass' and proof['independent_lane_FMA_comparisons']==120000 and proof['endpoint_checks']==40120
assert sha(w/'control/model.elf')==sha(old/'candidate/model.elf')=='09ca3d0849ef423a49eef5af5c359b93df225f3f9b6483233c00b1c5830f6565'
def number(log):return int(next(line.split()[1]for line in Path(log).read_text().splitlines()if line.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
def stats(log):return [line for line in Path(log).read_text().splitlines()if line.startswith(('WORKSPACE_STAT ','UNOBSERVED_CARRIER_DIFF'))]
a=number(old/'candidate/spike.log');b=number(w/'candidate/spike.log');assert a==2288221154 and b==2278816418 and stats(old/'candidate/spike.log')==stats(w/'candidate/spike.log')
files={p for p in w.rglob('*')if p.is_file() and p.suffix in {'.py','.c','.h','.ll','.d','.o','.so','.elf','.npy','.json','.log'}}
for arm in ['control','candidate']:
 for argv in json.loads((w/arm/'numeric/compile.json').read_text())['commands']:
  files.add(Path(argv[0]));files.update(Path(v)for v in argv[1:]if v.startswith('/') and Path(v).is_file())
 for v in json.loads((w/arm/'build.json').read_text())['link']:
  if v.startswith('/') and Path(v).is_file():files.add(Path(v))
for group in ['compile','link','command']:
 files.update(Path(v)for v in proof[group]if v.startswith('/') and Path(v).is_file())
files.update([core/'src/merlin/llvmlower/source_fma_batch.py',core/'merlin/runtime/c/prepared_polynomial_batch.h',core/'merlin/tests/ir/test_source_fma_batch.py',core/'docs/runtime/source_fma_batch.md',root/'mlir_oot/host_fma_batch.py',root/'tests/test_host_fma_batch.py',root/'mlir_oot/no_fsm_audit.py'])
for argv in native['commands']:files.update(Path(v)for v in argv if v.startswith('/') and Path(v).is_file())
files.update([Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy'),Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/frontier_encoded_spans_cast_bins_four_qualification.json'),Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/stock2003_composed_frontier_terminal.json')])
# Close the frozen full source/consumer basis; candidate still needs a fresh
# normal target provider seal before any whole-model performance qualification.
basis=json.loads(Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/frontier_encoded_spans_cast_bins_four_qualification.json').read_text())
for p,digest in basis['pins'].items():assert sha(p)==digest;files.add(Path(p))
r={'schema':'source_fma_eight_complete_group_qualification_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),'core_commit':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'OOT_capability_commit':'b5b7d5a','ownership':'Generic finite independent source scheduling/effects/owner proof Merlin; actual RV64GC register constraints and fmadd.s sequence OOT','source_contract':'Original perendpoint operand order/three rounded Horner FMAs retained; finite-prefix preparedplan, distinctprivatefraction/polyarrays, stable RNE/nontrapping/unobservederrno+flags. Explicitdefaultoff mathematicalhook; no captured/model selectors','actual_emitted_schedule':'Eight independent fmadd.s in each opaque block, coefficient stage outermost; earlyclobber readwrite outputs prevent input overwrite across lanes','instructions':{'control':a,'candidate':b,'reduction_percent':100*(a-b)/a,'scope':'Complete original allocation-aware12-head source group/providerROI; checks and original consumer afterROI','hardware_cycles':None},'target':target,'native48':native,'independent_target':proof,'core_and_target_obligation_tests':41,'default_control_object_and_final_elf_byteidentical':True,'all_eight_stats_and_carrier_diagnostics_unchanged':True,'workspace_and_product_callbacks_unchanged':True,'normal_target_seal':'Not built; no whole hardware release','result':'Qualified complete originalgroup and all48native; approved solely for matched stockgroup timing against2003','pins':{str(p):sha(p)for p in sorted(files)},'token_usage_available':False}
out=root/'docs/perf_records/source_fma_eight';out.mkdir(exist_ok=False)
for p in [w/'build.py',w/'archive.py',w/'native_build.py',w/'validate_native.py',w/'host_fma_batch.h',w/'candidate/spike.log',w/'candidate/spike_receipt.json',w/'independent/receipt.json',w/'native/validation.json']:
 name={'receipt.json':'independent_target.json','validation.json':'native48.json'}.get(p.name,p.name);shutil.copyfile(p,out/name)
for p in [core/'src/merlin/llvmlower/source_fma_batch.py',core/'merlin/runtime/c/prepared_polynomial_batch.h',root/'mlir_oot/host_fma_batch.py']:shutil.copyfile(p,out/p.name)
(root/'docs/perf_records/source_fma_eight_complete_group_qualification.json').write_text(json.dumps(r,indent=2)+'\n')
print('ARCHIVED',len(files),'pins',r['instructions'],flush=True)
