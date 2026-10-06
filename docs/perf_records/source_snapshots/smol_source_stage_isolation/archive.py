"""Archive stage isolation against the unchanged original whole gate."""
from pathlib import Path
import json,hashlib,shutil
w=Path(__file__).resolve().parent
docs=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records')
snapshot=docs/'source_snapshots/smol_source_stage_isolation';snapshot.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];pins={}
for stage in ['qk','pv','both']:
 p=w/f'exact_{stage}';r=json.loads((p/'native_validation.json').read_text())
 assert r['actual_runtime_calls']==48 and r['full_group_source_fallback_calls']==0 and not r['callback_errors']
 assert r['original_atol']==.03125 and r['original_rtol']==.02
 rows.append({'source_exact_stage':stage,'failed_count':r['failed_count'],'maxabs':r['maxabs'],
  'bitwise_mismatches':r['bitwise_mismatches'],'allclose':r['allclose'],'native_seconds':r['native_wall_seconds'],
  'receipt':str(p/'native_validation.json'),'receipt_sha256':sha(p/'native_validation.json')})
 for f in p.glob('*'):
  if f.is_file():pins[str(f)]=sha(f)
 for key,name in [('model_llvm_sha256','build/lower/model.ll'),('model_object_sha256','native_model.o'),('library_sha256','native.so'),('bridge_source_sha256','native_bridge.c'),('installation_sha256','installation.json')]:
  f=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/bounded_native')/name
  assert sha(f)==r[key];pins[str(f)]=sha(f)
assert [r['failed_count']for r in rows]==[92,168,0] and rows[-1]['bitwise_mismatches']==0
for name in ['prepare.py','validate_stage.py','validate_both.py','ordered_dot.c','ordered_dot.so','preparation.json','exact_qk.log','exact_pv.log','exact_both.log']:
 f=w/name;pins[str(f)]=sha(f)
 if f.suffix!='.so':shutil.copyfile(f,snapshot/name)
old=docs/'smol_source_group_center_journey.json';pins[str(old)]=sha(old)
for p,h in json.loads(old.read_text()).get('pins',{}).items():
 assert sha(p)==h;pins[p]=h
source=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/native_center_evaluator.py');pins[str(source)]=sha(source)
parent=source.parents[2]/'docs/perf_records/closed_source_center_only_native_screen.json'
pins[str(parent)]=sha(parent)
for p,h in json.loads(parent.read_text())['pins'].items():
 assert sha(p)==h;pins[p]=h
base=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005')
for name,key in [('bundle/golden.npy','original_golden_sha256'),('host_outlined/model.so','original_runtime_integer_provider_sha256')]:
 f=base/name;assert sha(f)==r[key];pins[str(f)]=sha(f)
records=[json.loads((w/f'exact_{stage}/runtime_group_calls.json').read_text())for stage in ['qk','pv','both']]
assert len(records[0])==len(records[1])==len(records[2])==48
assert records[0][0]['input_raw_sha256']==records[1][0]['input_raw_sha256']==records[2][0]['input_raw_sha256']
receipt={'schema':'golden_source_stage_sensitivity_journey_v1','status':'diagnostic_closed',
 'original_gate':{'atol':.03125,'rtol':.02,'elements':1600},'historical_pure_center_failed_count':121,
 'stage_isolation':rows,'original_group0_live_inputs_identical':True,
 'same_frozen_host_llvm_object_library_and_bridge':True,'all48_calls_zero_fallback':True,
 'compiler_policy_enabled':False,'hardware_admitted':False,'target_cycles':'UNKNOWN',
 'scope':'Native source sensitivity only. Original source nonlinear DAG and whole gate unchanged. New ordered F32 fmaf helper retains increasing-K source order in QK and/or PV; held radix3 center approximation serves the other stage. Both-exact positive control closes all1600 originalbits, validating stage substitution. Approximation in either entire stage independently fails. Existing numeric_witness/legacy policy labels belong to registered native diagnostic base, not a new production proof.',
 'observations':'Fail counts are not additive; recomputed downstream inputs and quantization amplify source rounding differences. No algorithm profitability or new target lowering permission follows.',
 'pins':pins}
(docs/'smol_source_stage_isolation_journey.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(Path(__file__),snapshot/'archive.py');print(json.dumps({'status':'pass','pins':len(pins),'rows':rows}),flush=True)
