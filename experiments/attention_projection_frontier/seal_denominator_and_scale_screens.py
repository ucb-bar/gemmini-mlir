"""Preserve one fixed-policy accuracy negative and exact collision refusal."""
from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parents[2];core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records={};files=set()
for name in ('selected_bf16_denominator','scaled_endpoint_census'):
 folder=base/'out/observation_frontier'/name
 q=json.loads((folder/'qualification.json').read_text());assert q['snapshots']==48 and q['product_calls']==23040 and not q['errors'] and q['counts'][0]==48 and q['counts'][1]==0
 records[name]={k:v for k,v in q.items()if k not in ('commands','pins','counts')}
 records[name]['named_lifecycle']={'groups':q['counts'][0],'source_fallback':q['counts'][1],'preparations':q['counts'][4],'actual_product_calls':q['product_calls']}
 records[name]['legacy_counter_note']='Opaque pointer slots in historical get_counts retained only in original qualification, not interpreted as statistics.'
 for p in folder.rglob('*'):
  if p.is_file():files.add(p.resolve())
 for command in q['commands']:
  for value in command:
   p=Path(value)
   if p.is_file():files.add(p.resolve())
assert records['selected_bf16_denominator']['elementwise_failures']==114
assert records['scaled_endpoint_census']['scaled_endpoint_counts']==[150994944,0,205173,205173,0]
for name in ('screen_selected_bf16_denominator','census_scaled_endpoint_collisions','capture_initial_bounds','seal_denominator_and_scale_screens'):files.add(base/f'experiments/attention_projection_frontier/{name}.py')
for name in ('selected_bf16_denominator_driver.py','scaled_endpoint_census_driver.py','selected_bf16_denominator.log','scaled_endpoint_census.log'):files.add(base/'out'/name)
files.add(core/'src/merlin/llvmlower/selected_bf16_denominator.py');files.add(core/'merlin/tests/runtime/test_selected_bf16_denominator.py')
bundle=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle');files.add(bundle/'golden.npy')
result=dict(status='RETAINED_NEGATIVE_NO_TARGET_OR_HARDWARE',policy='One fixed selected BF16 denominator; not a source enclosure or threshold sweep. QK/PV unchanged, original whole atol.03125/rtol.02 unchanged. Old interval work intentionally retained in native feasibility.',records=records,helper_tests=20,scaled_collision_conclusion='No extra point intervals after source scaling; all205173 equal scaled pairs were already equal inputs. Refuse transformation on this cost evidence; no measured cardinality used for production eligibility.',pins={str(p):sha(p)for p in sorted(files)})
(base/'docs/perf_records/selected_denominator_and_scale_collision_negative.json').write_text(json.dumps(result,indent=2)+'\n');print(len(files))
