"""Pin the completed, losing current-source packed RHS comparison."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
M=Path(__file__).resolve().parent
O=M.parents[4]
D=O/'docs/perf_records'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
checkpoint=read(D/'tiny_packed_rhs_current_checkpoint.json')
readiness=read(D/'tiny_packed_rhs_immutable_readiness.json')
pair=read(M/'pair_result.json');q=read(M/'qualification.json');census=read(M/'command_traffic_census.json')
assert pair['status']=='pass' and pair['warm_change_percent']>0
files={Path(p) for p in checkpoint['pins']}|{Path(p) for p in readiness['pins']}
for receipt in (checkpoint,readiness):
 for p,h in receipt['pins'].items():assert sha(p)==h,p
files.update(p for p in M.rglob('*') if p.is_file())
files.update(Path(p) for p in q['captured_inputs'])
files.add(Path(q['baseline_device_object_path']))
files.update(Path(p) for p in census['rtl_and_startup_sources'])
files.add(Path(census['old_hardware_profile_path']))
files.add(Path(census['catalog_path']))
files.update([D/'tiny_packed_rhs_current_checkpoint.json',D/'tiny_packed_rhs_immutable_readiness.json'])
for arm in ('dense','packed'):
 r=read(M/arm/'gsim_receipt.json')
 assert r['returncode']==0 and r['finish']['done'] and r['finish']['exit_code']==0
 assert sha(M/arm/'gsim.stdout')==pair['arms'][arm]['stdout_sha256']==r['stdout_sha256']
 assert sha(M/arm/'build/layer.elf')==pair['arms'][arm]['elf_sha256']==q['elfs'][arm]['sha256']
 assert q['audits'][arm]['status']=='pass' and not q['audits'][arm]['forbidden'] and not q['audits'][arm]['unknown']
 files.update([Path(r['engine']['path']),Path(r['engine']['receipt']['receipt_path'])])
assert (M/'dense/text.bin').read_bytes()==(M/'packed/text.bin').read_bytes()
assert q['original_control_typed_body_structurally_identical'] and q['same_all_symbol_addresses']
assert census['selected_original']['requested_bytes']==census['selected_packed']['requested_bytes']
assert census['selected_original']['commands']==census['selected_packed']['commands']
record={
 'schema':'compiler_optimization_journey_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
 'hypothesis':checkpoint['hypothesis'],'ownership':checkpoint['ownership'],
 'actual_emitted_change':checkpoint['actual_change'],
 'before':{'warm_complete_primitive_gsim_cycles':pair['arms']['dense']['warm_mean'],'cold_cycles':pair['arms']['dense']['events'][0][2]},
 'after':{'warm_complete_primitive_gsim_cycles':pair['arms']['packed']['warm_mean'],'cold_cycles':pair['arms']['packed']['events'][0][2],'change_percent':pair['warm_change_percent'],'whole_hardware_cycles':None},
 'scope':pair['roi_scope'],'gate':checkpoint['numeric_scope'],
 'result':'REJECTED: +1.258620741% slower in complete same-address current-source GSIM. Do not enable normal packed argument routing or admit a whole hardware arm.',
 'requested_B_bytes_per_call':11534336,'requested_B_bytes_reduction':0,
 'traffic_attribution':'Exact static command/byte census only. Frontend row-page transitions are a hypothesis, not measured events. Mmode bare has no pagewalk claim. Dynamic mesh, physical DRAM traffic, and current whole-model host/device partition remain unknown.',
 'normal_route_enabled':False,'prior_scope_refusal_retained':str(M.parent/'initial_scope_refusal.json'),
 'capture_metadata_correction':str(M/'capture_actual_recipe.json'),
 'older_profile_scope':census['stock1901_scope'],
 'whole_performance_projection':None,'token_usage_available':False,
 'pins':{str(p):sha(p) for p in sorted(files)},
}
out=D/'tiny_packed_rhs_current_completed_negative';out.mkdir(exist_ok=False)
for p in M.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.c','.json','.log','.stdout'}:
  dest=out/p.relative_to(M);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
shutil.copyfile(M.parent/'initial_scope_refusal.json',out/'initial_scope_refusal.json')
record['pins'][str(M.parent/'initial_scope_refusal.json')]=sha(M.parent/'initial_scope_refusal.json')
(D/'tiny_packed_rhs_current_completed_negative.json').write_text(json.dumps(record,indent=2)+'\n')
print('PACKED_COMPLETED_NEGATIVE',len(record['pins']),pair['warm_change_percent'],flush=True)
