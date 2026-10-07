"""Exact algebraic composition with a measured complete-cost regression."""
from pathlib import Path
import ast,hashlib,json,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/scaled-fused-radix';C=B/'out/artifacts/probes/prepared-polynomial-constants'
core=Path('/scratch/agustin/tmp/merlin-scaled-fused-radix-20261007');module='src/merlin/llvmlower/radix_integer_reconstruct.py'
old=subprocess.check_output(['git','-C',str(core),'show','eb15a85ce:'+module],text=True)
olddefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
newdefs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((core/module).read_text()).body if isinstance(n,ast.FunctionDef)}
assert all(newdefs[name]==value for name,value in olddefs.items())
def value(path,key):
 return next(int(x.split()[1]) for x in path.read_text().splitlines() if x.startswith(key+' '))
oldi=value(C/'strict/stdout','WORKSPACE_GROUP_INSTRUCTIONS');newi=value(W/'strict/stdout','WORKSPACE_GROUP_INSTRUCTIONS')
assert (oldi,newi)==(1560212849,1565985371)
control=[x for x in (C/'strict/stdout').read_text().splitlines() if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]
candidate=[x for x in (W/'strict/stdout').read_text().splitlines() if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]
assert control==candidate
assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in (W/'strict/stdout').read_text()
native=json.loads((W/'native/validation.json').read_text());assert native['bitwise_mismatches']==0 and native['calls'][0:3]==[48,0,0] and native['product_calls']==23040
rows=json.loads((C/'attribution/attribution.json').read_text())['rows']
reconstruct=sum(x['instructions'] for x in rows if x['function']=='evaluate_products' and x['file'].endswith('provider.c') and 128<=x['line']<=135)
scale=sum(x['instructions'] for x in rows if x['function']=='evaluate_products' and x['file'].endswith('provider.c') and x['line']==532)
paths=[p for p in W.rglob('*') if p.is_file()]
paths += [core/module,core/'merlin/tests/ir/test_scaled_fused_radix.py',core/'docs/reference/scaled_fused_radix.md',C/'candidate/model.elf',C/'candidate/target_numeric/provider.c',C/'candidate/target_numeric/provider.o',C/'strict/stdout',C/'attribution/attribution.json']
paths += [Path(__file__),*[B/'experiments/attention_projection_frontier'/x for x in ['build_scaled_fused_radix.py','qualify_scaled_fused_group.py','attribute_scaled_fused.py']]]
negatives=['two_signed_byte_checked_residual_negative.json','two_signed_byte_rms_residual_negative.json','sparse_dyadic_cached_complete_group_negative.json','sparse_dyadic_recode_complete_group_negative.json','smol_one_endpoint_word_negative_journey.json','root_smol_fused_radix_stock_release_20261007.json']
paths += [B/'docs/perf_records'/x for x in negatives]
for item in json.loads((W/'target_build.json').read_text())['link']:
 p=Path(item)
 if p.is_file():paths.append(p)
record={'schema':'scaled_fused_radix_complete_negative_v1','status':'NEGATIVE_COMPLETE_RETIRED_INSTRUCTIONS','core_head':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),'scope':'same original12-head complete group/consumer; retired instructions not stock cycles','baseline_facilities':'integer_reconstruction and fuse_integer_reconstruction already consumed by current2098; no per-term F64 reconstruction remains','default_existing_emitter_ast_identity':True,'control_instructions':oldi,'candidate_instructions':newi,'increase_percent':(newi/oldi-1)*100,'native':native,'same_group_stats_and_carriers':True,'control_exclusive_evaluate_cost':{'total':331152594,'integer_reconstruction':reconstruct,'separate_scaling':scale,'bounds_norms_control':331152594-reconstruct-scale},'candidate_scaled_helper_instructions':139629024,'selected_scaled_path':'all executed reconstruction PCs bind new scaled helper; original fallback reconstruction/scaling has no executed PCs','source_semantics':'same complete i32 planes/i64 prefix/final f64 conversion/scaling parenthesization; positive widened binary32 power scales checked; all source bounds/observer/replay preserved','resources':'workspace and480callbacks/readback unchanged; no newallocation; removes intermediate center pass but added checked-loop codegen dominates','limitations':['not a hardware measurement','no whole-model cycle projection','generic option remains experimental/unpromoted','shared helper PCs outside exclusive provider frame remain unscoped'],'prior_evidence':negatives,'pins':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
out=B/'docs/perf_records/scaled_fused_radix_complete_group_negative.json';out.write_text(json.dumps(record,indent=2)+'\n');print(out,len(record['pins']))
