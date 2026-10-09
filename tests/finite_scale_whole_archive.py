"""Seal unchanged full target gates plus public proof-admission hardening."""
import json
import subprocess
from pathlib import Path
from finite_scale_whole_prepare import HERE,OUT,CORE,sha,save,load

T=OUT/'whole'
Q=T/'target_validation_m16g.json'
record=json.loads(Q.read_text());assert record['status']=='pass' and record['spike_full_output_match'] and record['torch_allclose'] and record['no_fsm']
api=HERE/'docs/perf_records/finite_scale_guard_api_reclosure.json';hardened=json.loads(api.read_text());assert hardened['status']=='pass' and hardened['actual_finite_lookup_C_byteidentical']
pins={str(api):sha(api),str(Q):sha(Q)}
for path,digest in hardened['pins'].items():assert sha(path)==digest,path;pins[path]=digest
for receipt in [T/'build.json']:
 proof=json.loads(receipt.read_text());pins[str(receipt)]=sha(receipt)
 for path,digest in proof['pins'].items():assert sha(path)==digest,path;pins[path]=digest
for field in ('elf_path','reference_path','torch_golden_path','spike_console_path','normal_lower_recipe_path','controlled_link_path'):
 path=record[field];key={'elf_path':'elf_sha256','reference_path':'reference_sha256','torch_golden_path':'torch_golden_sha256','spike_console_path':'spike_console_sha256','normal_lower_recipe_path':'normal_lower_recipe_sha256','controlled_link_path':'controlled_link_sha256'}[field]
 assert sha(path)==record[key];pins[path]=record[key]
for path in [*T.rglob('*'),*HERE.glob('tests/finite_scale*.py'),CORE/'src/merlin/llvmlower/source_expression_interval.py',CORE/'merlin/tests/ir/test_scaled_integer_finite_llvm.py']:
 if path.is_file():pins[str(path)]=sha(path)
stock=Path('/scratch/agustin/tmp/gemmini-current-profile-20261007/out/tiny_finite_scale_stock/stock2075_terminal.json');actual=json.loads(stock.read_text())
assert actual['schema']=='finite_scale_complete_M8_stock2075_terminal_v1'
assert actual['status']=='complete_scope_observation'
assert actual['job']['id']==2075 and actual['job']['state']=='DONE' and actual['job']['phase']=='DONE' and actual['job']['exit_code']==0
assert actual['staged_identity']['observed_before_teardown'] and actual['staged_identity']['job_id']==2075
for path,digest in actual['pins'].items():assert sha(path)==digest,path;pins[path]=digest
parser_path=next(Path(path) for path in actual['pins'] if path.endswith('/mlir_oot/closed_i8_interval_capsule.py'))
declaration_path=next(Path(path) for path in actual['pins'] if path.endswith('/stock_adapter/declaration.json'))
expected_path=next(Path(path) for path in actual['pins'] if path.endswith('/expected.bin'))
uart_path=next(Path(path) for path in actual['pins'] if path.endswith('/stock2075_uart.txt'))
elf_path=next(Path(path) for path in actual['pins'] if path.endswith('/timing/build/layer.elf'))
assert actual['staged_identity']['objects']['elf']['sha256']==sha(elf_path)
parser=load('whole_stock_parser',parser_path)
assert parser.parse_report(uart_path.read_text(),json.loads(declaration_path.read_text()),expected_path.read_bytes())==actual['report']
assert actual['report']['candidate_mean_cycles']<actual['report']['control_mean_cycles']
pins[str(stock)]=sha(stock)
for key,value in actual.items():
 if key.endswith('_sha256') and key[:-7] in actual and isinstance(actual[key[:-7]],str):
  path=Path(actual[key[:-7]])
  if path.is_file():assert sha(path)==value;pins[str(path)]=value
save(HERE/'docs/perf_records/finite_scale_guard_whole.json',{'schema':'source_finite_broadcast_guard_original_whole_review_v1','status':'pass','core_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=CORE,text=True).strip(),'core_parent':'3a1e24c77258d577326978e5e21cdf7f369618be','candidate_ELF_path':record['elf_path'],'candidate_ELF_sha256':record['elf_sha256'],'standard_reference_validation':str(Q),'standard_reference_validation_sha256':sha(Q),'baseline2070_ELF_byteidentical':True,'unchanged_nonmodel_leaves':11,'all155devicebindings':True,'original_tokens':8,'original_layers':22,'original_compiled_output_words':256000,'original_source_all_words_exact':True,'original_Torch_gate':{'atol':.03125,'rtol':.02,'pass':True},'strict_final_ELF_noFSM':True,'native_original22context_i8_words':991232,'native_mode_sticky_cases':616,'source_producer_chains':44,'source_helpers':22,'public_proof_admission_hardened':True,'actual_lookup_and_scanner_C_unchanged_after_hardening':True,'functional_retired_instructions':121135311,'whole_hardware_cycles':'UNKNOWN','complete_M8_stock2075_terminal':str(stock),'complete_M8_stock2075_terminal_sha256':sha(stock),'capsule_scope_only':'Single complete original M8 context including both scans/table/products/interval/coldsource/finish/stores/frame; no whole-cycle forecast.','whole_submission':'ROOT review/sole queue owner required','retained_failure_scopes':['Missing harness environment before first capsule link','Native fenv sampled after NumPy ufunc; corrected immediate sampling with all binaries unchanged','First whole Spike omitted original16GiB memory extent and failed during payload loading; no model executed','Superseded metadata-only reclosure cancelled before receipt after discovering historical test snapshot also required'], 'pins':pins,'token_usage_available':False})
print('SEALED_WHOLE',len(pins),sha(HERE/'docs/perf_records/finite_scale_guard_whole.json'),flush=True)
