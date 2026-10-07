"""Hardened table/route admission reproduces every measured emitted C byte."""
import hashlib
import json
import subprocess
from pathlib import Path
from finite_scale_whole_prepare import OLD,HERE,OUT,CORE,SOURCE,sha,save,load
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import IntervalEffectContract,find_closed_scalar_i8_observers,build_source_interval_table,emit_source_interval_i8_lookup
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers,emit_finite_scale_helper

R=HERE/'docs/perf_records/finite_scale_guard'
old=R/'accepted_source_expression_interval_c37b.py'
text=subprocess.check_output(['git','show','c37b2f274:src/merlin/llvmlower/source_expression_interval.py'],cwd=CORE,text=True)
old.write_text(text)
prep=HERE/'docs/perf_records/finite_scale_guard_preparation.json';record=json.loads(prep.read_text());current=CORE/'src/merlin/llvmlower/source_expression_interval.py'
assert sha(old)==record['pins'][str(current)]
base=load('hardened_base',OLD/'tests/source_continuation_whole_build.py');typed=base.N/'typed_prepacket.generic.mlir'
effects=IntervalEffectContract(True,True,True,True,True)
observers,refused=find_closed_scalar_i8_observers(parse_mlir_text(typed.read_text()),effects=effects);assert len(observers)==22 and not refused
bindings=json.loads((OUT/'selected/target/host_llvm/source_binding.json').read_text())
helpers=bind_finite_scale_helpers(SOURCE.read_text(),routes=bindings['source_bindings']['routes'],observers=observers,effects=effects,immutable_inputs=True,fresh_disjoint_output=True)
table=build_source_interval_table(observers[0].expression,effects=effects,leading_bits=16,max_table_bytes=512*1024)
new=emit_source_interval_i8_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation',leading_bits=16,finite_inputs=helpers,finite_table=table)
assert new==(OUT/'selected/lookup.c').read_text()
ordinary=emit_source_interval_i8_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation',leading_bits=16)
assert ordinary==(OUT/'control/lookup.c').read_text()
for native in ('target','native'):
 mode=(OUT/f'selected/{native}/host_llvm/mode_guard.c').read_text()
 for i,helper in enumerate(helpers):
  scanner=emit_finite_scale_helper(helper,wrapper_symbol=f'prepared_{i}',finite_symbol=f'rne_{i}',fallback_symbol=f'source_{i}')
  assert scanner in mode
P=HERE/'out/artifacts/probes/finite-scale-M8-v2-20261007'
source=(OLD/'out/artifacts/probes/closed-i8-interval-result-20261007/source.ll').read_text();route=json.loads((OLD/'out/artifacts/probes/closed-i8-interval-result-20261007/source_binding.json').read_text())
cap=bind_finite_scale_helpers(source,routes=route['routes'],observers=observers,effects=effects,immutable_inputs=True,fresh_disjoint_output=True)
newcap=emit_source_interval_i8_lookup(table_name='table_b16',activation_name='finite_source_activation',quantizer_name='finite_quantize',lookup_name='finite_lookup_activation',leading_bits=16,finite_inputs=cap,finite_table=table)
assert newcap==(P/'lookup.c').read_text()
assert emit_finite_scale_helper(cap[0],wrapper_symbol='prepared_M8',finite_symbol='finite_M8_rne',fallback_symbol='control_M8')==(P/'scale_guard.c').read_text()
oldtest=R/'accepted_test_scaled_integer_finite_llvm_c37b.py'
testpath=CORE/'merlin/tests/ir/test_scaled_integer_finite_llvm.py'
oldtest.write_text(subprocess.check_output(['git','show','c37b2f274:merlin/tests/ir/test_scaled_integer_finite_llvm.py'],cwd=CORE,text=True))
resolved={}
for path,digest in record['pins'].items():
 actual=old if path==str(current) else oldtest if path==str(testpath) else Path(path)
 assert sha(actual)==digest,path
 resolved[str(actual)]=digest
save(HERE/'docs/perf_records/finite_scale_guard_api_reclosure.json',{'schema':'source_finite_lookup_semantic_api_hardening_v1','status':'pass','core_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=CORE,text=True).strip(),'original_preparation_receipt':str(prep),'original_preparation_sha256':sha(prep),'actual_default_lookup_C_byteidentical':True,'actual_finite_lookup_C_byteidentical':True,'all22_target_and_native_scanner_C_byteidentical':True,'complete_M8_lookup_and_scanner_C_byteidentical':True,'no_qualified_object_or_ELF_rebuilt_or_changed':True,'new_refusals':'Missing explicit table, different expression, different partition/layout, table without finite input binding. Actual recipe supplies matching immutable table.','source_pin_resolution':{'original_path':str(current),'original_sha256':record['pins'][str(current)],'original_snapshot':str(old),'snapshot_sha256':sha(old),'current_sha256':sha(current)},'pins':{**resolved,str(Path(__file__).resolve()):sha(__file__),str(current):sha(current),str(CORE/'out/artifacts/probes/finite-scale-binding-20261007/hardened_tests.log'):sha(CORE/'out/artifacts/probes/finite-scale-binding-20261007/hardened_tests.log')},'token_usage_available':False})
print('HARDENED_API_EMISSION_BYTEIDENTICAL',len(helpers),flush=True)
