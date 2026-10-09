"""Seal original qualifications plus explicit documentation-only source closure."""
import ast
import hashlib
import json
from pathlib import Path
from finite_scale_whole_prepare import OLD,HERE,OUT,CORE,SOURCE,sha,save

N=OUT
P=HERE/'out/artifacts/probes/finite-scale-M8-v2-20261007'
R=HERE/'docs/perf_records/finite_scale_guard'
qualification=json.loads((N/'qualification.json').read_text())
source=CORE/'src/merlin/llvmlower/source_expression_interval.py'
oldtext=source.read_text().replace('    Explicit finite_inputs additionally requires validated source/LLVM producer\n    bindings and dominating successful immutable scale scans. It removes only\n    their redundant finite checks; ordinary emission is byte-identical.\n','')
assert hashlib.sha256(oldtext.encode()).hexdigest()==qualification['pins'][str(source)]
accepted=R/'accepted_source_expression_interval.py';accepted.write_text(oldtext)
def semantic_ast(text):
 tree=ast.parse(text)
 for node in ast.walk(tree):
  if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
 return ast.dump(tree,include_attributes=False)
assert semantic_ast(oldtext)==semantic_ast(source.read_text())
pins={str(accepted):sha(accepted),str(source):sha(source)}
for receipt in [N/'qualification.json',N/'native_resume.json',P/'all_modes/qualification.json',P/'timing/qualification.json']:
 rec=json.loads(receipt.read_text());pins[str(receipt)]=sha(receipt)
 for path,digest in rec.get('pins',{}).items():
  actual=accepted if path==str(source) and digest!=sha(source) else Path(path)
  assert actual.is_file() and sha(actual)==digest,path
  pins[str(actual)]=digest
for p in [P/'stock_adapter/declaration.json',OLD/'mlir_oot/closed_i8_interval_capsule.py',OLD/'out/artifacts/probes/source-interval-calibration-v2-20261006/expected.bin',CORE/'out/artifacts/probes/finite-scale-binding-20261007/focused_tests.log',*[CORE/('src/merlin/llvmlower/'+n) for n in ('scaled_integer_finite.py','scaled_integer_finite_llvm.py')],*[CORE/('merlin/tests/ir/'+n) for n in ('test_scaled_integer_finite.py','test_scaled_integer_finite_llvm.py')],*HERE.glob('tests/finite_scale*.py')]:pins[str(p)]=sha(p)
timing=json.loads((P/'timing/qualification.json').read_text())
record={'schema':'source_finite_broadcast_guard_preparation_review_v1','status':'pass','hypothesis':'Move repeated input finite checks to once-per-owned immutable channel scale scans while retaining every original product/table/certificate/sourcecontinuation/finish.','ownership':'Generic source domain/binding/scanner/host emission Merlin; target frm/ABI/harness OOT.','default_policy':'disabled/ordinary bytes unchanged','typed_helpers':22,'typed_producer_chains':44,'original_per_element_finite_checks':1982464,'scale_scan_words':247808,'scan_cost_included_in_capsule':True,'whole_native_original256000_exact':True,'whole_original_Torch_gate':{'atol':.03125,'rtol':.02,'pass':True},'actual22_context_words':991232,'native_context_environment_cases':616,'native_control_model_object_byteexact2070':True,'strict_target_M8_all5modes7sticky':True,'strict_target_M8_original45056bytes':True,'complete_ABBA_elf_path':timing['elf_path'],'complete_ABBA_elf_sha256':timing['elf_sha256'],'whole_target_qualification':'pending','actual_cycles':'UNKNOWN','no_cycle_projection':True,'core_parent':'3a1e24c77258d577326978e5e21cdf7f369618be','OOT_parent':'c285ff68d2900668259c34f7e03dfe76d7b9bef1','documentation_pin_resolution':{'original_path':str(source),'original_sha256':qualification['pins'][str(source)],'accepted_snapshot_path':str(accepted),'accepted_snapshot_sha256':sha(accepted),'current_sha256':sha(source),'executable_AST_identical':True,'only_change':'Function docstring explaining explicit finite_inputs contract; original qualification unchanged.'},'retained_negative_attempts':['Missing explicit harness GCC environment before first link; no ELF qualification.','Native flags sampled after NumPy guard ufunc; corrected immediate sampling with exact unchanged compiled objects.'],'parser_path':str(OLD/'mlir_oot/closed_i8_interval_capsule.py'),'declaration_path':str(P/'stock_adapter/declaration.json'),'expected_path':str(OLD/'out/artifacts/probes/source-interval-calibration-v2-20261006/expected.bin'),'token_usage_available':False,'pins':pins}
save(HERE/'docs/perf_records/finite_scale_guard_preparation.json',record)
print('SEALED',len(pins),sha(HERE/'docs/perf_records/finite_scale_guard_preparation.json'))
