"""Archive source-derived rectangular contraction qualification and complete cost."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
J=Path(__file__).resolve().parent
O=J.parents[3]
C=Path('/scratch/agustin/tmp/merlin-scalar-contraction-rectangular-20261006')
T=C/'out/artifacts/probes/rectangular-tests'
D=O/'docs/perf_records'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
cost=read(J/'capsule_result.json');proof=read(J/'qualification.json');independent=read(J/'independent_seed_corrected/qualification.json');capture=read(J/'capture/validation.json')
assert cost['status']==proof['status']==independent['status']=='pass'
assert proof['all_original_qk2048_and_pv16384_words_exact'] and proof['all_five_target_rounding_modes_and_stickyflags_match']
assert cost['ELF_sha256']==proof['timing_elf_sha256']==sha(J/'timing_build/layer.elf')
assert sha(J/'gsim.stdout')==cost['gsim_stdout_sha256']
assert '88 passed' in (T/'pytest_semantic_contract_configured.log').read_text()
assert subprocess.check_output(['git','-C',str(C),'status','--porcelain'],text=True)==''
files=set(p for p in J.rglob('*') if p.is_file())
files.update(C/name for name in ['src/merlin/llvmlower/scalar_contraction.py','src/merlin/llvmlower/AGENT.md','merlin/tests/ir/test_scalar_contraction.py','src/merlin/llvmlower/bufferized_result_identity.py','src/merlin/llvmlower/impr_features.py','src/merlin/llvmlower/pipeline.py','src/merlin/llvmlower/lower.py'])
files.update(p for p in T.rglob('*') if p.is_file())
langref=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-project/llvm/docs/LangRef.rst')
arith=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-project/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td')
files.update([langref,arith])
gsim=read(J/'gsim_receipt.json')
files.update([Path(gsim['engine']['path']),Path(gsim['engine']['receipt']['receipt_path'])])
source_context=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
files.update(source_context/name for name in ['device_host_abi/model.mlir','model.prepared.mlir','mlir_rt.o','device_catalog/kernel.o','device_catalog/device_catalog.json'])
record={
 'schema':'compiler_optimization_journey_v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
 'hypothesis':'A typed two-row/four-column contraction tile can keep eight independent scalar accumulators while sharing only coordinate-identical immutable A/B reads across rows/columns. Preserve each original rounded multiply/add and increasing-K order, seed and output map. Complete materialization/public ABI cost determines profitability.',
 'ownership':'Portable typed tensor scheduling/default-off source feature Merlin; actual target/source bindings, numeric/executable/cost receipts OOT. No new CPU ISA or accelerator instructions.',
 'core_commit':subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip(),
 'core_tree':subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD^{tree}'],text=True).strip(),
 'emitted_change':'Explicit scalar_contraction_accumulator_2x4_outputs alternative replaces scalar_contraction_accumulator_8_outputs for structurally proven f32 source contractions; K2 annotation and canonical buffer identity remain matched. Eight output accumulators, source mul/add operand order and K order retained. Partial M/N tiles, incompatible projection maps, unknown scalar bodies/attrs and strict ancestors refuse.',
 'actual_source_scope':'Original32x8x8 QK64 and32x8x64 PV8 contractions with original full-model inputs/initialization/outputs. Original1989 readonly native capture preserves all22calls/family and all256000compiledwords+Torch. Experimental selection only locates immutable source bindings; production eligibility has no labels/model IDs/ordinals/content selectors.',
 'before':{'qk_complete_gsim_mean_cycles':cost['families']['qk']['control_complete_mean_cycles'],'pv_complete_gsim_mean_cycles':cost['families']['pv']['control_complete_mean_cycles']},
 'after':{'qk_complete_gsim_mean_cycles':cost['families']['qk']['rectangular_complete_mean_cycles'],'pv_complete_gsim_mean_cycles':cost['families']['pv']['rectangular_complete_mean_cycles'],'qk_saved_percent':cost['families']['qk']['saved_percent'],'pv_saved_percent':cost['families']['pv']['saved_percent'],'whole_hardware_cycles':None},
 'gate':{'native_source_suite_cases':88,'actual_independent_RV64GC_cases':8,'independent_all_five_FRM_and_stickyflags':True,'independent_actual_raw_words_nonfinite_exact':True,'captured_original_target_raw_words':18432,'captured_five_FRM_and_stickyflags':True,'input_guards':True,'poisoned_destination_tails':128,'all_executable_zeroFSM':True,'source_context_full256000_and_Torch_capture':True},
 'NaN_diagnostic':{'strict_payload_comparison':'Preserved87PASS/1FAIL x86 native diagnostic: both source operands NaN gave quiet propagated payload12345 versus quiet propagated payload00001. Both are legal unconstrained source results. Existing original Tiny source/golden/gate unchanged.','primary_source':str(langref),'section':'Behavior of Floating-Point NaN values, lines4107-4157','contract':'Unconstrained arith.mulf/addf permit canonical or propagated NaN payload choices. Non-NaN raw words exact; empty-K nonarithmetic seed bits exact. Strict/constrained FP refuses; no new floating-environment permission. Actual independent RV64 checks happen to match every raw word/flag in all5FRM.'},
 'retained_harness_refusal':{'path':str(J/'independent/qualify.log'),'reason':'The initial independent harness incorrectly required dead writable tensor seed storage to remain physically unchanged and reused it for the next paired call. Upstream may forward its source allocation. Corrected independent harness supplies identical fresh seed per call and leaves source tensor alias/lifetime authority upstream; immutable source fixtures/A/B remain guarded. Original captured positivezero-internal capsules were unaffected.'},
 'timing_scope':cost['scope'],'result':'POSITIVE complete source-bound capsules; default remains off. Whole normal source-compatible256000/Torch/155/noFSM qualification and stock comparison still required; no whole gain projection or hardware admission.',
 'whole_prediction':None,'token_usage_available':False,'pins':{str(p):sha(p) for p in sorted(files)},
}
out=D/'tiny_rectangular_contraction_capsule';out.mkdir(exist_ok=False)
for p in J.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.c','.json','.log','.stdout'}:
  dest=out/p.relative_to(J);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for name in ['qk_source.mlir','pv_source.mlir']:shutil.copyfile(J/name,out/name)
for parent in ['independent','independent_seed_corrected']:
 for p in (J/parent).glob('*.mlir'):shutil.copyfile(p,out/parent/p.name)
(out/'native_tests').mkdir()
for p in T.glob('*.log'):shutil.copyfile(p,out/'native_tests'/p.name)
(D/'tiny_rectangular_contraction_capsule_journey.json').write_text(json.dumps(record,indent=2)+'\n')
print('RECTANGULAR_SOURCE_COST_ARCHIVED',len(files),flush=True)
