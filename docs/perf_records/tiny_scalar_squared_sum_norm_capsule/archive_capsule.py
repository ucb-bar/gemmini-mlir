from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re, shutil, subprocess
J=Path(__file__).resolve().parent
O=J.parents[3]
C=Path('/scratch/agustin/tmp/merlin-scalar-squared-sum-20261006')
OLD=Path('/scratch/agustin/tmp/merlin-bufferized-destination-identity-20261006')
I=J.parent/'tiny-squared-sum-independent-v2-20261006'
N=J.parent/'tiny-broadcast-math-20261005'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
q=read(J/'qualification.json'); g=read(J/'gsim_receipt.json'); d=read(J/'default_identity.json'); i=read(I/'receipt.json')
assert q['native_all_exact'] and q['strict_all5_rounding_modes_and_flags'] and q['audit']['status']=='pass'
assert g['returncode']==0 and g['finish']['done'] and g['finish']['exit_code']==0
assert g['engine']['receipt_status']=='bound' and g['stdout_sha256']==sha(J/'gsim.stdout')
assert sha(J/'build/layer.elf')==q['elf_sha256']
assert d['normal_default_executable_IR_byteidentical'] and i['status']=='pass' and i['raw_final_f32_word_comparisons']==840
rows=[tuple(map(int,r))for r in re.findall(r'SQUARED_SUM_NORM_COMPLETE_CYCLES (\d+) (\d+) (\d+)',(J/'gsim.stdout').read_text())]
assert [r[:2]for r in rows]==[(0,0),(0,1),(1,1),(1,0)]
assert 'SOURCE_SQUARED_SUM_NORM_COMPLETE PASS' in (J/'gsim.stdout').read_text()
before=sum(r[2]for r in rows if r[1]==0)/2;after=sum(r[2]for r in rows if r[1]==1)/2
assert before>after
for n in ['source.mlir','weight.bin','activation.bin','expected.bin']:assert sha(J/n)==sha(N/n)
for p,h in d['pins'].items():assert sha(p)==h
core_files=['src/merlin/llvmlower/scalar_squared_sum.py','src/merlin/llvmlower/pipeline.py','src/merlin/llvmlower/impr_features.py','src/merlin/llvmlower/scalar_pointwise_packet.py','src/merlin/llvmlower/bufferized_result_identity.py','src/merlin/llvmlower/broadcast_math_hoist.py','src/merlin/llvmlower/late_quant_rne.py','merlin/tests/ir/test_scalar_squared_sum.py']
files=set(C/n for n in core_files)
files.update(J/n for n in ['archive_capsule.py','build_capsule.py','reproduce_control.py','source.mlir','weight.bin','activation.bin','expected.bin','main.c','main.o','data.S','data.o','qualification.json','ready.json','spike.log','gsim.stdout','gsim_receipt.json','default_identity.json','default_identity.log','build.log','build/layer.elf'])
for arm in ['control','accumulated','control_reproduced']:
 files.update(p for p in (J/arm).rglob('*')if p.is_file()and p.suffix in{'.json','.ll','.mlir','.o','.so','.txt'})
files.update(p for p in (J/'build').rglob('*')if p.is_file()and p.suffix in{'.json','.ld','.o'})
files.update(I/n for n in ['qualify.py','source.mlir','main.c','receipt.json','spike.log','qualify.log','build/layer.elf'])
for arm in ['control','selected']:
 files.update(p for p in (I/arm).rglob('*')if p.is_file()and p.suffix in{'.json','.ll','.mlir','.o'})
files.add(Path(q['original_runtime_object']))
files.add(Path(g['engine']['path']));files.add(Path(g['engine']['receipt']['receipt_path']))
for n in ['clang','mlir-opt','mlir-translate']:files.add(Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')/n)
files.add(Path(i['argv'][0]))
files.update(Path(n)for n in ['/tmp/tiny-squared-sum-tests-full-20261006.log','/tmp/tiny-squared-sum-structure-20261006.log','/tmp/tiny-squared-sum-tests-20261006.log'])
files.update(Path(p)for p in d['pins'])
# The two early attempts refused during setup/link; record their retained diagnostic inputs.
failed=[]
for directory,reason in [('tiny-squared-sum-independent-20261006','MERLIN_TARGET_PATH missing: backend selection refused before link'),('tiny-squared-sum-independent-v1-20261006','frozen runtime __errno unresolved: linker refused; fresh v2 supplies harness-only errno accessor')]:
 path=J.parent/directory
 logs=[p for p in path.rglob('*.log')if p.is_file()]
 files.update(logs)
 failed.append(dict(path=str(path),reason=reason,scope='Harness setup failure; no numerical comparison or timing result',diagnostic_paths=[str(p)for p in logs]))
result=dict(schema='compiler_optimization_journey_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Keep a closed ordered squared-sum reduction accumulator in scalar SSA so upstream lowering avoids storing the running tensor value after every channel, without changing any arithmetic rounding or observable tensor ownership.',ownership='Merlin generic typed scalar reduction, normal feature dispatch and upstream alias ownership; OOT experiment source/target/GSIM qualification only. No new target ISA implementation.',core_commit=subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip(),core_tree=subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD^{tree}'],text=True).strip(),default_off=True,feature='scalar_squared_sum_accumulator',emitted_change='Exact linalg.generic input-map permutation/output-map projection; source mulf(x,x) then addf(product,accumulator) in increasing final reduction axis becomes scalar SCF accumulator and one final tensor insert per output coordinate. Original seed, cast precision, operation attributes, operand order, empty K and live input/seed uses retained. Upstream bufferization owns aliases.',before=dict(scope='Complete first original8×2048 normalization including quantized output on current1983-compatible norm policies, original input/weight/captured expected bytes and original1880 mlir_rt.o',gsim_roi_cycles=[r[2]for r in rows if r[1]==0],gsim_mean_cycles=before,strict_retired_instructions=[274926,274926]),after=dict(gsim_roi_cycles=[r[2]for r in rows if r[1]==1],gsim_mean_cycles=after,saved_cycles=before-after,saved_percent=100*(before-after)/before,strict_retired_instructions=[258550,258550],whole_hardware_cycles=None),cost_scope='Actual bound GSIM ABBA complete function, reductions/division/epsilon/source rsqrt/source rounded weight multiplication/quantization/output/allocations/traffic. Numeric/flags/guard checks run outside ROI. GSIM simulator total includes setup/checks and is NOT model ROI. Spike mcycle is retired instructions, NOT hardware cycles.',gate=dict(original_native_i8_words_exact=16384,strict_original_i8_words_exact=16384,all_five_rounding_modes_and_matching_flags=True,dirty_tail_guard_bytes=128,independent_target_raw_f32_comparisons=840,independent_initial_sticky_flags=8,independent_dirty_prefix_suffix_words=32,independent_signed_zero_subnormal_nan_inf_overflow=True,zero_FSM_all_executable_sections=True,focused_source_native_and_existing_tests_pass=106,default_control_native_and_target_LLVM_byteidentical_prechange=True,default_before_core=d['before_core_commit'],default_after_core=d['after_core_commit']),alias_scope='Compiler proof is tensor SSA and upstream alias authority, without adding noalias. Independent aliased input/seed/live-use gates compare actual control/candidate buffer mutation. Native test arguments copied independently because a writable source seed can legally become the source output.',result='Positive complete capsule, prepare fresh whole original256000/Torch/155-binding/native/strict qualification. No whole speedup or stock promotion inferred.',failed_harness_attempts=failed,token_usage_available=False,campaign_checkpoints='Root records shared campaign usage; exact per-agent/experiment allocation unavailable.',pins={str(p):sha(p)for p in sorted(files)})
out=O/'docs/perf_records/tiny_scalar_squared_sum_norm_capsule'
out.mkdir(exist_ok=False)
for n in ['archive_capsule.py','build_capsule.py','reproduce_control.py','source.mlir','main.c','qualification.json','spike.log','gsim.stdout','gsim_receipt.json','default_identity.json','default_identity.log']:
 shutil.copyfile(J/n,out/n)
for n in ['qualify.py','source.mlir','main.c','receipt.json','spike.log','qualify.log']:
 dst=out/'independent';dst.mkdir(exist_ok=True);shutil.copyfile(I/n,dst/n)
(O/'docs/perf_records/tiny_scalar_squared_sum_norm_capsule_journey.json').write_text(json.dumps(result,indent=2)+'\n')
print('SCALAR_SQUARED_SUM_COMPLETE_CAPSULE_ARCHIVED',before,after,result['after']['saved_percent'],len(files),flush=True)
