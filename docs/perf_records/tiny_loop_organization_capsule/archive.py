"""Freeze complete local cost and normal-emission equivalence, separately scoped."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

W=Path(__file__).resolve().parent
O=W.parents[3]
R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006')
C=Path('/scratch/agustin/tmp/merlin-tiny-loop-organization-20261006')
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
result=json.loads((W/'capsule_result.json').read_text())
reclosure=json.loads((W/'normal_feature_reclosure/receipt.json').read_text())
engine=json.loads((W/'gsim_receipt.json').read_text())
assert result['DONE'] and result['saved_percent']>0
assert engine['returncode']==0 and engine['finish']['done']
assert reclosure['actual_object_byteidentical']
assert sha(W/'normal_feature_reclosure/model.o')==sha(W/'outlined.o')==reclosure['measured_after_RNE_object_sha256']
base=W.parent/'tiny-broadcast-math-20261005/whole'
rows=[]
for p in [base/'model.o',W/'whole_merged.o']:
    line=subprocess.run([str(L/'llvm-size'),str(p)],check=True,capture_output=True,text=True).stdout.splitlines()[-1]
    rows.append({'path':str(p),'sha256':sha(p),'text_bytes':int(line.split()[0]),'meaning':'Object text includes all compiled functions and readonly text contents; not executed working set or cycles.'})
paths=[W/'build_capsule.py',Path(__file__),W/'capsule_qualification.json',W/'capsule_result.json',W/'gsim_receipt.json',W/'gsim.stdout',W/'strict_spike.log',W/'timing_spike.log',W/'strict_audit.json',W/'timing_audit.json',W/'strict_main.c',W/'timing_main.c',W/'normal_feature_reclosure/receipt.json',W/'normal_feature_reclosure/model.ll',W/'normal_feature_reclosure/model.o',W/'timing_build/layer.elf',W/'strict_build/layer.elf',R/'capture/receipt.json',R/'source_receipt.json',R/'source_capsule.mlir',R/'capsule/data.o']
paths.extend(R/'capture'/(n+'.npy')for n in ['a','scale_a','b','scale_b','expected'])
paths.extend(W/(n+'.'+ext)for n in ['control','outlined']for ext in ['o','native.ll','target.ll','so','npy'])
paths.extend(C/p for p in ['src/merlin/llvmlower/llvm_loop_outline.py','src/merlin/llvmlower/pipeline.py','merlin/tests/ir/test_llvm_loop_outline.py','docs/reference/llvm_loop_organization.md'])
paths.extend(L/p for p in ['clang','opt','llvm-nm'])
receipt={
    'schema':'compiler_optimization_journey_v1',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'hypothesis':'Exact loop extraction and function commoning can reduce large scalar host code while retaining source order, effects and public symbols.',
    'ownership':'Generic source/LLVM organization and normal compiler policy Merlin; actual target capsule/audit/receipts OOT. No target ISA implementation added.',
    'core_commit':'058a86465',
    'emitted_change':'LLVM loop-extract, new-helper-only noinline, mergefunc and verification; explicit default-off normal feature. Original scalar arithmetic and tensor/writer ABI unchanged.',
    'before_after_cycles':{'scope':'Complete original first2rows/all5632preDown channels,11264i8results, all input reads/dequant/poly/reciprocal/three rounded products/quant/stores/temporary allocation/finalcopy. Warm ABBA, validation outside ROI. Pinned GSIM memory regime, not stock FireSim or whole model.','paired_cycles':result['paired_complete_gsim_cycles'],'control_mean':result['before_mean'],'candidate_mean':result['after_mean'],'saved_percent':result['saved_percent']},
    'before_after_code':{'scope':'Separate unchanged1926whole host IR code-only screen.18→588→248 functions, original frozen host/runtime/device objects untouched. Function extraction afterRNE+bridge for this code-size screen; whole normal feature uses its actual pipeline placement and needs fresh gates.','objects':rows,'text_reduction_percent':100*(rows[0]['text_bytes']-rows[1]['text_bytes'])/rows[0]['text_bytes']},
    'gate':{'native_original_i8_words':11264,'target_five_rounding_modes_compared_words':56320,'matching_sticky_flags':True,'dirty_guard_bytes':128,'immutable_input_bytes':405504,'native_input_arrays_unchanged':True,'no_FSM':True,'GSIM_D​ONE':True,'normal_feature_measured_object_byteidentical':True,'original_whole256000_native_strict':'PENDING separate whole qualification'},
    'result':'Weak positive0.900895% local capsule. Proceed independent normal full-source qualification; no whole-cycle prediction or hardware admission.',
    'no_whole_speed_projection':True,
    'token_usage_available':False,
    'pins':{str(p):sha(p)for p in paths},
}
# ASCII receipt keys only.
receipt['gate']['GSIM_DONE']=receipt['gate'].pop('GSIM_D​ONE')
(W/'journey.json').write_text(json.dumps(receipt,indent=2)+'\n')
archive=O/'docs/perf_records/tiny_loop_organization_capsule'
archive.mkdir(exist_ok=False)
for name in ['archive.py','build_capsule.py','capsule_qualification.json','capsule_result.json','gsim_receipt.json','gsim.stdout','strict_spike.log','timing_spike.log','strict_audit.json','timing_audit.json','strict_main.c','timing_main.c','control.native.ll','control.target.ll','outlined.native.ll','outlined.target.ll']:
    (archive/name).write_bytes((W/name).read_bytes())
(archive/'normal_feature_reclosure.json').write_bytes((W/'normal_feature_reclosure/receipt.json').read_bytes())
(O/'docs/perf_records/tiny_loop_organization_capsule_journey.json').write_bytes((W/'journey.json').read_bytes())
print('OUTLINE_CAPSULE_ARCHIVED',result['saved_percent'],receipt['before_after_code']['text_reduction_percent'])
