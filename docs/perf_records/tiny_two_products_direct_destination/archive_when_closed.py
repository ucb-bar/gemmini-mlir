"""Archive the corrected materialization experiment after its real cycle gate."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

D = Path(__file__).resolve().parent
W = D / 'complete'
O = D.parents[3]
T = D.parent / 'tiny-two-multiplications-capsule-20261006'
C = Path('/scratch/agustin/tmp/merlin-bufferized-destination-identity-20261006')
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

result = read(W/'capsule_result.json')
engine = read(W/'gsim_receipt.json')
capture = read(T/'capture/receipt.json')
assert result['DONE'] and engine['finish']['done'] and engine['returncode'] == 0
assert result['native_original_f32_words'] == 16384 and result['zero_FSM']
assert result['all_five_rounding_modes_and_sticky_flags_match']
assert result['strict_compared_words'] == 81920
assert read(W/'independent_target_v1/qualification.json')['status'] == 'pass'
assert read(W/'typed_source_witness.json')['exact_complete_ssa_maps_types_constant_and_attribute_match']
assert sha(W/'control.o') == sha(T/'control.o')
assert 'call ptr @malloc' not in (W/'packet.ll').read_text()
assert 'call void @llvm.memcpy.' not in (W/'packet.ll').read_text()
assert subprocess.check_output(['git','-C',str(C),'status','--porcelain'],text=True) == ''

files = {p for p in W.rglob('*') if p.is_file()}
files.update({p for p in D.iterdir() if p.is_file()})
files.update({p for p in (T/'capture').iterdir() if p.is_file()})
bundle = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
files.update(bundle/name for name in ['model.mlir','inputs.npz','extra.npz',
    'weights.safetensors','weights.safetensors.manifest.json','golden.npy',
    'input_order.json','meta.json','capture_receipt.json'])
files.update([Path(read(W/'typed_source_witness.json')['source_path']),
              Path(capture['source_gate_path']),
              D.parent/'tiny-multiplication-packet4-whole-20261006/whole/host_llvm/expanded.native.ll'])
files.update(C/name for name in [
    'src/merlin/llvmlower/bufferized_result_identity.py',
    'src/merlin/llvmlower/scalar_pointwise_packet.py',
    'src/merlin/llvmlower/pipeline.py', 'src/merlin/llvmlower/impr_features.py',
    'src/merlin/llvmlower/abi.py','src/merlin/perf/layer_bench/__init__.py',
    'merlin/tests/ir/test_bufferized_result_identity.py',
    'merlin/tests/ir/test_scalar_pointwise_packet.py'])
files.update([T/'control.o',T/'packet.o',T/'control.ll',T/'packet.ll',
              T/'capsule_result.json',T/'gsim_receipt.json',T/'typed_source_witness.json',
              O/'docs/perf_records/tiny_two_multiplications_negative_journey.json',
              Path('/tmp/bufferized-result-identity-tests-final-20261006.log'),
              Path('/tmp/bufferized-result-identity-tests-extra-20261006.log')])
files.update(Path(p) for p in capture['boundary_pins'])
files.update(L/name for name in ['clang','opt','mlir-opt','mlir-translate','llvm-objdump'])
files.update([Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc'),
              Path('/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/emulator'),
              Path('/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/build_receipt.json')])
for arm in ['control','packet']:
    subprocess.run([str(L/'llvm-objdump'),'-d',str(W/(arm+'.o'))],
                   stdout=(W/(arm+'.disassembly.txt')).open('w'),check=True)
    files.add(W/(arm+'.disassembly.txt'))
for p, expected in capture['boundary_pins'].items():
    assert sha(p) == expected
for value in capture['captured_tensors'].values():
    assert sha(value['path']) == value['sha256']
old = read(T/'capsule_result.json')
receipt = {
    'schema':'compiler_optimization_journey_v1',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'hypothesis':'Upstream canonicalization exposes an unchanged loop-carried memref identity, enabling public output forwarding before result conversion and removing a complete temporary/copy.',
    'ownership':'Generic pipeline/materialization and source schedule in Merlin; target ISA/executable audit/GSIM evidence in OOT. No target branches, pointer no-alias or new arithmetic permissions.',
    'core_commit':subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip(),
    'core_tree':subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD^{tree}'],text=True).strip(),
    'emitted_change':'Explicit canonicalize_bufferized_result_identity inserts upstream canonicalize immediately before buffer-results-to-out-params. Actual source capsule composes this general choice with the separately selected exactly-two-multiply packet4; default and >=3 eligibility remain unchanged.',
    'typed_source_witness':read(W/'typed_source_witness.json'),
    'original_capture':capture,
    'original_input_checkpoint_gate':{'full_scope':'22 layers/eight tokens/256000 output words',
        'torch_atol':.03125,'torch_rtol':.02,'no_new_capture_or_golden':True},
    'before':{'complete_cycles':[x[2] for x in result['paired_complete_gsim_cycles'] if x[1] == 0], 'mean':result['before_mean'], 'static_malloc_calls':0,'output_copy_bytes':0,'control_object_reproduced_byteexact':True},
    'after':{'complete_cycles':[x[2] for x in result['paired_complete_gsim_cycles'] if x[1] == 1], 'mean':result['after_mean'], 'saved_percent':result['saved_percent'],'static_malloc_calls':0,'output_copy_bytes':0},
    'rejected_previous_materialization':{'complete_mean_cycles':old['after_mean'],'slower_percent':-old['saved_percent'],'malloc_calls':6,'malloc_requests_bytes':[65600,16,2,0,2,0],'output_copy_bytes':65536,'record':str(O/'docs/perf_records/tiny_two_multiplications_negative_journey.json')},
    'gate':{'original_native_words':16384,'five_frm_original_comparisons':81920,'sticky_flags_match':True,'independent_corner_tail_comparisons':1260,'immutable_input_bytes':139264,'dirty_output_guard_bytes':128,'dirty_private_arena':True,'zero_FSM':True,'engine_DONE':True,'engine_rc':0,'generic_native_alias_empty_tail_ownership_tests':15,'existing_packet_tests':56},
    'scope':result['scope'],
    'gsim_engine':engine['engine'],
    'result':'Complete capsule positive; root review and original full-model qualification required.' if result['saved_percent'] > 0 else 'Reject promotion: complete capsule still costs more after the materialization fix.',
    'whole_hardware_prediction':None,
    'instruction_counter_is_not_hardware_cycles':True,
    'token_usage_available':False,
    'pins':{str(p):sha(p) for p in sorted(files)},
}
out = O/'docs/perf_records/tiny_two_products_direct_destination'
out.mkdir(exist_ok=False)
for p in sorted(W.rglob('*')):
    if p.is_file() and p.suffix in {'.json','.py','.log','.txt','.mlir'} and p.name != 'run_lowering.py':
        dst = out/p.relative_to(W)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
for name in ['control.ll','packet.ll']:
    shutil.copyfile(W/name,out/name)
for name in ['archive_when_closed.py','screen.py','screen.json','lower.log']:
    shutil.copyfile(D/name,out/name)
(O/'docs/perf_records/tiny_two_products_direct_destination_journey.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('DIRECT_DESTINATION_ARCHIVED',len(files),result['saved_percent'],flush=True)
