"""Archive the rejected source-exact two-product schedule and its owned bytes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess

W = Path(__file__).resolve().parent
O = W.parents[3]
C = Path('/scratch/agustin/tmp/merlin-tiny-two-multiply-20261006')
L = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

result = read(W / 'capsule_result.json')
capture = read(W / 'capture/receipt.json')
typed = read(W / 'typed_source_witness.json')
independent = read(W / 'independent_target_v1/qualification.json')
assert result['DONE'] and result['saved_percent'] < 0
assert result['zero_FSM'] and capture['original_compiled_bits_exact']
assert capture['torch_original_gate_pass'] and capture['outputs'] == 256000
assert typed['exact_complete_ssa_maps_types_constant_and_attribute_match']
assert subprocess.check_output(['git', '-C', str(C), 'status', '--porcelain'], text=True) == ''

memory = {}
for arm in ['control', 'packet']:
    text = (W / (arm + '.ll')).read_text().split('define void @_mlir_ciface_', 1)[0]
    malloc = [int(n) for n in re.findall(r'call ptr @malloc\(i64 (\d+)\)', text)]
    memory[arm] = {
        'static_malloc_calls': len(malloc), 'constant_malloc_requests_bytes': malloc,
        'static_llvm_memcpy_calls': text.count('call void @llvm.memcpy.'),
        'final_output_copy_logical_bytes': 65536 if arm == 'packet' else 0,
        'source_output_words': 16384,
    }
    subprocess.run([str(L / 'llvm-objdump'), '-d', str(W / (arm + '.o'))],
                   stdout=(W / (arm + '.disassembly.txt')).open('w'), check=True)

files = {p for p in W.rglob('*') if p.is_file()}
files.update(C / name for name in [
    'src/merlin/llvmlower/scalar_pointwise_packet.py',
    'src/merlin/llvmlower/pipeline.py', 'src/merlin/llvmlower/impr_features.py',
    'src/merlin/llvmlower/abi.py', 'merlin/tests/ir/test_scalar_pointwise_packet.py',
    'src/merlin/llvmlower/AGENT.md'])
files.update(L / name for name in ['clang', 'opt', 'mlir-opt', 'mlir-translate', 'llvm-objdump'])
files.update(Path(p) for p in capture['boundary_pins'])
files.update([Path(typed['source_path']), Path(capture['source_gate_path']),
              Path('/tmp/tiny-two-mul-tests-20261006.log')])
for arm in ['control', 'packet']:
    recipe = read(W / (arm + '_lower/lowering_recipe.json'))
    for entry in recipe['sources'].values():
        files.add(Path(entry['path']))
files.update([C / 'src/merlin/perf/layer_bench/__init__.py',
              Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc'),
              Path('/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/emulator'),
              Path('/scratch/agustin/projects/oscar-merlin/out/build/rtl_engines/gemmini/gsim/build_receipt.json')])
for path, expected in capture['boundary_pins'].items():
    assert sha(path) == expected, path
for value in capture['captured_tensors'].values():
    assert sha(value['path']) == value['sha256']

receipt = {
    'schema': 'compiler_optimization_journey_v1',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'hypothesis': 'Interleave independent conversion/two-multiply/residual chains to expose scalar instruction latency.',
    'ownership': 'Typed schedule and materialization in Merlin; target ISA legality, executable audit and GSIM qualification in OOT.',
    'core_commit': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD'], text=True).strip(),
    'core_tree': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
    'emitted_change': 'Explicit separate four-lane exactly-two-f32-multiplication selector before upstream bufferization. Existing >=3 multiplication selector and default pipeline are unchanged. Each cast, multiply and addition retains its original rounding/order.',
    'source_typed_witness': typed,
    'capture_scope': 'Read-only before/after taps in the accepted complete 22-layer eight-token native image, preserving all 256000 compiled output bits and the original Torch gate. The capsule uses original 1x8x2048 dequant/residual operands, not a recapture or substituted golden.',
    'before': {'complete_gsim_cycles': [492115, 488297], 'mean': 490206.0, 'spike_retired_instructions_per_roi': 213115},
    'after': {'complete_gsim_cycles': [591132, 590128], 'mean': 590630.0, 'slower_percent': 20.48608136171324, 'spike_retired_instructions_per_roi': 479407},
    'gate': {'original_native_capsule_words_exact': 16384, 'five_frm_raw_output_comparisons': 81920, 'sticky_fp_flags_match': True, 'independent_target_corner_tail_comparisons': 1260, 'immutable_inputs_bytes': 139264, 'dirty_output_guards_bytes': 128, 'dirty_private_allocator': True, 'zero_FSM': True, 'DONE': True, 'gsim_rc': 0},
    'materialization_evidence': memory,
    'scope': result['scope'],
    'result': 'Rejected. No full-model candidate compilation or stock admission. The actual complete paired capsule loses 20.4861%; the packet IR also exposes an avoidable destination-forwarding problem worth a separate general compiler experiment.',
    'harness_refusals_retained': ['missing_ciface_attempt: original entry lacked llvm.emit_c_interface; no arithmetic failure', 'independent_target: all checks passed but required DONE was absent; independent_target_v1 adds DONE in a fresh directory'],
    'hardware_whole_prediction': None,
    'instruction_counter_is_not_hardware_cycles': True,
    'token_usage_available': False,
    'pins': {str(p): sha(p) for p in sorted(files)},
}
out = O / 'docs/perf_records/tiny_two_multiplications_negative'
out.mkdir(exist_ok=False)
for p in sorted(W.rglob('*')):
    if p.is_file() and p.suffix in {'.py', '.json', '.mlir', '.log', '.txt'}:
        # The lowering runner is large but is the actual executable rewrite proof.
        dst = out / p.relative_to(W)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
(O / 'docs/perf_records/tiny_two_multiplications_negative_journey.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('ARCHIVED_REJECTED', len(files), receipt['after']['slower_percent'], memory, flush=True)
