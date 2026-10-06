"""Archive normal rectangular source contraction composition against verified2002."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

J = Path(__file__).resolve().parent
P = J.parent
O = P.parents[2]
C = Path('/scratch/agustin/tmp/merlin-scalar-contraction-rectangular-20261006')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
BUNDLE = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
BASE_NATIVE = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
CORE_FILES = [
    'src/merlin/llvmlower/scalar_squared_sum.py',
    'src/merlin/llvmlower/scalar_contraction.py',
    'merlin/tests/ir/test_scalar_contraction.py',
    'src/merlin/llvmlower/llvm_loop_outline.py',
    'merlin/tests/ir/test_llvm_loop_outline.py',
    'merlin/tests/ir/test_scalar_squared_sum.py',
    'src/merlin/llvmlower/bufferized_result_identity.py',
    'src/merlin/llvmlower/impr_features.py',
    'src/merlin/llvmlower/scalar_pointwise_packet.py',
    'src/merlin/llvmlower/pipeline.py',
    'src/merlin/llvmlower/lower.py',
    'src/merlin/llvmlower/late_quant_rne.py',
    'src/merlin/llvmlower/broadcast_math_hoist.py',
    'src/merlin/llvmlower/quant_hoist.py',
    'src/merlin/llvmlower/abi.py',
    'src/merlin/runtime/dispatch_runtime.py',
    'merlin/tests/ir/test_bufferized_result_identity.py',
    'merlin/tests/ir/test_scalar_pointwise_packet.py',
]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def check_mapping(mapping):
    for name, expected in mapping.items():
        assert sha(name) == expected, name

def archive(case_name, before_name, stock_job, stock_cycles, before_instructions, label):
    case = P / case_name
    whole = case / 'whole'
    before = P / before_name / 'whole'
    target = read(whole / 'spike_validation.json')
    native = read(whole / 'host/validation.json')
    normal = read(whole / 'normal_lower_recipe.json')
    link = read(whole / 'controlled_link.json')
    reference = read(whole / 'reference_validation.json')
    before_target = read(before / 'spike_validation.json')
    assert target['status'] == native['status'] == 'pass'
    assert native['original_compiled_bits_exact'] and native['allclose']
    assert reference['spike_full_output_match'] and reference['torch_allclose']
    assert target['outputs'] == native['outputs'] == 256000
    assert target['spike_output_sha256'] == 'ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
    assert target['torch_atol'] == .03125 and target['torch_rtol'] == .02
    assert target['nofsm_audit']['status'] == 'pass'
    assert not target['nofsm_audit']['forbidden'] and not target['nofsm_audit']['unknown']
    assert link['only_changed_object'] == 'model.o' and link['baseline_byte_exact']
    assert normal['all_original_source_bound_device_symbol_reference_counts_conserved']
    assert normal['original_source_bound_calls'] == 155
    assert 'scalar_squared_sum_accumulator' in normal['features']
    assert 'packet_scalar_pointwise_two_multiplications_4' in normal['features']
    assert 'canonicalize_bufferized_result_identity' in normal['features']
    assert 'packet_scalar_pointwise_broadcast_2' not in normal['features']
    assert 'packet_scalar_pointwise_multiplication_4' in normal['features']
    assert 'hoist_broadcast_source_rsqrt' in normal['features']
    assert 'outline_llvm_loops_merge_identical' in normal['features']
    assert set(normal['features']) - set(read(case / 'control/normal_lower_recipe.json')['features']) == {'scalar_contraction_accumulator_2x4_outputs'}
    assert set(read(case / 'control/normal_lower_recipe.json')['features']) - set(normal['features']) == {'scalar_contraction_accumulator_8_outputs'}
    assert 'unroll_scalar_contraction_reduction_by_2' in normal['features']
    census = read(case/'source_census.json')
    assert census['actual_source_operations'] == 45 and census['recurring_QK_PV_operations'] == 44
    assert census['additional_eligible_outer_product_operations'] == 1
    assert normal['normal_control_llvm_target_native_object_byteidentical2002']
    assert 'packet_scalar_pointwise_fma_division_2' in normal['features']
    assert before_target['functional_instructions_not_hardware_cycles'] == before_instructions
    assert target['before_verified_stock_job'] == stock_job
    assert target['before_verified_stock_cycles'] == stock_cycles
    check_mapping(normal['source_abi_pins'])
    check_mapping(link['baseline_objects'])
    check_mapping(link['candidate_objects'])
    check_mapping(native['native_boundary_sources'])
    assert sha(whole / 'model.elf') == target['elf_sha256'] == link['elf_sha256']
    assert sha(case / 'baseline_reproduced.elf') == sha(before / 'model.elf') == link['baseline_reproduced_sha256']
    for kind in ['expanded.ll', 'expanded.native.ll']:
        assert sha(case / 'control/host_llvm' / kind) == sha(before / 'host_llvm' / kind)
    assert sha(case / 'control/model.o') == sha(before / 'model.o')

    files = set(Path(p) for p in normal['source_abi_pins'])
    files.update(Path(p) for p in link['baseline_objects'])
    files.update(Path(p) for p in link['candidate_objects'])
    files.update(Path(p) for p in native['native_boundary_sources'])
    files.update(C / name for name in CORE_FILES)
    files.update(LLVM / name for name in ['clang', 'opt', 'llvm-nm', 'llvm-link', 'mlir-opt', 'mlir-translate'])
    files.update(BUNDLE / name for name in ['model.mlir', 'inputs.npz', 'extra.npz', 'weights.safetensors', 'weights.safetensors.manifest.json', 'golden.npy', 'input_order.json', 'meta.json', 'capture_receipt.json'])
    files.update(BASE_NATIVE / name for name in ['output.npy', 'validation.json'])
    files.update(before / name for name in ['model.o', 'model.elf', 'spike_validation.json', 'host/output.npy'])
    files.update([Path(__file__), case / 'baseline_reproduced.elf', O / 'mlir_oot/late_quant_rne.py', O / 'mlir_oot/no_fsm_audit.py'])
    for arm in ['control', 'whole']:
        files.update(p for p in (case / arm).rglob('*') if p.is_file() and p.suffix in {'.json', '.ll', '.mlir', '.o', '.so', '.npy', '.log', '.py', '.elf'})
    files.update(case / name for name in ['build_whole.py', 'validate_whole_native.py', 'qualify_whole_strict.py', 'build.log', 'native.log', 'strict.log'])
    # Compile/link executable and linker script bytes are part of the actual recipe.
    for argv in [normal['candidate_compile_argv'], link['candidate_link_argv'], native['compile_argv']]:
        files.add(Path(argv[0]))
    files.add(Path(link['candidate_link_argv'][link['candidate_link_argv'].index('-T') + 1]))
    plan = next(Path(p) for p in normal['source_abi_pins'] if p.endswith('/quant_hoist_args.json'))
    values = plan.with_name('quant_hoist_values.npz')
    if values.exists():
        files.add(values)
    hardware_receipt = Path(f'/scratch/agustin/tmp/firesim-golden-recovery-20261005/job{stock_job}_verified.json')
    if hardware_receipt.exists():
        files.add(hardware_receipt)
    files.add(O/'docs/perf_records/tiny_scalar_squared_sum_norm_capsule_journey.json')
    files.add(O/'docs/perf_records/tiny_two_products_direct_destination_whole_qualification.json')
    files.add(O/'docs/perf_records/tiny_rectangular_contraction_capsule_journey.json')
    files.add(O/'docs/perf_records/tiny_scalar_squared_sum_outline_whole_artifact_closure.json')
    files.update(p for p in case.glob('source_census*') if p.is_file())
    instructions = target['functional_instructions_not_hardware_cycles']
    result = {
        'schema': 'compiler_optimization_journey_v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'An explicit generic2M×4N scalar accumulator tile can reduce exact source contraction read repetition while preserving each output increasingK arithmetic and source ownership. The source-compatible whole composition needs independent gates and actual hardware costs; local gains are not whole predictions.',
        'ownership': 'Typed source contraction matching, exact rectangular scalar SSA scheduling and source ordered numeric/ownership proof Merlin; target executable/stock qualification OOT. No new ISA implementation.',
        'core_commit': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD'], text=True).strip(),
        'core_tree': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
        'emitted_change': 'Replace only explicit scalar_contraction_accumulator_8_outputs with scalar_contraction_accumulator_2x4_outputs on verified2002, retainK2/SQS/outline/normhoist/multiplication4/two-products4/identity/FMAdivision2/RNE. Source eligibility reaches44 recurring QK/PV plus1K1 outer product by typed maps/scalar body. Exact increasingK/mul/add operand order/seeds/precision/stores/source tensor ownership retained. Only model.o differs,155 source device bindings unchanged; no broadcast or device change.',
        'before': {'stock_job': stock_job, 'verified_stock_cycles': stock_cycles, 'strict_retired_instructions': before_instructions, 'elf_sha256': sha(before / 'model.elf')},
        'after': {'strict_retired_instructions': instructions, 'instruction_change_percent': 100 * (instructions - before_instructions) / before_instructions, 'hardware_cycles': None, 'elf_sha256': target['elf_sha256']},
        'gate': {'native_original_bits_exact': 256000, 'target_original_digest_words': 256000, 'torch_atol': .03125, 'torch_rtol': .02, 'torch_pass': True, 'DONE': True, 'rank_mismatch': 0, 'zero_FSM_all_executable_sections': True, 'all155_sourcebound_reference_counts_conserved': True, 'fresh_normal_control_target_native_IR_object_ELF_byteidentical': True, 'inherited_marker': link['inherited_marker'], 'marker_is_unique': False},
        'scope': link['scope'],
        'source_coverage': census,
        'retained_scope_refusal': read(case/'source_census_scope_refusal.json'),
        'independent_proof_scope': '338-pin capsule archive retains88native and8actualRV64 independent cases/five rounding modes/sticky flags/tails/alias/emptyK/dirty guards. Unconstrained source NaN payload choices follow upstream primary semantics; no strict-fenv eligibility and no original model gate change. Writable tensor seeds may be forwarded by upstream; fresh seeds used for each independent call. Original diagnostics retained.',
        'local_cost_scope': {'composition_actual_hardware_cycles': None, 'whole_prediction': None, 'independent_gains_not_summed': True},
        'result': 'Qualified fresh normal2×4 source contraction alternative against verified2002. Root review/release pending; actual hardware unmeasured.',
        'instruction_counter_is_not_hardware_cycles': True,
        'token_usage_available': False,
        'pins': {str(p): sha(p) for p in sorted(files)},
    }
    out = O / 'docs/perf_records' / label
    out.mkdir(exist_ok=False)
    for name in ['build_whole.py', 'validate_whole_native.py', 'qualify_whole_strict.py', 'build.log', 'native.log', 'strict.log']:
        shutil.copyfile(case / name, out / name)
    for name in ['normal_lower_recipe.json', 'controlled_link.json', 'reference_validation.json', 'spike_validation.json', 'model.nofsm_audit.json', 'spike.log']:
        shutil.copyfile(whole / name, out / name)
    shutil.copyfile(whole / 'host/validation.json', out / 'native_validation.json')
    shutil.copyfile(whole / 'lower/lowering_recipe.json', out / 'upstream_lowering_recipe.json')
    shutil.copyfile(case / 'control/normal_lower_recipe.json', out / 'control_normal_lower_recipe.json')
    shutil.copyfile(__file__, out / 'archive_whole.py')
    for path in case.glob('source_census*'):
        if path.is_file():
            shutil.copyfile(path, out/path.name)
    (O / 'docs/perf_records' / (label + '_qualification.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(label, 'FULL_QUALIFIED', instructions, result['after']['instruction_change_percent'], len(files), flush=True)

assert subprocess.check_output(['git', '-C', str(C), 'status', '--porcelain'], text=True) == ''
archive('tiny-rectangular-whole-20261006', 'tiny-squared-sum-outline-whole-20261006', 2002, 423831503, 134945424, 'tiny_rectangular_contraction_whole')
