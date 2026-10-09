"""Archive source-compatible loop organization on the qualified two-product direct-destination arm."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

J = Path(__file__).resolve().parent
P = J.parent
O = P.parents[2]
C = Path('/scratch/agustin/tmp/merlin-bufferized-destination-identity-20261006')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
BUNDLE = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
BASE_NATIVE = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/whole_2/host')
CORE_FILES = [
    'src/merlin/llvmlower/bufferized_result_identity.py',
    'src/merlin/llvmlower/llvm_loop_outline.py',
    'merlin/tests/ir/test_llvm_loop_outline.py',
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
    assert 'packet_scalar_pointwise_two_multiplications_4' in normal['features']
    assert 'canonicalize_bufferized_result_identity' in normal['features']
    assert 'packet_scalar_pointwise_broadcast_2' not in normal['features']
    assert 'packet_scalar_pointwise_multiplication_4' in normal['features']
    assert 'hoist_broadcast_source_rsqrt' in normal['features']
    assert 'outline_llvm_loops_merge_identical' in normal['features']
    assert 'packet_scalar_pointwise_fma_division_2' in normal['features']
    assert before_target['functional_instructions_not_hardware_cycles'] == before_instructions
    assert target['before_stock_job'] == stock_job
    assert target['before_stock_hardware_status'] == 'pending'
    assert target['before_verified_stock_cycles'] is None
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
        files.update(p for p in (case / arm).rglob('*') if p.is_file() and p.suffix in {'.json', '.ll', '.mlir', '.o', '.so', '.npy', '.log', '.py'})
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
    files.add(O/'docs/perf_records/tiny_two_products_direct_destination_journey.json')
    files.add(O/'docs/perf_records/tiny_two_products_direct_destination_whole_qualification.json')
    files.add(O/'docs/perf_records/tiny_loop_organization_capsule_journey.json')
    files.add(O/'docs/perf_records/tiny_two_multiplications_negative_journey.json')
    instructions = target['functional_instructions_not_hardware_cycles']
    result = {
        'schema': 'compiler_optimization_journey_v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Source-compatible loop organization/identical-helper merging may compose with the independent exact two-product schedule and generic buffer identity exposure. Complete whole qualification and actual stock cycles are required; no whole forecast from the older local0.9% outline result.',
        'ownership': 'Generic typed tensor schedule and normal compiler policy Merlin; target qualification and final executable/stock evidence OOT. No new ISA implementation.',
        'core_commit': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD'], text=True).strip(),
        'core_tree': subprocess.check_output(['git', '-C', str(C), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
        'emitted_change': 'Retain the complete qualified1983 two-product/direct-destination normal source route; add only explicit outline_llvm_loops_merge_identical. Original norm hoist, >=3multiplicationpacket4, exactly-two-multiplicationpacket4, bufferized identity exposure, FMA/divisionpacket2 and bounded-RNE retained. All source arithmetic/precision/effects preserved; only model.o differs in frozen final link.',
        'before': {'stock_job': stock_job, 'hardware_status': 'queued, outcome unknown at qualification', 'verified_stock_cycles': None, 'strict_retired_instructions': before_instructions, 'elf_sha256': sha(before / 'model.elf')},
        'separate_verified_organization_reference': {'stock_job':1975,'cycles':450035885,'scope':'Outline plus >=3multiplypacket4; no newtwo-product/identity changes. Not the isolated control for this composition.'},
        'after': {'strict_retired_instructions': instructions, 'instruction_change_percent': 100 * (instructions - before_instructions) / before_instructions, 'hardware_cycles': None, 'elf_sha256': target['elf_sha256']},
        'gate': {'native_original_bits_exact': 256000, 'target_original_digest_words': 256000, 'torch_atol': .03125, 'torch_rtol': .02, 'torch_pass': True, 'DONE': True, 'rank_mismatch': 0, 'zero_FSM_all_executable_sections': True, 'all155_sourcebound_reference_counts_conserved': True, 'fresh_normal_control_target_native_IR_object_ELF_byteidentical': True, 'inherited_marker': link['inherited_marker'], 'marker_is_unique': False},
        'scope': link['scope'],
        'local_cost_scope': {'underlying_two_product_complete_capsule_gsim_saved_percent': 52.490690440277795, 'prior_loop_organization_complete_capsule_saved_percent':0.90089475, 'same_source_features': 'Underlying1983 route corrected original1x8x2048 two-product/residual complete capsule (52.49%win). Current only-change is loop organization, whose prior1x2x5632 complete capsule improved0.9009% on earlier compatible source. Neither capsule predicts this whole result.', 'whole_prediction': None},
        'result': 'Qualified source-compatible normal loop organization composition on1983 two-product/direct-destination route; root review/release pending. Both current control1983 and candidate hardware unmeasured.1975 is separate verified organization reference.',
        'instruction_counter_is_not_hardware_cycles': True,
        'token_usage_available': False,
        'pins': {str(p): sha(p) for p in sorted(files)},
    }
    out = O / 'docs/perf_records' / label
    out.mkdir(exist_ok=False)
    for name in ['build_whole.py', 'validate_whole_native.py', 'qualify_whole_strict.py', 'build.log', 'native.log', 'strict.log']:
        shutil.copyfile(case / name, out / name)
    for name in ['normal_lower_recipe.json', 'controlled_link.json', 'reference_validation.json', 'spike_validation.json', 'model.nofsm_audit.json', 'spike.log', 'spike_pc_histogram.log']:
        shutil.copyfile(whole / name, out / name)
    shutil.copyfile(whole / 'host/validation.json', out / 'native_validation.json')
    shutil.copyfile(whole / 'lower/lowering_recipe.json', out / 'upstream_lowering_recipe.json')
    shutil.copyfile(case / 'control/normal_lower_recipe.json', out / 'control_normal_lower_recipe.json')
    shutil.copyfile(__file__, out / 'archive_whole.py')
    (O / 'docs/perf_records' / (label + '_qualification.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(label, 'FULL_QUALIFIED', instructions, result['after']['instruction_change_percent'], len(files), flush=True)

assert subprocess.check_output(['git', '-C', str(C), 'status', '--porcelain'], text=True) == ''
archive('tiny-two-products-outline-composed-whole-20261006', 'tiny-two-products-destination-whole-20261006', 1983, None, 135853970, 'tiny_two_products_outline_composed_whole')
