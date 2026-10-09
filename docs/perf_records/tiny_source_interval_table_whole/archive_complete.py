"""Seal the original source-wide table proof and immutable complete timing pair."""
from pathlib import Path
import hashlib
import json

ROOT = Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006')
CORE = Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006')
P = CORE / 'out/artifacts/probes/source-expression-interval-table-20261006'
T = Path(__file__).resolve().parent
D = T / 'complete_m2'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

cost = json.loads((D / 'paired_cost.json').read_text())
finish = json.loads((D / 'gsim_receipt.json').read_text())
qualification = json.loads((D / 'qualification.json').read_text())
assert finish['returncode'] == 0
assert finish['finish']['done'] and finish['finish']['exit_code'] == 0
assert cost['candidate_mean_cycles'] < cost['baseline_mean_cycles']
assert all(cost['gates'].values())
assert sha(qualification['elf_path']) == cost['qualified_elf_sha256']

pins = {}
for base in (P, D):
    for path in sorted(base.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            pins[str(path)] = sha(path)
for path in (T / 'build_complete_pair.py', T / 'run_complete_gsim.py',
             T / 'pc_census.py', Path(__file__)):
    pins[str(path)] = sha(path)
for receipt in (P / 'native_screen.json', P / 'independent_proof_checks.json',
                P / 'compiled_observer_checks.json', P / 'rational_primitive_checks.json',
                P / 'requested_traffic.json', P / 'lookup_emitter_ownership.json',
                D / 'qualification.json'):
    for path, expected in json.loads(receipt.read_text()).get('pins', {}).items():
        actual = sha(path)
        assert actual == expected, (receipt, path, expected, actual)
        pins[path] = actual
plan = json.loads((P / 'source_plan.json').read_text())
assert sha(plan['source_path']) == plan['source_sha256']
pins[plan['source_path']] = plan['source_sha256']
for path in (Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'),
             Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-link'),
             Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/opt'),
             Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'),
             Path(finish['engine']['path']),
             Path(finish['engine']['receipt']['receipt_path'])):
    pins[str(path)] = sha(path)

record = {
    'schema': 'tiny_source_expression_interval_table_complete_gsim_v1',
    'hypothesis': 'A source-wide binary32 interval table can remove repeated source polynomial FMA and division when the unchanged closed i8 consumer observes one certified bin.',
    'ownership': {'Merlin': 'typed source DAG/complete observation closure, portable interval proof/table/lookup emitter and unchanged rounded observer algorithm', 'OOT': 'RV64GC FRM guard, existing source RNE ISA legalization, target objects/link/zeroFSM audit and execution'},
    'actual_change': 'Replace only the proved unobserved activation endpoint with a table interval-certified carrier. Retain source preparation, every original final rounded multiply/clamp/RNE/store, exact original activation fallback, and whole-helper fallback for unsupported rounding modes.',
    'policy': {'fixed_leading_binary32_bits': 20, 'suffix_bits': 12, 'physical_table_bytes': 8388608, 'no_capture_or_golden_table_inputs': True, 'valid_cells': 1040131, 'invalid_cells': 8445, 'table_source': 'compiler-produced immutable source-bound table shipped in ELF, no runtime generation excluded', 'runtime': 'explicit RNE; finite operands and typed positive finite nonzero quant factor; nontrapping FP flags unobserved; unsupported cells/data or disagreeing source i8 bins replay original source; other FRMs execute complete baseline'},
    'proof': {'scope': 'Every binary32 word in every admitted fixed cell, by Cartesian interval extension of original typed arithmetic; endpoint checks are independent validation only.', 'document': str(P / 'PROOF.md'), 'original_float_escape': 'none; sole live i8 destination observation proved, unknown FP/call/store observers refuse', 'binding': 'Original complete typed source/hash and exact scalar LLVM DAG/constants/order; label/SSA renaming accepted; literal/FMA/float-store/unknown-observer mutations refuse'},
    'numeric_gates': {'original_full_first_section_i8_words': 45056, 'all_original_i8_exact': True, 'original_lookup_accepted': 45043, 'original_source_replays': 13, 'unobserved_carrier_word_differences': 45042, 'compiled_independent_scalar_cases': 19924, 'every_valid_cell_compiled_endpoint_checks': 2080262, 'exact_rational_FMA_checks': 6331, 'exact_rational_add_mul_div_cases_each': 5121, 'source_quant_boundary_cases': 785, 'original_complete_M2_native_strict_i8_words': 11264, 'all5_actual_target_rounding_modes': True, 'nonRNE_native_source_words_and_flags': '21 rounding-mode/sticky-preset cases exact', 'input_hashes_and_128_dirty_guards': True, 'executable_zeroFSM': True},
    'cost': {'scope': cost['scope'], 'engine': cost['engine'], 'raw_ABBA': cost['events'], 'baseline_mean_cycles': cost['baseline_mean_cycles'], 'candidate_mean_cycles': cost['candidate_mean_cycles'], 'fraction_change': cost['fraction_change'], 'inside_each_ROI': 'complete source preparations, table reads/addressing, guard/interval finish, original fallback, unchanged final arithmetic/observer/stores and outer FRM guard', 'outside_ROI': 'initialization, warm calls, complete5FRM source parity loop, per-ROI original output/guards checks, final input hashes', 'simulator_total_not_ROI': finish['finish'], 'samples': '2 per arm; no statistical confidence or stock FireSim result', 'max_simulator_cycles': 45000000, 'wall_timeout_seconds': 3600},
    'traffic': {'M2_requested_table_bytes': 90112, 'M2_unique_cells': 8601, 'M2_unique64byte_lines': 3159, 'M2_minimum_distinct64byte_fill_bytes': 202176, 'full8rows_unique_cells': 20336, 'full8rows_unique64byte_lines': 4632, 'physical_cache_misses_DRAM': 'UNKNOWN; hypothetical LRU models separately labeled, not hardware facts'},
    'actual_CPU_PC_classes': json.loads((D / 'pc_census.json').read_text()),
    'instructions_are_not_cycles': 'Baseline568958 vs candidate630936 body dispatches per M2, both CSR-free; candidate has more instructions yet lower actual GSIM cycles. Serialized wrapper CSR dispatch retries excluded. No workload or target-cycle fitting.',
    'result': 'POSITIVE complete local GSIM pair, retained for generic compiler promotion review. No production/default/whole-model256000 or stock FireSim qualification yet, no whole speedup projection.',
    'normal_build_or_missing_abstraction': 'Generic table/source/observer proof and explicit numeric/effect admission must be promoted into Merlin with a normal build constant-data/lookup seam; target FRM/ISA ABI glue remains OOT. Current source-bound extraction is an experimental binding, not a normal automatic policy.',
    'tokens': {'token_usage_available': False, 'scope': 'Root owns shared campaign checkpoints; exclusive per-agent/per-optimization allocation unavailable'},
    'pins': pins,
}
target = ROOT / 'docs/perf_records/tiny_source_interval_table_complete_gsim.json'
assert not target.exists()
target.write_text(json.dumps(record, indent=2) + '\n')
print('SOURCE_TABLE_COMPLETE_ARCHIVED', len(pins), cost['fraction_change'], flush=True)
