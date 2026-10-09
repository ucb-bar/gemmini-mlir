"""Reclose the owned floor and row-arithmetic experiments; no new execution.

The counters come from a functional Spike mcycle proxy. They are not hardware
cycles, a calibrated timing estimate, or whole-model performance.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mlir_oot.no_fsm_audit import audit_elf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    probes = root / 'out/artifacts/probes/smol-monotone-polynomial-20261006'
    pins = {}

    def pin(path, expected=None):
        path = Path(path).absolute()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None:
            assert actual == expected, str(path)
        if str(path) in pins:
            assert actual == pins[str(path)]
        pins[str(path)] = actual
        return actual

    control_path = root / 'docs/perf_records/smol_directed_zero_error_complete_group_target.json'
    pin(control_path)
    control = json.loads(control_path.read_text())
    arms = {}
    original_markers = control['strict']['markers']
    control_counter = next(int(x.split()[1]) for x in original_markers if x.startswith('ENDPOINT_CYCLES '))
    invariant_prefixes = ('DIRECTED_MATH ', 'ENDPOINT_PLANES ', 'ENDPOINT_REPLAY ',
                          'FRONTIER_REFINE ', 'ENDPOINT_OUTPUT ')
    invariant_markers = [x for x in original_markers if x.startswith(invariant_prefixes)]
    for name in ('floor_bits', 'row_invariants'):
        target = probes / f'{name}_target'
        build_path = target / 'build_receipt.json'
        strict_path = target / 'timed.spike.json'
        native_path = probes / name / 'result.json'
        for path in (build_path, strict_path, native_path):
            pin(path)
        build = json.loads(build_path.read_text())
        strict = json.loads(strict_path.read_text())
        native = json.loads(native_path.read_text())
        assert strict['rc'] == 0 and strict['pass'] is True
        assert 'ENDPOINT PASS' in strict['markers']
        assert [x for x in strict['markers'] if x.startswith(invariant_prefixes)] == invariant_markers
        assert native['original_i8_words_exact'] and native['original_scale_words_exact']
        assert (native['i8_words'], native['scale_words'], native['changed_endpoint_words']) == (196608, 256, 125)
        assert native['counts']['replay_total_fmas'] == 4461440
        assert build['accuracy_gate_changed'] is False and build['default_changed'] is False
        for collection in ('pins', 'objects'):
            for path, expected in build.get(collection, {}).items():
                pin(path, expected)
        for path, expected in native['pins'].items():
            pin(path, expected)
        pin(native['tap_path'], native['tap_sha256'])
        pin(target / 'timed.elf', strict['elf_sha256'])
        pin(target / 'timed.spike.log', strict['log_sha256'])
        assert strict['elf_sha256'] == build['elf_sha256']
        audit = audit_elf((target / 'timed.elf').read_bytes())
        assert audit['status'] == 'pass'
        assert audit == build['nofsm_audit']
        for key in ('control_build', 'control_strict'):
            if f'{key}_path' in build:
                pin(build[f'{key}_path'], build[f'{key}_sha256'])
        pin(build['compile_argv'][0])
        for token in build['link_argv']:
            if token.endswith('.ld'):
                pin(token)
        counter = next(int(x.split()[1]) for x in strict['markers'] if x.startswith('ENDPOINT_CYCLES '))
        stages = next([int(v) for v in x.split()[1:]] for x in strict['markers'] if x.startswith('ENDPOINT_STAGE '))
        arms[name] = dict(native=native, build=build, strict=strict, audit=audit,
                          spike_counter_instruction_proxy=counter,
                          stage_instruction_proxies=stages,
                          reduction_vs_directed_zero_percent=100*(control_counter-counter)/control_counter)
    floor = arms['floor_bits']['spike_counter_instruction_proxy']
    row = arms['row_invariants']['spike_counter_instruction_proxy']
    receipt = dict(
        schema='exact_source_row_arithmetic_qualification_v1',
        scope='Complete original first source attention group: 12 heads, 256 queries, 1024 keys. Native integer stand-in and strict target primitive execution qualify identical original quantized observations. Counter is functional Spike instruction proxy, never stock FireSim cycles or whole-model prediction.',
        control_counter_instruction_proxy=control_counter,
        control_receipt=str(control_path),
        row_reduction_vs_floor_percent=100*(floor-row)/floor,
        invariant_markers=invariant_markers,
        arms=arms,
        obligations=[
            'Stable source RNE and gradual underflow; nontrapping arithmetic and unobserved exception flags.',
            'Frozen distinct private arrays prove row factors and denominators immutable across endpoint writes. Arbitrary C pointers grant no equivalent hoisting permission.',
            'Original binary32 source multiplications preserve signed zero; no multiply-to-FMA replacement.',
            'All original quantized words and escaping BF16 scales remain exact; changed carrier words are protected by complete consumer observation proofs.',
            'Production caller-workspace provider under its normal compiler flags is a separate qualification and is not enabled by these capsule results.',
        ],
        ownership=dict(portable_floor_and_interval_algorithms='Merlin',
                       fixed_rounding_CPU_ISA_and_Gemmini_product_schedules='OOT',
                       row_hoisting_legality_and_provider_compilation='Merlin'),
        performance_targets_met=False,
        stock_hardware_measured=False,
        token_usage_available=False,
        pins=pins,
    )
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(pins_rehashed=len(pins), control=control_counter,
                         floor=floor, row=row, row_reduction_vs_floor_percent=receipt['row_reduction_vs_floor_percent'])))


if __name__ == '__main__':
    main()
