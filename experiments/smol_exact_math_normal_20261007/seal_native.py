"""Qualify the explicit existing exact-math selection without a hardware verdict."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CORE = Path('/scratch/agustin/tmp/merlin-host-llvm-helper-main-20261007')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')


def identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def main():
    native = json.loads((ROOT / 'native/validation.json').read_text())
    assert native['elements'] == 1600 and native['bitwise_mismatches'] == 0 and native['allclose']
    assert native['product_calls'] == 23040 and not native['callback_errors']
    assert native['calls'][:3] == [48, 0, 0] and native['calls'][4:7] == [12, 0, 0]
    build = ROOT / 'build'
    audit = json.loads((build / 'model.nofsm_audit.json').read_text())
    assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
    assert audit['elf_sha256'] == identity(build / 'model.elf')['sha256']
    recipe = json.loads((build / 'compilation_recipe.json').read_text())
    assert recipe['status'] == 'completed'
    undefined = subprocess.check_output([str(LLVM / 'llvm-nm'), '-u', str(build / 'model.o')], text=True)
    assert '__truncsfbf2' not in undefined
    (ROOT / 'model_undefined_symbols.txt').write_text(undefined)
    census = json.loads((ROOT / 'llvm_census.json').read_text())
    inherited = Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007/docs/perf_records/prepared_endpoint_dag_normal_whole_qualification.json')
    prior = json.loads(inherited.read_text())
    for path, expected in prior['pins'].items():
        assert identity(path)['sha256'] == expected, path
    paths = [Path(__file__), ROOT / 'build_normal.py', ROOT / 'validate_native.py',
             ROOT / 'run_full_strict.py', ROOT / 'selection.json', ROOT / 'llvm_census.json',
             ROOT / 'strict_admission.json', ROOT / 'model_undefined_symbols.txt', inherited,
             LLVM / 'llvm-nm', LLVM / 'clang', CORE / 'src/merlin/llvmlower/roundeven_intrinsic.py']
    paths += [p for folder in (build, ROOT / 'native') for p in folder.rglob('*')
              if p.is_file() and not p.is_symlink() and p.suffix in ('.o', '.elf', '.mlir', '.ll', '.json', '.c', '.so')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pins = list(pool.map(identity, sorted(set(paths))))
    receipt = {
        'schema': 'root.smol_exact_math.normal_native_qualification.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'QUALIFIED_NATIVE_TARGET_BUILT_WHOLE_SPIKE_PENDING',
        'core_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CORE, text=True).strip(),
        'feature': 'lower_exact_math_inline', 'generic_implementation_preexisted_on_main': True,
        'inherited_source_pins_reclosed': len(prior['pins']), 'inherited_source_packet': identity(inherited),
        'native_original_gate': {'elements': 1600, 'atol': 0.03125, 'rtol': 0.02,
                                'bitwise_mismatches': 0, 'allclose': True, 'maxabs': 0.0},
        'native_scope': 'actual full48 current normal source/provider composition; native exact integer product stand-ins, not hardware',
        'native_calls': {'groups': 48, 'preparations': 12, 'product_callbacks': 23040,
                         'fallback': 0, 'lifetime_errors': 0},
        'llvm_census': census, 'model_object_has_bf16_truncation_helper_import': False,
        'target_elf': identity(build / 'model.elf'), 'final_nofsm': audit,
        'whole_target_execution': 'PENDING', 'whole_firesim_cycles': 'UNKNOWN',
        'comparison_scope': 'new current3430 normal recipe versus earlierf90 source; no isolated pass cycle attribution',
        'numeric_policy': 'original explicit approximate_source_roundoff_rms4 retained; original final gate unchanged',
        'file_pins': pins,
    }
    (ROOT / 'native_qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'qualification': identity(ROOT / 'native_qualification.json'), 'pins': len(pins)}))


if __name__ == '__main__':
    main()
