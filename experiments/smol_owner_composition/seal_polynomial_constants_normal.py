"""Seal fresh explicit row-proof normal source, target build and native execution."""
from pathlib import Path
import hashlib
import json
from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/artifacts/probes/prepared-polynomial-constants/normal'
CORE = Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    native = json.loads((OUT / 'native_v3/validation.json').read_text())
    assert native['allclose'] and native['elements'] == 1600 and native['bitwise_mismatches'] == 0
    assert native['product_calls'] == 23040 and native['callback_errors'] == []
    assert native['calls'][:3] == [48, 0, 0] and native['calls'][4:7] == [12, 0, 0]
    elf = OUT / 'build_v3/model.elf'
    audit = audit_elf(elf.read_bytes())
    assert audit['status'] == 'pass'
    (OUT / 'normal_final_nofsm.json').write_text(json.dumps(audit, indent=2) + '\n')
    current = json.loads((OUT / 'build_v3/device_signatures.json').read_text())
    previous = json.loads((ROOT / 'out/normal_composition/build_v6/device_signatures.json').read_text())
    assert current['routed'] == previous['routed'] and current['signatures'] == previous['signatures']
    assert current['skipped'] == previous['skipped'] == []
    paths = list((OUT / 'drivers').glob('*.py')) + list((OUT / 'provider').glob('*/*'))
    paths += list((OUT / 'source').glob('*')) + list((OUT / 'native_v3').glob('*'))
    paths += list(OUT.glob('*.json')) + list(OUT.glob('*.log'))
    paths += [elf, OUT / 'build_v3/model.o', OUT / 'build_v3/lower/model.ll', OUT / 'build_v3/compilation_recipe.json', OUT / 'build_v3/device_signatures.json', OUT / 'build_v3/host_provider/host_provider.json', OUT / 'provider/build.json']
    paths += [CORE / 'src/merlin/llvmlower/exact_row_radix_pack.py', CORE / 'src/merlin/llvmlower/fused_encoded_witness.py', CORE / 'merlin/tests/runtime/test_exact_row_radix_pack.py', Path(__file__), ROOT / 'experiments/smol_owner_composition/build_exact_row_normal.py']
    paths += list((ROOT / 'out/normal_composition/capability_snapshot').glob('*'))
    paths += [CORE / 'src/merlin/llvmlower/prepared_polynomial_constants.py', CORE / 'src/merlin/llvmlower/rounded_polynomial_monotonicity.py', ROOT / 'experiments/attention_projection_frontier/bind_polynomial_constants.py', ROOT / 'experiments/smol_owner_composition/build_polynomial_constants_normal.py', ROOT / 'experiments/smol_owner_composition/resume_polynomial_constants_normal.py', ROOT / 'experiments/smol_owner_composition/resume_polynomial_constants_toolchain.py']
    paths = [p for p in paths if p != OUT / 'qualification.json']
    def check_recipe(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and Path(value['path']).is_file():
                assert sha(value['path']) == value['sha256'], value['path']
                paths.append(Path(value['path']))
            for child in value.values():
                check_recipe(child)
        elif isinstance(value, list):
            for child in value:
                check_recipe(child)
    check_recipe(json.loads((OUT / 'build_v3/compilation_recipe.json').read_text()))
    for command in native['commands']:
        paths.extend(Path(x) for x in command if Path(x).is_file())
    receipt = dict(schema='smol_prepared_polynomial_constants_normal_source_v1', core_commit='8b73513b7', option='exact_row_radix_pack + exact rounded polynomial constant context', numerical_policy='unchanged explicit approximate_source_roundoff_rms4', native=native, target_elf_sha256=sha(elf), no_fsm=audit['status'], ordinary_routed_contractions=297, ordinary_signatures=19, ordinary_catalog_binding='identical routed records and signatures to original normal build_v6', target_execution='UNMEASURED; compilation is not target execution', whole_performance='UNKNOWN', pins={str(p.resolve()): sha(p) for p in paths if p.is_file()})
    (OUT / 'qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'pins': len(receipt['pins']), 'elf': receipt['target_elf_sha256'], 'native': 'PASS'}))


if __name__ == '__main__':
    main()
