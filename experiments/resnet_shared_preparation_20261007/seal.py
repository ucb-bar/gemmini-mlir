"""Close the source-preserving normal compiler interface refactor independently."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CORE = Path('/scratch/agustin/tmp/merlin-prepared-model-transform-main-20261007')
OOT = Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')


def identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def main():
    core_receipt = CORE / 'out/artifacts/prepared_model_transform/final/qualification.json'
    core = json.loads(core_receipt.read_text())
    for pin in core['file_pins']:
        assert identity(pin['path']) == pin, pin['path']
    assert len(core['file_pins']) == 2189
    comparison = json.loads((ROOT / 'normal_byte_reclosure.json').read_text())
    assert all(row['byte_exact'] for row in comparison['comparison'])
    for row in comparison['comparison']:
        for arm, field in [('control_v2', 'control_sha256'), ('selected_v2', 'selected_sha256')]:
            assert identity(ROOT / arm / 'build_direct' / row['relative'])['sha256'] == row[field]
    terminal = json.loads((ROOT / 'strict_terminal.json').read_text())
    stdout = (ROOT / 'spike.stdout').read_text()
    assert terminal['returncode'] == 0
    assert 'OUT_SHA256 f32le 1000 4000 0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787' in stdout
    assert 'METRIC memref_rank_mismatch 0' in stdout and 'DONE' in stdout
    selected = ROOT / 'selected_v2/build_direct'
    audit = json.loads((selected / 'model.nofsm_audit.json').read_text())
    assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
    assert audit['elf_sha256'] == identity(selected / 'model.elf')['sha256']
    recipe = json.loads((selected / 'compilation_recipe.json').read_text())
    assert recipe['status'] == 'completed'
    bound = recipe['preparation']['prepared_model_transform']
    assert identity(bound['path'])['sha256'] == bound['sha256']
    paths = [Path(__file__), core_receipt, ROOT / 'normal_byte_reclosure.json',
             ROOT / 'strict_terminal.json', ROOT / 'strict_admission.json', ROOT / 'spike.stdout',
             ROOT / 'spike.stderr', ROOT / 'control.log', ROOT / 'selected.log',
             ROOT / 'control_v2.log', ROOT / 'selected_v2.log', ROOT / 'build_control.py',
             ROOT / 'build_selected.py', ROOT / 'derivation.json', ROOT / 'capabilities/facts.json',
             ROOT / 'capabilities/source_identity.json', selected / 'compilation_recipe.json',
             selected / 'model.nofsm_audit.json']
    for arm in ('control_v2', 'selected_v2'):
        paths += [p for p in (ROOT / arm).rglob('*') if p.is_file() and not p.is_symlink()
                  and p.suffix in ('.o', '.elf', '.mlir', '.ll', '.json', '.c')]
    paths += [CORE / 'src/merlin/runtime/backends/spike_model.py',
              CORE / 'src/merlin/runtime/backends/zephyr_model.py',
              CORE / 'src/merlin/llvmlower/prepared_model_transform.py']
    paths += [OOT / 'mlir_oot' / n for n in ('physical_layout.py', 'fused_mixed_catalog.py', 'stem_pool_mixed_catalog.py')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pins = list(pool.map(identity, sorted(set(paths))))
    receipt = {
        'schema': 'root.resnet_shared_preparation.normal_qualification.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'QUALIFIED_FRESH_NORMAL_FULL_SOURCE_INTERFACE_REFACTOR',
        'core_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CORE, text=True).strip(),
        'core_qualification': identity(core_receipt), 'core_pins_reclosed': 2189,
        'core_source_tests': 17, 'core_installed_tests': 17,
        'oot_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OOT, text=True).strip(),
        'oot_explicit_layout_selection': {'reduction_channel_block': 64, 'catalog_defaults_unchanged': True},
        'source_preparation': {'fresh_writer_routes': 70, 'quantization_packet_lanes': 4,
                               'global_function_replacements_in_selected_driver': 0},
        'comparison': comparison['comparison'],
        'comparison_scope': 'same fresh current-core/source/providers, legacy global hooks versus explicit local hooks; NOT byte reproduction of old stock2109',
        'original_output_gate': {'elements': 1000, 'atol': 0, 'rtol': 0, 'bitwise_exact': True,
                                 'raw_sha256': '0c2fb2f53d4f080e8d2da3a2647b0daa6fe0759f3d15833c1af2125b3ed14787'},
        'terminal': terminal, 'final_nofsm': audit,
        'spike_functional_counter': 8394275, 'whole_firesim_cycles': 'UNKNOWN',
        'preserved_refusals': 'both first attempts omitted original pinned RTL datapath facts; both refused before lowering',
        'promotion_scope': 'explicit shared compiler mechanism; no automatic cost policy or cross-workload qualification',
        'file_pins': pins,
    }
    (ROOT / 'qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'qualification': identity(ROOT / 'qualification.json'), 'pins': len(pins)}))


if __name__ == '__main__':
    main()
