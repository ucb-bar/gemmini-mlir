"""Controlled2062 final link and original whole target qualification.

Requires the independently completed profitable M8 cycle receipt. The existing
normal callback objects/native gates are immutable inputs; only model.o changes.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
Y = Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006')
P = Y / 'out/artifacts/probes/closed-i8-interval-result-20261007'
N = Y / 'out/artifacts/probes/closed-i8-interval-normal-2062-20261007'
OLD = Y / 'out/artifacts/probes/masked-contraction-normal-whole-v3-20261007'
O = N / 'whole'
GCC = Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
RAW = 'ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
CONTROL_ELF = '77dbb5d85b4ddd2a0bc69e445398e85141aa404eedecc1c733998c5ebe9a79f7'
GOLD = Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle/golden.npy')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, record):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')

def reclose(path):
    record = json.loads(path.read_text())
    assert all((sha(name) == digest for name, digest in record.get('pins', {}).items()))
    return record

def main():
    cost_path = Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006/out/artifacts/probes/closed-i8-interval-normal-2062-20261007/stock_cost_successor/cost.json')
    cost = reclose(cost_path)
    control_cycles = cost['control_mean_cycles']
    candidate_cycles = cost['candidate_mean_cycles']
    assert cost['status'] == 'pass' and candidate_cycles < control_cycles
    assert cost['engine_kind'] == 'stock' and cost['engine_completion_verified']
    for seal in ('closed_i8_interval_preparation', 'closed_i8_interval_code_adjunct'):
        reclose(Y / f'docs/perf_records/{seal}.json')
    native = reclose(N / 'qualification.json')
    contexts = reclose(N / 'actual22contexts/qualification.json')
    assert native['status'] == contexts['status'] == 'pass'
    assert native['all155devicebindings'] and native['control_model_object_byteexact2062']
    assert contexts['original_i8_words'] == 991232
    original = reclose(OLD / 'build.json')
    original_elf = OLD / 'selected/target/model.elf'
    original_model = OLD / 'selected/target/model.o'
    assert sha(original_elf) == CONTROL_ELF
    assert sha(N / 'control/target/model.o') == sha(original_model)
    output = N / 'selected/native/host/output.npy'
    array = np.load(output)
    assert array.shape == (1, 8, 32000) and array.dtype == np.float32
    assert hashlib.sha256(array.astype('<f4').tobytes()).hexdigest() == RAW
    assert np.allclose(array, np.load(GOLD), atol=0.03125, rtol=0.02)
    O.mkdir(exist_ok=False)
    case = O / 'target'
    case.mkdir()
    candidate_model = N / 'selected/target/model.o'
    os.link(candidate_model, case / 'model.o')
    commands = []

    def run(argv):
        commands.append(list(map(str, argv)))
        return subprocess.run(list(map(str, argv)), check=True, capture_output=True, text=True)
    baseline_argv = [str(N / 'control/target/model.o') if name == str(original_model) else name for name in original['candidate_link_argv']]
    baseline_argv[-1] = str(O / 'baseline_reproduced.elf')
    run(baseline_argv)
    assert sha(O / 'baseline_reproduced.elf') == CONTROL_ELF
    (O / 'baseline_reproduced.elf').unlink()
    os.link(original_elf, O / 'baseline_reproduced.elf')
    argv = [str(case / 'model.o') if name == str(N / 'control/target/model.o') else name for name in baseline_argv]
    argv[-1] = str(case / 'model.elf')
    objects = {name: sha(name) for name in argv if name.endswith('.o')}
    original_nonmodel = {name: digest for name, digest in original['candidate_objects'].items() if name != str(original_model)}
    assert {name: digest for name, digest in objects.items() if name != str(case / 'model.o')} == original_nonmodel
    assert len(original_nonmodel) == 11
    run(argv)
    assert objects == {name: sha(name) for name in objects}
    elf = case / 'model.elf'
    audit = audit_elf(elf.read_bytes())
    assert audit['status'] == 'pass'
    save(case / 'model.nofsm_audit.json', audit)
    (case / 'model.dump').write_text(run([GCC.with_name('riscv64-unknown-elf-objdump'), '-dr', case / 'model.o']).stdout)
    (case / 'model.nm').write_text(run([LLVM / 'llvm-nm', '--print-size', case / 'model.o']).stdout)
    bindings = N / 'selected/target/host_llvm/source_binding.json'
    source = reclose(bindings)
    assert source['all_original_device155_references_conserved']
    assert len(source['source_bindings']['routes']) == 44 and len(source['whole_helper_guards']) == 22
    assert source['readonly_bytes'] == 512 * 1024 and source['leading_bits'] == 16
    build = {'schema': 'closed_i8_interval_controlled2062_whole_build_v1', 'status': 'pass', 'baseline_byte_exact': True, 'baseline_elf_sha256': CONTROL_ELF, 'baseline_compile_object_byte_exact': True, 'candidate_elf_sha256': sha(elf), 'candidate_model_object_sha256': sha(case / 'model.o'), 'candidate_objects': objects, 'original_11_nonmodel_objects_unchanged': True, 'all155devicebindings': True, 'candidate_link_argv': argv, 'baseline_reproduction_argv': baseline_argv, 'commands': commands, 'control_job': 2062, 'control_cycles': 410147055, 'only_changed_linked_object': 'model.o', 'scope': 'Normal pre-object host transform on frozen2062 source, controlled whole link; all8tokens/22layers/256000outputs. Not a fresh full capture/upstream pipeline.', 'whole_cycles': 'UNKNOWN', 'capsule_cost_path': str(cost_path), 'capsule_cost_sha256': sha(cost_path), 'inherited_marker': '37bdf9be0856', 'marker_contract': 'Inherited nonunique1880 marker; exact finalELF/objects identify candidate.', 'token_usage_available': False, 'pins': {str(path): sha(path) for path in [Path(__file__), cost_path, bindings, N / 'qualification.json', N / 'actual22contexts/qualification.json', OLD / 'build.json', original_elf, O / 'baseline_reproduced.elf', *case.iterdir(), *map(Path, objects)] if path.is_file()}}
    save(O / 'build.json', build)
    command = [str(GCC.with_name('spike')), '-g', '--extension=gemmini', '--isa=rv64gc', '-m0x80000000:0x400000000', str(elf)]
    started = datetime.now(timezone.utc).isoformat()
    with (case / 'spike.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=3600)
    console = (case / 'spike.log').read_text().replace('\r', '')
    metrics = {line.split()[1]: line.split()[2] for line in console.splitlines() if line.startswith('METRIC ')}
    passed = result.returncode == 0 and 'DONE' in console and (metrics.get('memref_rank_mismatch') == '0') and (metrics.get('build_hash') == '37bdf9be0856') and (console.count('OUT_SHA256 f32le 256000 1024000 ' + RAW) == 1)
    record = {'schema': 'closed_i8_interval_original_whole_target_v1', 'status': 'pass' if passed else 'fail', 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'argv': command, 'elf_path': str(elf), 'elf_sha256': sha(elf), 'exit_code': result.returncode, 'reference_path': str(output), 'reference_sha256': sha(output), 'torch_golden_path': str(GOLD), 'torch_golden_sha256': sha(GOLD), 'torch_atol': 0.03125, 'torch_rtol': 0.02, 'torch_allclose': True, 'spike_console_path': str(case / 'spike.log'), 'spike_console_sha256': sha(case / 'spike.log'), 'spike_output_sha256': RAW if passed else None, 'spike_full_output_match': passed, 'metrics': metrics, 'functional_instructions_not_hardware_cycles': int(metrics.get('cycles', '0')), 'normal_lower_recipe_path': str(bindings), 'normal_lower_recipe_sha256': sha(bindings), 'controlled_link_path': str(O / 'build.json'), 'controlled_link_sha256': sha(O / 'build.json'), 'no_fsm': True, 'control_job': 2062, 'control_cycles': 410147055, 'whole_hardware_cycles': 'UNKNOWN', 'hardware_submission': 'NONE', 'token_usage_available': False}
    save(case / 'spike_validation.json', record)
    assert passed and sha(elf) == build['candidate_elf_sha256']
    adapter = {**record, 'schema': 'reference_validation_v1', 'all_original_compiled_words_exact': True, 'original_reference_elements': 256000, 'original_reference_shape': [1, 8, 32000], 'source_qualification_path': str(case / 'spike_validation.json'), 'source_qualification_sha256': sha(case / 'spike_validation.json'), 'inherited_marker': build['inherited_marker'], 'marker_contract': build['marker_contract'], 'scope': build['scope']}
    save(case / 'reference_validation.json', adapter)
    print('CLOSED_I8_CONTROLLED2062_WHOLE_TARGET_PASS', sha(elf), metrics, flush=True)
if __name__ == '__main__':
    main()
