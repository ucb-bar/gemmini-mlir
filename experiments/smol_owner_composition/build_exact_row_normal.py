"""Fresh normal source build with explicit generic exact-row radix selection.

Retains the existing source preparation, ordinary device routing, numeric policy,
and callback ABI. Generated drivers are preserved alongside their exact inputs.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/exact_row_normal'
BASE = ROOT / 'experiments/smol_owner_composition'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one source binding: {old}')
    return text.replace(old, new)


def main():
    OUT.mkdir(exist_ok=False)
    drivers = OUT / 'drivers'
    drivers.mkdir()
    records = {}
    for name in ('build_provider.py', 'prepare_source.py', 'build_normal.py', 'validate_native.py'):
        source = (BASE / name).read_text()
        selected = source.replace("Path(__file__).resolve().parents[2]", repr(str(ROOT)))
        # Preserve Path semantics in the original drivers.
        selected = selected.replace(repr(str(ROOT)) + "/'out/normal_composition", "Path(" + repr(str(ROOT)) + ")/'out/exact_row_normal")
        if name == 'build_provider.py':
            selected = selected.replace('from merlin.llvmlower.prepared_attention_rhs import prepare_attention_rhs_owner', 'from merlin.llvmlower.prepared_attention_rhs import prepare_attention_rhs_owner\nfrom merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix')
            selected = replace_once(selected, "        (dest / 'provider.c').write_text(selected)", "        selected = prepare_exact_row_radix(selected)\n        (dest / 'provider.c').write_text(selected)")
        if name == 'prepare_source.py':
            selected = selected.replace('from merlin.llvmlower.prepared_attention_rhs import prepare_attention_rhs_owner', 'from merlin.llvmlower.prepared_attention_rhs import prepare_attention_rhs_owner\nfrom merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix')
            selected = replace_once(selected, " if (provider/'target_numeric/provider.c').read_text()!=expected:", " expected=prepare_exact_row_radix(expected)\n if (provider/'target_numeric/provider.c').read_text()!=expected:")
        selected = selected.replace("'build_v6'", "'build_v1'").replace("'build_v5", "'build_v1").replace("'native_v3'", "'native_v1'")
        destination = drivers / name
        destination.write_text(selected)
        compile(selected, str(destination), 'exec')
        records[name] = {'original_sha256': hashlib.sha256(source.encode()).hexdigest(), 'selected_sha256': hashlib.sha256(selected.encode()).hexdigest()}
    (OUT / 'driver_binding.json').write_text(json.dumps({'explicit_option': 'exact_row_radix_pack', 'drivers': records}, indent=2) + '\n')
    python = '/scratch/agustin/projects/oscar-merlin/.venv/bin/python'
    def run(name, *args):
        with (OUT / (name + '.log')).open('w') as log:
            subprocess.run([python, str(drivers / name), *map(str, args)], check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    run('build_provider.py', '--baseline', '/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate', '--output', OUT / 'provider')
    run('prepare_source.py')
    shutil.copyfile(OUT / 'source/normal_prepared.mlir', OUT / 'source/normal_prepared_original.mlir')
    run('build_normal.py')
    run('validate_native.py')
    print('EXACT_ROW_NORMAL_BUILD_NATIVE_PASS', flush=True)


if __name__ == '__main__':
    main()
