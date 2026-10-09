"""Explicit source-bound owner composition; no automatic production selection."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess

from merlin.llvmlower.prepared_polynomial_word_results import prepare_polynomial_word_results, consume_polynomial_word_results
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output directory required')
    policy = ApproximateSourceRoundoffPolicy(*([True] * 8))
    policy.validate()
    args.output.mkdir(parents=True)
    records = {}
    for role in ('native_numeric', 'target_numeric'):
        src = args.baseline / role
        manifest = json.loads((src / 'compile.json').read_text())
        # Every retained compile input must still agree with its sealed receipt.
        for path, expected in manifest['pins'].items():
            if sha(path) != expected:
                raise ValueError(f'changed baseline dependency: {path}')
        dest = args.output / role
        dest.mkdir()
        for path in src.iterdir():
            if path.suffix in ('.h', '.c'):
                shutil.copyfile(path, dest / path.name)
        source = (src / 'provider.c').read_text()
        selected = consume_polynomial_word_results(source)
        header = prepare_polynomial_word_results((src/'prepared_polynomial_batch.h').read_text(), SourceNumericContract(*([True]*7),standard_fp_classification=True,classification_interposition_unobserved=True))
        (dest/'prepared_polynomial_batch.h').write_text(header)
        (dest / 'provider.c').write_text(selected)
        commands = []
        for old in manifest['commands']:
            command = [part.replace(str(src), str(dest)) for part in old]
            subprocess.run(command, check=True)
            commands.append(command)
        records[role] = dict(commands=commands, source_sha256=sha(dest / 'provider.c'), baseline_source_sha256=sha(src / 'provider.c'))
    lib = ctypes.CDLL(str((args.output / 'native_numeric/provider.so').resolve()))
    sizes = {}
    for symbol in ('group_provider_workspace_bytes', 'group_provider_workspace_alignment'):
        query = getattr(lib, symbol)
        query.restype = ctypes.c_size_t
        sizes[symbol] = query()
    receipt = dict(schema='smol_prepared_polynomial_word_results_v1', policy=policy.numerical_policy,
                   scope='Provider composition only; normal source binding and all48 execution pending.', sizes=sizes,
                   roles=records, pins={str(p.resolve()):sha(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output / 'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(sizes), flush=True)


if __name__ == '__main__':
    main()
