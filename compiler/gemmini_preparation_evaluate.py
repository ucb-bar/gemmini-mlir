#!/usr/bin/env python3
"""Freeze ordinary preparation-CSE variants and check actual emitted products.

This is a development evaluator, never an ordinary compilation action. It
requests no broker execution and makes no cycle or runtime-custody claim.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'compiler_library')]
from gemmini_conv_evaluate import copy_package, digest, manifest_emit, tree_identity


def prepare(submission, output, sources):
    submission, output = Path(submission).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('choose a fresh output directory outside the compiler')
    output.mkdir(parents=True)
    parent = tree_identity(submission)
    frozen = output / 'compiler'
    copy_package(submission, frozen)
    if tree_identity(frozen) != parent or tree_identity(submission) != parent:
        raise ValueError('compiler changed during snapshot; retry')
    records = []
    for source in sources:
        source = Path(source).resolve()
        original = source.read_bytes()
        saved = output / (digest(original) + '.mlir')
        saved.write_bytes(original)
        records.append(dict(original_path=str(source), path=str(saved), source_sha256=digest(original)))
    modes, products = {}, []
    script = '''import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/'compiler_library')]\nfrom gemmini_backend import compile_source\ncb,ll=compile_source(Path(sys.argv[2]).read_text())\nPath(sys.argv[3]).write_text(json.dumps(cb,sort_keys=True,indent=2)+'\\n')\nPath(sys.argv[4]).write_text(ll+'\\n')\n'''
    tools = Path('/work/merlin/toolchains/llvm-a47bddcc/install/bin')
    for mode in ('off', 'on'):
        package = output / ('compiler-' + mode)
        copy_package(frozen, package)
        declaration = json.loads((package / 'TUNING_KNOBS.json').read_text())
        settings = package / declaration['settings_file']
        choices = json.loads(settings.read_text())
        choices['graph_preparation_reuse'] = mode
        settings.write_text(json.dumps(choices, indent=2) + '\n')
        modes[mode] = dict(path=str(package), compiler_sha256=tree_identity(package))
        for record in records:
            location = output / record['source_sha256'] / mode
            ordinary, receipt = manifest_emit(package, Path(record['path']), location)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-c', script, str(package), record['path'],
                str(location/'cb.json'), str(location/'replay.mlir')], env=env, check=True)
            if ordinary != (location/'replay.mlir').read_bytes():
                raise ValueError('direct observation differs from ordinary manifest artifact')
            subprocess.run([str(tools/'mlir-translate'), '--mlir-to-llvmir', str(location/'stdout.mlir'),
                '-o', str(location/'kernel.ll')], check=True)
            cb = json.loads((location/'cb.json').read_text())
            params = cb['params'].get('graph_preparation_reuse', {})
            products.append(dict(mode=mode, source_sha256=record['source_sha256'], **receipt,
                command_buffer_sha256=digest((location/'cb.json').read_bytes()),
                kernel_abi=cb['kernel_abi'], preparations=params,
                llvm_bytes=len(ordinary), llvm_translation_pass=True,
                replay_byte_identical=True,
                materialization_load_sites=ordinary.count(b'llvm.load'),
                materialization_store_sites=ordinary.count(b'llvm.store')))
    changes = []
    for record in records:
        pair = {r['mode']:r for r in products if r['source_sha256']==record['source_sha256']}
        if pair['off']['program_sha256'] == pair['on']['program_sha256']:
            raise ValueError('preparation reuse did not change the ordinary emitted product')
        if pair['off']['kernel_abi']['outputs'] != pair['on']['kernel_abi']['outputs']:
            raise ValueError('published outputs changed')
        if not pair['on']['preparations'].get('rewrites'):
            raise ValueError('boundary preparation reuse did not reach ordinary emission')
        if not (len(pair['on']['kernel_abi']['args']) < len(pair['off']['kernel_abi']['args'])
                and pair['on']['materialization_store_sites'] < pair['off']['materialization_store_sites']
                and pair['on']['materialization_load_sites'] < pair['off']['materialization_load_sites']):
            raise ValueError('reuse did not eliminate actual pointer-boundary materialization')
        changes.append(dict(source_sha256=record['source_sha256'],
            baseline_program_sha256=pair['off']['program_sha256'],
            candidate_program_sha256=pair['on']['program_sha256'],
            baseline_llvm_bytes=pair['off']['llvm_bytes'], candidate_llvm_bytes=pair['on']['llvm_bytes']))
    report = dict(schema='merlin.gemmini.preparation_reuse_experiment.v1',
        parent_compiler_sha256=parent, sources=records, variants=modes,
        products=products, changes=changes, timing='UNKNOWN; default remains off',
        custody='ordinary manifest products verified; no broker runtime custody claim')
    (output/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(output/'report.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', default='submission')
    parser.add_argument('--output', required=True)
    parser.add_argument('sources', nargs='+')
    options = parser.parse_args()
    prepare(options.submission, options.output, options.sources)
