#!/usr/bin/env python3
"""Freeze and check descriptor-identity variants through the ordinary emitter.

Development-only adapter. Production compilation never invokes this module.
Submission stays off; optional submission requests one matched advisory sweep.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'compiler_library')]
from gemmini_conv_evaluate import copy_package, digest, manifest_emit, tree_identity


def prepare(submission, output, source, checked, variant_name):
    submission, output, source, checked = map(lambda p: Path(p).resolve(),
                                              (submission, output, source, checked))
    if output.exists() or output.is_relative_to(submission):
        raise ValueError('choose a fresh output outside submission')
    if (not variant_name or not variant_name[0].isalnum() or len(variant_name) > 40
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                   for c in variant_name)):
        raise ValueError('invalid variant name')
    variant = WORKSPACE / 'variants' / variant_name
    if variant.exists():
        raise ValueError('variant already exists')
    declaration = json.loads((submission / 'TUNING_KNOBS.json').read_text())
    settings_file = declaration['settings_file']
    settings = json.loads((submission / settings_file).read_text())
    if settings.get('host_tensor_views', 'off') != 'off':
        raise ValueError('implicit current baseline must have host_tensor_views off')
    if 'on' not in declaration['knobs']['host_tensor_views']['values']:
        raise ValueError('host_tensor_views on is not declared')
    output.mkdir(parents=True)
    parent = tree_identity(submission)
    off = output / 'compiler-off'
    copy_package(submission, off)
    if tree_identity(off) != parent or tree_identity(submission) != parent:
        raise ValueError('submission changed during snapshot')
    variant.parent.mkdir(exist_ok=True)
    marker = variant.parent / ('.library-pass-pending-' + variant.name)
    marker.write_text('Descriptor identity experiment checks pending.\n')
    copy_package(off, variant)
    settings['host_tensor_views'] = 'on'
    (variant / settings_file).write_text(json.dumps(settings, indent=2) + '\n')
    saved = output / 'source.mlir'
    saved.write_bytes(source.read_bytes())
    products = {}
    replay = '''import sys,json
from pathlib import Path
p=Path(sys.argv[1]);sys.path[:0]=[str(p),str(p/'compiler_library')]
from gemmini_backend import compile_source
cb,ll=compile_source(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_text(json.dumps(cb,sort_keys=True,indent=2)+'\\n')
Path(sys.argv[4]).write_text(ll+'\\n')
'''
    tools = Path('/work/merlin/toolchains/llvm-a47bddcc/install/bin')
    for mode, package in (('off', off), ('on', variant)):
        location = output / mode
        ordinary, receipt = manifest_emit(package, saved, location)
        expected = (checked / ('connected-' + mode + '.mlir')).read_bytes() + b'\n'
        if ordinary != expected:
            raise ValueError('ordinary product differs from checked product plus CLI newline: ' + mode)
        subprocess.run([sys.executable, '-c', replay, str(package), str(saved),
                        str(location / 'cb.json'), str(location / 'replay.mlir')],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), check=True)
        if ordinary != (location / 'replay.mlir').read_bytes():
            raise ValueError('ordinary emitter and compiler API differ')
        subprocess.run([str(tools / 'mlir-opt'), str(location / 'stdout.mlir'),
                        '-o', str(location / 'verified.mlir')], check=True)
        subprocess.run([str(tools / 'mlir-translate'), '--mlir-to-llvmir',
                        str(location / 'verified.mlir'), '-o', str(location / 'kernel.ll')], check=True)
        cb = json.loads((location / 'cb.json').read_text())
        products[mode] = dict(**receipt, package=str(package),
            compiler_sha256=tree_identity(package), llvm_bytes=len(ordinary),
            checked_product_plus_cli_newline=True, direct_replay_identical=True,
            llvm_verified=True, kernel_abi=cb['kernel_abi'],
            realized_recipes=cb['params'].get('host_tensor_views', []),
            command_buffer_sha256=digest((location / 'cb.json').read_bytes()))
    if products['off']['kernel_abi'] != products['on']['kernel_abi']:
        raise ValueError('kernel ABI changed')
    recipes = products['on']['realized_recipes']
    if not recipes or not any(r['traversal_unit_axes_removed'] for r in recipes):
        raise ValueError('identity traversal removal did not reach production emission')
    if products['off']['program_sha256'] == products['on']['program_sha256']:
        raise ValueError('candidate did not change emitted product')
    if tree_identity(submission) != parent:
        raise ValueError('submission changed during experiment preparation')
    report = dict(schema='merlin.gemmini.preparation_identity_experiment.v1',
        parent_compiler_sha256=parent, source_path=str(source),
        frozen_source=str(saved), source_sha256=digest(saved.read_bytes()),
        products=products, variant='variants/' + variant_name,
        timing='UNKNOWN; no promotion', default='off',
        scope='immutable descriptor identity; original output publications retained')
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    marker.unlink()
    return report


def submit(report, output, capsule, suite):
    """One sweep: implicit current off baseline and the frozen on variant."""
    if tree_identity(WORKSPACE / 'submission') != report['parent_compiler_sha256']:
        raise ValueError('current baseline differs from frozen off compiler')
    if tree_identity(Path(report['products']['on']['package'])) != report['products']['on']['compiler_sha256']:
        raise ValueError('on variant changed after checks')
    argv = [sys.executable, str(WORKSPACE / 'agent_selfcheck.py'), '--submission',
            'submission', '--sweep', report['variant'], '--suite', suite,
            '--capsules', capsule, '--wait-budget', '5']
    process = subprocess.run(argv, cwd=WORKSPACE, capture_output=True, text=True)
    record = dict(argv=argv, returncode=process.returncode,
                  stdout=process.stdout, stderr=process.stderr)
    (Path(output) / 'sweep.json').write_text(json.dumps(record, indent=2) + '\n')
    if process.returncode not in (0, 3):
        raise RuntimeError('sweep failed; inspect sweep.json before any retry')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', default='submission')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checked', required=True)
    parser.add_argument('--variant-name', required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--capsule', default='JC_resnet50_prepared_connected')
    parser.add_argument('--suite', default='scalability')
    args = parser.parse_args()
    report = prepare(args.submission, args.output, args.source, args.checked, args.variant_name)
    if args.submit:
        print(json.dumps(submit(report, args.output, args.capsule, args.suite), indent=2))
    print(Path(args.output) / 'report.json')
