#!/usr/bin/env python3
"""Ordinary CLI experiment bridge for exact ordered host reduction expansion.

Only a frozen opt-in variant names this file as its manifest entrypoint. The
production manifest and settings continue to use the original gemmini-opt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'compiler_library')]


def compiler_main():
    from gemmini_host_pool_experiment import compile_ordered_source
    argv = sys.argv[1:]
    inputs = [x for x in argv if not x.startswith('-')]
    if len(inputs) != 1:
        raise ValueError('expected one input MLIR file')
    source = Path(inputs[0]).read_text()
    if '--verify-diagnostics' in argv:
        from gemmini_frontend import normalize_source
        normalize_source(source)
        return
    analysis = next((x.split('=',1)[1] for x in argv if x.startswith('--emit-analysis-bundle=')), None)
    evidence_sink = None
    if analysis:
        from gemmini_activity import enable_activity
        enable_activity()
        from gemmini_evidence import write_evidence
        evidence_sink = lambda receipts, cb, llvm: write_evidence(analysis, source, cb, llvm+'\n', receipts)
    cb, llvm = compile_ordered_source(source, evidence_sink=evidence_sink)
    output = next((x.split('=',1)[1] for x in argv if x.startswith('--emit-command-buffer=')), None)
    if output:
        Path(output).write_text(json.dumps(cb, indent=2)+'\n')
    if '--emit-target-artifact' in argv or output is None:
        print(llvm)


def prepare(output, source, checked, variant_name):
    import yaml
    from gemmini_conv_evaluate import copy_package, digest, manifest_emit, tree_identity
    workspace = PACKAGE.parent
    output, source, checked = [Path(p).resolve() for p in (output, source, checked)]
    if output.exists() or output.is_relative_to(PACKAGE):
        raise ValueError('choose a fresh output outside submission')
    if (not variant_name or len(variant_name) > 40 or not variant_name[0].isalnum()
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                   for c in variant_name)):
        raise ValueError('invalid variant name')
    variant = workspace / 'variants' / variant_name
    if variant.exists():
        raise ValueError('variant already exists')
    output.mkdir(parents=True)
    parent = tree_identity(PACKAGE)
    off = output / 'compiler-off'
    copy_package(PACKAGE, off)
    if tree_identity(off) != parent or tree_identity(PACKAGE) != parent:
        raise ValueError('submission changed during snapshot')
    variant.parent.mkdir(exist_ok=True)
    marker = variant.parent / ('.library-pass-pending-' + variant.name)
    marker.write_text('Ordered source reduction expansion checks pending.\n')
    copy_package(off, variant)
    manifest = yaml.safe_load((variant / 'manifest.yaml').read_text())
    manifest['entrypoints']['tool'] = 'gemmini_host_pool_bridge.py'
    (variant / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    (variant / 'gemmini_host_pool_bridge.py').chmod(0o755)
    saved = output / 'source.mlir'
    saved.write_bytes(source.read_bytes())
    checked_report = json.loads((checked / 'report.json').read_text())
    if digest(saved.read_bytes()) != checked_report['source_sha256']:
        raise ValueError('source differs from independently checked pool source')
    products = {}
    tools = workspace / 'install/bin'
    for mode, package in (('off', off), ('on', variant)):
        location = output / mode
        product, receipt = manifest_emit(package, saved, location)
        if product != (checked / (mode + '-connected.mlir')).read_bytes():
            raise ValueError('ordinary CLI product differs from checked connected product: ' + mode)
        declaration = yaml.safe_load((package / 'manifest.yaml').read_text())
        bindings = dict(tool=str(package / declaration['entrypoints']['tool']),
                        input_mlir=str(saved), output_json=str(location / 'cb.json'))
        argv = [v.format(**bindings) for v in declaration['commands']['emit_command_buffer']['argv']]
        subprocess.run(argv, cwd=package,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(package/'compiler_library')),
            check=True, capture_output=True)
        cb = json.loads((location / 'cb.json').read_text())
        subprocess.run([str(tools/'mlir-opt'), str(location/'stdout.mlir'), '-o',
                        str(location/'verified.mlir')], check=True)
        subprocess.run([str(tools/'mlir-translate'), '--mlir-to-llvmir',
                        str(location/'verified.mlir'), '-o', str(location/'kernel.ll')], check=True)
        products[mode] = dict(**receipt, package=str(package),
            compiler_sha256=tree_identity(package), checked_product_identical=True,
            llvm_verified=True, llvm_bytes=len(product), kernel_abi=cb['kernel_abi'],
            tasks=[(t['kind'], t['source_op_indices']) for t in cb['params']['global_program_plan']['tasks']],
            source_recipe=cb['params'].get('host_pool_ordered_expansion'),
            command_buffer_sha256=digest((location/'cb.json').read_bytes()))
    if products['on']['tasks'] != products['off']['tasks'] or products['on']['kernel_abi'] != products['off']['kernel_abi']:
        raise ValueError('source placement, tasks or published outputs changed')
    if products['on']['source_recipe']['realized_recipes'] != checked_report['products']['on']['realized_recipes']:
        raise ValueError('ordinary bridge recipe differs from checked recipe')
    if tree_identity(PACKAGE) != parent:
        raise ValueError('submission changed while preparing experiment')
    report = dict(schema='merlin.gemmini.host_pool_bridge_experiment.v1',
        source_sha256=digest(saved.read_bytes()), original_source=str(source), frozen_source=str(saved),
        parent_compiler_sha256=parent, products=products,
        checked_report=str(checked/'report.json'), variant='variants/'+variant_name,
        timing='UNKNOWN; no promotion', default='original manifest without pool bridge')
    (output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    marker.unlink()
    return report


if __name__ == '__main__':
    if '--prepare-experiment' not in sys.argv:
        compiler_main()
    else:
        import argparse
        from gemmini_preparation_experiment import submit
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--prepare-experiment', action='store_true')
        parser.add_argument('--output', required=True)
        parser.add_argument('--source', required=True)
        parser.add_argument('--checked', required=True)
        parser.add_argument('--variant-name', required=True)
        parser.add_argument('--submit', action='store_true')
        args = parser.parse_args()
        report = prepare(args.output, args.source, args.checked, args.variant_name)
        if args.submit:
            print(json.dumps(submit(report, args.output, 'JC_resnet50_prepared_connected',
                                    'scalability'), indent=2))
        print(Path(args.output)/'report.json')
