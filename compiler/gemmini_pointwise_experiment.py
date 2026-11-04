#!/usr/bin/env python3
"""Freeze shared pointwise lifetime alternatives inside ordinary grouped conv.

Static adapter sites exist only in variant copies. Outer exact-scale grouping,
all preparation/output copies and the ordinary terminal encoder remain intact.
"""
import argparse
import json
from pathlib import Path
from gemmini_conv_evaluate import (WORKSPACE, copy_package, tree_identity,
    source_record, manifest_emit, consume_replay)


def prepare(output, prefix, sources):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sources').mkdir()
    production = WORKSPACE / 'submission'
    parent = tree_identity(production)
    frozen = output / 'compiler'
    copy_package(production, frozen)
    if tree_identity(production) != parent or tree_identity(frozen) != parent:
        raise ValueError('production changed while freezing')
    original = (frozen / 'gemmini_conv.py').read_text()
    marker = '    input_rows = None\n'
    array_marker = '    auto_admission=None\n'
    if original.count(marker) != 1 or original.count(array_marker) != 1:
        raise ValueError('ordinary convolution adapter sites changed')
    variants = {}
    for mode in ('incumbent', 'resident', 'array'):
        variant = WORKSPACE / 'variants' / (prefix + '-' + mode)
        copy_package(frozen, variant)
        if mode != 'incumbent':
            transformed = original.replace(marker,
                '    from gemmini_pointwise_adapter import select_pointwise\n'
                '    retention_mode,pairs,selection=select_pointwise(source,retention_mode,pairs,selection)\n' + marker)
            transformed = transformed.replace(array_marker,
                '    from gemmini_pointwise_adapter import select_array_mode\n'
                '    array_mode=select_array_mode(source,selection,array_mode)\n' + array_marker)
            transformed = transformed.replace(
                '        from gemmini_conv_array import plan_array_retention\n',
                '        from gemmini_pointwise_adapter import plan_pointwise_array_retention as plan_array_retention\n')
            (variant / 'gemmini_conv.py').write_text(transformed)
            (variant / 'POINTWISE_EXPERIMENT.json').write_text(json.dumps(dict(mode=mode))+'\n')
        variants[mode] = dict(package=str(variant), compiler_sha256=tree_identity(variant))
    report = dict(parent_compiler_sha256=parent, variants=variants, sources=[],
        emissions=[], changes=[], requests={}, production_defaults_changed=False,
        pass_order='exact scale grouping and RHS/bias copies -> original Window2D -> isolated capacity proposal -> shared rolling input -> selected compute epochs -> shared complete RHS residency -> optional shared output group and array retention -> ordinary terminal effects -> inverse output copy',
        target_numerics='UNKNOWN', target_timing='UNKNOWN',
        contiguous_pointwise_view='refused for stride two; original convolution M projection retained',
        pointwise_source_contract='elementwise ScalarGraph source; does not authorize contraction flattening')
    for split, path in sources:
        _, source = source_record(path, split, output / 'sources')
        report['sources'].append(source)
        rows = {}
        for mode, entry in variants.items():
            location = output / 'products' / source['source_sha256'] / mode
            data, receipt = manifest_emit(Path(entry['package']), Path(source['frozen_path']), location)
            receipt.update(consume_replay(Path(entry['package']), Path(source['frozen_path']), location, data))
            rows[mode] = dict(mode=mode, split=split, source_sha256=source['source_sha256'], **receipt)
            report['emissions'].append(rows[mode])
        report['changes'].append(dict(source_sha256=source['source_sha256'], split=split,
            **{m: rows[m]['program_sha256'] != rows['incumbent']['program_sha256']
               for m in ('resident', 'array')}))
        (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--train', action='append', required=True)
    parser.add_argument('--source-only-neighbor', action='append', default=[])
    args = parser.parse_args()
    report = prepare(args.output, args.prefix,
        [('training', p) for p in args.train] +
        [('source_only_neighbor', p) for p in args.source_only_neighbor])
    print(json.dumps(dict(changes=report['changes'], variants=report['variants']), indent=2))
