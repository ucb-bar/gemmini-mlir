#!/usr/bin/env python3
"""Freeze actual full-resident spatial-group alternatives for ordinary selfcheck.

The isolated adapter supplies a bounded argument before private input union and
shared joint-storage construction. Checked owners feed the ordinary terminal
consumer. No production routing, settings, planners or defaults are changed.
"""
import argparse
import json
from pathlib import Path
from gemmini_conv_evaluate import (WORKSPACE, copy_package, tree_identity,
    digest, source_record, manifest_emit, consume_replay)


def prepare(output, prefix, sources):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('use a fresh output directory')
    output.mkdir(parents=True)
    (output / 'sources').mkdir()
    production = WORKSPACE / 'submission'
    before = tree_identity(production)
    frozen = output / 'compiler'
    copy_package(production, frozen)
    if tree_identity(production) != before or tree_identity(frozen) != before:
        raise ValueError('production changed while freezing')
    baseline = WORKSPACE / 'variants' / (prefix + '-incumbent')
    copy_package(frozen, baseline)
    variants = {'incumbent': dict(package=str(baseline), compiler_sha256=before)}
    original_array = (frozen / 'gemmini_conv_array.py').read_text()
    original_conv = (frozen / 'gemmini_conv.py').read_text()
    for limit in (2, 4):
        mode = 'rows' + str(limit)
        variant = WORKSPACE / 'variants' / (prefix + '-' + mode)
        copy_package(frozen, variant)
        # Explicit source argument on the copied owner, before layout or laws
        # are constructed. Every remaining proof and consumer stays ordinary.
        array = original_array.replace(
            'def plan_array_retention(schedule,rhs,input_rows, *, multirow=False):',
            'def plan_array_retention(schedule,rhs,input_rows, *, multirow=False, full_resident_output_rows=None):')
        marker = '    retained_input=input_rows\n'
        if array == original_array or array.count(marker) != 1:
            raise ValueError('current array owner lacks the reviewed adapter site')
        array = array.replace(marker,
            '    if full_resident_output_rows is not None and type(rhs) is not ConvolutionRhsPhases:\n'
            '        if type(full_resident_output_rows) is not int or full_resident_output_rows not in (2,4):\n'
            "            raise ValueError('invalid isolated full-resident grouping bound')\n"
            '        rows_bound=min(rows_bound,full_resident_output_rows)\n' + marker)
        (variant / 'gemmini_conv_array.py').write_text(array)
        target = '        from gemmini_conv_array import plan_array_retention\n'
        if original_conv.count(target) != 1:
            raise ValueError('current convolution owner lacks the reviewed static adapter site')
        (variant / 'gemmini_conv.py').write_text(original_conv.replace(target,
            '        from gemmini_conv_group_adapter import plan_array_retention\n'))
        (variant / 'CONV_GROUP_EXPERIMENT.json').write_text(json.dumps(
            dict(full_resident_output_rows=limit), indent=2) + '\n')
        variants[mode] = dict(package=str(variant), compiler_sha256=tree_identity(variant),
            full_resident_output_rows=limit)
    report = dict(parent_compiler_sha256=before, variants=variants, sources=[],
        emissions=[], changes=[], requests={},
        hypothesis='smaller spatial groups trade array retention for lower ACC/readout pressure',
        pass_order='ordinary regional admission -> selected array ordering proof -> isolated full-resident row bound -> exact input union -> shared joint storage and output group -> unchanged terminal consumer',
        legality='primitive/operand/ACC capacities and selected fresh/stay ordering remain proved by ordinary shared owners',
        target_timing='UNKNOWN', production_defaults_changed=False)
    for split, path in sources:
        _, source = source_record(path, split, output / 'sources')
        report['sources'].append(source)
        for mode, entry in variants.items():
            location = output / 'products' / source['source_sha256'] / mode
            data, receipt = manifest_emit(Path(entry['package']), Path(source['frozen_path']), location)
            receipt.update(consume_replay(Path(entry['package']), Path(source['frozen_path']), location, data))
            report['emissions'].append(dict(mode=mode, source_sha256=source['source_sha256'],
                split=split, **receipt))
        rows = {e['mode']: e for e in report['emissions'] if e['source_sha256'] == source['source_sha256']}
        report['changes'].append(dict(source_sha256=source['source_sha256'], split=split,
            **{mode: rows[mode]['program_sha256'] != rows['incumbent']['program_sha256']
               for mode in ('rows2', 'rows4')}))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
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


if __name__ == '__main__':
    main()
