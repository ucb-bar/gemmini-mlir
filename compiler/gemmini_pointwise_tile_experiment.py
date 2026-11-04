#!/usr/bin/env python3
"""Isolated real pointwise transport/row-tile alternatives, from the current ordinary compiler.

The shared resident-input owner supplies the wider DMA and exact channel-tail
map. The ordinary terminal emits that wider command with its selected block
stride. No schedule or checking projection substitutes for the emitted product.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def _options():
    return json.loads((Path(__file__).resolve().parent / 'POINTWISE_TILES.json').read_text())


def _pointwise(window):
    return (window.kernel == (1, 1) and window.stride in ((1, 1), (2, 2))
            and window.dilation == (1, 1) and not any(window.padding))


def row_tile(window, edge, original):
    if _options()['balanced_rows'] and _pointwise(window):
        width = window.output_spatial[1]
        blocks = (width + edge - 1) // edge
        return (width + blocks - 1) // blocks
    return original


def retile_source(source):
    """Retile only an already admitted prepared grouped contraction."""
    plan = source.plan
    rows = row_tile(plan.window, plan.primitive, plan.row_tile)
    return replace(source, plan=replace(plan, row_tile=rows))


def transport(source):
    from gemmini_conv_policy import prepare_rolling_input
    prepared, ring = prepare_rolling_input(source)
    if ring is None or not _pointwise(source.plan.window):
        return prepared, ring
    root = Path(__file__).resolve().parent
    raw = (root / 'data/rtl_facts.json').read_bytes()
    facts = json.loads(raw)
    pin = json.loads((root / 'data/COMMAND_ORDERING_SELECTION.json').read_text())
    if pin['base_facts']['sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('pointwise wide input selected facts differ')
    extent = pin['ordering']['extents']['MVIN.spad_addr']
    blocks = extent.get('blocks', {})
    cols = blocks.get('cols', {})
    stride = blocks.get('stride', {})
    rows = extent.get('rows', {})
    edge = source.plan.primitive
    if (extent.get('status') != 'derived' or blocks.get('block') != edge
            or cols.get('operand') != 'rs2' or cols.get('offset') != 32
            or type(cols.get('width')) is not int
            or rows.get('operand') != 'rs2' or rows.get('offset') != 48
            or type(rows.get('width')) is not int
            or stride.get('code') != 1 or stride.get('operand') != 'rs1'
            or stride.get('offset') != 16 or stride.get('state', {}).get('value') != 0):
        raise ValueError('pointwise wide input needs selected stream0 block addressing')
    request = facts['facts']['timing_structure']['memory_clients']['clients']['dma_read']['max_request_bytes']
    channels = _options()['stage_channels']
    if (type(channels) is not int or channels % edge or channels > request
            or channels > (1 << cols['width']) - 1
            or ring.stage_rows > (1 << rows['width']) - 1
            or ring.row_pitch >= 1 << stride['width']):
        raise ValueError('pointwise wide input exceeds selected DMA or encoded extent')
    return prepared, replace(ring, stage_channels=channels)


def prepare(output, prefix, base, sources):
    from gemmini_conv_evaluate import (WORKSPACE, copy_package, tree_identity,
        source_record, manifest_emit, consume_replay)
    import shutil
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sources').mkdir()
    base = Path(base).resolve()
    before = tree_identity(base)
    frozen = output / 'compiler'
    copy_package(base, frozen)
    if tree_identity(base) != before or tree_identity(frozen) != before:
        raise ValueError('frozen array parent changed while capturing')
    original = (frozen / 'gemmini_conv.py').read_text()
    variants = {}
    for mode, options in (('incumbent', None),
                          ('stage64', dict(stage_channels=64, balanced_rows=False)),
                          ('stage64-balanced', dict(stage_channels=64, balanced_rows=True))):
        variant = WORKSPACE / 'variants' / (prefix + '-' + mode)
        copy_package(frozen, variant)
        if options:
            shutil.copyfile(Path(__file__), variant / Path(__file__).name)
            source = original
            marker = '    input_rows = None\n'
            stage = '        source, input_rows = prepare_rolling_input(source)\n'
            if source.count(marker) != 1 or source.count(stage) != 1:
                raise ValueError('ordinary prepared pointwise adapter sites changed')
            source = source.replace(marker,
                "    if _prepared_columns and selection.get('pointwise_experiment',{}).get('admitted'):\n"
                '        from gemmini_pointwise_tile_experiment import retile_source\n'
                '        source = retile_source(source)\n'
                '        retention_mode,pairs,selection=select_pointwise(source,retention_mode,pairs,selection,mode=pointwise_mode)\n' + marker)
            source = source.replace(stage,
                "        if _prepared_columns and selection.get('pointwise_experiment',{}).get('admitted'):\n"
                '            from gemmini_pointwise_tile_experiment import transport\n'
                '            source, input_rows = transport(source)\n'
                '        else:\n' + '    ' + stage)
            (variant / 'gemmini_conv.py').write_text(source)
            (variant / 'POINTWISE_TILES.json').write_text(json.dumps(options, indent=2)+'\n')
        variants[mode] = dict(package=str(variant), compiler_sha256=tree_identity(variant), options=options)
    report = dict(parent_compiler_sha256=before, parent=str(base), variants=variants,
        sources=[], emissions=[], changes=[], production_defaults_changed=False,
        pass_order='unchanged exact scale layout copies -> original Window2D -> pointwise capacity admission -> admitted-only row tile option -> rebound capacity admission -> shared input ring with selected64-column stage option -> unchanged epochs and complete RHS residency -> unchanged array/output-group owner -> ordinary actual wide DMA and terminal -> inverse copy',
        hypothesis='reduce CPU issue from source input DMA/configuration and balanced-row control; bytes, physical occupancy and timing are separate',
        timing='UNKNOWN; parent coordinates requests')
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
               for m in ('stage64', 'stage64-balanced')}))
        (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--base', default='submission')
    parser.add_argument('--train', action='append', required=True)
    parser.add_argument('--source-only-neighbor', action='append', default=[])
    args = parser.parse_args()
    report = prepare(args.output, args.prefix, args.base,
        [('training', p) for p in args.train] +
        [('source_only_neighbor', p) for p in args.source_only_neighbor])
    print(json.dumps(dict(changes=report['changes'], variants=report['variants']), indent=2))
