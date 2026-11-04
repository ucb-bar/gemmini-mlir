"""Check realized source choices against the artifacts consumed by compilation.

These observations grant no permissions and cannot reconstruct live contracts.
Only the adapter which already selected the ordinary source passes calls capture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _check_file(record):
    actual = _identity(record['path'])
    if actual != record:
        raise ValueError('stale host recipe artifact: ' + str(record['path']))


def capture_source_preparation(workdir, source_report, lowered_llvm):
    """Bind the selected worker outcome to actual prepared input and LLVM."""
    work = Path(workdir)
    if source_report.get('schema') != 'merlin.source_optimization_selection.v1':
        raise ValueError('selected source preparation has no realized shared selection')
    saved_report = work / 'source-optimization-selection.json'
    if json.loads(saved_report.read_text()) != source_report:
        raise ValueError('returned source selection differs from its actual lowering artifact')
    selection = source_report['selection']
    request = selection['request']
    if (request['producer_packing'] is not True or request['padding_gather'] is not True
            or request['maximum_added_operations'] != 4096
            or selection['original_maximum_added_operations'] != 4096
            or selection['original_permitted_passes'] != [
                'automatic-host-permutations', 'automatic-host-gathers']
            or _digest(selection) != source_report['selection_sha256']):
        raise ValueError('requested source preparation choices were ignored or changed')
    prepared_input = _identity(work / 'model.tensor-preparations.mlir')
    if request['source_sha256'] != prepared_input['sha256']:
        raise ValueError('source selection does not bind the actual lower_model input')
    files = [prepared_input, _identity(saved_report)]
    passes = {}
    for name, consumer in source_report['consumers'].items():
        if name not in ('automatic-host-permutations', 'automatic-host-gathers'):
            raise ValueError('source preparation contains an unselected consumer')
        path = work / (name + '.json')
        realized = json.loads(path.read_text())
        if any(realized[key] != consumer[key] for key in (
                'source_file_sha256', 'output_file_sha256', 'rewrites')):
            raise ValueError('source pass outcome differs from its actual lowering artifact')
        files.append(_identity(path))
        passes[name] = dict(status='rewritten' if consumer['rewrites'] else 'unchanged',
            **consumer, diagnostic=realized.get('diagnostic'))
    if set(passes) != {'automatic-host-permutations', 'automatic-host-gathers'}:
        raise ValueError('selected source preparation omitted a pass outcome')
    recipe_path = work / 'lowering_recipe.json'
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('schema') != 'merlin.lowering_recipe.v1' or recipe.get('status') != 'returned':
        raise ValueError('source preparation has no completed actual lowering recipe')
    prepared = recipe['sources']['prepared_mlir']
    _check_file(prepared)
    native = _identity(lowered_llvm)
    if any(native[key] != recipe['returned_llvm_ir'][key] for key in ('sha256', 'bytes')):
        raise ValueError('selected recipe does not describe the returned host LLVM')
    files.extend([_identity(recipe_path), prepared, native])
    return dict(schema='submission.host_recipe_realization.v1',
        selection_sha256=source_report['selection_sha256'], files=files, passes=passes,
        scope='artifact correspondence only; no numerical or performance qualification')


def validate_source_preparation(result):
    """Refuse stale selected artifacts before producing a mixed program plan."""
    realization = result.stats.get('adapter_host_recipe_realization')
    if not realization or realization.get('schema') != 'submission.host_recipe_realization.v1':
        raise ValueError('selected source preparation has no bound realization')
    for record in realization['files']:
        _check_file(record)
    host = result.stats.get('adapter_host_optimization')
    if host is None or host.get('selection') != 'source_preparation':
        raise ValueError('selected source preparation was not materialized in the combined program')
    if host.get('source_realization') != realization:
        raise ValueError('combined host recipe dropped or changed the source realization')
    source_report = result.stats.get('adapter_host_source_preparation')
    if (not source_report or host.get('source_preparation') != source_report
            or _digest(source_report['selection']) != realization['selection_sha256']):
        raise ValueError('combined host recipe changed the realized source selection')
    combined = host.get('emitted_llvm_dialect')
    if not combined or Path(combined['path']).resolve() != Path(
            result.stats['adapter_combined_llvm_dialect']).resolve():
        raise ValueError('selected source recipe does not bind the emitted combined module')
    _check_file(combined)
    return realization
