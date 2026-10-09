"""Align measured sections by source/geometry; never selects compiler behavior."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from mlir_oot.golden_device_profile import parse_profile

root = Path.cwd()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
paths = {key: root / value for key, value in {
    'terminal': 'docs/perf_records/stock2021_current2013_profile_terminal.json',
    'uart': 'docs/perf_records/stock2021_current2013_profile_uart.txt',
    'current_build': 'out/artifacts/probes/root-resnet-current2013-profile-20261006/profile_build.json',
    'reference_timing': 'docs/perf_records/q1013_diagnostic_reference_firesim.json',
    'previous_geometry_binding': 'docs/perf_records/q1013_exact1988_stock1999_profile_alignment.json',
}.items()}
paths['current_catalog'] = Path('/scratch/agustin/tmp/gemmini-segmented-current-20261006/out/segmented_current/candidate/whole_normal/build_direct/device_catalog/device_catalog.json')
paths['driver'] = Path(__file__).resolve()
stock = json.loads(paths['terminal'].read_text())
build = json.loads(paths['current_build'].read_text())
reference = json.loads(paths['reference_timing'].read_text())
old = json.loads(paths['previous_geometry_binding'].read_text())
catalog = json.loads(paths['current_catalog'].read_text())
assert stock['state'] == stock['phase'] == 'DONE' and stock['exit_code'] == 0
assert sha(paths['uart']) == stock['uart_sha256']
assert stock['elf_sha256'] == build['elf_sha256']
current = parse_profile(paths['uart'].read_text(), build)
assert current == stock['boundary_profile']
assert stock['bitstream_sha256'] == reference['bitstream_sha256']
assert stock['bitstream_archive_sha256'] == reference['bitstream_archive_sha256']
metadata = {row['id']: row for row in build['boundaries']}
by_region, residual = {}, []
for event in current['events']:
    row = metadata[event[1]]
    if row['category'] == 'residual':
        residual.append(event)
        continue
    region = row['source_region']
    if region is None:
        bindings = [item for item in catalog['bindings'] if item['symbol'] == row['symbol']]
        assert len(bindings) == 1
        region = bindings[0]['region']
    assert region not in by_region
    by_region[region] = (row, event)
assert len(by_region) == 54 and len(residual) == 16
paired, categories = [], Counter()
for match in old['paired_layers']:
    region = match['current_region']
    row, event = by_region[region]
    shape = row['shape']
    if row['category'] == 'pooled_stem':
        assert shape == match['current1988_shape']
        geometry = match['geometry']
    elif row['category'] == 'dense':
        dimensions = next(item['dimensions'] for item in catalog['kernels'] if item['symbol'] == row['symbol'])
        geometry = {key: dimensions[key] for key in ('m', 'k', 'n')}
    elif 'h' in shape:
        stride = shape['stride']
        geometry = {
            'm': ((shape['h'] + stride - 1) // stride) * ((shape['w'] + stride - 1) // stride),
            'k': shape['cin'] * 9,
            'n': shape['cout'],
        }
    else:
        geometry = {key: shape[key] for key in ('m', 'k', 'n')}
    assert geometry == match['geometry'], (region, geometry, match['geometry'])
    category = match['category']
    categories[category] += event[3]
    paired.append({
        'reference_index': match['reference_index'], 'reference_layer': match['reference_layer'],
        'current_region': region, 'category': category, 'geometry': geometry,
        'current_symbol': row['symbol'], 'current2021_callback_cycles': event[3],
        'reference1876_cycles': match['reference1876_cycles'],
        'difference_cycles': event[3] - match['reference1876_cycles'],
        'preceding_host_gap': event[2], 'call_index': event[0],
    })
categories['residual'] = sum(event[3] for event in residual)
categories['host_intervals'] = current['host_gap_counter']
refcategories = {row['category']: row['reference1876_cycles'] for row in old['class_comparison']}
assert sum(categories.values()) == current['forward_counter']
assert sum(refcategories.values()) == reference['kernel_cycles']
comparison = [dict(category=key, reference1876_cycles=refcategories[key],
                   current2021_cycles=categories[key], difference_cycles=categories[key] - refcategories[key])
              for key in refcategories]
record = {
    'schema': 'q1013_current2013_stock2021_profile_alignment_v1',
    'control_job': 2013, 'control_cycles': 32553639, 'profile_job': 2021,
    'profile_outer_cycles': stock['kernel_cycles'], 'forward_cycles': current['forward_counter'],
    'primitive_callback_cycles': current['device_counter'], 'host_gap_cycles': current['host_gap_counter'],
    'tail_cycles': current['tail_counter'], 'instrumented_forward_delta': current['forward_counter'] - 32553639,
    'reference_cycles': reference['kernel_cycles'], 'class_comparison': comparison, 'paired_layers': paired,
    'largest_current_host_gaps': [dict(call_index=event[0], source_region=metadata[event[1]]['source_region'] or 'classifier', cycles=event[2])
                                  for event in sorted(current['events'], key=lambda event: event[2], reverse=True)[:8]],
    'pins': {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()},
    'source_numeric_equivalence': False, 'causal_savings_claim': False,
    'reference51717_scope': 'Outside reference layer timers, not total reference CPU time. Callback costs include CPU command issue, DMA and fences.',
    'scope': 'Exact current2013 stock2021 profile and54source/geometry bound reference layers. Previous receipt supplies matching only; all current counts are from new UART. Different numeric/input/output timing boundaries remain.',
    'token_usage_available': False,
}
output = root / 'docs/perf_records/q1013_current2013_stock2021_profile_alignment.json'
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'class_comparison': comparison, 'largest_host_gaps': record['largest_current_host_gaps'][:4]}, indent=2))
