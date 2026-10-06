"""Cross read counts and address extents; preserve the released gather kernel.

This is a calibration fixture, never a production code selection. Four source
corners train two-term hypotheses; middle axes and a second seed are withheld.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PARENT = Path(__file__).resolve().with_name('gather_service_battery_probe.py')
spec = importlib.util.spec_from_file_location('released_gather_fixture', PARENT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def draws(count, seed):
    words = []
    value = seed
    for _ in range(count):
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        value &= 0xFFFFFFFF
        words.append(value)
    return words


def generate(work):
    base.generate(work)
    old = json.loads((work / 'manifest.json').read_text())
    source = (work / 'battery.c').read_text()
    original = source
    cases = []
    specifications = [(count, extent, 0x31415927)
                      for count in (1024, 4096, 16384)
                      for extent in (64 * 1024, 512 * 1024, 8 * 1024 * 1024)]
    specifications += [(4096, extent, 0x27182819)
                       for extent in (64 * 1024, 512 * 1024, 8 * 1024 * 1024)]
    checksum = 14695981039346656037
    for count, extent, seed in specifications:
        indices = [value & (extent // 8 - 1) for value in draws(count, seed)]
        expected = base.SEED
        for index in indices:
            expected = (((expected << 7) | (expected >> 57)) & base.MASK64) ^ base.pattern(index)
        partition = 'training' if count in (1024, 16384) and extent in (65536, 8388608) else 'heldout'
        case = dict(id=len(cases), family='gather', size=count, kernel='gather',
                    partition=partition, working_set_bytes=extent, stream_seed=seed,
                    gather_value_load_bytes=count * 8, index_stream_load_bytes=count * 4,
                    requested_cpu_load_bytes=count * 12, requested_cpu_store_bytes=8,
                    source_and_destination_extent_bytes=extent,
                    unique_requested_words=len(set(indices)),
                    unique_requested_64B_regions=len({index // 8 for index in indices}),
                    requested_region_granule_bytes=64,
                    requested_region_granule_is_hardware_cache_line='UNKNOWN',
                    indices=indices, expected_word=expected, expected_fflags=0)
        cases.append(case)
        for _ in range(2):
            for byte in range(8):
                checksum = ((checksum ^ ((expected >> (byte * 8)) & 255)) * 1099511628211) & base.MASK64
    original_table = source[source.index('static const struct {uint32_t words;'):source.index('\nstatic uint64_t checksum')]
    new_table = 'static const struct {uint32_t words,reads,seed;uint64_t expected;} cases[12]={' + ','.join(
        '{' + ','.join((str(c['working_set_bytes'] // 8), str(c['size']), str(c['stream_seed']),
                        'UINT64_C(' + str(c['expected_word']) + ')')) + '}' for c in cases) + '};'
    source = source.replace(original_table, new_table)
    replacements = {
        '#define READS 4096': '#define READS 16384',
        'for(int id=0;id<3;id++)': 'for(int id=0;id<12;id++)',
        'uint32_t x=0x31415927;for(uint32_t i=0;i<READS;i++)':
            'for(uint32_t i=0;i<READS;i++)indices.data[i]=UINT32_C(0xdeadc0de);\n  uint32_t x=cases[id].seed;for(uint32_t i=0;i<cases[id].reads;i++)',
        'output.context.count=READS;': 'output.context.count=cases[id].reads;',
        'x=0x31415927;for(uint32_t i=0;i<READS;i++)':
            'x=cases[id].seed;for(uint32_t i=0;i<cases[id].reads;i++)',
        'digest(output.context.value);if(report(id,repeat,r))return 13;':
            'for(uint32_t i=cases[id].reads;i<READS;i++)if(indices.data[i]!=UINT32_C(0xdeadc0de))return 16;\n  if(output.context.count!=cases[id].reads)return 17;\n  digest(output.context.value);if(report(id,repeat,r))return 13;',
        'UINT64_C(' + str(int(old['expected_checksum'], 16)) + ')': 'UINT64_C(' + str(checksum) + ')',
        'SERVICE_PASS cases=3 repeats=2': 'SERVICE_PASS cases=12 repeats=2',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('released source fixture changed: ' + before)
        source = source.replace(before, after)
    # Retain the complete released gather/measure bodies, including fences and
    # their shared call boundary, byte for byte. Only fixture setup varies.
    start, end = original.index('static NOINLINE void gather('), original.index('static int report(')
    unchanged = original[start:end]
    if unchanged not in source:
        raise ValueError('kernel or counter-window implementation changed')
    manifest = dict(old, cases=cases, expected_checksum=f'{checksum:016x}',
                    index_stream='Xorshift32, explicit per-case counts/seeds; second seed cases held out',
                    partition_rule='Four primary-seed count/extent corners training; all middle axes and all second-seed cases held out together with both repeats',
                    training_ids=[c['id'] for c in cases if c['partition'] == 'training'],
                    heldout_ids=[c['id'] for c in cases if c['partition'] == 'heldout'],
                    released_kernel_and_counter_source_sha256=hashlib.sha256(unchanged.encode()).hexdigest(),
                    source_generator_dependency_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                    max_index_count=16384,
                    setup='Full8MiBtable initialized once before all windows; full16384-index buffer reset outside every ROI, active indices regenerated and immutable/padding/guards verified after each ROI. Entire table immutable after all windows. No cold/warm-cache assertion.',
                    inference_scope='Requested payload/regions, not physical misses/DDR. Operational in-order indexed gather including index loads, accumulator and finish.')
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (work / 'battery.c').write_text(source)
    print('CROSSED_GATHER_GENERATED', len(cases), f'{checksum:016x}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    generate(args.workdir.resolve())
