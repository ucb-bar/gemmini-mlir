"""Observe stationary operand requests with an isolated functional extension.

This experiment extends a hash-bound observational provider, preserving its old
fields and every target instruction. Counts describe requests, never latency.
The ISA/model seam belongs to this OOT provider; shared fitting stays in Merlin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError('unrecognized observational provider seam')
    return source.replace(old, new, 1)


def build(prior: Path, output: Path) -> None:
    prior = prior.resolve()
    previous = json.loads((prior / 'build.json').read_text())
    for row in [*previous['outputs'].values(), previous['hook'], previous['builder']]:
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('prior observational provider changed')
    source = output / 'source'
    source.mkdir(parents=True, exist_ok=False)
    shutil.copy2(prior / 'build.py', source / 'build.py')
    header = (prior / 'telemetry.h').read_text()
    header = replace_once(header,
        'uint64_t first_event = 0, last_event = 0;',
        'uint64_t first_event = 0, last_event = 0;\n'
        '    uint64_t preload_real_stationary_b = 0, preload_retained_stationary_b = 0;\n'
        '    uint64_t preload_unknown_mode = 0;')
    header = replace_once(header,
        'row.last_event = event; ++row.count;',
        'row.last_event = event; ++row.count;\n'
        '    if (funct == 6) {\n'
        '      if (!ex_valid) ++row.preload_unknown_mode;\n'
        '      else if (state.mode == State::WS) {\n'
        '        if (state.preload_sp_addr == std::numeric_limits<uint32_t>::max())\n'
        '          ++row.preload_retained_stationary_b;\n'
        '        else ++row.preload_real_stationary_b;\n'
        '      }\n'
        '    }')
    header = replace_once(header,
        '<< ",\\\"padded_compute_rows\\\":" << row.padded_compute_rows << \'}\';',
        '<< ",\\\"padded_compute_rows\\\":" << row.padded_compute_rows\n'
        '           << ",\\\"preload_real_stationary_b\\\":" << row.preload_real_stationary_b\n'
        '           << ",\\\"preload_retained_stationary_b\\\":" << row.preload_retained_stationary_b\n'
        '           << ",\\\"preload_unknown_mode\\\":" << row.preload_unknown_mode << \'}\';')
    header = replace_once(header, 'gemmini_primitive_operand_telemetry_v1',
                          'gemmini_primitive_operand_telemetry_v2')
    (source / 'telemetry.h').write_text(header)
    command = ['python', str(source / 'build.py'), '--source',
               str(Path(previous['inputs']['gemmini.cc']['path']).parent),
               '--riscv', str(Path(previous['spike_source']['path']).parent.parent),
               '--output', str(output / 'engine')]
    subprocess.run(command, check=True)
    record = {'schema': 'stationary_preload_observer_build_v1',
              'previous_build': pin(prior / 'build.json'),
              'new_build': pin(output / 'engine/build.json'),
              'script': pin(Path(__file__)), 'production_modified': False,
              'semantics': 'After functional preload: explicit WS and non-GARBAGE preload_sp_addr requests real stationary operand. Unknown execute mode is retained. OS is outside this feature.',
              'scope': 'Functional requests only; no hardware capability, duration, overlap or cycle floor.'}
    (output / 'observer_build.json').write_text(json.dumps(record, indent=2)+'\n')


def replay(prior_replay: Path, engine: Path, output: Path) -> None:
    previous = json.loads(prior_replay.read_text())
    for row in previous['pins'].values():
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('previous functional replay changed')
    output.mkdir(parents=True, exist_ok=False)
    command = list(previous['command'])
    command[0] = str(engine / 'spike')
    indexes = [i for i, item in enumerate(command) if item.startswith('--extlib=')]
    if len(indexes) != 1:
        raise ValueError('exactly one explicit extension required')
    command[indexes[0]] = '--extlib=' + str(engine / 'libgemmini_telemetry.so')
    env = dict(os.environ)
    env.update(previous['env'])
    env['MERLIN_GEMMINI_TELEMETRY'] = str(output / 'operands.json')
    start = time.monotonic()
    with (output / 'stdout').open('wb') as stdout, (output / 'histogram').open('wb') as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                timeout=180)
    elapsed = time.monotonic() - start
    if result.returncode:
        raise ValueError('functional replay did not complete')
    for name in ('stdout', 'histogram'):
        if (output / name).read_bytes() != Path(previous['pins'][name]['path']).read_bytes():
            raise ValueError('target execution or original numeric gate changed')
    old = json.loads(Path(previous['pins']['telemetry']['path']).read_text())
    new = json.loads((output / 'operands.json').read_text())
    keys = ('preload_real_stationary_b', 'preload_retained_stationary_b', 'preload_unknown_mode')
    stripped = dict(new)
    stripped['schema'] = old['schema']
    stripped['rows'] = [{key: value for key, value in row.items() if key not in keys}
                        for row in new['rows']]
    if stripped != old:
        raise ValueError('any original telemetry field changed')
    totals = {key: sum(row[key] for row in new['rows']) for key in keys}
    if totals['preload_unknown_mode']:
        raise ValueError('stationary preload feature contains unknown execute modes')
    record = {'schema': 'stationary_preload_observer_replay_v1',
              'command': command, 'exit_code': result.returncode, 'elapsed_seconds': elapsed,
              'old_stdout_histogram_and_all_telemetry_fields_identical': True,
              'features': totals, 'target_cycles': 'UNKNOWN', 'production_modified': False,
              'pins': {name: pin(path) for name, path in {
                  'previous': prior_replay, 'build': engine / 'build.json',
                  'elf': Path(previous['pins']['elf']['path']), 'stdout': output / 'stdout',
                  'histogram': output / 'histogram', 'telemetry': output / 'operands.json',
                  'script': Path(__file__)}.items()}}
    (output / 'replay_identity.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'features': totals, 'elapsed_seconds': elapsed}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--prior', required=True, type=Path)
    builder.add_argument('--output', required=True, type=Path)
    runner = sub.add_parser('replay')
    runner.add_argument('--prior-replay', required=True, type=Path)
    runner.add_argument('--engine', required=True, type=Path)
    runner.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.prior, args.output)
    else:
        replay(args.prior_replay, args.engine, args.output)
