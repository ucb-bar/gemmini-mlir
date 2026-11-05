#!/usr/bin/env python3
"""Preserve measured compiler submissions and their experiment records in Git.

Reads historical evidence only. Never imports a compiler, starts a service, or
launches an experiment. The only publication operation is an explicit --push.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

TASK = Path('/work/merlin/tasks/mlsys-paper-20261024')
REPO = TASK / 'gemmini-mlir'
MERLIN = TASK / 'merlin'
OUT = MERLIN / 'out/artifacts/artifact-evaluation/20261104/compiler-branches'
MEASUREMENTS = Path('/work/merlin/tasks/disk-reclamation-20261021/merlin/out/artifacts/perf-studies/gemmini/v4/latest').resolve()
STUDIES = Path('/work/merlin/tasks/el4-grader-throughput-20261024/.tmp/cca-abstractions-next-20261028')
PAPER_COMMIT = '24181ce016544dac2742825dbc401230ebb85100'
EXCLUDED = {'build', '__pycache__', '.git'}
REMOTE = 'https://github.com/ucb-bar/gemmini-mlir'


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}


def source_files(root):
    root = Path(root)
    return [p for p in sorted(root.rglob('*'))
            if p.is_file() and not EXCLUDED.intersection(p.relative_to(root).parts)]


def tree_digest(root, files=None):
    root = Path(root)
    digest = hashlib.sha256()
    for path in files if files is not None else source_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def git(*argv, data=None):
    return subprocess.check_output(['git', '-C', str(REPO), *argv], input=data)


def csv_bytes(rows):
    if not rows:
        return b''
    fields = list(dict.fromkeys(k for row in rows for k in row))
    output = io.StringIO()
    writer = csv.DictWriter(output, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def branch_for(label):
    direct = {
        'F10-FULL-NOREF-NOCCA': 'paper/construction/full-noref/final-4h',
        'F10-INFRA-REF-NOCCA': 'paper/tooling/infra-ref/final-4h',
        'F10-SCHED-REF-NOCCA': 'paper/tooling/scheduling-ref/final-4h',
        'CCA-EXO-R3': 'paper/refinement/cca-exo-r3/final-8h-total',
        'FULL-REF': 'paper/construction/full-ref/final-12h',
        'FULL-REF-4H': 'paper/construction/full-ref/checkpoint-4h',
        'fresh10cca-live': 'paper/historical/fresh10cca/checkpoint-4h',
        'fresh10cca-final12h': 'paper/historical/fresh10cca/final-12h',
        'fresh10seed-live': 'paper/historical/fresh10seed/checkpoint-4h',
        'fresh10seed-final12h': 'paper/historical/fresh10seed/final-12h',
    }
    if label in direct:
        return direct[label]
    if label.startswith('EL'):
        return 'paper/scaffolding/' + label.lower() + '/final-4h'
    return 'paper/cca/' + label.lower().removeprefix('cca-') + '/final-8h-total'


def measurement_rows():
    """Retain record identity. Do not overwrite repeats or select a best sample."""
    by_system = {}
    for path in sorted((MEASUREMENTS / 'measurements').glob('*/results.csv')):
        receipt = record(path)
        for row in csv.DictReader(path.open()):
            row = dict(row, evidence_path=str(path), evidence_sha256=receipt['sha256'],
                       measurement_image=path.parent.name)
            by_system.setdefault(row['system'], []).append(row)
    # The master table records the selected results for the original nine systems.
    path = MEASUREMENTS / 'master_long.csv'
    receipt = record(path)
    selected = {}
    for row in csv.DictReader(path.open()):
        if row['record_origin'] == 'native_measured':
            row = dict(row, evidence_path=str(path), evidence_sha256=receipt['sha256'])
            selected.setdefault(row['system'], []).append(row)
    return by_system, selected


def metrics(rows):
    """Summaries remain per image/profile to avoid pooling incompatible runs."""
    groups = {}
    for row in rows:
        key = (row.get('measurement_image', 'selected_master'),
               row.get('cache_profile', row.get('measurement_profile', 'unknown')))
        groups.setdefault(key, []).append(row)
    output = []
    for (image, profile), group in sorted(groups.items()):
        rankable = [r for r in group if str(r.get('rankable')).lower() == 'true']
        values = [float(r.get('median_cycles') or r.get('validated_cycles'))
                  for r in rankable if r.get('median_cycles') or r.get('validated_cycles')]
        output.append({
            'image': image, 'cache_profile': profile, 'records': len(group),
            'distinct_cases': len({r['case_id'] for r in group}),
            'rankable_records': len(rankable),
            'reported_mismatch_records': sum(float(r.get('mismatches') or r.get('mismatch_elements') or 0) > 0 for r in group),
            'reported_guard_failure_records': sum(float(r.get('guard_failures') or 0) > 0 for r in group),
            'geometric_mean_cycles_on_rankable_records': math.exp(sum(map(math.log, values)) / len(values)) if values else None,
            'scope': 'Descriptive cycles within this image/profile, not a speedup or a pooled author-trial estimate.',
        })
    return {'schema': 'merlin.archived-measurement-summary.v1', 'groups': output,
            'new_execution_performed': False,
            'utilization_percent_formula': '100 * nominal_MACs / (256 * completed_call_cycles)',
            'utilization_scope': 'Nominal-work throughput relative to peak, not observed processing-element occupancy.'}


def fsm_pairs(rows, by_system):
    """Recompute local controls only when image, profile and numerical inputs join."""
    baselines = {(r['measurement_image'], r['cache_profile'], r['case_id']): r
                 for r in by_system.get('gemmini-fsm', [])}
    groups = {}
    for row in rows:
        key = (row['measurement_image'], row['cache_profile'])
        group = groups.setdefault(key, {'pairs': [], 'excluded': []})
        baseline = baselines.get((*key, row['case_id']))
        reason = None
        if baseline is None:
            reason = 'No same-image/profile FSM record'
        elif any(str(r.get('rankable')).lower() != 'true' for r in [row, baseline]):
            reason = 'One or both records are not rankable'
        elif any((r.get('mismatches') or '0') != '0' or (r.get('guard_failures') or '0') != '0' for r in [row, baseline]):
            reason = 'Numerical mismatch or guard failure'
        elif any(not row.get(k) or row.get(k) != baseline.get(k)
                 for k in ['input_sha256s', 'reference_sha256s', 'output_dtype', 'elf_sha256']):
            reason = 'Input, reference, dtype or linked-image identities do not match'
        if reason:
            group['excluded'].append({'case': row['case_id'], 'reason': reason})
        else:
            group['pairs'].append({'case': row['case_id'],
                                   'fsm_cycles': float(baseline['median_cycles']),
                                   'compiler_cycles': float(row['median_cycles']),
                                   'fsm_over_compiler': float(baseline['median_cycles']) / float(row['median_cycles']),
                                   'hardware_qualification': row.get('hardware_qualification')})
    out = []
    for (image, profile), group in sorted(groups.items()):
        ratios = [p['fsm_over_compiler'] for p in group['pairs']]
        out.append({'image': image, 'profile': profile,
                    'paired_cases': len(ratios),
                    'geometric_mean_fsm_over_compiler': math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
                    **group})
    return {'scope': 'Recorded same-image controls. These do not replace the common FSM reference used by the published plots.',
            'comparisons': out}


def campaign_setup(source):
    run_path = source.get('campaign_run')
    output = {'intake': source, 'authoring_model': None, 'reasoning_effort': None,
              'recorded_policy': None, 'recorded_completion': None, 'records': []}
    if not run_path:
        output['qualification'] = 'Historical checkpoint. Campaign identity must be recovered from the original snapshot records.'
        return output
    run = Path(run_path)
    environment = run / 'environment.yaml'
    if environment.exists():
        text = environment.read_text()
        for key, target in [('model', 'authoring_model'), ('effort', 'reasoning_effort')]:
            match = re.search(r'^' + key + r':\s*(\S+)\s*$', text, re.MULTILINE)
            if match:
                output[target] = match.group(1).strip('\"\'')
        output['records'].append(record(environment))
    config_path = run / 'joint_campaign.json'
    if config_path.exists():
        d = read(config_path)
        policy = d['policy']
        output['recorded_policy'] = {k: policy.get(k) for k in [
            'duration_s', 'drain_s', 'final_model_s', 'compiler_execution_mode',
            'prohibited_instruction_roles', 'compiler_origin', 'operator_measurement_policy']}
        output['records'].append(record(config_path))
    summary = run / 'joint_campaign_summary.json'
    if summary.exists():
        d = read(summary)
        output['recorded_completion'] = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
        output['records'].append(record(summary))
    profile_path = run / 'AUTHOR_PROFILE.json'
    if profile_path.exists():
        profile = read(profile_path)
        output['author_profile'] = {k: profile.get(k) for k in [
            'name', 'level', 'seeded', 'tools', 'capabilities', 'live_cca',
            'cca_bank_service', 'cca_bank_sha256', 'raw_reference_grant',
            'optimization_guidance', 'prior_reference_tasks']}
        output['records'].append(record(profile_path))
    # Model and effort belong to the launch records, not the compiler's own README.
    candidates = [run / 'run_manifest.json', run / 'manifest.json', run / 'launch.json']
    for path in candidates:
        if path.exists():
            output['records'].append(record(path))
    output['interpretation'] = [
        'Native campaign completion flags describe the construction test suite, not the separate 89-case frozen-kernel panel.',
        'Configured duration, final-model reservation, drain and observed wall time are separate fields. Do not infer author time from a single duration field.',
        'reference_analysis can use a precomputed expert corpus even when live_cca is false.',
    ]
    return output


VERIFY_SCRIPT = '''#!/usr/bin/env python3
"""Check archived source bytes and modes without executing the compiler."""
import hashlib, json, os
from pathlib import Path
root = Path(__file__).resolve().parent.parent
inventory = json.loads((root / "experiment/source-files.json").read_text())
digest = hashlib.sha256()
for row in inventory["files"]:
    path = root / row["branch_path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sha256"], row["branch_path"]
    assert bool(path.stat().st_mode & 0o111) == row["executable"], row["branch_path"]
    digest.update(row["source_path"].encode() + b"\\0" + data + b"\\0")
assert digest.hexdigest() == inventory["tree_sha256"], "source-tree digest"
print("Verified", len(inventory["files"]), "compiler files:", digest.hexdigest())
'''


def checked_payload(source, by_system, selected):
    label = source['label']
    root = Path(source['copied_source'])
    files = source_files(root)
    actual = tree_digest(root, files)
    if actual != source['expected']:
        raise ValueError(f'{label}: source identity changed: {actual} != {source["expected"]}')
    inventory = []
    payload = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        branch_path = 'compiler/' + rel
        inventory.append({'source_path': rel, 'branch_path': branch_path,
                          'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size,
                          'executable': bool(path.stat().st_mode & 0o111)})
        payload[branch_path] = str(path)
    rows = by_system.get(label, [])
    picked = selected.get(label, [])
    mismatches = [r for r in rows + picked if r.get('compiler_source_sha256') and r['compiler_source_sha256'] != actual]
    if mismatches:
        raise ValueError(f'{label}: {len(mismatches)} measurement records refer to another compiler source')
    manifest = {
        'schema': 'merlin.paper-compiler-experiment.v1', 'system_id': label,
        'display_name': source.get('display_name', label), 'branch': source.get('branch', branch_for(label)),
        'paper_repository': 'copparihollmann/merlin-mlsys-edit', 'paper_commit': PAPER_COMMIT,
        'compiler': {'path': 'compiler', 'tree_sha256': actual, 'source': str(root),
                     'file_count': len(files), 'digest_algorithm': 'SHA256(sorted relative UTF8 path, NUL, file bytes, NUL), excluding build, __pycache__, .git'},
        'experiment_setup': 'experiment/setup.json',
        'results': {'raw_observations': 'experiment/measurement-records.csv',
                    'selected_master_rows': 'experiment/selected-master-records.csv',
                    'summary': 'experiment/metrics.json'},
        'verification': {'source_hash': 'matched_original_record',
                         'measurement_source_hashes': 'matched' if rows or picked else 'no_measurement_records_selected',
                         'new_execution': 'not_performed', 'compiler_build_rerun': 'not_performed'},
        'dependencies': {'framework_export': source.get('framework'),
                         'generic_library': source.get('generic_library'),
                         'environment_record': 'experiment/toolchains.json',
                         'scope': 'This branch preserves the compiler package. Evaluator, toolchain, simulator and hardware dependencies are tracked separately.'},
        'historical_role': 'reference-assisted historical control with a curated map' if label.startswith('fresh10') else 'measured experiment compiler',
    }
    # Preserve shared dependencies outside compiler/ so its original digest stays valid.
    lib = source.get('generic_library')
    if lib and Path(lib).is_dir():
        lib_root = Path(lib)
        lib_files = source_files(lib_root)
        manifest['dependencies']['generic_library_tree_sha256'] = tree_digest(lib_root, lib_files)
        manifest['dependencies']['packaged_generic_library'] = 'dependencies/generic-library'
        for path in lib_files:
            payload['dependencies/generic-library/' + path.relative_to(lib_root).as_posix()] = str(path)
    extras = {
        'experiment/manifest.json': encoded(manifest),
        'experiment/setup.json': encoded(campaign_setup(source)),
        'experiment/source-files.json': encoded({'tree_sha256': actual, 'files': inventory}),
        'experiment/measurement-records.csv': csv_bytes(rows),
        'experiment/selected-master-records.csv': csv_bytes(picked),
        'experiment/metrics.json': encoded({'raw_observations': metrics(rows), 'selected_master': metrics(picked),
                                           'same_image_fsm_controls': fsm_pairs(rows, by_system)}),
        'experiment/verify_snapshot.py': VERIFY_SCRIPT.encode(),
        'experiment/toolchains.json': (MEASUREMENTS / 'toolchains.json').read_bytes(),
        'experiment/protocol.json': (MEASUREMENTS / 'protocol.json').read_bytes(),
        'experiment/runtime-provenance.json': (MEASUREMENTS / 'runtime-provenance.json').read_bytes(),
    }
    extras['README.md'] = f'''# {source.get('display_name', label)}

This branch preserves the exact compiler submission used by experiment `{label}`. It is an archival experiment version, with its original source under `compiler/` and its setup and results under `experiment/`.

| Item | Record |
|---|---|
| Experiment branch | `{manifest['branch']}` |
| Original run | `{source.get('campaign_run', 'See setup.json')}` |
| Source digest | `{actual}` |
| Source files | {len(files)} |
| Paper revision | `{PAPER_COMMIT}` |

Start with [the manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), and [recorded metrics](experiment/metrics.json). The [measurement records](experiment/measurement-records.csv) retain image and cache-profile identities. [Selected master rows](experiment/selected-master-records.csv) preserve the original panel selection where it exists. Rows from different images are not silently pooled.

Verify the compiler bytes without executing it:

```sh
python3 experiment/verify_snapshot.py
```

The original `compiler/manifest.yaml` declares the compiler entrypoints. Shared imported code shipped within the frozen compiler is retained because it is part of the measured source identity. Additional captured generic-library dependencies, when available, are under `dependencies/`. The manifest pins the remaining evaluator and toolchain records. Reconstruction of a full execution environment remains part of the artifact package.

The archived measurements are historical observations. No compiler, agent, simulator or FPGA experiment was rerun to prepare this branch. Construction-suite completion and the frozen 89-case kernel panel have different scopes. Read the setup and per-record statuses before interpreting an aggregate.
'''.encode()
    folder = OUT / 'metadata' / label
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in extras.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        payload[name] = str(path)
    return {'label': label, 'branch': manifest['branch'], 'source_sha256': actual,
            'source': str(root), 'source_files': len(files), 'payload': payload,
            'measurement_records': len(rows), 'selected_master_records': len(picked)}


def replace_metadata(package, rel, value):
    path = Path(package['payload'][rel])
    path.write_bytes(encoded(value))


def add_metadata(package, rel, data):
    root = OUT / 'metadata' / package['label']
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    package['payload'][rel] = str(path)


def extend_studies():
    """Add both arms of the studies shown in the scheduling/library figures."""
    plan = read(OUT / 'publication-plan.json')
    names = [
        ('scheduling', 'followup-scheduling-h3a-30min-v1'),
        ('scheduling', 'scheduling-completion-authors'),
        ('scheduling', 'scheduling-completion-matched-60-r1'),
        ('scheduling', 'followup-scheduling-reference-free-recovery-r1'),
        ('scheduling', 'followup-scheduling-extension-v1'),
        ('transformations', 'followup-library-oracle-h2conv-30min-v3'),
        ('cca-controlled', 'followup-cca-h1d-p1r'),
        ('cca-controlled', 'followup-cca-h1d-p2r'),
    ]
    ttm = STUDIES / 'figures-extra/time-to-mechanism/figure-data.json'
    time_studies = {s['id']: s for s in read(ttm)['studies']}
    ledger_path = STUDIES / 'ledger/ledger.json'
    ledger = {r['id']: r for r in read(ledger_path)['records']}
    for group, name in names:
        run = STUDIES / 'queue/runs' / name
        try:
            checkpoints = list(run.glob('checkpoints/*/observations.json'))
            final = [p for p in checkpoints if 'final_stop' in p.parent.name]
            if len(final) != 1:
                raise ValueError(f'Expected one final-stop record, found {len(final)}')
            checkpoint = read(final[0])
            observations = checkpoint['observations']
            if len(observations) != 2:
                raise ValueError('Expected two frozen arms')
            for arm, observation in enumerate(observations):
                role = 'control' if arm == 0 else 'treatment'
                label = name + '-' + role
                if any(p['label'] == label for p in plan['packages']):
                    continue
                cfg_path = Path(observation['config']['path'])
                cfg = read(cfg_path)
                source = observation['source']
                if not source.get('present'):
                    raise ValueError(f'{role}: frozen source missing')
                source_row = {
                    'label': label, 'display_name': name + ' / ' + role,
                    'branch': f'paper/{group}/{name}/{role}-final',
                    'copied_source': source['path'], 'expected': source['sha256'],
                    'framework': cfg.get('framework'), 'snapshot_kind': 'final_stop',
                }
                package = checked_payload(source_row, {}, {})
                manifest = read(package['payload']['experiment/manifest.json'])
                manifest['historical_role'] = 'paired-study frozen compiler'
                manifest['experiment_id'] = name
                manifest['arm'] = role
                manifest['verification']['measurement_source_hashes'] = 'See source-pinned study records. No new execution performed.'
                setup = {k: cfg.get(k) for k in [
                    'model', 'effort', 'wall_s', 'cpu_quota_percent', 'cpus',
                    'no_helpers', 'workers', 'tooling', 'reference_guidance',
                    'study_name', 'study_role', 'study_arm', 'schedule_guidance',
                    'library_guidance', 'cca_guidance', 'experiment_protocol',
                    'initial_submission_sha256', 'training_cases', 'suite']}
                setup['authoring_model'] = cfg.get('model')
                setup['reasoning_effort'] = cfg.get('effort')
                setup['seed_source'] = cfg.get('source_seed')
                setup['records'] = [record(final[0]), record(cfg_path), record(run / 'runs.json')]
                setup['frozen_observation'] = observation
                setup['framework_export'] = cfg.get('framework')
                setup['remaining_dependencies'] = {k: cfg.get(k) for k in [
                    'python', 'parser_dependencies', 'read_only_dependencies',
                    'library', 'library_sha', 'inputs', 'facts', 'descriptor']}
                summary = {
                    'schema': 'merlin.archived-paired-study-summary.v1',
                    'study': name, 'arm': role, 'new_execution_performed': False,
                    'scope': 'Saved study observations. Preserve published and native qualification limits.',
                    'recorded_effort': None, 'time_to_mechanism': None,
                }
                if name in ledger:
                    study = ledger[name]
                    effort = study.get('effort', {}).get('arms', [])
                    summary['recorded_effort'] = next((a for a in effort if a['arm'] == arm), None)
                    summary['native_correctness_summary'] = study.get('correctness')
                    summary['contrast'] = study.get('contrast')
                    summary['source_records'] = [record(ledger_path)]
                if name in time_studies:
                    study = time_studies[name]
                    summary['time_to_mechanism'] = next(a for a in study['arms'] if a['arm'] == arm)
                    summary.setdefault('source_records', []).append(record(ttm))
                if group == 'transformations':
                    data_path = STUDIES / 'figures-paper-v3/H2-library/bundle/data/figure-data.json'
                    data = read(data_path)
                    summary['published_hardware_panel'] = data['firesim']
                    summary['modeled_timeline'] = data['timeline']['baseline' if arm == 0 else '+library']
                    summary.setdefault('source_records', []).append(record(data_path))
                    summary['qualification'] = 'The control hangs on hardware. Modeled progress is not hardware correctness. Oracle was withheld from both authors.'
                manifest['results']['summary'] = 'experiment/metrics.json'
                manifest['dependencies']['scope'] = 'Frozen submission and exact dependency identities. Full evaluator/runtime restoration is tracked by the artifact package.'
                replace_metadata(package, 'experiment/manifest.json', manifest)
                replace_metadata(package, 'experiment/setup.json', setup)
                replace_metadata(package, 'experiment/metrics.json', summary)
                # Do not attach the unrelated 89-case performance protocol.
                replace_metadata(package, 'experiment/protocol.json', {
                    'study': name, 'config': record(cfg_path),
                    'runs': record(run / 'runs.json'),
                    'declared_budget_s': cfg.get('wall_s'),
                    'numerical_and_hardware_protocol': 'See source-pinned study records and saved evaluation metadata.',
                    'new_execution_performed': False,
                })
                # The shared toolchain record is a reference, not an assertion that
                # this study used exactly the same binaries as the 89-case panel.
                replace_metadata(package, 'experiment/toolchains.json', {
                    'study': name, 'python': cfg.get('python'),
                    'recorded_environment': {'path': str(cfg_path), 'sha256': record(cfg_path)['sha256']},
                    'scope': 'Exact tool paths are retained in the original config/environment. Binary reconstruction is pending the dependency audit.',
                })
                del package['payload']['experiment/runtime-provenance.json']
                readme = Path(package['payload']['README.md'])
                readme.write_text(f'''# {name}: {role}

This branch preserves the final compiler for the {role} arm of `{name}`. The source under `compiler/` matches the native final-stop checkpoint `{source['sha256']}`.

The [experiment manifest](experiment/manifest.json) identifies the version. The [setup](experiment/setup.json) records the model, budget, tools, resources, dependencies and seed. The [metrics](experiment/metrics.json) preserve available effort and evaluation records, including their scope and source identities. The [protocol](experiment/protocol.json) points to the original declaration.

```sh
python3 experiment/verify_snapshot.py
```

This command verifies archived source bytes only. No compiler, agent, simulator or FPGA experiment was rerun during preservation. Development timing, modeled cycles, numerical checks and hardware measurements remain distinct. Reconstruction of the evaluator and toolchain is tracked separately in the artifact package.
''')
                plan['packages'].append(package)
                print('PREPARED', label, source['sha256'], flush=True)
        except (ValueError, KeyError, OSError) as error:
            plan['gaps'].append({'study': name, 'error': str(error)})
            print('GAP', name, str(error), flush=True)
    (OUT / 'publication-plan.json').write_bytes(encoded(plan))
    print('Total branches', len(plan['packages']), 'gaps', len(plan['gaps']), flush=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    study = read(MEASUREMENTS / 'study.json')
    by_system, selected = measurement_rows()
    packages = []
    errors = []
    for source in study['sources']:
        try:
            item = checked_payload(source, by_system, selected)
            packages.append(item)
            print(json.dumps({k: v for k, v in item.items() if k != 'payload'}), flush=True)
        except (ValueError, OSError) as error:
            errors.append({'label': source['label'], 'error': str(error)})
            print('GAP', source['label'], str(error), flush=True)
    plan = {'schema': 'merlin.paper-compiler-branches.v1', 'repository': REMOTE,
            'paper_commit': PAPER_COMMIT, 'measurement_root': str(MEASUREMENTS),
            'packages': packages, 'gaps': errors}
    (OUT / 'publication-plan.json').write_bytes(encoded(plan))
    print('Prepared', len(packages), 'compiler branches. Gaps:', len(errors), flush=True)


def refresh_metrics():
    plan = read(OUT / 'publication-plan.json')
    by_system, selected = measurement_rows()
    for package in plan['packages']:
        prior = read(package['payload']['experiment/metrics.json'])
        if 'paired' in prior.get('schema', ''):
            if package['label'].startswith('followup-cca-h1d-'):
                path = STUDIES / 'figures-paper-v3/H1-cca/h1d-effect-map-v3/figure-data.json'
                d = read(path)
                # Preserve the public, anonymized figure view without copying
                # sealed input definitions or oracle implementations.
                prior['published_cca_study'] = {k: d[k] for k in [
                    'pairs', 'rows', 'noise_floor', 'verdict_rule', 'mechanisms']}
                prior.setdefault('source_records', []).append(record(path))
                prior['qualification'] = 'Saved ordered hardware observations. Full consumed-ELF/domain qualification remains a separate audit item.'
                replace_metadata(package, 'experiment/metrics.json', prior)
            continue
        rows = by_system.get(package['label'], [])
        picked = selected.get(package['label'], [])
        images = []
        for image in sorted({r['measurement_image'] for r in rows}):
            folder = MEASUREMENTS / 'measurements' / image
            hardware_path = folder / 'hardware.json'
            manifest_path = folder / 'image-manifest.json'
            entry = {'image': image, 'records': []}
            if hardware_path.exists():
                d = read(hardware_path)
                entry['hardware'] = d.get('hardware')
                entry['elf_sha256'] = d.get('elf_sha256')
                entry['execution_observed'] = d.get('execution_observed')
                entry['records'].append(record(hardware_path))
            if manifest_path.exists():
                d = read(manifest_path)
                entry['harness'] = {k: d.get(k) for k in [
                    'harness_version', 'harness_sha256', 'cache_profiles',
                    'samples', 'correctness_phases', 'warm_policy', 'cold_policy',
                    'measurement', 'preparation_scope', 'runtime_provenance_sha256']}
                entry['records'].append(record(manifest_path))
            receipt = folder / 'receipt.json'
            if receipt.exists():
                entry['records'].append(record(receipt))
            images.append(entry)
        add_metadata(package, 'experiment/evaluation-images.json', encoded(images))
        manifest = read(package['payload']['experiment/manifest.json'])
        manifest['evaluation_images'] = 'experiment/evaluation-images.json'
        replace_metadata(package, 'experiment/manifest.json', manifest)
        result = {
            'raw_observations': metrics(rows), 'selected_master': metrics(picked),
            'same_image_fsm_controls': fsm_pairs(rows, by_system)}
        if package['label'] in ['EL1-r5', 'EL2-r5', 'EL3-r5']:
            ladder = Path('/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/artifacts/figures/paper-v1/data/el_ladder')
            arm = package['label'].split('-')[0]
            development = {'scope': 'Historical development across compiler versions. These counts are separate from final frozen-compiler correctness.', 'records': {}}
            for path in sorted(ladder.glob('*.csv')):
                values = [r for r in csv.DictReader(path.open()) if r.get('run') == arm]
                if not values:
                    continue
                relative = 'experiment/development/' + path.name
                add_metadata(package, relative, csv_bytes(values))
                development['records'][path.name] = {'source': record(path), 'record_count': len(values), 'preserved_rows': relative}
                if path.name == 'token_timeline.csv':
                    first_hour = [r for r in values if float(r['elapsed_s']) <= 3600]
                    development['uncached_input_tokens_at_60min'] = max(float(r['cum_uncached_input']) for r in first_hour)
                if path.name == 'correctness_timeseries.csv':
                    first_hour = [r for r in values if r['table'] == 'ladder' and r['suite'] == 'cases' and float(r['elapsed_min']) <= 60]
                    development['historical_l2_pass_union_at_60min'] = max(int(r['cum_l2_pass_cases']) for r in first_hour)
            result['construction_development'] = development
        replace_metadata(package, 'experiment/metrics.json', result)
    (OUT / 'publication-plan.json').write_bytes(encoded(plan))
    print('Refreshed recorded metrics without running an experiment', flush=True)


def create_index(published, base, previous=None):
    branch = 'paper/index'
    if not previous and git('for-each-ref', '--format=%(refname)', 'refs/heads/' + branch).strip():
        raise ValueError('Index branch already exists. Inspect it before creating a replacement.')
    folder = OUT / 'index'
    folder.mkdir(exist_ok=True)
    lines = [
        '# Merlin paper experiment compilers', '',
        f'This directory identifies the frozen compiler submissions used in paper revision `{PAPER_COMMIT}`. Each experiment branch contains its original compiler, setup manifest and saved result records. Source identities have been checked against the original experiment records. No agent, compiler, simulator or FPGA experiment was rerun for this archive.', '',
        '| Compiler / run | Experiment branch | Commit |', '|---|---|---|']
    for p in published:
        lines.append(f"| {p['label']} | [{p['branch']}]({REMOTE}/tree/{p['branch']}) | [`{p['commit'][:12]}`]({p['url']}) |")
    lines.extend(['',
        'Each branch has:', '',
        '- `compiler/`: exact frozen submission, with executable file modes preserved.',
        '- `experiment/manifest.json`: experiment identity, source digest and evidence status.',
        '- `experiment/setup.json`: starting point, model, budget, tools and dependency records.',
        '- `experiment/metrics.json`: recorded result metrics, grouped by their evaluation scope.',
        '- `experiment/source-files.json` and `verify_snapshot.py`: per-file hashes and a source verification command.',
        '- Measurement CSVs where available. Historical failures and excluded observations retain their status.', '',
        'Use the commit SHA or source digest when citing a version. The archival commits descend from the repository landing version for navigation. This parentage does not reconstruct the agents\' development histories.', '',
        'The 89-case kernel panel, construction-suite checks, modeled development curves and smaller paired studies have different denominators. A native campaign completion flag is not the frozen kernel panel\'s numerical verdict. Comparisons in different images are kept separate. The common FSM reference in the paper figures and same-image FSM controls are explicitly distinguished.', '',
        'The framework export, runtime, test inputs, toolchains, simulator and hardware-image dependencies remain part of the wider artifact package. These branches preserve the compiler versions and their existing evidence. Full execution reproducibility remains to be checked later.', '',
        'Historical curated-reference runs are retained for provenance and are not the primary Merlin + Exo result. The primary Exo-refined compiler is `CCA-EXO-R3`. The existing `mlsys27/test-generation-coverage` branch records the separate coverage implementation.', '',
        'See [branches.json](branches.json) for machine-readable identities and [the preservation utility](tools/prepare_compiler_branches.py) for the source-to-branch mapping.', ''])
    (folder / 'README.md').write_text('\n'.join(lines))
    (folder / 'branches.json').write_bytes(encoded({'schema': 'merlin.paper-compiler-index.v1', 'paper_commit': PAPER_COMMIT, 'branches': published}))
    (folder / 'tools').mkdir(exist_ok=True)
    (folder / 'tools/prepare_compiler_branches.py').write_bytes(Path(__file__).read_bytes())
    # A separate index keeps the editable working tree and its staging area intact.
    index = TASK / '.tmp/artifact-branch-index'
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    def indexed(*argv, data=None):
        return subprocess.check_output(['git', '-C', str(REPO), *argv], input=data, env=env)
    indexed('read-tree', '--empty')
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            blob = git('hash-object', '-w', '--stdin', data=path.read_bytes()).decode().strip()
            indexed('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path.relative_to(folder).as_posix())
    tree = indexed('write-tree').decode().strip()
    commit = git('commit-tree', tree, '-p', previous or base, data=b'docs(artifacts): index paper compiler versions and recorded experiments\n').decode().strip()
    git('update-ref', 'refs/heads/' + branch, commit, previous or '0' * 40)
    return {'branch': branch, 'commit': commit, 'url': REMOTE + '/tree/' + branch}


def attach_current_validation():
    """Keep the later current-library controls distinct from historical rows."""
    import statistics
    plan = read(OUT / 'publication-plan.json')
    package = next(p for p in plan['packages'] if p['label'] == 'CCA-EXO-R3')
    local = TASK / '.tmp'
    root = local / 'fsm-rerun'
    declaration = read(root / 'rerun-manifest.json')
    observed = read(root / 'hardware-validation-summary.json')
    rows = read(root / 'verified-rerun-results.json')
    assert declaration['merlin_objects_unchanged'] == 89
    assert observed['status'] == 'verified_hardware_execution'
    assert observed['elf_sha256'] == declaration['elf_sha256']
    assert all(observed[k] == 0 for k in ['numerical_mismatches', 'guard_failures', 'input_failures', 'done_code'])
    groups = {}
    for row in rows:
        assert len(row['fsm_samples']) == len(row['merlin_samples']) == 20
        assert statistics.median(row['fsm_samples']) == row['fsm_median']
        assert statistics.median(row['merlin_samples']) == row['merlin_median']
        assert math.isclose(row['fsm_over_merlin'], row['fsm_median'] / row['merlin_median'], rel_tol=1e-12)
        groups.setdefault(row['case_id'], []).append(row)
    assert len(groups) == 89 and len(rows) == 178
    combined = []
    for case, values in sorted(groups.items()):
        assert sorted(r['pass_id'] for r in values) == [0, 1]
        fsm = math.sqrt(math.prod(r['fsm_median'] for r in values))
        compiler = math.sqrt(math.prod(r['merlin_median'] for r in values))
        combined.append({'case_id': case, 'fsm_cycles': fsm, 'compiler_cycles': compiler, 'fsm_over_compiler': fsm / compiler})
    for subdir, names in [
        ('fsm-rerun', ['rerun-manifest.json', 'hardware-validation-summary.json', 'verified-rerun-results.json', 'verified-order-and-trend.json', 'current-upstream-comparison.json']),
        ('fsm-counterfactual-execution', ['manifest.json', 'hardware-validation-summary.json', 'verified-results.json']),
        ('fsm-decode-strides', ['manifest.json', 'hardware-validation-summary.json', 'verified-results.json']),
        ('fsm-fresh-inputs', ['manifest.json', 'hardware-validation-summary.json', 'verified-results.json']),
        ('gu-hardware-ablation', ['manifest.json', 'hardware-validation-summary.json', 'verified-results.json']),
    ]:
        for name in names:
            source = local / subdir / name
            add_metadata(package, 'experiment/later-validation/' + subdir + '/' + name, source.read_bytes())
    latest = {'schema': 'merlin.recorded-current-library-validation.v1',
              'new_execution_performed': False,
              'recorded_fsm_commit': declaration['fsm_upstream_commit'],
              'warmups_per_implementation_per_order': 20,
              'measurements_per_implementation_per_order': 20,
              'opposite_orders_in_one_boot': 2,
              'compiler_objects_unchanged': 89,
              'case_count': 89, 'order_records': 178,
              'geometric_mean_fsm_over_compiler': math.exp(sum(math.log(r['fsm_over_compiler']) for r in combined) / len(combined)),
              'cases': combined, 'saved_execution_summary': observed,
              'source_records': [record(root / n) for n in ['rerun-manifest.json', 'hardware-validation-summary.json', 'verified-rerun-results.json']],
              'scope': 'Later validation. Keep separate from the five-call historical/common-reference results and from manual adapted FSM policies.',
              'record_phase_note': 'Build manifests retain their original built_not_executed status. The separate hardware-validation summaries record the subsequent execution.'}
    metrics_path = 'experiment/metrics.json'
    m = read(package['payload'][metrics_path])
    m['later_current_library_validation'] = latest
    replace_metadata(package, metrics_path, m)
    manifest_path = 'experiment/manifest.json'
    m = read(package['payload'][manifest_path])
    m['results']['later_validation_records'] = 'experiment/later-validation'
    replace_metadata(package, manifest_path, m)
    (OUT / 'publication-plan.json').write_bytes(encoded(plan))
    print('Attached later recorded validation, exact arithmetic checked on 178 order records', flush=True)


def update_metadata(push=False):
    """Advance metadata only. Refuse any compiler-subtree change."""
    registry = read(OUT / 'branches.json')
    plan = read(OUT / 'publication-plan.json')
    packages = {p['label']: p for p in plan['packages']}
    index_file = TASK / '.tmp/artifact-metadata-index'
    env = dict(os.environ, GIT_INDEX_FILE=str(index_file))
    changed = []
    for item in registry['branches']:
        parent = item['commit']
        assert git('rev-parse', item['branch']).decode().strip() == parent
        subprocess.run(['git', '-C', str(REPO), 'read-tree', parent], env=env, check=True)
        for rel, filename in packages[item['label']]['payload'].items():
            if not (rel.startswith('experiment/') or rel == 'README.md'):
                continue
            blob = git('hash-object', '-w', '--stdin', data=Path(filename).read_bytes()).decode().strip()
            subprocess.run(['git', '-C', str(REPO), 'update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + rel], env=env, check=True)
        tree = subprocess.check_output(['git', '-C', str(REPO), 'write-tree'], env=env).decode().strip()
        if tree == git('rev-parse', parent + '^{tree}').decode().strip():
            continue
        commit = git('commit-tree', tree, '-p', parent, data=b'docs(experiment): retain later recorded validation alongside historical measurements\n').decode().strip()
        assert git('rev-parse', parent + ':compiler') == git('rev-parse', commit + ':compiler')
        git('update-ref', 'refs/heads/' + item['branch'], commit, parent)
        item.update(commit=commit, url=REMOTE + '/tree/' + commit)
        changed.append(item)
    if changed:
        old_index = registry['index']['commit']
        registry['index'] = create_index(registry['branches'], git('rev-parse', 'origin/main').decode().strip(), previous=old_index)
    registry['pushed'] = False if changed else registry['pushed']
    registry['backup_covers_current_refs'] = not changed
    (OUT / 'branches.json').write_bytes(encoded(registry))
    if push and changed:
        names = [p['branch'] for p in changed] + [registry['index']['branch']]
        subprocess.run(['git', '-C', str(REPO), 'push', '--atomic', 'origin',
                        *['refs/heads/' + n + ':refs/heads/' + n for n in names]], check=True)
        registry['pushed'] = True
        (OUT / 'branches.json').write_bytes(encoded(registry))
    print('Updated metadata branches:', len(changed), flush=True)


def create_commits(push=False):
    plan = read(OUT / 'publication-plan.json')
    base = git('rev-parse', 'origin/main').decode().strip()
    existing = set(git('for-each-ref', '--format=%(refname)').decode().splitlines())
    for p in plan['packages']:
        ref = 'refs/heads/' + p['branch']
        if ref in existing:
            raise ValueError(f'Ref already exists. Inspect it rather than overwriting: {ref}')
    name = git('config', 'user.name').strip()
    email = git('config', 'user.email').strip()
    stamp = str(int(time.time())).encode()
    log = OUT / 'fast-import.log'
    marks = {}
    mark = 0
    with log.open('wb') as stderr:
        proc = subprocess.Popen(['git', '-C', str(REPO), 'fast-import', '--quiet', '--done'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        stream = proc.stdin
        assert stream is not None
        for p in plan['packages']:
            entries = []
            for rel, path in sorted(p['payload'].items()):
                file = Path(path)
                data = file.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest not in marks:
                    mark += 1
                    marks[digest] = mark
                    stream.write(f'blob\nmark :{mark}\ndata {len(data)}\n'.encode())
                    stream.write(data + b'\n')
                mode = '100755' if file.stat().st_mode & 0o111 else '100644'
                entries.append(f'M {mode} :{marks[digest]} {json.dumps(rel, ensure_ascii=False)}\n'.encode())
            msg = f"archive({p['label']}): preserve measured compiler and experiment records\n".encode()
            stream.write(f"commit refs/heads/{p['branch']}\n".encode())
            stream.write(b'committer ' + name + b' <' + email + b'> ' + stamp + b' +0000\n')
            stream.write(f'data {len(msg)}\n'.encode() + msg)
            stream.write(f'from {base}\ndeleteall\n'.encode())
            for entry in entries:
                stream.write(entry)
            stream.write(b'\n')
        stream.write(b'done\n')
        stream.close()
        proc.stdin = None
        proc.communicate()
        if proc.returncode:
            raise RuntimeError(log.read_text())
    verify_created(push=push, base=base)


def verify_created(push=False, base=None):
    plan = read(OUT / 'publication-plan.json')
    base = base or git('rev-parse', 'origin/main').decode().strip()
    published = []
    for p in plan['packages']:
        commit = git('rev-parse', p['branch']).decode().strip()
        # Check the bytes committed to Git, not just their original source files.
        raw = git('archive', '--format=tar', commit, 'compiler')
        import tarfile
        inventory = read(p['payload']['experiment/source-files.json'])
        recorded_modes = {r['source_path']: r['executable'] for r in inventory['files']}
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            digest = hashlib.sha256()
            members = {m.name.removeprefix('compiler/'): m for m in archive.getmembers() if m.isfile()}
            if set(members) != set(recorded_modes):
                raise ValueError(f'{p["label"]}: Git archive file inventory differs')
            # The native digest sorts pathlib components. Full-string sorting
            # differs for names such as x.py and x/y.py. Use the original order.
            for row in inventory['files']:
                rel = row['source_path']
                member = members[rel]
                data = archive.extractfile(member).read()
                if sha(data) != row['sha256']:
                    raise ValueError(f'{p["label"]}: Git blob differs for {rel}')
                digest.update(rel.encode() + b'\0' + data + b'\0')
                if bool(member.mode & 0o111) != recorded_modes[rel]:
                    raise ValueError(f'{p["label"]}: executable mode differs for {rel}')
            if digest.hexdigest() != p['source_sha256']:
                raise ValueError(f'{p["label"]}: Git archive differs from recorded compiler')
        published.append({k: v for k, v in p.items() if k != 'payload'} | {'commit': commit, 'url': REMOTE + '/tree/' + commit})
        print('VERIFIED', p['branch'], commit, flush=True)
    index = create_index(published, base)
    (OUT / 'branches.json').write_bytes(encoded({'repository': REMOTE, 'branches': published, 'index': index, 'pushed': False}))
    if push:
        refs = ['refs/heads/' + p['branch'] + ':refs/heads/' + p['branch'] for p in plan['packages']]
        refs.append('refs/heads/' + index['branch'] + ':refs/heads/' + index['branch'])
        subprocess.run(['git', '-C', str(REPO), 'push', '--atomic', 'origin', *refs], check=True)
        remote = git('ls-remote', '--heads', 'origin', *[p['branch'] for p in plan['packages']]).decode()
        observed = {line.split()[1]: line.split()[0] for line in remote.splitlines()}
        assert all(observed.get('refs/heads/' + p['branch']) == p['commit'] for p in published)
        index_remote = git('ls-remote', '--heads', 'origin', index['branch']).decode().split()[0]
        assert index_remote == index['commit']
        (OUT / 'branches.json').write_bytes(encoded({'repository': REMOTE, 'branches': published, 'index': index, 'pushed': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'extend-studies', 'refresh-metrics', 'attach-current-validation', 'update-metadata', 'create', 'verify-created'])
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    if (OUT / 'organization.json').exists():
        parser.error('Publication refs have been reorganized. This original importer is retained for provenance only. Use compiler-branches/branches.json and paper/index/runs.json for current identities. Do not recreate retired branch names.')
    if args.action == 'prepare':
        if args.push:
            parser.error('--push requires create')
        prepare()
    elif args.action == 'extend-studies':
        if args.push:
            parser.error('--push requires create')
        extend_studies()
    elif args.action == 'refresh-metrics':
        if args.push:
            parser.error('--push requires create')
        refresh_metrics()
    elif args.action == 'verify-created':
        verify_created(push=args.push)
    elif args.action == 'attach-current-validation':
        if args.push:
            parser.error('--push requires a publication action')
        attach_current_validation()
    elif args.action == 'update-metadata':
        update_metadata(push=args.push)
    else:
        create_commits(push=args.push)
