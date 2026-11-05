#!/usr/bin/env python3
"""Organize existing paper snapshots without changing compilers or measurements.

Plan first, preserve every snapshot as an annotated tag, then retire only the
explicitly recorded old branches. No compiler or experiment is executed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

TASK = Path('/work/merlin/tasks/mlsys-paper-20261024')
MERLIN = TASK / 'merlin'
REPO = TASK / 'gemmini-mlir'
ROOT = MERLIN / 'out/artifacts/artifact-evaluation/20261104'
OUT = ROOT / 'compiler-branches'
WORK = OUT / 'organization-20261105'
REMOTE = 'https://github.com/ucb-bar/gemmini-mlir'

# Original record IDs stay intact. Friendly names are aliases, not new trials.
# False means that a permanent tag is sufficient for navigation.
DIRECT = {
    'F10-FULL-NOREF-NOCCA': ('construction/no-expert-kernels-4h', 'Merlin construction without expert kernels', True),
    'CCA-EXO-R3': ('refinement/exo-kernels-4plus4h', 'Merlin refinement with Exo kernels', True),
    'EL1-r5': ('scaffolding/e1-raw-rtl-4h', 'E1: raw RTL and C++', True),
    'EL2-r5': ('scaffolding/e2-cpp-support-4h', 'E2: C++ interface support', True),
    'EL3-r5': ('scaffolding/e3-merlin-support-4h', 'E3: Merlin compiler support', True),
    'CCA-RESNET50-R2': ('cca/resnet50-kernels-4plus4h', 'CCA study with ResNet-50 kernels, revision 2', True),
    'CCA-TINYLLAMA-R2': ('cca/tinyllama-kernels-4plus4h', 'CCA study with TinyLlama kernels, revision 2', True),
    'CCA-RAW': ('cca/raw-expert-files-4plus4h', 'Raw expert-file continuation', False),
    'CCA-EXO': ('cca/exo-kernels-revision1-4plus4h', 'Exo continuation, revision 1', False),
    'CCA-EXO-R2': ('cca/exo-kernels-revision2-4plus4h', 'Exo continuation, revision 2', False),
    'CCA-RESNET50': ('cca/resnet50-kernels-revision1-4plus4h', 'ResNet-50 continuation, revision 1', False),
    'CCA-TINYLLAMA': ('cca/tinyllama-kernels-revision1-4plus4h', 'TinyLlama continuation, revision 1', False),
    'CCA-ALL': ('cca/all-sources-revision1-4plus4h', 'All-source continuation, revision 1', False),
    'CCA-ALL-R2': ('cca/all-sources-revision2-4plus4h', 'All-source continuation, revision 2', False),
    'F10-INFRA-REF-NOCCA': ('tooling/compiler-support-with-references-4h', 'Compiler-support study with references', False),
    'F10-SCHED-REF-NOCCA': ('tooling/scheduling-support-with-references-4h', 'Scheduling-support study with references', False),
    'FULL-REF-4H': ('historical/curated-references-checkpoint-4h', 'Curated-reference construction, four-hour checkpoint', False),
    'FULL-REF': ('historical/curated-references-final-12h', 'Curated-reference construction, final twelve-hour snapshot', False),
    'fresh10cca-live': ('historical/cca-campaign-checkpoint-4h', 'Earlier CCA campaign, four-hour checkpoint', False),
    'fresh10cca-final12h': ('historical/cca-campaign-final-12h', 'Earlier CCA campaign, final twelve-hour snapshot', False),
    'fresh10seed-live': ('historical/seed-campaign-checkpoint-4h', 'Earlier seed campaign, four-hour checkpoint', False),
    'fresh10seed-final12h': ('historical/seed-campaign-final-12h', 'Earlier seed campaign, final twelve-hour snapshot', False),
}
STUDIES = {
    'followup-scheduling-h3a-30min-v1': ('scheduling/main-30min', 'Checked scheduling, main thirty-minute study', True),
    'scheduling-completion-authors': ('scheduling/study02-60min', 'Checked scheduling, sixty-minute study 2', False),
    'scheduling-completion-matched-60-r1': ('scheduling/study03-60min', 'Checked scheduling, sixty-minute study 3', False),
    'followup-scheduling-reference-free-recovery-r1': ('scheduling/study04-60min', 'Checked scheduling, sixty-minute study 4', False),
    'followup-scheduling-extension-v1': ('scheduling/extension-60min', 'Checked scheduling, capability-extension study', False),
    'followup-library-oracle-h2conv-30min-v3': ('transformations/convolution-30min', 'Convolution transformation-library study', True),
    'followup-cca-h1d-p1r': ('cca-controlled/pair1-30min', 'Controlled CCA pair 1', False),
    'followup-cca-h1d-p2r': ('cca-controlled/pair2-30min', 'Controlled CCA pair 2', False),
}


def git(*args, data=None, env=None):
    return subprocess.check_output(['git', '-C', str(REPO), *args], input=data, env=env)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text())


def remote_refs():
    return {line.split()[1]: line.split()[0] for line in git('ls-remote', '--heads', '--tags', 'origin').decode().splitlines()}


def identity(label, setup):
    if label in DIRECT:
        return DIRECT[label]
    study, arm = label.rsplit('-', 1)
    suffix, title, primary = STUDIES[study]
    if study.startswith('followup-cca-'):
        # Queue slot order reverses in pair 2. Name the actual CCA access,
        # never infer it from the preservation utility's positional label.
        role = setup['study_role']
        assert role in ('raw', 'full')
        variant = 'cca-on' if role == 'full' else 'cca-off'
    elif 'h2conv' in study:
        variant = 'seed-library' if arm == 'control' else 'extended-library'
    else:
        variant = 'seed-scheduler' if arm == 'control' else 'checked-scheduler'
    return suffix + '/' + variant, title + ': ' + variant.replace('-', ' '), primary


def collect_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from collect_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_paths(item)
    elif isinstance(value, str) and value.startswith('/work/'):
        yield value


def plan():
    WORK.mkdir(exist_ok=True)
    previous = read(OUT / 'branches.json')
    if previous.get('organization'):
        raise ValueError('Already organized. Inspect organization.json rather than recreating aliases.')
    for name in ['branches.json', 'publication-plan.json', 'verification.json', 'SHA256SUMS']:
        target = WORK / ('before-' + name)
        data = (OUT / name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Refusing to overwrite the prior inventory: ' + name)
        target.write_bytes(data)
    observed = remote_refs()
    snapshots = []
    for old in previous['branches']:
        assert observed['refs/heads/' + old['branch']] == old['commit']
        setup = json.loads(git('show', old['commit'] + ':experiment/setup.json'))
        suffix, title, primary = identity(old['label'], setup)
        paths = sorted(set(collect_paths(setup)))
        aliases = sorted({old['label'], old['branch'], old['source_sha256'], old['commit'],
                          *paths, *(Path(p).name for p in paths if Path(p).name not in ('config.json', 'settings.json', 'RULES.md'))})
        snapshots.append(dict(old, previous_branch=old['branch'],
                              branch='paper/' + suffix if primary else None,
                              tag='mlsys-paper-v1/' + suffix,
                              display_name=title, aliases=aliases,
                              setup_path='experiment/setup.json', metrics_path='experiment/metrics.json',
                              recorded_study_role=setup.get('study_role'),
                              recorded_reference_guidance=setup.get('reference_guidance')))
    assert len(snapshots) == 38
    assert len({s['tag'] for s in snapshots}) == len(snapshots)
    assert sum(s['branch'] is not None for s in snapshots) == 11
    assert observed['refs/heads/paper/index'] == previous['index']['commit']
    for s in snapshots:
        for ref in ['refs/tags/' + s['tag']] + (['refs/heads/' + s['branch']] if s['branch'] else []):
            assert ref not in observed, ref
            assert not git('for-each-ref', '--format=%(refname)', ref).strip(), ref
    result = {'schema': 'merlin.paper-ref-organization.v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'previous_index': previous['index'],
              'repository_main': observed['refs/heads/main'],
              'unrelated_remote_heads': {k: v for k, v in observed.items() if k.startswith('refs/heads/') and not k.startswith('refs/heads/paper/')},
              'snapshots': snapshots,
              'scope': 'Rename publication refs and documentation only. No source or measurement edits, no experiment execution, no task cleanup.'}
    (WORK / 'plan.json').write_bytes(encoded(result))
    print('Planned 11 experiment branches, 38 permanent snapshot tags, and an updated index.')


def write_tree(files, parent=None):
    env = dict(os.environ, GIT_INDEX_FILE=str(TASK / '.tmp/ae-organize-index'))
    git('read-tree', parent or '--empty', env=env)
    for rel, data in files.items():
        blob = git('hash-object', '-w', '--stdin', data=data).decode().strip()
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + rel, env=env)
    return git('write-tree', env=env).decode().strip()


def make_tag(snapshot):
    name = git('config', 'user.name').decode().strip()
    email = git('config', 'user.email').decode().strip()
    message = (f"object {snapshot['commit']}\ntype commit\ntag {snapshot['tag']}\n"
               f'tagger {name} <{email}> {int(time.time())} +0000\n\n'
               f"{snapshot['display_name']}\nOriginal record ID: {snapshot['label']}\n"
               f"Compiler SHA256: {snapshot['source_sha256']}\n"
               'Preserves the complete prior snapshot, including setup and measurement records.\n')
    return git('mktag', data=message.encode()).decode().strip()


def branch_docs(s):
    old_readme = git('show', s['commit'] + ':README.md').decode()
    new_readme = '# ' + s['display_name'] + '\n' + old_readme[old_readme.index('\n'):]
    new_readme = new_readme.replace('`' + s['previous_branch'] + '`', '`' + s['branch'] + '`')
    new_readme += (f"\n## Run identity\n\nOriginal result ID: `{s['label']}`. Permanent measured-snapshot tag: `{s['tag']}`. "
                   'This branch adds navigation metadata to that snapshot. Compiler and dependency trees and all saved measurements are unchanged. '
                   'The tag retains the original complete commit.\n\n'
                   'Names containing `4plus4h` mean four hours of seed construction followed by four additional hours of refinement. '
                   'A `30min` suffix is the local study budget after its existing seed, not the cost of constructing the seed.\n\n'
                   f'See [the experiment index]({REMOTE}/tree/paper/index) for all original run IDs, paths, commits and source digests.\n')
    m = json.loads(git('show', s['commit'] + ':experiment/manifest.json'))
    m.update(branch=s['branch'], display_name=s['display_name'],
             original_snapshot_commit=s['commit'], snapshot_tag=s['tag'], previous_branch=s['previous_branch'])
    ident = {k: s[k] for k in ['label', 'display_name', 'branch', 'tag', 'commit', 'source_sha256', 'previous_branch', 'aliases']}
    ident['snapshot_commit'] = ident.pop('commit')
    return {'README.md': new_readme.encode(), 'experiment/manifest.json': encoded(m),
            'experiment/identity.json': encoded(ident)}


LOOKUP = '''#!/usr/bin/env python3
"""Find a published compiler by run ID, path, old branch, tag or source hash."""
import json
from pathlib import Path
import sys

rows = json.loads(Path(__file__).with_name('runs.json').read_text())['snapshots']
query = ' '.join(sys.argv[1:]).strip()
if not query:
    raise SystemExit('Usage: python3 lookup_run.py RUN_ID_OR_PATH_OR_SHA')
def keys(row):
    return row['aliases'] + [row['tag'], row['branch'] or '', row['display_name']]
exact = [r for r in rows if query in keys(r)]
matches = exact or [r for r in rows if any(query.lower() in k.lower() for k in keys(r))]
if not matches:
    raise SystemExit('No matching run. See runs.csv for the complete inventory.')
for row in matches:
    print(json.dumps({k: row.get(k) for k in ['label', 'display_name', 'branch', 'tag',
          'commit', 'branch_commit', 'source_sha256', 'snapshot_url', 'branch_url',
          'recorded_study_role']}, indent=2))
'''


def index_files(p):
    rows = p['snapshots']
    lines = ['# Merlin paper compiler versions', '',
             'Start with the experiment branches below. Every preserved compiler also has a permanent annotated snapshot tag. '
             'Original run IDs and paths remain searchable in [runs.csv](runs.csv) and [runs.json](runs.json). '
             'Names are aliases for existing runs, not new experiments.', '',
             '| Experiment | Branch | Original result ID |', '|---|---|---|']
    for s in rows:
        if s['branch']:
            lines.append(f"| {s['display_name']} | [{s['branch']}]({s['branch_url']}) | `{s['label']}` |")
    lines += ['', '## Find a compiler from a run', '',
              'After checking out this index branch, use the original result ID, a full run path, an old branch name, or a source digest:', '',
              '```sh', 'python3 lookup_run.py CCA-EXO-R3', 'python3 lookup_run.py F10-FULL-NOREF-NOCCA',
              'python3 lookup_run.py followup-cca-h1d-p2r-control', '```', '',
              'The lookup returns the snapshot tag, commit, source digest, and experiment branch when one is maintained. '
              'To retrieve a snapshot without relying on a moving branch:', '',
              '```sh', 'git fetch origin tag mlsys-paper-v1/refinement/exo-kernels-4plus4h',
              'git switch --detach mlsys-paper-v1/refinement/exo-kernels-4plus4h',
              'python3 experiment/verify_snapshot.py', '```', '',
              '## Naming and budgets', '',
              '- `paper/` branches are entry points for the principal construction, refinement, scaffolding, scheduling and transformation conditions.',
              '- `mlsys-paper-v1/` tags preserve all 38 original complete commits, including supporting studies and historical checkpoints.',
              '- `4plus4h` means four hours of construction and four additional hours of refinement, not eight new hours of refinement.',
              '- Thirty- and sixty-minute study budgets exclude the prior work that produced their seed compiler.',
              '- Revision and study numbers identify distinct recorded trials. They do not imply matched conditions or independent replications.',
              '- `seed-scheduler` retains the inherited scheduling support. It does not mean an absence of compiler libraries.', '',
              '## All preserved snapshots', '',
              '| Experiment | Original result ID | Permanent tag | Original commit |', '|---|---|---|---|']
    for s in rows:
        lines.append(f"| {s['display_name']} | `{s['label']}` | [{s['tag']}]({REMOTE}/tree/{s['tag']}) | [`{s['commit'][:12]}`]({s['snapshot_url']}) |")
    lines += ['', '## Evidence and package contents', '',
              'Each snapshot contains the original compiler in `compiler/`, setup in `experiment/setup.json`, '
              'results in `experiment/metrics.json`, and a source-hash verifier in `experiment/verify_snapshot.py`. '
              'Where captured, shared dependencies are retained in `dependencies/`. Principal branches add only navigation documentation and identity metadata. '
              'Compiler, dependency and measurement bytes match their tagged snapshots.', '',
              'The 89-case hardware panel, construction tests and small paired studies have different evaluation scopes. '
              'Measurements from different images and the common plotted FSM reference remain separate. '
              'The later current-library and adapted-FSM controls are attached to the Exo-refined snapshot under `experiment/later-validation/`.', '',
              'Controlled CCA pair 2 reverses the positional queue labels: the old ID ending in `control` is the CCA-on arm, '
              'and the ID ending in `treatment` is CCA-off. The tag names follow the recorded `study_role` and `tooling`, '
              'rather than the queue position. Original IDs and observations are unchanged.', '',
              'This is a verified source-and-record archive, not a fully reconstructed execution environment. '
              'Four historical snapshots retain vendored Z3 shared libraries, and two early checkpoint model fields remain unresolved. '
              'Framework exports, runtime, toolchains, evaluators and hardware dependencies still require full artifact closure. '
              'No compiler, agent, simulator or FPGA experiment was rerun for this organization.', '',
              'Snapshot commits descend from a repository landing commit for storage. This is not the agent development history. '
              'No repository main branch or unrelated branch was modified. The separate `mlsys27/test-generation-coverage` branch is outside this reorganization.', '']
    stream = io.StringIO()
    fields = ['label', 'display_name', 'branch', 'tag', 'commit', 'branch_commit', 'source_sha256', 'previous_branch', 'source', 'snapshot_url']
    writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    registry = {'schema': 'merlin.paper-compiler-run-index.v2', 'snapshots': rows}
    return {'README.md': '\n'.join(lines).encode(), 'runs.csv': stream.getvalue().encode(),
            'runs.json': encoded(registry), 'branches.json': encoded(registry),
            'lookup_run.py': LOOKUP.encode(),
            'tools/organize_compiler_refs.py': Path(__file__).read_bytes(),
            'tools/prepare_compiler_branches.py': Path(__file__).with_name('prepare_compiler_branches.py').read_bytes()}


def apply():
    p = read(WORK / 'plan.json')
    if (WORK / 'prepared.json').exists():
        raise ValueError('Prepared already. Inspect the existing plan.')
    updates = []
    for s in p['snapshots']:
        s['tag_object'] = make_tag(s)
        updates.append(f"create refs/tags/{s['tag']} {s['tag_object']}")
        s['snapshot_url'] = REMOTE + '/tree/' + s['commit']
        s['branch_commit'] = None
        s['branch_url'] = None
        if s['branch']:
            tree = write_tree(branch_docs(s), parent=s['commit'])
            commit = git('commit-tree', tree, '-p', s['commit'], data=b'docs(experiment): identify published condition and original run\n').decode().strip()
            # Refuse changes beyond the intended navigation metadata.
            changed = set(git('diff', '--name-only', s['commit'], commit).decode().splitlines())
            assert changed <= {'README.md', 'experiment/manifest.json', 'experiment/identity.json'}
            for subtree in ['compiler', 'dependencies']:
                a = git('ls-tree', s['commit'], subtree)
                b = git('ls-tree', commit, subtree)
                assert a == b, (s['label'], subtree)
            assert git('show', s['commit'] + ':experiment/metrics.json') == git('show', commit + ':experiment/metrics.json')
            s.update(branch_commit=commit, branch_url=REMOTE + '/tree/' + s['branch'])
            updates.append(f"create refs/heads/{s['branch']} {commit}")
    old_index = p['previous_index']['commit']
    tree = write_tree(index_files(p))
    index = git('commit-tree', tree, '-p', old_index, data=b'docs(artifacts): organize paper compilers by experiment and preserve snapshot tags\n').decode().strip()
    updates.append(f'update refs/heads/paper/index {index} {old_index}')
    git('update-ref', '--stdin', data=('start\n' + '\n'.join(updates) + '\nprepare\ncommit\n').encode())
    p['index'] = {'branch': 'paper/index', 'commit': index, 'url': REMOTE + '/tree/paper/index'}
    (WORK / 'prepared.json').write_bytes(encoded(p))
    bundle = OUT / 'gemmini-mlir-paper-compilers-20261105.bundle'
    refs = ['refs/heads/paper/index'] + ['refs/heads/' + s['branch'] for s in p['snapshots'] if s['branch']] + ['refs/tags/' + s['tag'] for s in p['snapshots']]
    git('bundle', 'create', str(bundle), *refs)
    git('bundle', 'verify', str(bundle))
    p['backup'] = {'path': str(bundle), 'bytes': bundle.stat().st_size,
                   'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(), 'verified': True}
    (WORK / 'prepared.json').write_bytes(encoded(p))
    print('Prepared new refs and verified their backup. Remote branches have not yet changed.')


def push():
    p = read(WORK / 'prepared.json')
    before = remote_refs()
    assert before['refs/heads/paper/index'] == p['previous_index']['commit']
    refs, leases = [], []
    for s in p['snapshots']:
        old = 'refs/heads/' + s['previous_branch']
        assert before[old] == s['commit'], old
        refs.append('refs/tags/' + s['tag'] + ':refs/tags/' + s['tag'])
        leases.append('--force-with-lease=refs/tags/' + s['tag'] + ':')
        if s['branch']:
            refs.append('refs/heads/' + s['branch'] + ':refs/heads/' + s['branch'])
            leases.append('--force-with-lease=refs/heads/' + s['branch'] + ':')
    refs.append('refs/heads/paper/index:refs/heads/paper/index')
    leases.append('--force-with-lease=refs/heads/paper/index:' + p['previous_index']['commit'])
    # First establish and verify all successors. Never remove the only remote ref.
    git('push', '--atomic', *leases, 'origin', *refs)
    after = remote_refs()
    for s in p['snapshots']:
        assert after['refs/tags/' + s['tag']] == s['tag_object']
        assert after['refs/tags/' + s['tag'] + '^{}'] == s['commit']
        if s['branch']:
            assert after['refs/heads/' + s['branch']] == s['branch_commit']
    assert after['refs/heads/paper/index'] == p['index']['commit']
    deletion = [':refs/heads/' + s['previous_branch'] for s in p['snapshots']]
    leases = ['--force-with-lease=refs/heads/' + s['previous_branch'] + ':' + s['commit'] for s in p['snapshots']]
    git('push', '--atomic', *leases, 'origin', *deletion)
    final = remote_refs()
    for s in p['snapshots']:
        assert 'refs/heads/' + s['previous_branch'] not in final
        assert final['refs/tags/' + s['tag'] + '^{}'] == s['commit']
        if s['branch']:
            assert final['refs/heads/' + s['branch']] == s['branch_commit']
    assert final['refs/heads/main'] == p['repository_main']
    assert all(final[k] == v for k, v in p['unrelated_remote_heads'].items())
    remaining = {k for k in final if k.startswith('refs/heads/paper/')}
    expected = {'refs/heads/paper/index'} | {'refs/heads/' + s['branch'] for s in p['snapshots'] if s['branch']}
    assert remaining == expected
    git('update-ref', '--stdin', data=('start\n' + '\n'.join(
        f"delete refs/heads/{s['previous_branch']} {s['commit']}" for s in p['snapshots']) + '\nprepare\ncommit\n').encode())
    p.update(pushed=True, old_branches_retired=True, all_remote_refs_verified=True,
             completed_utc=datetime.now(timezone.utc).isoformat())
    (WORK / 'organization.json').write_bytes(encoded(p))
    (OUT / 'organization.json').write_bytes(encoded(p))
    registry = {'schema': 'merlin.paper-compiler-publication.v2', 'repository': REMOTE,
                'organization': str(OUT / 'organization.json'), 'index': p['index'],
                'branches': [dict(s, commit=s['branch_commit'], snapshot_commit=s['commit']) for s in p['snapshots'] if s['branch']],
                'snapshots': p['snapshots'], 'pushed': True, 'backup_covers_current_refs': True, 'backup': p['backup']}
    (OUT / 'branches.json').write_bytes(encoded(registry))
    print('Published and verified 11 experiment branches, 38 annotated tags and the index. Retired 38 old branch names.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply', 'push'])
    args = parser.parse_args()
    {'plan': plan, 'apply': apply, 'push': push}[args.action]()
