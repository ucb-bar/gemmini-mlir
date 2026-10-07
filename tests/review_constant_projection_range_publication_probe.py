"""Reclose the constant-weight proof delivery before its authorized merge."""

import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from review_closed_i8_stock_capsule_probe import require, sha


def review():
    root = Path(__file__).resolve().parents[1]
    core = Path('/scratch/agustin/tmp/merlin-constant-projection-range-upstream-20261007')
    delivery = core / 'out/artifacts/delivery/constant-projection-range-currentmain-20261007'
    q = json.loads((delivery / 'delivery.json').read_text())
    plan = json.loads((delivery / 'publication_plan.json').read_text())
    receipt = json.loads((delivery / 'pull_request.json').read_text())
    output = root / 'docs/perf_records/root_merlin_constant_projection_PR52_review_20261007.json'
    require(not output.exists(), 'fresh review receipt required')
    for path, digest in q['pins'].items():
        require(sha(path) == digest, 'changed qualified delivery: ' + path)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(core), *args], text=True).strip()

    require(git('rev-parse', 'HEAD') == q['head'] == plan['head'] == receipt['head']['sha'], 'head differs')
    require(git('rev-parse', 'HEAD^') == q['base'] == plan['base'], 'single current-main topic required')
    changed = git('diff', '--name-only', q['base'], q['head']).splitlines()
    require(set(changed) == {'src/merlin/llvmlower/integer_producer_range.py',
                            'merlin/tests/ir/test_constant_integer_producer_range.py'}, 'unexpected topic files')
    wheel = delivery / 'wheel/merlin-0.0.1-py3-none-any.whl'
    modules = data = native = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if '.dist-info/' in name or name.endswith('/'):
                continue
            source = (core / 'merlin' / name.removeprefix('merlin/_data/')) if name.startswith('merlin/_data/') else core / 'src' / name
            payload = archive.read(name)
            require(source.read_bytes() == payload == (delivery / 'installed' / name).read_bytes(),
                    'source/wheel/install mismatch: ' + name)
            if name.endswith('.py'):
                modules += 1
            elif name.startswith('merlin/_data/'):
                data += 1
            else:
                native += 1
    require(modules == q['source_wheel_install_identical_modules'] == 974, 'module extent differs')
    require(q['source31_tests'] == q['installed_outside_checkout31_tests'] == 31 and
            '31 passed' in (delivery / 'source31_tests.log').read_text() and
            '31 passed' in (delivery / 'installed31_tests.log').read_text(), 'qualification tests differ')
    original = subprocess.check_output(['git', '-C', str(core), 'show', q['base'] + ':src/merlin/llvmlower/integer_producer_range.py'])
    current = (core / 'src/merlin/llvmlower/integer_producer_range.py').read_bytes()
    require(len(original) == q['existing_implementation_prefix_unchanged_bytes'] and current.startswith(original),
            'existing integer proof was changed')
    credential = subprocess.run(['git', '-C', str(core), 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True, check=True,
        timeout=20, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    request = urllib.request.Request('https://api.github.com/repos/ucb-bar/merlin/pulls/52',
        headers={'Authorization': 'Bearer ' + fields['password'], 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        pr = json.load(response)
    remote = dict(reversed(line.split()) for line in git('ls-remote', 'origin', 'refs/heads/main',
                  'refs/heads/' + plan['branch']).splitlines())
    require(pr['state'] == 'open' and not pr['merged'] and pr['head']['sha'] == q['head'] and
            pr['base']['ref'] == 'main' and remote['refs/heads/main'] == q['base'] and
            remote['refs/heads/' + plan['branch']] == q['head'] and
            hashlib.sha256(pr['body'].encode()).hexdigest() == plan['body_sha256'] == sha(delivery / 'pr_body.md'),
            'actual publication differs')
    result = {
        'schema': 'root_constant_projection_range_PR52_premerge_review_v1',
        'status': 'SOURCE_WHEEL_INSTALLED_REMOTE_REVIEWED', 'PR': pr['html_url'],
        'head': q['head'], 'base_main': q['base'], 'changed_files': changed,
        'delivery_pins_reclosed': len(q['pins']), 'source_wheel_install_modules_exact': modules,
        'packaged_data_exact': data, 'bundled_native_exact': native,
        'source_tests': 31, 'outside_installed_tests': 31,
        'binding_obligations': q['binding_requirements'],
        'production_route_enabled': False, 'target_or_workload_selector': False,
        'scope': 'Immutable integer coefficients, signed ordered prefix bounds and inclusive exact binary conversion domain; no floating reassociation.',
        'authorization': 'User authorized finishing and merging existing PRs; future PR creation requires explicit approval.',
        'pins': {str(p.resolve()): sha(p) for p in
                 (delivery / 'delivery.json', delivery / 'publication_plan.json', delivery / 'pull_request.json',
                  delivery / 'pr_body.md', wheel, Path(__file__))},
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'PR', 'head', 'source_wheel_install_modules_exact', 'source_tests')}))


if __name__ == '__main__':
    review()
