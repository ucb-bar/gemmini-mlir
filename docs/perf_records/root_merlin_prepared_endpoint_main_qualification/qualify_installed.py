from pathlib import Path
import hashlib, json, shutil, subprocess, os, zipfile

ROOT = Path('/scratch/agustin/tmp/merlin-prepared-endpoint-main-20261007')
WORK = Path(__file__).resolve().parent
WHEEL = Path('/scratch/agustin/tmp/merlin-prepared-endpoint-dag-20261007/out/artifacts/endpoint-delivery/merlin-0.0.1-py3-none-any.whl')
SITE = WORK / 'site'
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

assert sha(WHEEL) == '64ee4ad0697d2db2b0dede7143271702f8ff25fc08aa51483ce8a2a4f455fb84'
pins = {str(WHEEL): sha(WHEEL)}
with zipfile.ZipFile(WHEEL) as z:
    modules = sorted((ROOT / 'src/merlin').rglob('*.py'))
    headers = sorted((ROOT / 'merlin/runtime/c').glob('*.h'))
    for path in modules + headers:
        member = ('merlin/' + str(path.relative_to(ROOT / 'src/merlin'))) if path in modules else ('merlin/_data/runtime/c/' + path.name)
        assert path.read_bytes() == z.read(member) == (SITE / member).read_bytes(), path
        pins[str(path)] = sha(path)
        pins[str(SITE / member)] = sha(SITE / member)
    assert len(modules) == 1014 and len(headers) == 30
outside = WORK / 'outside'
outside.mkdir(exist_ok=False)
test = ROOT / 'merlin/tests/runtime/test_prepared_endpoint_dag.py'
shutil.copy2(test, outside / test.name)
env = dict(os.environ, PYTHONPATH=str(SITE), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
python = '/scratch/agustin/projects/oscar-merlin/.venv/bin/python'
probe = subprocess.run([python, '-c', 'import merlin.llvmlower.prepared_endpoint_dag as p;print(p.__file__)'], env=env, cwd=outside, text=True, capture_output=True, check=True)
assert probe.stdout.strip() == str(SITE / 'merlin/llvmlower/prepared_endpoint_dag.py')
(WORK / 'installed_import.log').write_text(probe.stdout)
with (WORK / 'installed_tests.log').open('w') as f:
    r = subprocess.run([python, '-m', 'pytest', '-q', str(outside / test.name), '--basetemp', str(WORK / 'pytest-installed')], env=env, cwd=outside, stdout=f, stderr=subprocess.STDOUT, timeout=120)
assert r.returncode == 0
assert '12 passed' in (WORK / 'installed_tests.log').read_text()
for p in [test, outside / test.name, Path(__file__), WORK / 'install_wheel.log', WORK / 'installed_tests.log', WORK / 'installed_import.log', WORK / 'source_tests.log', WORK / 'no_target.log', WORK / 'no_regex.log']:
    pins[str(p)] = sha(p)
head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
record = dict(schema='root_prepared_endpoint_main_package_qualification_v1', status='PASS', core_commit=head,
    source_tests=12, installed_tests=12, python_modules_byte_identical=1014, runtime_headers_byte_identical=30,
    installed_import=probe.stdout.strip(), installed_from_verified_wheel=True, default_opt_in=True,
    semantics='Private immutable complete row epochs; invalidation precedes refinement, checked fallback retained, original source-order evaluation, no new numerical permission.',
    ownership='Reusable host interval preparation in Merlin; target binding and ISA in OOT.', pins=pins)
(WORK / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k != 'pins'}))
