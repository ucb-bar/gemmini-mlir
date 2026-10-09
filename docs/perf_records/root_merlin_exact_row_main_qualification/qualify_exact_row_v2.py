"""Qualify the generic typed observer on main and from its installed wheel."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/exact_row_qualification_v2'
OUTSIDE = Path('/scratch/agustin/tmp/merlin-exact-row-installed-v2-20261007')
WORK = OUTSIDE / 'work'
OUT.mkdir(exist_ok=False)
WORK.mkdir(parents=True, exist_ok=True)
UV = '/home/agustin/.local/bin/uv'
PYTHON = '/scratch/agustin/projects/oscar-merlin/.venv/bin/python'
environment = os.environ.copy()
for key in ['PYTHONPATH', 'MERLIN_REPO_ROOT', 'MERLIN_OUT_ROOT']:
    environment.pop(key, None)
environment.update(MERLIN_CLANG='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang',
    MERLIN_MLIR_INSTALL='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install',
    MERLIN_COMPILER_PYTHON='/scratch/agustin/projects/model2MLIR/.venv/bin/python')
commands = []

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(label, argv, cwd=WORK, env=None):
    log = OUT / (label + '.log')
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, argv)), cwd=cwd, env=environment if env is None else env,
            stdout=stream, stderr=subprocess.STDOUT)
    commands.append(dict(label=label, argv=list(map(str, argv)), cwd=str(cwd),
        exit_code=result.returncode, log=str(log), log_sha256=sha(log)))
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    assert result.returncode == 0, label

tests = ['test_exact_row_radix_pack.py', 'test_fused_encoded_witness.py']
source_env = dict(environment, PYTHONPATH=str(ROOT/'src'))
run('source_tests', [PYTHON, '-m', 'pytest', '-q',
    '--basetemp='+str(OUT/'source_tmp'), *[ROOT/'merlin/tests/runtime'/name for name in tests]], ROOT, source_env)
assert '44 passed' in (OUT/'source_tests.log').read_text()
run('build_wheel', [UV, 'build', '--wheel', '--out-dir', OUT/'dist'], ROOT)
run('create_venv', [UV, 'venv', '--python', PYTHON, OUTSIDE/'venv'])
installed = OUTSIDE/'venv/bin/python'
site = Path(subprocess.check_output([installed, '-c',
    'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())
deps = subprocess.check_output([PYTHON, '-c',
    'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip()
(site/'qualification_dependencies.pth').write_text(deps+'\n')
wheel, = (OUT/'dist').glob('*.whl')
run('install_wheel', [UV, 'pip', 'install', '--no-deps', '--python', installed, wheel])
modules = ['merlin/llvmlower/exact_row_radix_pack.py', 'merlin/llvmlower/fused_encoded_witness.py', 'merlin/common/paths.py']
module_pins, resources = {}, {}
with zipfile.ZipFile(wheel) as archive:
    for name in modules:
        source = ROOT/'src'/name
        assert archive.read(name) == source.read_bytes() == (site/name).read_bytes()
        module_pins[name] = sha(source)
    for source in sorted((ROOT/'merlin/runtime/c').glob('*.h')):
        name = 'merlin/_data/runtime/c/'+source.name
        assert archive.read(name) == source.read_bytes() == (site/name).read_bytes()
        resources[name] = sha(source)
for name in tests:
    shutil.copyfile(ROOT/'merlin/tests/runtime'/name, WORK/name)
    assert (ROOT/'merlin/tests/runtime'/name).read_bytes() == (WORK/name).read_bytes()
shutil.copyfile(ROOT/'merlin/tests/runtime/test_source_attention_frontier.py', WORK/'test_source_attention_frontier.py')
assert (ROOT/'merlin/tests/runtime/test_source_attention_frontier.py').read_bytes()==(WORK/'test_source_attention_frontier.py').read_bytes()
runner = WORK/'run_installed.py'
runner.write_text('''import importlib,json,pathlib,sys
import pytest
site=pathlib.Path(sys.argv[1]).resolve()
for name in ["merlin.llvmlower.exact_row_radix_pack","merlin.llvmlower.fused_encoded_witness"]:
    assert pathlib.Path(importlib.import_module(name).__file__).resolve().is_relative_to(site)
from merlin.common.paths import checkout_root,data_path
assert checkout_root() is None
assert data_path("runtime","c").resolve().is_relative_to(site)
code=pytest.main(sys.argv[2:])
origins={}
for name,module in list(sys.modules.items()):
    if (name=="merlin" or name.startswith("merlin.")) and getattr(module,"__file__",None):
        path=pathlib.Path(module.__file__).resolve()
        assert path.is_relative_to(site),(name,path)
        origins[name]=str(path)
pathlib.Path("import_origins.json").write_text(json.dumps(origins,indent=2)+"\\n")
raise SystemExit(code)
''')
run('installed_tests', [installed, '-P', runner, site, '-q',
    '--basetemp='+str(WORK/'test_tmp'), *[WORK/name for name in tests]])
assert '44 passed' in (OUT/'installed_tests.log').read_text()
receipt = dict(schema='exact_row_radix_source_wheel_installed_qualification_v1',
    status='pass', source_parent='c0f40f8f8d100b841c26fe8bbd6c09e79b6de17c',
    reviewed_original_topic='609a3be07',
    source_tests=44, installed_tests=44, wheel=str(wheel), wheel_sha256=sha(wheel),
    source_wheel_installed_module_identity=module_pins, bundled_header_identity=resources,
    tests_byte_identical=True, fixture_byte_identical=True, prior_harness_failures_preserved=str(ROOT/'out/exact_row_qualification'), installed_test_resource_fix='Existing fused encoder tests now resolve bundled headers through data_path instead of cwd-relative merlin_dir', installed_import_origins=str(WORK/'import_origins.json'),
    commands=commands, scope='Generic explicit source-exact complete-row canonical radix128 packing/widening proof. Mandatory finite BF16 scan, source-normal exponent span, exact unsaturated grid, original numerical/effect/rounding and default fallback unchanged. Exhaustive BF16, strided/tail/refusal/UBSan cases plus existing canonical witness gate. No target, measured/golden selector, automatic profit policy or hardware/whole cycle claim',
    token_usage_available=False)
(OUT/'qualification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(status='pass', source_tests=44, installed_tests=44, modules=len(modules), installed_headers=len(resources))), flush=True)
