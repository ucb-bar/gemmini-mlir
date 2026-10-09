"""Qualify the generic typed observer on main and from its installed wheel."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'out/artifacts/probes/polynomial-constants-main-qualification/results'
OUTSIDE = Path('/scratch/agustin/tmp/merlin-polynomial-constants-installed-20261007')
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

tests = [('ir','test_prepared_polynomial_constants.py'),('runtime','test_rounded_polynomial_monotonicity.py')]
source_env = dict(environment, PYTHONPATH=str(ROOT/'src'))
run('source_tests', [PYTHON, '-m', 'pytest', '-q',
    '--basetemp='+str(OUT/'source_tmp'), *[ROOT/'merlin/tests'/folder/name for folder,name in tests]], ROOT, source_env)
assert '30 passed' in (OUT/'source_tests.log').read_text()
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
modules = ['merlin/llvmlower/prepared_polynomial_constants.py','merlin/llvmlower/rounded_polynomial_monotonicity.py','merlin/llvmlower/source_numeric_capability.py','merlin/common/paths.py']
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
for folder,name in tests:
    shutil.copyfile(ROOT/'merlin/tests'/folder/name, WORK/name)
    assert (ROOT/'merlin/tests'/folder/name).read_bytes() == (WORK/name).read_bytes()
runner = WORK/'run_installed.py'
runner.write_text('''import importlib,json,pathlib,sys
import pytest
site=pathlib.Path(sys.argv[1]).resolve()
for name in ["merlin.llvmlower.prepared_polynomial_constants","merlin.llvmlower.rounded_polynomial_monotonicity","merlin.llvmlower.source_numeric_capability"]:
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
    '--basetemp='+str(WORK/'test_tmp'), *[WORK/name for _,name in tests]])
assert '30 passed' in (OUT/'installed_tests.log').read_text()
receipt = dict(schema='prepared_polynomial_constants_source_wheel_installed_qualification_v1',status='pass',
    source_parent='e9194ac080c0eb0a006e677284dea68249378481',reviewed_original_topic='8b73513b721484fd34a03b60cc1c83d529f0a9d7',
    source_tests=30,installed_tests=30,wheel=str(wheel),wheel_sha256=sha(wheel),
    source_wheel_installed_module_identity=module_pins,bundled_header_identity=resources,tests_byte_identical=True,
    installed_import_origins=str(WORK/'import_origins.json'),commands=commands,
    scope='Explicit private immutable prepared polynomial context; complete rounded source theorem and effect contract, exact eight plan words/RNE/zero implementation budget. Both distinct input endpoints/source order and F32 denominator remain unchanged. Unsupported context retains checked helper. No numerical/default/target policy promotion or whole hardware claim',
    formatting_only_against_original_topic=True,token_usage_available=False)
(OUT/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='pass',source_tests=30,installed_tests=30,modules=len(modules),installed_headers=len(resources))),flush=True)
