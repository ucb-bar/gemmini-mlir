"""Explicit support discovery and command pins refuse changed inputs before execution."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / 'support/gemmini_gsim'


def child(body):
    env = dict(os.environ, MERLIN_TARGET_PATH=str(PROVIDER))
    result = subprocess.run([sys.executable, '-c', body], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_is_explicit_and_does_not_import_legacy_backend():
    child('''import sys
from merlin.runtime.backends.base import get_backend
b=get_backend('gemmini')
assert b.__file__.endswith('/support/gemmini_gsim/backend.py')
assert not any('gemmini_codegen' in n or 'gemmini_layer_bench' in n for n in sys.modules)
''')


def test_recipe_refuses_missing_caller_configuration():
    child('''import os
from merlin.runtime.backends.base import get_backend
b=get_backend('gemmini');os.environ.pop('MERLIN_RISCV_GCC',None)
try:b.harness_build_recipe()
except ValueError as e:assert 'MERLIN_RISCV_GCC' in str(e)
else:raise AssertionError('missing compiler accepted')
''')


BASE = '''import tempfile,json,os
from pathlib import Path
from dataclasses import replace
from merlin.runtime.backends.base import get_backend
b=get_backend('gemmini');tmp=tempfile.TemporaryDirectory();root=Path(tmp.name)
elf=root/'probe.elf';elf.write_bytes(b'ELF input')
emu=root/'engine';emu.write_bytes(b'engine');emu.chmod(0o755)
receipt=root/'build_receipt.json';receipt.write_text('{}')
engine={'available':True,'refused':False,'flavour':'binary','receipt_status':'bound','path':str(emu),'binary_sha256':b._sha(emu),'receipt':{'receipt_path':str(receipt),'receipt_sha256':b._sha(receipt)}}
b._engine=lambda:engine
'''


@pytest.mark.parametrize('cycles', ['None','True','0','-1','1<<63','1.5'])
def test_command_requires_explicit_integer_cycle_bound(cycles):
    child(BASE+f'''
try:b.prepare_gsim_command(elf,expected_elf_sha256=b._sha(elf),expected_engine_provenance=engine,max_cycles={cycles})
except ValueError as e:assert 'max_cycles' in str(e)
else:raise AssertionError('invalid bound accepted')
''')


@pytest.mark.parametrize('tamper', ["elf.write_bytes(b'changed')", "receipt.write_text('changed')", "engine['path']='other'", "cmd=replace(cmd,argv=cmd.argv+('other',))", "cmd=replace(cmd,env_overrides=(('UNBOUND','1'),))"])
def test_prepared_command_revalidates_inputs_and_exact_argv(tamper):
    child(BASE+'''
cmd=b.prepare_gsim_command(elf,expected_elf_sha256=b._sha(elf),expected_engine_provenance=engine,max_cycles=123)
assert cmd.revalidate()['status']=='unchanged'
'''+tamper+'''
try:cmd.revalidate()
except ValueError:pass
else:raise AssertionError('changed command inputs accepted')
''')


def test_unbound_engine_and_wrong_expected_elf_are_refused():
    child(BASE+'''
for bad_digest,status in [('0'*64,'bound'),(b._sha(elf),'absent')]:
 engine['receipt_status']=status
 try:b.prepare_gsim_command(elf,expected_elf_sha256=bad_digest,expected_engine_provenance=engine,max_cycles=123)
 except ValueError:pass
 else:raise AssertionError('unbound input accepted')
''')


def test_no_executable_discovery_without_explicit_selection():
    child("""import os
os.environ.pop('MERLIN_TARGET_PATH',None)
from merlin.runtime.backends.base import get_backend
try:get_backend('gemmini')
except (KeyError,ValueError):pass
else:raise AssertionError('backend implicitly selected')
""")
