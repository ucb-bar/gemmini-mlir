"""Preserve property fixture; prevent pure-check hoisting in paired cost loops."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from mlir_oot.no_fsm_audit import audit_elf

work = Path(__file__).parent
old = work / 'machine_counter'
arm = work / 'aligned_matched'
arm.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
recipe = json.loads((old / 'build.json').read_text())
for path, digest in recipe['pins'].items():
    assert sha(path) == digest, path
core = Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
shutil.copy2(core / 'merlin/runtime/c/benchmark_buffer.h', arm / 'benchmark_buffer.h')
shutil.copy2(old / 'native_fixture.py', arm / 'native_fixture.py')
source = (old / 'check.c').read_text()
for name in ('byte', 'word'):
    before = f'for(int i=0;i<16;i++)assert({name}_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);'
    after = f'for(int i=0;i<16;i++){{__asm__ volatile("":::"memory");assert({name}_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);}}'
    assert source.count(before) == 1
    source = source.replace(before, after)
(arm / 'check.c').write_text(source)
command = [a.replace(str(old), str(arm)) for a in recipe['compile']]
link = [a.replace(str(old), str(arm)) for a in recipe['link']]
subprocess.run(command, check=True)
subprocess.run(link, check=True)
audit = audit_elf((arm / 'check.elf').read_bytes())
assert audit['status'] == 'pass'
deps = (arm / 'check.d').read_text().replace('\\\n', ' ').split(':', 1)[1].split()
pins = {str(Path(p).resolve()): sha(p) for p in
        [*command, *link, *deps, str(work/'prepare_matched.py'), str(arm/'native_fixture.py')]
        if Path(p).is_file()}
(arm / 'build.json').write_text(json.dumps({
    'compile': command, 'link': link, 'pins': pins, 'nofsm': audit,
    'scope': 'Exact checker/fill full target property fixture;16 actual read checks each arm, source memory compiler barrier prevents hoisting. Functional instruction proxy only.',
    'previous_cost_rejected': 'Compiler hoisted15 of16 byte checks; word helper lost alignment after outlining. Prior property PASS retained, prior cost is not a paired measurement.',
}, indent=2) + '\n')
shutil.copy2(old / 'watch.py', arm / 'watch.py')
print(json.dumps({'elf_sha256': sha(arm/'check.elf'), 'nofsm': audit['status']}))
