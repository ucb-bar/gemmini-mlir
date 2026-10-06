"""Preserve the failed counter probe; qualify machine CSR on the same checker."""
from pathlib import Path
import hashlib,json,shutil,subprocess
from mlir_oot.no_fsm_audit import audit_elf
work=Path(__file__).parent;arm=work/'machine_counter';arm.mkdir(exist_ok=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name in('benchmark_buffer.h','native_fixture.py'):shutil.copy2(work/name,arm/name)
source=(work/'check.c').read_text().replace('rdinstret %0','csrr %0,mcycle')
assert source!=(work/'check.c').read_text()
(arm/'check.c').write_text(source)
receipt=json.loads((work/'build.json').read_text())
command=[arg.replace(str(work),str(arm))for arg in receipt['compile']]
subprocess.run(command,check=True)
link=[arg.replace(str(work/'check.o'),str(arm/'check.o')).replace(str(work/'check.elf'),str(arm/'check.elf'))for arg in receipt['link']]
subprocess.run(link,check=True)
audit=audit_elf((arm/'check.elf').read_bytes());assert audit['status']=='pass'
deps=(arm/'check.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
receipt.update(compile=command,link=link,pins={str(Path(path).resolve()):sha(path)for path in[*command,*link,*deps,str(arm/'native_fixture.py'),str(work/'prepare_counter_fix.py')]if Path(path).is_file()},nofsm=audit,counter_scope='Functional Spike mcycle instruction proxy only; no hardware timing.',prior_counter_failure='User instret CSR unavailable in forced RV64GC without counter extension. Minimal markers pass aligned/unaligned checks and fail precisely at rdinstret; prior log/ELF retained unchanged.')
(arm/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
watch=(work/'watch.py').read_text().replace('code==0and','code == 0 and ').replace('readchecker retiredinstructions only, notFireSimcycles','Functional Spike mcycle instruction proxy, notFireSimcycles')
(arm/'watch.py').write_text(watch)
print(json.dumps({'elf':sha(arm/'check.elf'),'nofsm':audit['status']}))
