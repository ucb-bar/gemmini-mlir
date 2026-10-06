"""Screen a source-proved fractional polynomial table, keeping all device work fixed."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from mlir_oot.no_fsm_audit import audit_elf

root=Path.cwd();work=root/'out/artifacts/probes/smol-polynomial-table-20261006'
control=root/'out/artifacts/probes/smol-minmax-20261006/core_selected'
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
arm=work/'table10';arm.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
recipe=json.loads((control/'compile.json').read_text())
for path,digest in recipe['pins'].items():assert sha(path)==digest,path
for path in control.iterdir():
 if path.suffix in('.h','.c'):shutil.copy2(path,arm/path.name)
shutil.copy2(core/'merlin/runtime/c/monotone_polynomial_table.h',arm/'monotone_polynomial_table.h')
source=(arm/'provider.c').read_text()
source=source.replace('#include "monotone_bit_polynomial.h"','#include "monotone_bit_polynomial.h"\n#include "monotone_polynomial_table.h"')
needle='if(!(root_prepared.fast_valid))return 0;'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n int32_t polynomial_knots[1025];merlin_monotone_polynomial_table root_table=merlin_monotone_polynomial_table_prepare(&root_prepared,polynomial_knots,1025,10);')
old='merlin_monotone_bit_polynomial_apply(x,&root_prepared)';assert source.count(old)==1
source=source.replace(old,'merlin_monotone_polynomial_table_apply(x,&root_table)')
(arm/'provider.c').write_text(source)
commands=[[a.replace(str(control),str(arm))for a in command]for command in recipe['commands']]
for command in commands:subprocess.run(command,check=True)
old=root/'out/artifacts/probes/smol-word-enclosure-20261006/consumer_capsules_ordered/control'
previous=root/'out/artifacts/probes/smol-minmax-20261006/core_selected/provider.o'
link=json.loads((old/'build.json').read_text())['link']
link=[a.replace(str(old/'model.elf'),str(arm/'model.elf')).replace(str(previous),str(arm/'provider.o'))for a in link]
subprocess.run(link,check=True)
audit=audit_elf((arm/'model.elf').read_bytes());assert audit['status']=='pass'
deps=(arm/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
pins={str(Path(p).resolve()):sha(p)for p in[*deps,*link,*commands[0],*commands[1],str(work/'prepare.py')]if Path(p).is_file()}
(arm/'build.json').write_text(json.dumps({'scope':'One explicit1024-bin fractional table with curvature/rounding proof. Complete original group; original compiled consumer and goldens unchanged; instruction proxy only. Private immutable knot lifetime is one soft_details call; no mutable external cache.','commands':commands,'link':link,'pins':pins,'nofsm':audit,'default_policy_changed':False,'source_accuracy_gate_changed':False,'table_bytes':4100,'hardware_cycles':None},indent=2)+'\n')
shutil.copy2(old/'watch.py',arm/'watch.py')
print(json.dumps({'elf_sha256':sha(arm/'model.elf'),'nofsm':audit['status']}))
