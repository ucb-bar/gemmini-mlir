from pathlib import Path
import json,shutil,subprocess,hashlib
from mlir_oot.no_fsm_audit import audit_elf
from merlin.llvmlower.coarse_absolute_upper import prepare_coarse_absolute_upper, CoarseAbsoluteUpperPlan
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/encoded_row_equality');core=Path('/scratch/agustin/tmp/merlin-coarse-absolute-upper-20261006')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name in ['control_numeric','numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir(exist_ok=False);source=old/('native_numeric_frozen'if name=='native_numeric_frozen'else'numeric_frozen')
 for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
 shutil.copyfile(core/'merlin/runtime/c/absolute_upper_dot_bounds.h',d/'absolute_upper_dot_bounds.h')
 s=(source/'provider.c').read_text();(d/'provider.c').write_text(s if name=='control_numeric'else prepare_coarse_absolute_upper(s, CoarseAbsoluteUpperPlan(6)))
 cmds=json.loads((source/'compile.json').read_text())['commands'];actual=[]
 for cmd in cmds:
  cmd=[x.replace(str(source),str(d))for x in cmd];subprocess.run(cmd,check=True);actual.append(cmd)
 (d/'compile.json').write_text(json.dumps({'commands':actual,'pins':{str(p):sha(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
assert sha(w/'control_numeric/provider.o')==sha(old/'numeric_frozen/provider.o')
for arm in ['control','candidate']:
 d=w/arm;d.mkdir(exist_ok=False);link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(w/('numeric_frozen'if arm=='candidate'else'control_numeric')/'provider.o')if x==str(old/'numeric_frozen/provider.o')else x for x in link];subprocess.run(link,check=True)
 if arm=='control':assert sha(d/'model.elf')==sha(old/'candidate/model.elf')
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(d/'model.elf'),'scope':'Exact encoded-row complete source-consumer group. Candidate adds coarse positive absolute upper products only; actual dynamic workspace query/allocation is inside ROI.'},indent=2)+'\n')
 if arm=='control':continue
 shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
 with(d/'watch.log').open('w')as log:process=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (d/'process.json').write_text(json.dumps({'pid':process.pid}));print('watch',arm,process.pid,sha(d/'model.elf'),flush=True)
