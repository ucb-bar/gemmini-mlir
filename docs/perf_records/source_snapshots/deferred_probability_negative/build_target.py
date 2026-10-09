from pathlib import Path
import json,hashlib,subprocess,shutil
from merlin.llvmlower.deferred_probability import defer_probability_certification
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parent;old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=json.loads((w/'native_group.json').read_text());assert native['candidate']['status']==1 and native['candidate']['original_i8_mismatches']==0 and native['candidate']['original_scale_mismatches']==0
recipe=json.loads((old/'numeric_frozen/compile.json').read_text())
for p,h in recipe['pins'].items():assert sha(p)==h,p
for arm in ['control','candidate']:
 d=w/(arm+'_target');d.mkdir(exist_ok=True);source=old/'numeric_frozen'
 for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
 text=(source/'provider.c').read_text();(d/'provider.c').write_text(text if arm=='control'else defer_probability_certification(text))
 commands=[]
 for cmd in recipe['commands']:
  cmd=[x.replace(str(source),str(d))for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd)
 if arm=='control':assert sha(d/'provider.o')==sha(source/'provider.o')
 link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'provider.o')if x==str(source/'provider.o')else str(d/'model.elf')if x==str(old/'candidate/model.elf')else x for x in link];subprocess.run(link,check=True)
 if arm=='control':assert sha(d/'model.elf')==sha(old/'candidate/model.elf')
 audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass';audit['elf_sha256']=sha(d/'model.elf');(d/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'compile':commands,'link':link,'elf_sha256':sha(d/'model.elf')},indent=2)+'\n')
 if arm=='candidate':
  shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as log:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid})+'\n');print('TARGET_WATCH',p.pid,flush=True)
