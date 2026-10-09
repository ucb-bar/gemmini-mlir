from pathlib import Path
import json,ast,shutil,subprocess,hashlib
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');core=Path('/scratch/agustin/tmp/merlin-encoded-row-equality-20261006');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from merlin.llvmlower.encoded_row_equality import prepare_encoded_row_equality
emitter=core/'src/merlin/llvmlower/encoded_row_equality.py'
for name in ['numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir(exist_ok=True)
 for p in (old/name).glob('*.h'):shutil.copyfile(p,d/p.name)
 shutil.copyfile(core/'merlin/runtime/c/encoded_row_equality.h',d/'encoded_row_equality.h')
 s=prepare_encoded_row_equality((old/name/'provider.c').read_text());(d/'provider.c').write_text(s)
 if name=='numeric_frozen':commands=json.loads((old/name/'compile.json').read_text())['commands']
 else:commands=[json.loads((old/name/'manifest.json').read_text())['compile']]
 cmds=[]
 for c in commands:
  c=[x.replace(str(old/name),str(d))for x in c];subprocess.run(c,check=True);cmds.append(c)
 (d/'compile.json').write_text(json.dumps({'commands':cmds,'source_transform':str(emitter),'source_transform_sha256':h(emitter),'pins':{str(p):h(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
for arm in ['control','candidate']:
 d=w/arm;d.mkdir(exist_ok=True);link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(w/'numeric_frozen/provider.o')if arm=='candidate'and x==str(old/'numeric_frozen/provider.o')else x for x in link];subprocess.run(link,check=True)
 if arm=='control':assert h(d/'model.elf')==h(old/'candidate/model.elf')
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':h(d/'model.elf')},indent=2)+'\n')
 if arm=='candidate':
  shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as log:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid}));print('watch',p.pid,flush=True)
