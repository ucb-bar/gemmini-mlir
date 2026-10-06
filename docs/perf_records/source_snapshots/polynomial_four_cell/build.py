from pathlib import Path
import json,shutil,subprocess,hashlib
from merlin.llvmlower.prepared_polynomial_batch import prepare_polynomial_batch_four
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');core=Path('/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['default_numeric','numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir(exist_ok=True);source=old/('native_numeric_frozen'if name=='native_numeric_frozen'else'numeric_frozen')
 for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
 shutil.copyfile(core/'merlin/runtime/c/prepared_polynomial_batch.h',d/'prepared_polynomial_batch.h')
 text=(source/'provider.c').read_text();(d/'provider.c').write_text(text if name=='default_numeric'else prepare_polynomial_batch_four(text))
 commands=[json.loads((source/'manifest.json').read_text())['compile']]if name=='native_numeric_frozen'else json.loads((source/'compile.json').read_text())['commands']
 actual=[]
 for c in commands:
  c=[x.replace(str(source),str(d))for x in c];subprocess.run(c,check=True);actual.append(c)
 (d/'compile.json').write_text(json.dumps({'commands':actual,'pins':{str(p):sha(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
assert sha(w/'default_numeric/provider.o')==sha(old/'numeric_frozen/provider.o')
for arm in ['control','candidate']:
 d=w/arm;d.mkdir(exist_ok=True);link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(w/('numeric_frozen'if arm=='candidate'else'default_numeric')/'provider.o')if x==str(old/'numeric_frozen/provider.o')else x for x in link];subprocess.run(link,check=True)
 if arm=='control':assert sha(d/'model.elf')==sha(old/'candidate/model.elf')
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(d/'model.elf')},indent=2)+'\n')
 if arm=='candidate':
  shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as log:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid}));print('watch',p.pid,flush=True)
