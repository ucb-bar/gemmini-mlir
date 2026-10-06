from pathlib import Path
import json,shutil,subprocess,hashlib
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/root-absolute-product-bounds-20261006');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Diagnostic partition at an explicit 1/32 extra readback bytes / scalar MAC.
# Three magnitude planes have nine exact cross terms, five grouped i32 outputs:
# extra bytes=5*4*m*n, scalar MAC=9*m*n*k. No source/value/golden selector.
for bucket,condition in [('high_readback','20*32>9*k'),('low_readback','20*32<=9*k')]:
 case=w/bucket;case.mkdir();recipe={'scope':'Diagnostic arithmetic-work partition only, not automatic profitability policy','threshold':{'bytes':1,'scalar_mac':32},'extra_readback_bytes':'20*m*n','extra_scalar_macs':'9*m*n*k','predicate':condition,'physical_scope':'Original fully tiled dimensions only; logical MAC count equals physical16x16 padded work for these source-bound shapes','core_proof_commit':'2f45db902'}
 for name in ['numeric_frozen','native_numeric_frozen']:
  d=case/name;d.mkdir();source=old/name
  for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
  text=(source/'provider.c').read_text();assert text.count('\n if(exact){\n')==1;text=text.replace('\n if(exact){\n','\n if(exact && ('+condition+')){\n');(d/'provider.c').write_text(text)
  commands=json.loads((source/'compile.json').read_text())['commands'];actual=[]
  for cmd in commands:
   cmd=[x.replace(str(source),str(d))for x in cmd];subprocess.run(cmd,check=True);actual.append(cmd)
  (d/'compile.json').write_text(json.dumps({'commands':actual,'pins':{str(p):sha(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
 d=case/'candidate';d.mkdir();link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(case/'numeric_frozen/provider.o')if x==str(old/'numeric_frozen/provider.o')else x for x in link];subprocess.run(link,check=True)
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n');(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(d/'model.elf')},indent=2)+'\n');(case/'cost_policy.json').write_text(json.dumps(recipe,indent=2)+'\n')
 shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
 with(d/'watch.log').open('w')as f:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (d/'process.json').write_text(json.dumps({'pid':p.pid}));print(bucket,p.pid,flush=True)
