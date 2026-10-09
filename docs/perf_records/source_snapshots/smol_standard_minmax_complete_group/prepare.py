from pathlib import Path
import hashlib, json, shutil, subprocess
from mlir_oot.no_fsm_audit import audit_elf

root=Path.cwd(); work=root/'out/artifacts/probes/smol-minmax-20261006'
old=root/'out/artifacts/probes/smol-absolute-values-20261006/absolute'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
original=json.loads((old/'compile.json').read_text())
source_build=json.loads((old/'build.json').read_text())
for path,digest in original['pins'].items():assert sha(path)==digest,path
arm=work/'guarded';arm.mkdir(exist_ok=False)
changes={}
for path in old.iterdir():
 if path.suffix in('.h','.c'):shutil.copy2(path,arm/path.name)
for path in arm.iterdir():
 if path.suffix in('.h','.c')and path.name!='numeric_capability.h':
  text=path.read_text()
  for name,macro in [('fminf','MERLIN_SOURCE_F32_MIN'),('fmaxf','MERLIN_SOURCE_F32_MAX'),('fmin','MERLIN_SOURCE_F64_MIN'),('fmax','MERLIN_SOURCE_F64_MAX')]:
   count=text.count(name+'(')
   if count:changes.setdefault(path.name,{})[name]=count;text=text.replace(name+'(',macro+'(')
  path.write_text(text)
shutil.copy2(work/'guarded_minmax.h',arm/'guarded_minmax.h')
with(arm/'numeric_capability.h').open('a')as file:file.write('\n#include "guarded_minmax.h"\n')
commands=[[arg.replace(str(old),str(arm))for arg in cmd]for cmd in original['commands']]
for command in commands:subprocess.run(command,check=True)
link=[arg.replace(str(old),str(arm))for arg in source_build['link']]
subprocess.run(link,check=True)
audit=audit_elf((arm/'model.elf').read_bytes());assert audit['status']=='pass'
deps=(arm/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
pins={str(Path(path).resolve()):sha(path)for path in[*link,*deps,*commands[0],*commands[1],str(work/'prepare.py'),str(work/'guarded_minmax.h')]if Path(path).is_file()}
(arm/'build.json').write_text(json.dumps({'scope':'Guarded standard min/max experiment; source zero/NaN operands retain original library. Complete original12-head group instructionproxy, nothardwarecycles.','commands':commands,'link':link,'changes':changes,'pins':pins,'nofsm':audit,'control_build':str(old/'build.json'),'control_build_sha256':sha(old/'build.json'),'control_instructions':3370349620,'contract':{'standard_min_max':True,'min_max_interposition_unobserved':True,'errno_unobserved':True,'nontrapping':True,'exception_flags_unobserved':True,'source_signed_zero_nan_library_paths_preserved':True},'production_policy_changed':False},indent=2)+'\n')
shutil.copy2(old/'watch.py',arm/'watch.py')
print(json.dumps({'elf':sha(arm/'model.elf'),'changes':changes,'nofsm':audit['status']}))
