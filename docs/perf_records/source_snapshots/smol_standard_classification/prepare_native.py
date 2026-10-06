from pathlib import Path
import json,hashlib,shutil,subprocess
root=Path.cwd();w=root/'out/artifacts/probes/smol-classification-20261006/native_frozen';w.mkdir(exist_ok=False)
old=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/numeric_rows_floor_frozen')
r=json.loads((old/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in r['local_pins'].items():assert sha(old/p)==h,p
for p in old.iterdir():
 if p.suffix in ('.h','.c','.py'):shutil.copy2(p,w/p.name)
for p in w.iterdir():
 if p.suffix in ('.h','.c')and p.name not in ('numeric_capability.h','source_f32_math.h'):
  p.write_text(p.read_text().replace('isfinite(','MERLIN_SOURCE_ISFINITE('))
for name in ('numeric_capability.h','source_f32_math.h'):
 shutil.copy2(root/'out/artifacts/probes/smol-classification-20261006/classification'/name,w/name)
commands=[[v.replace(str(old),str(w))for v in cmd]for cmd in r['compile_commands']]
for cmd in commands:subprocess.run(cmd,check=True)
pins={p.name:sha(p)for p in w.iterdir()if p.suffix in('.h','.c','.py','.so')}
deps=(w/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
r['compile']=commands[0];r['compile_commands']=commands;r['local_pins']=pins
r['transitive_compile_dependencies']={str(Path(p).resolve()):sha(p)for p in[*deps,commands[0][0]]}
r['numeric_capability']['inline_classification']=True
r['numeric_capability']['standard_fp_classification']=True
r['numeric_capability']['classification_interposition_unobserved']=True
r['scope']='Current row+floor source math and source observations unchanged; explicit finite classification builtin. Native stand-in only, target production capsule and complete source consumer gate separately required.'
r['control_manifest']=str(old/'manifest.json');r['control_manifest_sha256']=sha(old/'manifest.json')
(w/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'frozen':str(w),'so_sha256':sha(w/'provider.so'),'target_complete_group':'pass3.527309706Binstructions','full48gate':'pending'}))
