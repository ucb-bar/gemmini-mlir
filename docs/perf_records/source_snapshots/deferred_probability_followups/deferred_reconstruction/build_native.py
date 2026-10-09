from pathlib import Path
import json,hashlib,subprocess,shutil
from merlin.llvmlower.deferred_probability import defer_probability_certification, separate_reconstruction_proof
w=Path(__file__).resolve().parent;old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed/native_numeric_frozen');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();recipe=json.loads((old/'compile.json').read_text())
for p,h in recipe['pins'].items():assert sha(p)==h,p
for arm in ['control','candidate']:
 d=w/arm;d.mkdir(exist_ok=True)
 for p in old.glob('*.h'):shutil.copyfile(p,d/p.name)
 if arm=='candidate':shutil.copyfile('/scratch/agustin/tmp/merlin-deferred-probability-20261006/merlin/runtime/c/encoded_row_equality.h',d/'encoded_row_equality.h')
 text=(old/'provider.c').read_text();(d/'provider.c').write_text(text if arm=='control' else separate_reconstruction_proof(defer_probability_certification(text)))
 commands=[]
 for cmd in recipe['commands']:
  cmd=[x.replace(str(old),str(d))for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd)
 (d/'compile.json').write_text(json.dumps({'commands':commands,'pins':{str(p):sha(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
 if arm=='control':assert sha(d/'provider.so')==sha(old/'provider.so')
print('NATIVE_BUILD_PASS')
