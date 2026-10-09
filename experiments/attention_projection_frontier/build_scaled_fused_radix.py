"""Exact terminal scaling fused into the existing completed readout loop."""
from pathlib import Path
import json,shutil,subprocess
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups
from merlin.llvmlower.radix_integer_reconstruct import c_scaled_fused_header
B=Path(__file__).resolve().parents[2];C=B/'out/artifacts/probes/prepared-polynomial-constants/candidate';W=B/'out/artifacts/probes/scaled-fused-radix';W.mkdir(parents=True,exist_ok=False)
roles={}
for role in ('native_numeric','target_numeric'):
 D=W/role;D.mkdir()
 for p in (C/role).iterdir():
  if p.suffix in ('.c','.h'):shutil.copyfile(p,D/p.name)
 (D/'scaled_fused_radix.h').write_text(c_scaled_fused_header(plan_radix_product_groups(radix_bits=7,digits=3,reduction_length=192)))
 s=(D/'provider.c').read_text()
 anchor='static int evaluate_products('
 assert s.count(anchor)==1;s=s.replace(anchor,'#include "scaled_fused_radix.h"\n'+anchor)
 old=''' merlin_radix_integer_fused_exact_f64(w->center,planes,(size_t)m*n);
 for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->center[r*n+c]*=w->astep[r]*bstep[c];'''
 assert s.count(old)==1
 s=s.replace(old,' if(!merlin_radix_scaled_fused_exact_f64(w->center,planes,m,n,w->astep,bstep)){\n'+old+'\n }')
 (D/'provider.c').write_text(s)
 commands=[]
 for oldcmd in json.loads((C/'build.json').read_text())['roles'][role]['commands']:
  cmd=[x.replace(str(C/role),str(D)) for x in oldcmd];subprocess.run(cmd,check=True);commands.append(cmd)
 roles[role]={'commands':commands}
(W/'build.json').write_text(json.dumps({'roles':roles},indent=2))
h=W/'native';h.mkdir()
s=(C.parent/'normal_native/run.py').read_text().replace(str(C.parent/'normal_native'),str(h)).replace(str(C/'native_numeric'),str(W/'native_numeric'))
(h/'run.py').write_text(s)
print(h/'run.py',flush=True)
