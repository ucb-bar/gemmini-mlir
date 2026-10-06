from pathlib import Path
import json,subprocess,shutil,hashlib
from merlin.llvmlower.binary32_separable_radius import prepare_binary32_separable_radius
from mlir_oot.host_binary32_radius import emit_binary32_radius_header
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed');core=Path('/scratch/agustin/tmp/merlin-binary32-source-radius-20261006')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['control_numeric','numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir();source=old/('native_numeric_frozen'if name=='native_numeric_frozen'else'numeric_frozen')
 for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
 text=(source/'provider.c').read_text()
 if name!='control_numeric':
  shutil.copyfile(core/'merlin/runtime/c/binary32_separable_radius.h',d/'binary32_separable_radius.h')
  text=prepare_binary32_separable_radius(text)
  if name=='numeric_frozen':prefix=emit_binary32_radius_header(host_isa='rv64gc')
  else:
   prefix='''#include <fenv.h>
#include <math.h>
#pragma STDC FENV_ACCESS ON
static float radius_fma_up(float a,float b,float c){int m=fegetround();fesetround(FE_UPWARD);volatile float r=fmaf(a,b,c);fesetround(m);return r;}
static float radius_add_up(float a,float b){int m=fegetround();fesetround(FE_UPWARD);volatile float r=a+b;fesetround(m);return r;}
static float radius_add_down(float a,float b){int m=fegetround();fesetround(FE_DOWNWARD);volatile float r=a+b;fesetround(m);return r;}
#pragma STDC FENV_ACCESS OFF
#define MERLIN_F32_RADIUS_FMA_UP(a,b,c) radius_fma_up(a,b,c)
#define MERLIN_F32_RADIUS_ADD_UP(a,b) radius_add_up(a,b)
#define MERLIN_F32_RADIUS_ADD_DOWN(a,b) radius_add_down(a,b)
'''
  (d/'binary32_capability.h').write_text(prefix)
 (d/'provider.c').write_text(text)
 commands=[]
 for cmd in json.loads((source/'compile.json').read_text())['commands']:
  cmd=[x.replace(str(source),str(d))for x in cmd]
  if name!='control_numeric':cmd[1:1]=['-include',str(d/'binary32_capability.h')]
  subprocess.run(cmd,check=True);commands.append(cmd)
 (d/'compile.json').write_text(json.dumps({'commands':commands,'pins':{str(p):sha(p)for p in d.iterdir()if p.is_file()}},indent=2)+'\n')
assert sha(w/'control_numeric/provider.o')==sha(old/'numeric_frozen/provider.o')
for arm in ['control','candidate']:
 d=w/arm;d.mkdir();link=json.loads((old/'candidate/build.json').read_text())['link'];link=[str(d/'model.elf')if x==str(old/'candidate/model.elf')else str(w/('control_numeric'if arm=='control'else'numeric_frozen')/'provider.o')if x==str(old/'numeric_frozen/provider.o')else x for x in link];subprocess.run(link,check=True)
 if arm=='control':assert sha(d/'model.elf')==sha(old/'candidate/model.elf')
 a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2));(d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(d/'model.elf')},indent=2))
 if arm=='candidate':
  shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as f:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid}));print(p.pid,sha(d/'model.elf'),flush=True)
