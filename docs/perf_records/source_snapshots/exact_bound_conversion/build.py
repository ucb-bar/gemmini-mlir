from pathlib import Path
import json,shutil,subprocess,hashlib
from mlir_oot.host_outward_fp import emit_fixed_outward_f64_header
from mlir_oot.no_fsm_audit import audit_elf
from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract,emit_exact_bound_conversion_permission
w=Path(__file__).parent.resolve();old=Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/word_soft_i64_group');core=Path('/scratch/agustin/tmp/merlin-exact-bound-conversion-20261006');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prefix=emit_exact_bound_conversion_permission(ExactBoundConversionContract(True,True,True,True,True))
native_header='''#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static inline float exact_bound_floor(double x){int old=fegetround();fesetround(FE_DOWNWARD);volatile double d=x;volatile float f=(float)d;fesetround(old);return f;}
static inline float exact_bound_ceil(double x){int old=fegetround();fesetround(FE_UPWARD);volatile double d=x;volatile float f=(float)d;fesetround(old);return f;}
#pragma STDC FENV_ACCESS OFF
#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) exact_bound_floor(x)
#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) exact_bound_ceil(x)
'''
for name in ['default_numeric','numeric_frozen','native_numeric_frozen']:
 d=w/name;d.mkdir(exist_ok=True);source=old/('native_numeric_frozen'if name=='native_numeric_frozen'else'numeric_frozen')
 for p in source.glob('*.h'):shutil.copyfile(p,d/p.name)
 for h in ['ordered_fma_bounds.h','prepared_fma_product_bounds.h','separable_fma_radius.h']:shutil.copyfile(core/'merlin/runtime/c'/h,d/h)
 (d/'provider.c').write_text((''if name=='default_numeric'else prefix)+(source/'provider.c').read_text())
 if name=='numeric_frozen':(d/'fixed_outward_f64.h').write_text(emit_fixed_outward_f64_header(name='certificate_f64',host_isa='rv64gc',exact_bound_f32=True))
 if name=='native_numeric_frozen':
  (d/'native_exact_conversion.h').write_text(native_header);commands=[json.loads((source/'manifest.json').read_text())['compile']]
 else:commands=json.loads((source/'compile.json').read_text())['commands']
 actual=[]
 for c in commands:
  c=[x.replace(str(source),str(d))for x in c]
  if name=='native_numeric_frozen':c[1:1]=['-include',str(d/'native_exact_conversion.h')]
  subprocess.run(c,check=True);actual.append(c)
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
