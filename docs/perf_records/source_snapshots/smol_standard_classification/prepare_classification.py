from pathlib import Path
import hashlib,json,shutil,subprocess
from merlin.llvmlower.source_numeric_capability import SourceNumericContract,emit_source_numeric_capability
from mlir_oot.no_fsm_audit import audit_elf
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
base=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint')
frozen=base/'numeric_rows_floor_target_frozen'
control=base/'numeric_capability_prepared_rows_floor_target'
work=Path.cwd()/'out/artifacts/probes/smol-classification-20261006'
work.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((frozen/'reclosure.json').read_text())
for p,h in r['compilation_pins'].items():assert sha(p)==h,p
link=json.loads((control/'build.json').read_text())
for p,h in link['pins'].items():assert sha(p)==h,p
contract=SourceNumericContract(True,True,True,True,True,True,True,True,True)
changes={}
for name,select in [('default_control',False),('classification',True)]:
 arm=work/name;arm.mkdir()
 for p in frozen.iterdir():
  if p.suffix in ('.h','.c'):shutil.copy2(p,arm/p.name)
 for p in arm.iterdir():
  if p.suffix in ('.h','.c') and p.name not in ('numeric_capability.h','source_f32_math.h'):
   t=p.read_text();count=t.count('isfinite(')
   if count:p.write_text(t.replace('isfinite(','MERLIN_SOURCE_ISFINITE('));changes[p.name]=count
 shutil.copy2(core/'merlin/runtime/c/source_f32_math.h',arm/'source_f32_math.h')
 (arm/'numeric_capability.h').write_text(emit_source_numeric_capability(contract,inline_fma=True,inline_bitcasts=True,inline_classification=select)+'\n#include "f32_floor_bits.h"\n#define MERLIN_MONOTONE_F32_FLOOR(x) merlin_f32_floor_bits(x)\n')
 cmds=[]
 for cmd in r['compile_commands']:
  cmd=[v.replace(str(frozen),str(arm)) for v in cmd]
  subprocess.run(cmd,check=True);cmds.append(cmd)
 (arm/'compile.json').write_text(json.dumps({'commands':cmds,'pins':{str(p):sha(p)for p in arm.iterdir()if p.suffix in ('.h','.c','.o','.ll','.d')},'undefined':subprocess.check_output(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm','-u',str(arm/'provider.o')],text=True),'default_object_byte_exact':sha(arm/'provider.o')==r['provider_object_sha256']},indent=2)+'\n')
 if not select:assert sha(arm/'provider.o')==r['provider_object_sha256'],'changed default object: must remeasure control'
 if select:
  cmd=[str(arm/'provider.o')if v==str(control/'provider.o')else str(arm/'model.elf')if v==str(control/'model.elf')else v for v in link['link']]
  subprocess.run(cmd,check=True)
  a=audit_elf((arm/'model.elf').read_bytes());assert a['status']=='pass'
  pins={v:sha(v)for v in cmd if Path(v).is_file()}
  deps=(arm/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
  pins.update({str(Path(p).resolve()):sha(p)for p in deps})
  (arm/'build.json').write_text(json.dumps({'schema':'classified_finite_production_provider_v1','scope':'Same complete12head production group; selected standard finite classification only. Actual source quant observations and accepted carriers unchanged; no hardware cycles or whole performance claim.','link':cmd,'pins':pins,'nofsm':a,'control_build':str(control/'build.json'),'control_sha256':sha(control/'build.json'),'control_instructions':5034507191,'classification_contract':vars(contract),'classification_macro_replacements':changes,'default_control_byte_exact':True},indent=2)+'\n')
  shutil.copy2(control/'watch.py',arm/'watch.py')
print(json.dumps({'work':str(work),'default_object_byte_exact':True,'changed':changes,'candidate_object_sha256':sha(work/'classification/provider.o'),'elf_sha256':sha(work/'classification/model.elf')}))
