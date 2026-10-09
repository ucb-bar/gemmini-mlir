from pathlib import Path
import json,hashlib,shutil,subprocess,ast
from merlin.llvmlower.source_numeric_capability import SourceNumericContract,emit_source_numeric_capability
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();old=root/'out/artifacts/probes/smol-classification-20261006/classification';w=root/'out/artifacts/probes/smol-absolute-values-20261006';w.mkdir(exist_ok=False)
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((old/'compile.json').read_text());b=json.loads((old/'build.json').read_text());contract=SourceNumericContract(*([True]*12));changed={}
for name,select in [('default_control',False),('absolute',True)]:
 arm=w/name;arm.mkdir()
 for p in old.iterdir():
  if p.suffix in('.h','.c'):shutil.copy2(p,arm/p.name)
 for p in arm.iterdir():
  if p.suffix in('.h','.c')and p.name not in('numeric_capability.h','source_f32_math.h'):
   t=p.read_text();count=t.count('fabs(')+t.count('fabsf(')
   if count:p.write_text(t.replace('fabsf(','MERLIN_SOURCE_F32_ABS(').replace('fabs(','MERLIN_SOURCE_F64_ABS('));changed[p.name]=count
 shutil.copy2(core/'merlin/runtime/c/source_f32_math.h',arm/'source_f32_math.h')
 (arm/'numeric_capability.h').write_text(emit_source_numeric_capability(contract,inline_fma=True,inline_bitcasts=True,inline_classification=True,inline_absolute_values=select)+'\n#include "f32_floor_bits.h"\n#define MERLIN_MONOTONE_F32_FLOOR(x) merlin_f32_floor_bits(x)\n')
 cmds=[[v.replace(str(old),str(arm))for v in cmd]for cmd in r['commands']]
 for cmd in cmds:subprocess.run(cmd,check=True)
 undefined=subprocess.check_output(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm','-u',str(arm/'provider.o')],text=True)
 (arm/'compile.json').write_text(json.dumps({'commands':cmds,'undefined':undefined,'object_sha256':sha(arm/'provider.o'),'pins':{str(p):sha(p)for p in arm.iterdir()if p.suffix in('.h','.c','.ll','.o','.d')},'default_object_byte_exact':sha(arm/'provider.o')==sha(old/'provider.o')},indent=2)+'\n')
 if not select:assert sha(arm/'provider.o')==sha(old/'provider.o'),'controlchanged; remeasurementnecessary'
 if select:
  cmd=[v.replace(str(old),str(arm))for v in b['link']]
  subprocess.run(cmd,check=True);audit=audit_elf((arm/'model.elf').read_bytes());assert audit['status']=='pass'
  deps=(arm/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
  (arm/'build.json').write_text(json.dumps({'scope':'Explicit standard absolute-value builtin only; same original complete12head production group; functional Spike instructionproxy notFireSim cycles.','link':cmd,'pins':{str(Path(p).resolve()):sha(p)for p in[*cmd,*deps]if Path(p).is_file()},'nofsm':audit,'contract':vars(contract),'changes':changed,'control_instructions':3527309706,'control_build':str(old/'build.json'),'control_build_sha256':sha(old/'build.json'),'default_object_byte_exact':True},indent=2)+'\n')
  shutil.copy2(old/'watch.py',arm/'watch.py')
proof=w/'representation_proof';proof.mkdir()
fixture=core/'merlin/tests/runtime/test_source_numeric_capability.py';frozen=proof/'original_fixture_test_source_numeric_capability.py';shutil.copy2(fixture,frozen)
module=ast.parse(frozen.read_text());source=next(ast.literal_eval(n.value)for n in module.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ABSOLUTE_SOURCE'for t in n.targets))
source=source.replace('int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};','unsigned long original;__asm__ volatile("csrr %0,frm":"=r"(original));')
source=source.replace('mode<4','mode<5').replace('if(fesetround(modes[mode]))return 1;','__asm__ volatile("csrw frm,%0"::"r"((unsigned long)mode));')
source=source.replace('if(fegetround()!=modes[mode])return 7;','unsigned long current;__asm__ volatile("csrr %0,frm":"=r"(current));if(current!=(unsigned long)mode)return 7;')
source=source.replace('#include <fenv.h>','#include <fenv.h>\n#include <stdio.h>')
source=source.replace(' return 0;\n}', ' __asm__ volatile("csrw frm,%0"::"r"(original));\n printf("ABSOLUTE PASS 565925\\n");\n return 0;\n}')
(proof/'absolute.c').write_text(emit_source_numeric_capability(contract,inline_absolute_values=True)+source)
cmd=r['commands'][0];cmd=cmd[:cmd.index('-I')]+['-I',str(w/'absolute'),'-c','-MD','-MF',str(proof/'absolute.d'),str(proof/'absolute.c'),'-o',str(proof/'absolute.o')];subprocess.run(cmd,check=True)
link=json.loads((root/'out/artifacts/probes/smol-classification-20261006/representation_proof/build.json').read_text())['link'];link=[v.replace('smol-classification-20261006/representation_proof/classification.elf','smol-absolute-values-20261006/representation_proof/absolute.elf').replace('smol-classification-20261006/representation_proof/classification.o','smol-absolute-values-20261006/representation_proof/absolute.o')for v in link]
subprocess.run(link,check=True);audit=audit_elf((proof/'absolute.elf').read_bytes());assert audit['status']=='pass'
(proof/'build.json').write_text(json.dumps({'compile':cmd,'link':link,'pins':{p:sha(p)for p in[*cmd,*link,str(frozen)]if Path(p).is_file()},'nofsm':audit,'expected_checks':565925,'mode_count':5,'source_payload_obligation':'NaNpayloadunobserved explicitlyrequired; actual qualifiedplatformadditionallyretainsNaNpayloadbits.'},indent=2)+'\n')
print(json.dumps({'probe':str(w),'control_byte_exact':True,'changed':changed}))
