"""Compile the general Merlin capability against the frozen original component."""
from pathlib import Path
import hashlib,json,shutil,subprocess
from merlin.llvmlower.source_numeric_capability import SourceNumericContract,emit_source_numeric_capability
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/smol-minmax-20261006'
old=root/'out/artifacts/probes/smol-absolute-values-20261006/absolute'
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
original=json.loads((old/'compile.json').read_text());control_build=json.loads((old/'build.json').read_text())
contract=SourceNumericContract(**control_build['contract'],standard_min_max=True,min_max_interposition_unobserved=True,min_max_signed_zero_nan_payload_unobserved=True)
for name,selected in [('core_default',False),('core_selected',True)]:
 arm=work/name;arm.mkdir(exist_ok=False)
 for path in old.iterdir():
  if path.suffix in('.h','.c'):shutil.copy2(path,arm/path.name)
 for path in arm.iterdir():
  if path.suffix in('.h','.c')and path.name!='numeric_capability.h':
   text=path.read_text()
   for source,macro in [('fminf','MERLIN_SOURCE_F32_MIN'),('fmaxf','MERLIN_SOURCE_F32_MAX'),('fmin','MERLIN_SOURCE_F64_MIN'),('fmax','MERLIN_SOURCE_F64_MAX')]:text=text.replace(source+'(',macro+'(')
   path.write_text(text)
 shutil.copy2(core/'merlin/runtime/c/source_f32_math.h',arm/'source_f32_math.h')
 prefix=emit_source_numeric_capability(contract,inline_fma=True,inline_bitcasts=True,inline_classification=True,inline_absolute_values=True,inline_min_max=selected)
 (arm/'numeric_capability.h').write_text(prefix+'\n#include "f32_floor_bits.h"\n#define MERLIN_MONOTONE_F32_FLOOR(x) merlin_f32_floor_bits(x)\n')
 commands=[[arg.replace(str(old),str(arm))for arg in cmd]for cmd in original['commands']]
 for command in commands:subprocess.run(command,check=True)
 expected=old/'provider.o'if not selected else work/'builtin/provider.o'
 assert sha(arm/'provider.o')==sha(expected),'actual object changed; reuse refused'
 receipt={'commands':commands,'pins':{str(path):sha(path)for path in arm.iterdir()if path.suffix in('.h','.c','.ll','.o','.d')},'object_sha256':sha(arm/'provider.o'),'comparison_object':str(expected),'object_byte_exact':True,'contract':vars(contract),'selection':selected}
 if selected:
  link=[arg.replace(str(old),str(arm))for arg in control_build['link']];subprocess.run(link,check=True)
  audit=audit_elf((arm/'model.elf').read_bytes());assert audit['status']=='pass'
  receipt.update(link=link,nofsm=audit,elf_sha256=sha(arm/'model.elf'))
  assert sha(arm/'model.elf')==sha(work/'builtin/model.elf'),'actual linked ELF changed; reuse refused'
 (arm/'compile.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'arm':name,'object_byte_exact':True,'elf_byte_exact':selected}))

native=work/'native_frozen';native.mkdir(exist_ok=False)
previous=root/'out/artifacts/probes/smol-absolute-values-20261006/native_frozen'
manifest=json.loads((previous/'manifest.json').read_text())
for path,digest in manifest['local_pins'].items():assert sha(previous/path)==digest,path
for path in previous.iterdir():
 if path.suffix in('.h','.c','.py'):shutil.copy2(path,native/path.name)
for path in native.iterdir():
 if path.suffix in('.h','.c')and path.name!='numeric_capability.h':
  text=path.read_text()
  for source,macro in [('fminf','MERLIN_SOURCE_F32_MIN'),('fmaxf','MERLIN_SOURCE_F32_MAX'),('fmin','MERLIN_SOURCE_F64_MIN'),('fmax','MERLIN_SOURCE_F64_MAX')]:text=text.replace(source+'(',macro+'(')
  path.write_text(text)
for name in('numeric_capability.h','source_f32_math.h'):shutil.copy2(work/'core_selected'/name,native/name)
commands=[[arg.replace(str(previous),str(native))for arg in cmd]for cmd in manifest['compile_commands']]
for command in commands:subprocess.run(command,check=True)
deps=(native/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
manifest.update(compile=commands[0],compile_commands=commands,local_pins={path.name:sha(path)for path in native.iterdir()if path.suffix in('.h','.c','.py','.so')},transitive_compile_dependencies={str(Path(path).resolve()):sha(path)for path in[*deps,commands[0][0]]},scope='Explicit general min/max capability. Unchanged source/consumer accuracy gate; all48 qualification pending; target complete group exact byteidentity to independently measured experiment.')
manifest['numeric_capability'].update(inline_min_max=True,standard_min_max=True,min_max_interposition_unobserved=True,min_max_signed_zero_nan_payload_unobserved=True)
manifest['control_manifest']=str(previous/'manifest.json');manifest['control_manifest_sha256']=sha(previous/'manifest.json')
(native/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'native_manifest':str(native/'manifest.json'),'so':sha(native/'provider.so'),'full48_status':'pending'}))
