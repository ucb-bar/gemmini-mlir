from pathlib import Path
import ast,json,hashlib,subprocess
from merlin.llvmlower.source_numeric_capability import SourceNumericContract,emit_source_numeric_capability
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/smol-classification-20261006';proof=work/'representation_proof';proof.mkdir(exist_ok=False)
core=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004')
fixture=core/'merlin/tests/runtime/test_source_numeric_capability.py'
module=ast.parse(fixture.read_text());source=next(ast.literal_eval(n.value)for n in module.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CLASSIFICATION_SOURCE'for t in n.targets))
source=source.replace('int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};','unsigned long original;__asm__ volatile("csrr %0,frm":"=r"(original));')
source=source.replace('mode<4','mode<5').replace('if(fesetround(modes[mode]))return 1;','__asm__ volatile("csrw frm,%0"::"r"((unsigned long)mode));')
source=source.replace('if(fegetround()!=modes[mode]||errno!=123)return 7;','unsigned long current;__asm__ volatile("csrr %0,frm":"=r"(current));if(current!=(unsigned long)mode||errno!=123)return 7;')
source=source.replace('#include <errno.h>','#include <errno.h>\n#include <stdio.h>')
source=source.replace(' return 0;\n}', ' __asm__ volatile("csrw frm,%0"::"r"(original));\n printf("CLASSIFICATION PASS 565925\\n");\n return 0;\n}')
assert 'modes['not in source
prefix=emit_source_numeric_capability(SourceNumericContract(True,True,True,True,True,True,True,True,True),inline_classification=True)
(proof/'classification.c').write_text(prefix+source)
base=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint')
r=json.loads((base/'numeric_rows_floor_target_frozen/reclosure.json').read_text());cmd=r['compile_commands'][0]
out=cmd[:cmd.index('-I')]+['-I',str(work/'classification'),'-c','-MD','-MF',str(proof/'classification.d'),str(proof/'classification.c'),'-o',str(proof/'classification.o')]
subprocess.run(out,check=True)
link=json.loads((base/'numeric_capability_prepared_rows_floor_target/build.json').read_text())['link']
# Retain exact qualified startup/linker/libs only; numerical accelerator objects
# do not participate in this independent CPU representation proof.
first=link.index('-o');last=link.index('-lm');objects=link[first+2:last]
link=link[:first+1]+[str(proof/'classification.elf')]+objects[:2]+[str(proof/'classification.o')]+link[last:]
subprocess.run(link,check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=audit_elf((proof/'classification.elf').read_bytes());assert audit['status']=='pass'
undefined=subprocess.check_output(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm','-u',str(proof/'classification.o')],text=True)
assert '__fpclassify'not in undefined
(proof/'build.json').write_text(json.dumps({'scope':'Independent BF16 exhaustiveness and F32/F64 IEEE representation boundaries/random words,5ambienttargetroundingmodes; noperformanceprediction','compile':out,'link':link,'pins':{v:sha(v)for v in[*out,*link,str(fixture)]if Path(v).is_file()},'nofsm':audit,'expected_checks':565925,'undefined':undefined},indent=2)+'\n')
print(proof)
