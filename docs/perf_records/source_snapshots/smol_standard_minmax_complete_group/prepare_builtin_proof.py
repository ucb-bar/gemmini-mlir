"""Independent differential under only the explicitly waived distinctions."""
from pathlib import Path
import hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/smol-minmax-20261006'
previous=root/'out/artifacts/probes/smol-absolute-values-20261006/representation_proof'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
source='#include <math.h>\n'+(work/'representation.c').read_text().replace('"guarded_minmax.h"','"builtin_minmax.h"')
helpers='''
static int equal32(float a,float b){
 uint32_t x=b32(a),y=b32(b);
 if((x&UINT32_C(0x7fffffff))==0&&(y&UINT32_C(0x7fffffff))==0)return 1;
 if((x&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000)&&
    (y&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000))return 1;
 return x==y;
}
static int equal64(double a,double b){
 uint64_t x=b64(a),y=b64(b);
 if((x&UINT64_C(0x7fffffffffffffff))==0&&(y&UINT64_C(0x7fffffffffffffff))==0)return 1;
 if((x&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000)&&
    (y&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000))return 1;
 return x==y;
}
'''
source=source.replace('static unsigned long checks;',helpers+'\nstatic unsigned long checks;')
for width in (32,64):
 for op in ('min','max'):
  old=f'b{width}(original_{op}{width}(x,y))==b{width}(MERLIN_SOURCE_F{width}_{op.upper()}(x,y))'
  new=f'equal{width}(original_{op}{width}(x,y),MERLIN_SOURCE_F{width}_{op.upper()}(x,y))'
  assert old in source;source=source.replace(old,new)
source=source.replace('GUARDED_MINMAX PASS','BUILTIN_MINMAX PASS')
(work/'representation_builtin.c').write_text(source)
receipt=json.loads((previous/'build.json').read_text())
command=receipt['compile'];command=command[:command.index('-I')]+['-I',str(work),'-c','-MD','-MF',str(work/'representation_builtin.d'),str(work/'representation_builtin.c'),'-o',str(work/'representation_builtin.o')]
subprocess.run(command,check=True)
link=[arg.replace(str(previous/'absolute.elf'),str(work/'representation_builtin.elf')).replace(str(previous/'absolute.o'),str(work/'representation_builtin.o'))for arg in receipt['link']]
subprocess.run(link,check=True)
audit=audit_elf((work/'representation_builtin.elf').read_bytes());assert audit['status']=='pass'
deps=(work/'representation_builtin.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
(work/'representation_builtin_build.json').write_text(json.dumps({'compile':command,'link':link,'pins':{str(Path(path).resolve()):sha(path)for path in[*command,*link,*deps,str(work/'prepare_builtin_proof.py'),str(work/'representation.c')]if Path(path).is_file()},'nofsm':audit,'expected_pair_checks':859360,'mode_count':5,'waived_distinctions':['min_max_signed_zero','min_max_nan_payload'],'unwaived':'Every other returned binary32/binary64 bit, including oneNaN/number numerical results, infinity signs, allfinitevalues. Operandsevaluatedonce/roundmodepreserved.'},indent=2)+'\n')
print(json.dumps({'elf':sha(work/'representation_builtin.elf'),'nofsm':audit['status']}))
