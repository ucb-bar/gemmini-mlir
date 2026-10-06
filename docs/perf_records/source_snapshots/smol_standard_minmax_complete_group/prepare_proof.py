from pathlib import Path
import hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
root=Path.cwd();work=root/'out/artifacts/probes/smol-minmax-20261006'
previous=root/'out/artifacts/probes/smol-absolute-values-20261006/representation_proof'
receipt=json.loads((previous/'build.json').read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
command=receipt['compile'];command=command[:command.index('-I')]+['-I',str(work),'-c','-MD','-MF',str(work/'representation.d'),str(work/'representation.c'),'-o',str(work/'representation.o')]
subprocess.run(command,check=True)
link=[arg.replace(str(previous/'absolute.elf'),str(work/'representation.elf')).replace(str(previous/'absolute.o'),str(work/'representation.o'))for arg in receipt['link']]
subprocess.run(link,check=True)
audit=audit_elf((work/'representation.elf').read_bytes());assert audit['status']=='pass'
deps=(work/'representation.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
(work/'representation_build.json').write_text(json.dumps({'compile':command,'link':link,'pins':{str(Path(path).resolve()):sha(path)for path in[*command,*link,*deps,str(work/'prepare_proof.py')]if Path(path).is_file()},'nofsm':audit,'expected_pair_checks':859360,'mode_count':5},indent=2)+'\n')
print(json.dumps({'elf':sha(work/'representation.elf'),'nofsm':audit['status']}))
