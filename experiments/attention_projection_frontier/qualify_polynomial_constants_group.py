"""Complete current-row source group, with only provider object replaced."""
from pathlib import Path
import hashlib,json,subprocess,time
from mlir_oot.no_fsm_audit import audit_elf
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants';D=W/'candidate';C=B/'out/exact_row_group/candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
link=json.load(open(C/'build.json'))['commands'][-1]
link=[str(D/'target_numeric/provider.o') if x==str(C/'target_numeric/provider.o') else str(D/'model.elf') if x==str(C/'model.elf') else x for x in link]
assert str(D/'model.elf') in link and str(D/'target_numeric/provider.o') in link
subprocess.run(link,check=True,capture_output=True)
audit=audit_elf((D/'model.elf').read_bytes());assert audit['status']=='pass'
(D/'target_build.json').write_text(json.dumps(dict(link=link,audit=audit,elf_sha256=sha(D/'model.elf'),control_sha256=sha(C/'model.elf')),indent=2)+'\n')
T=W/'strict';T.mkdir();cmd=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(D/'model.elf')];start=time.time()
with (T/'stdout').open('w') as out,(T/'stderr').open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
(T/'terminal.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,seconds=time.time()-start,elf_sha256=sha(D/'model.elf'),stdout_sha256=sha(T/'stdout'),stderr_sha256=sha(T/'stderr')),indent=2)+'\n')
print((T/'stdout').read_text(),flush=True);assert r.returncode==0
