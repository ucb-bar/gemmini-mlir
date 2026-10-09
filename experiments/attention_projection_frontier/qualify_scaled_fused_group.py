"""One complete original consumer screen after native whole acceptance."""
from pathlib import Path
import hashlib,json,subprocess,time
from mlir_oot.no_fsm_audit import audit_elf
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/scaled-fused-radix';C=B/'out/artifacts/probes/prepared-polynomial-constants/candidate'
native=json.loads((W/'native/validation.json').read_text())
assert native['bitwise_mismatches']==0 and native['allclose'] and native['calls'][0:3]==[48,0,0] and native['product_calls']==23040
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
link=json.loads((C/'target_build.json').read_text())['link']
link=[str(W/'target_numeric/provider.o') if x==str(C/'target_numeric/provider.o') else str(W/'model.elf') if x==str(C/'model.elf') else x for x in link]
assert str(W/'model.elf') in link and str(W/'target_numeric/provider.o') in link
subprocess.run(link,check=True,capture_output=True)
audit=audit_elf((W/'model.elf').read_bytes());assert audit['status']=='pass'
(W/'target_build.json').write_text(json.dumps({'link':link,'audit':audit,'candidate_sha256':sha(W/'model.elf'),'control_sha256':sha(C/'model.elf')},indent=2))
T=W/'strict';T.mkdir()
cmd=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(W/'model.elf')];start=time.time()
with (T/'stdout').open('w') as out,(T/'stderr').open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err)
(T/'terminal.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'seconds':time.time()-start,'elf_sha256':sha(W/'model.elf'),'stdout_sha256':sha(T/'stdout'),'stderr_sha256':sha(T/'stderr')},indent=2))
print((T/'stdout').read_text(),flush=True);assert r.returncode==0
