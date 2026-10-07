"""Complete original group timing screen; unchanged consumer and device objects."""
from pathlib import Path
import datetime,hashlib,json,subprocess,time
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parents[2]/'out/rounded_monotone_group'
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')
d=w/'candidate';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
link=json.loads((b/'build.json').read_text())['link']
cmd=[str(d/'model.elf') if x==str(b/'model.elf') else str(d/'target_numeric/provider.o') if x==str(b/'target_numeric/provider.o') else x for x in link]
subprocess.run(cmd,check=True,capture_output=True)
audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass'
(d/'target_build.json').write_text(json.dumps({'link':cmd,'audit':audit,'elf_sha256':sha(d/'model.elf'),'baseline':str(b/'model.elf'),'baseline_sha256':sha(b/'model.elf')},indent=2)+'\n')
t=w/'strict';t.mkdir()
run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')]
start=time.time()
with (t/'stdout').open('w') as out,(t/'stderr').open('w') as err:r=subprocess.run(run,stdout=out,stderr=err)
(t/'terminal.json').write_text(json.dumps({'command':run,'exit_code':r.returncode,'elapsed_seconds':time.time()-start,'elf_sha256':sha(d/'model.elf'),'stdout_sha256':sha(t/'stdout'),'stderr_sha256':sha(t/'stderr'),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print((t/'stdout').read_text(),flush=True)
