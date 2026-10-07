"""Same fixed4MAC catalog, explicit RMS4 plus unscaled residual bounds."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
from merlin.llvmlower.source_rms_residual_products import c_header
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];b=base/'out/two_signed_byte_group/candidate';w=base/'out/two_signed_byte_rms_group';d=w/'candidate';dest=d/'target_numeric';dest.mkdir(parents=True,exist_ok=False)
for p in (b/'target_numeric').iterdir():
 if p.suffix in ('.c','.h'):shutil.copyfile(p,dest/p.name)
(dest/'source_rms_residual_products.h').write_text(c_header(ApproximateSourceRoundoffPolicy(*([True]*8))))
s=(dest/'provider.c').read_text();s=s.replace('#include "source_rms_point_products.h"','#include "source_rms_point_products.h"\n#include "source_rms_residual_products.h"').replace('double uncertainty[CHUNK];','double uncertainty[CHUNK];double rms_rep_error[CHUNK];').replace('merlin_source_rms4_point_product_estimates(','merlin_source_rms4_residual_product_estimates(').replace('scratch->uncertainty,CHUNK))return 1;','scratch->uncertainty,scratch->rms_rep_error,CHUNK))return 1;');(dest/'provider.c').write_text(s)
old=json.loads((b/'build.json').read_text());commands=[]
for prior in old['commands'][:2]:
 cmd=[x.replace(str(b/'target_numeric'),str(dest)) for x in prior];subprocess.run(cmd,check=True);commands.append(cmd)
link=[str(d/'model.elf') if x==str(b/'model.elf') else str(dest/'provider.o') if x==str(b/'target_numeric/provider.o') else x for x in old['commands'][-1]];subprocess.run(link,check=True,capture_output=True);commands.append(link)
audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(d/'build.json').write_text(json.dumps(dict(commands=commands,audit=audit,baseline=str(b/'model.elf'),pins={str(p):sha(p) for p in d.rglob('*') if p.is_file()}),indent=2)+'\n')
t=w/'strict';t.mkdir();run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
with (t/'stdout').open('w') as out,(t/'stderr').open('w') as err:p=subprocess.run(run,stdout=out,stderr=err)
(t/'terminal.json').write_text(json.dumps(dict(command=run,returncode=p.returncode,seconds=time.time()-start,elf_sha256=sha(d/'model.elf')),indent=2)+'\n');print((t/'stdout').read_text(),flush=True)
