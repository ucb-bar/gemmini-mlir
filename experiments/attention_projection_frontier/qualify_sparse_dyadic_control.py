"""Match unchanged normal provider and unchanged complete original driver."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
base=Path(__file__).resolve().parents[2];w=base/'out/sparse_dyadic_group';d=w/'control';d.mkdir(parents=True,exist_ok=False)
normal=base/'out/normal_composition/provider';dest=d/'target_numeric';dest.mkdir()
for p in (normal/'target_numeric').iterdir():
 if p.suffix in ('.c','.h'):shutil.copyfile(p,dest/p.name)
commands=[]
for old in json.loads((normal/'build.json').read_text())['roles']['target_numeric']['commands']:
 cmd=[x.replace('out/normal_composition/provider/target_numeric',str(dest)) for x in old];subprocess.run(cmd,check=True);commands.append(cmd)
assert (dest/'provider.o').read_bytes()==(normal/'target_numeric/provider.o').read_bytes()
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')
link=json.loads((b/'build.json').read_text())['link'];link=[str(dest/'provider.o') if x==str(b/'target_numeric/provider.o') else str(d/'model.elf') if x==str(b/'model.elf') else x for x in link]
subprocess.run(link,check=True,capture_output=True);commands.append(link)
(d/'build.json').write_text(json.dumps(dict(commands=commands,provider_default_byte_identity=True),indent=2)+'\n')
t=w/'control_strict';t.mkdir();run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
with (t/'stdout').open('w') as out,(t/'stderr').open('w') as err:p=subprocess.run(run,stdout=out,stderr=err)
(t/'terminal.json').write_text(json.dumps(dict(command=run,returncode=p.returncode,seconds=time.time()-start,elf_sha256=hashlib.sha256((d/'model.elf').read_bytes()).hexdigest()),indent=2)+'\n');print((t/'stdout').read_text(),flush=True)
