from pathlib import Path
import json,hashlib,shutil,subprocess
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parent
old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed')
core=Path('/scratch/agustin/tmp/merlin-source-fma-eight-20261006')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for arm in ['control','candidate']:
 d=w/arm; d.mkdir(exist_ok=False); n=d/'numeric';n.mkdir()
 source=old/'numeric_frozen'
 for p in source.glob('*.h'):shutil.copyfile(p,n/p.name)
 shutil.copyfile(core/'merlin/runtime/c/prepared_polynomial_batch.h',n/'prepared_polynomial_batch.h')
 (n/'provider.c').write_text((source/'provider.c').read_text())
 if arm=='candidate': shutil.copyfile(w/'host_fma_batch.h',n/'host_fma_batch.h')
 commands=json.loads((source/'compile.json').read_text())['commands'];actual=[]
 for argv in commands:
  argv=[v.replace(str(source),str(n)) for v in argv]
  if arm=='candidate':argv[1:1]=['-include',str(n/'host_fma_batch.h')]
  subprocess.run(argv,check=True);actual.append(argv)
 (n/'compile.json').write_text(json.dumps({'commands':actual},indent=2)+'\n')
 if arm=='control':assert sha(n/'provider.o')==sha(source/'provider.o')
 link=json.loads((old/'candidate/build.json').read_text())['link']
 link=[str(d/'model.elf')if v==str(old/'candidate/model.elf')else str(n/'provider.o')if v==str(source/'provider.o')else v for v in link]
 subprocess.run(link,check=True)
 if arm=='control':assert sha(d/'model.elf')==sha(old/'candidate/model.elf')
 audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass'
 (d/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 (d/'build.json').write_text(json.dumps({'link':link,'elf_sha256':sha(d/'model.elf')},indent=2)+'\n')
 if arm=='candidate':
  shutil.copyfile(old/'candidate/watch.py',d/'watch.py')
  with(d/'watch.log').open('w')as log:p=subprocess.Popen(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(d/'watch.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (d/'process.json').write_text(json.dumps({'pid':p.pid}));print('STRICT_LIVE',p.pid,sha(d/'model.elf'),flush=True)
# Reuse original independent 1,003 interval-case data, not workload arrays for
# the numerical proof. New C and object/link destinations leave frozen originals intact.
r=json.loads(Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/prepared_polynomial_four_cell_qualification.json').read_text())['independent_proof']
d=w/'independent';d.mkdir(exist_ok=False)
oldproof=Path('/scratch/agustin/tmp/gemmini-polynomial-four-cell-20261006/out/polynomial_four_cell/independent_target/proof.c')
shutil.copyfile(oldproof,d/'proof.c')
argv=[v.replace('/scratch/agustin/tmp/merlin-polynomial-four-cell-20261006/merlin/runtime/c',str(core/'merlin/runtime/c')).replace(str(oldproof.parent),str(d)) for v in r['compile']]
shutil.copyfile(oldproof.parent/'exact_provider.h',d/'exact_provider.h')
argv[1:1]=['-include',str(w/'host_fma_batch.h')]
subprocess.run(argv,check=True)
link=[v.replace(str(oldproof.parent),str(d)) for v in r['link']];subprocess.run(link,check=True)
a=audit_elf((d/'model.elf').read_bytes());assert a['status']=='pass'
(d/'model.nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n')
with(d/'spike.log').open('w')as log:
 cmd=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(d/'model.elf')]
 result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=120)
text=(d/'spike.log').read_text();assert result.returncode==0 and 'POLYNOMIAL_FOUR 0' in text,text
(d/'receipt.json').write_text(json.dumps({'status':'pass','compile':argv,'link':link,'command':cmd,'elf_sha256':sha(d/'model.elf'),'log_sha256':sha(d/'spike.log'),'independent_cases':1003,'actual_rounding_modes':5,'endpoint_checks':40120,'noFSM':a},indent=2)+'\n')
print('INDEPENDENT_TARGET_PASS',flush=True)
