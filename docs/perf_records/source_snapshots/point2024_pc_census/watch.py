from pathlib import Path
import subprocess,json,hashlib,time,re
w=Path(__file__).resolve().parent;control=Path('/scratch/agustin/tmp/gemmini-probability-point-20261006/out/probability_points/candidate');elf=control/'model.elf';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();expected='ec3215988a150a0aea70eb10562b69c8bb617a4a5188138789790af247a01039';assert sha(elf)==expected
cmd=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(elf)]
intent={'command':cmd,'elf_sha256':expected,'scope':'Same-ELF PC histogram, original group; no code/timing instrumentation, not stock cycles','started':time.time()};(w/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
with (w/'spike.log').open('w')as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);(w/'process.json').write_text(json.dumps({'pid':p.pid}));rc=p.wait(timeout=900)
text=(w/'spike.log').read_text();control_text=(control/'spike.log').read_text();assert rc==0 and sha(elf)==expected
assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
for pattern in [r'^WORKSPACE_STAT (\d+) (\d+)$',r'^WORKSPACE_GROUP_INSTRUCTIONS (\d+)$']:
 assert re.findall(pattern,text,re.M)==re.findall(pattern,control_text,re.M)
intent.update(status='pass',returncode=rc,log_sha256=sha(w/'spike.log'),elapsed_seconds=time.time()-intent['started']);(w/'receipt.json').write_text(json.dumps(intent,indent=2)+'\n')
