from pathlib import Path
import hashlib,json,subprocess,time
work=Path(__file__).parent;start=time.monotonic()
command=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(work/'check.elf')]
with(work/'spike.log').open('w')as file:
 try:code=subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,timeout=600).returncode
 except subprocess.TimeoutExpired:code='timeout'
text=(work/'spike.log').read_text()
receipt={'command':command,'returncode':code,'status':'pass'if code == 0 and 'BENCHMARK_BUFFER EXACT PASS'in text else'fail','elapsed_seconds_functional_only':time.monotonic()-start,'elf_sha256':hashlib.sha256((work/'check.elf').read_bytes()).hexdigest(),'log_sha256':hashlib.sha256((work/'spike.log').read_bytes()).hexdigest(),'metric_scope':'Functional Spike mcycle instruction proxy, notFireSimcycles'}
(work/'spike_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
