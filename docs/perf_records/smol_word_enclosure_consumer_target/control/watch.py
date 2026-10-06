from pathlib import Path
import subprocess,json,hashlib,time
w=Path(__file__).parent;elf=w/'model.elf';sha=hashlib.sha256(elf.read_bytes()).hexdigest();start=time.time()
cmd=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(elf)]
with (w/'spike.log').open('w') as log:
 try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=7200);rc=r.returncode
 except subprocess.TimeoutExpired:rc='timeout'
text=(w/'spike.log').read_text();passed=rc==0 and 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
(w/'spike_receipt.json').write_text(json.dumps({'scope':'Complete original group actual product+ranked workspace bridge capsule; source refusal deliberately fatal, not normal production fallback qualification','elf_sha256':sha,'elapsed_seconds':time.time()-start,'command':cmd,'returncode':rc,'status':'pass'if passed else'fail','log_sha256':hashlib.sha256((w/'spike.log').read_bytes()).hexdigest(),'token_usage_available':False},indent=2)+'\n')
