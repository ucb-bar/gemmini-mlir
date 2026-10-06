from pathlib import Path
import subprocess,json,hashlib,time,re
import numpy as np
w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();elf=w/'build/model.elf';expected='043f0e592778a35609ce415b93d66cdc45e0c39c6339149111945fb155517fd1';assert sha(elf)==expected
native=json.loads((w/'native/validation.json').read_text());assert native['bitwise_mismatches']==0 and native['allclose'];reference=w/'native/output.npy';x=np.load(reference);digest=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest();assert x.size==1600
from mlir_oot.no_fsm_audit import audit_elf
assert audit_elf(elf.read_bytes())['status']=='pass'
cmd=['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--extension=gemmini','--isa=rv64gc','-m0x80000000:0x400000000',str(elf)]
intent={'scope':'Actual ordinary whole Smol original1600 gate with source-consumer provider, pooledworkspace and retainedsource fallback. Target correctness only; no performance promotion.','elf_sha256':expected,'reference_path':str(reference),'reference_sha256':sha(reference),'expected_output_sha256':digest,'command':cmd,'started_at':time.time(),'token_usage_available':False};(w/'target_intent.json').write_text(json.dumps(intent,indent=2)+'\n')
with (w/'target_spike.log').open('wb')as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);(w/'target_process.json').write_text(json.dumps({'spike_pid':p.pid,'started_at':time.time()},indent=2))
 try:rc=p.wait(timeout=21600)
 except subprocess.TimeoutExpired:rc=None
if rc is None:
 intent['status']='observation_deadline_process_still_live';(w/'target_observation.json').write_text(json.dumps(intent,indent=2));raise SystemExit(0)
text=(w/'target_spike.log').read_text().replace('\r','');passed=rc==0 and f'OUT_SHA256 f32le 1600 6400 {digest}' in text and re.search(r'^DONE$',text,re.M) and re.search(r'^METRIC memref_rank_mismatch 0$',text,re.M)
intent.update(returncode=rc,status='pass' if passed else 'fail',log_sha256=sha(w/'target_spike.log'),elapsed_seconds=time.time()-intent['started_at'],final_elf_unchanged=sha(elf)==expected);assert intent['final_elf_unchanged'];(w/'target_validation.json').write_text(json.dumps(intent,indent=2)+'\n');print(intent['status'],flush=True)
