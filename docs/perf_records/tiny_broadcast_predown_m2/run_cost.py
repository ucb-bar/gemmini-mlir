"""Time the already qualified immutable broadcast capsule; no rebuild."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
from merlin.perf.layer_bench import run_on_gsim

W = Path(__file__).resolve().parent
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
save = lambda p, x: Path(p).write_text(json.dumps(x, indent=2, default=str)+'\n')
record = json.loads((W/'capsule_qualification.json').read_text())
elf = W/'timing_build/layer.elf'
assert sha(elf)==record['elf_sha256']
assert record['all5_frm_and_sticky_flags_match'] and record['zero_FSM']
for case in record['cases']:
    assert sha(W/(case['name']+'.o'))==case['target_object_sha256']
result = run_on_gsim(elf,target='gemmini',timeout_s=1800,max_cycles=30000000,stdout_path=W/'gsim.stdout')
save(W/'gsim_receipt.json',asdict(result))
assert sha(elf)==record['elf_sha256']
assert result.returncode==0 and result.finish and result.finish.done
console=(W/'gsim.stdout').read_text()
assert 'ORIGINAL_BROADCAST_COMPLETE PASS' in console
rows=[[int(x) for x in line.split()[1:]] for line in console.splitlines() if line.startswith('BROADCAST_COMPLETE_CYCLES ')]
assert [(i,a) for i,a,c in rows]==[(0,0),(1,1),(2,1),(3,0)]
old=[c for i,a,c in rows if a==0]
new=[c for i,a,c in rows if a==1]
final={**record,'paired_complete_gsim_cycles':rows,'before_mean':sum(old)/2,'after_mean':sum(new)/2,'saved_percent':100*(sum(old)-sum(new))/sum(old),'DONE':True,'console_sha256':sha(W/'gsim.stdout'),'decision':'Retain positive local typed broadcast schedule screen for full qualification.' if sum(new)<sum(old) else 'Reject promotion: complete local broadcast cost loses.'}
save(W/'capsule_result.json',final)
print('BROADCAST_COMPLETE_RESULT',rows,final['saved_percent'],flush=True)
