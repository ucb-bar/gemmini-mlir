from pathlib import Path
from dataclasses import asdict
import hashlib,json,re
from merlin.perf.layer_bench import run_on_gsim
W=Path(__file__).resolve().parent
q=json.loads((W/'qualification.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'schema':'current_complete_transposed_stationary_gsim_pair_v1','arms':{},'status':'running','ROI':q['ROI'],'token_usage_available':False}
for arm in ['dense','transposed']:
 path=Path(q['elfs'][arm]['path']);assert sha(path)==q['elfs'][arm]['sha256']
 r=run_on_gsim(path,target='gemmini',max_cycles=60000000,timeout_s=4500,stdout_path=W/arm/'gsim.stdout')
 (W/arm/'gsim_receipt.json').write_text(json.dumps(asdict(r),indent=2,default=str)+'\n')
 assert r.completed and r.returncode==0
 log=(W/arm/'gsim.stdout').read_text();events=[list(map(int,x))for x in re.findall(r'TRANSPOSED_STATIONARY_COMPLETE_CYCLES (\d+) (\d+) (\d+)',log)]
 assert len(events)==3 and 'TRANSPOSED_STATIONARY_COMPLETE PASS' in log
 result['arms'][arm]={'events':events,'warm_mean':sum(e[2]for e in events[1:])/2,'elf_sha256':sha(path),'stdout_sha256':sha(W/arm/'gsim.stdout')}
 (W/'pair_result.json').write_text(json.dumps(result,indent=2)+'\n');print('TRANSPOSED_GSIM_COMPLETE',arm,events,flush=True)
result['warm_change_percent']=100*(result['arms']['transposed']['warm_mean']/result['arms']['dense']['warm_mean']-1)
result['status']='pass';result['whole_cycles']=None
(W/'pair_result.json').write_text(json.dumps(result,indent=2)+'\n');print('TRANSPOSED_CURRENT_PAIR_VERDICT',result['warm_change_percent'],flush=True)
