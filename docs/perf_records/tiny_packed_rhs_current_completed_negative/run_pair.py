from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,re
from merlin.perf.layer_bench import run_on_gsim
work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((work/'qualification.json').read_text())
assert proof['same_executable_bytes'] and proof['same_all_symbol_addresses'] and proof['original_control_typed_body_structurally_identical']
assert json.loads((work/'independent/qualification.json').read_text())['status']=='pass'
outputs={}
for arm in ['dense','packed']:
    elf=Path(proof['elfs'][arm]['path'])
    assert sha(elf)==proof['elfs'][arm]['sha256']
    print('START',arm,datetime.now(timezone.utc).isoformat(),flush=True)
    result=run_on_gsim(elf,target='gemmini',max_cycles=50000000,timeout_s=7200,stdout_path=work/arm/'gsim.stdout')
    (work/arm/'gsim_receipt.json').write_text(json.dumps(asdict(result),indent=2,default=str)+'\n')
    assert sha(elf)==proof['elfs'][arm]['sha256']
    assert result.completed,repr(result)
    console=(work/arm/'gsim.stdout').read_text()
    assert 'PACKED_RHS_COMPLETE PASS' in console
    events=[(int(a),int(r),int(c))for a,r,c in re.findall(r'PACKED_RHS_COMPLETE_CYCLES (\d+) (\d+) (\d+)',console)]
    assert len(events)==3 and all(a==int(arm=='packed')for a,r,c in events)
    outputs[arm]={'events':events,'warm_mean':sum(c for a,r,c in events if r>0)/2,'elf_sha256':sha(elf),'stdout_sha256':sha(work/arm/'gsim.stdout')}
    print('CLOSED',arm,outputs[arm],datetime.now(timezone.utc).isoformat(),flush=True)
ratio=outputs['packed']['warm_mean']/outputs['dense']['warm_mean']
(work/'pair_result.json').write_text(json.dumps({'schema':'current_complete_packed_rhs_gsim_pair_v1','status':'pass','arms':outputs,'warm_candidate_over_control':ratio,'warm_change_percent':100*(ratio-1),'roi_scope':'Same virtual addresses and executable bytes; complete actual8x5632x2048 primitive kernel, not whole Tiny nor FireSim. Cold repeat0 retained separately; warm repeats1/2. Offline exact parameter permutation outside runtime ROI. No bytes-saving or whole-cycle extrapolation.','token_usage_available':False},indent=2)+'\n')
print('PAIRED_RESULT',100*(ratio-1),flush=True)
