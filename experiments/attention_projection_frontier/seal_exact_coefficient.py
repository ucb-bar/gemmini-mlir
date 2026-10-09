"""Retain exact all48 proof and complete-group cost negative."""
from pathlib import Path
import json,hashlib
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007');w=base/'out/exact_coefficient_group';native=base/'out/observation_frontier/exact_coefficient_native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((native/'qualification.json').read_text());assert q['snapshots']==48 and q['product_calls']==23040 and not q['bit_mismatches'] and not q['errors'] and q['counts'][1]==0
text=(w/'strict/stdout').read_text();assert 'ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
roi=int(next(x.split()[1]for x in text.splitlines()if x.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
control=base/'out/sparse_dyadic_group/control_strict/stdout';ct=control.read_text();old=int(next(x.split()[1]for x in ct.splitlines()if x.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
assert [x for x in text.splitlines()if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]==[x for x in ct.splitlines()if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]
audit=audit_elf((w/'candidate/model.elf').read_bytes());assert audit['status']=='pass'
files={p for folder in (w,native)for p in folder.rglob('*')if p.is_file()}
files.update([control,base/'docs/perf_records/sparse_dyadic_complete_group_negative.json',core/'src/merlin/llvmlower/exact_coefficient_widen.py',core/'merlin/tests/runtime/test_exact_coefficient_widen.py'])
for name in ('build_exact_coefficient_group','qualify_exact_coefficient_native','seal_exact_coefficient'):files.add(base/f'experiments/attention_projection_frontier/{name}.py')
for name in ('exact_coefficient_group_driver.py','exact_coefficient_native_driver.py','exact_coefficient_group.log','exact_coefficient_native.log'):files.add(base/'out'/name)
record=dict(status='COST_NEGATIVE_NO_HARDWARE',control_instructions=old,candidate_instructions=roi,change_percent=(roi/old-1)*100,native=dict(groups=48,preparations=12,products=23040,fallback=0,original1600_bit_mismatches=0),tests=21,all_bf16_input_comparisons=196608*3,audit=audit,conclusion='Per-word exact identity is numerically valid but runtime branch/code costs exceed removed FP reconstruction. Fixed source/RMS4/default original helper unchanged.',pins={str(p):sha(p)for p in sorted(files)})
(base/'docs/perf_records/exact_coefficient_widen_cost_negative.json').write_text(json.dumps(record,indent=2)+'\n');print(len(files),record['change_percent'])
