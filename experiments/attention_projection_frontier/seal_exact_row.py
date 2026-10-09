"""Seal one positive complete source-group row-proof implementation screen."""
from pathlib import Path
import collections,ctypes,hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
from merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix
base=Path(__file__).resolve().parents[2];core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007');w=base/'out/exact_row_group';native=base/'out/observation_frontier/exact_row_native';target=base/'out/exact_row_target';attrdir=base/'out/exact_row_attribution'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((native/'qualification.json').read_text());assert q['snapshots']==48 and q['product_calls']==23040 and not q['bit_mismatches'] and not q['errors'] and q['counts'][1]==0
text=(w/'strict/stdout').read_text();assert 'ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
roi=int(next(x.split()[1]for x in text.splitlines()if x.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
control=base/'out/sparse_dyadic_group/control_strict/stdout';ct=control.read_text();old=int(next(x.split()[1]for x in ct.splitlines()if x.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
assert [x for x in text.splitlines()if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]==[x for x in ct.splitlines()if x.startswith(('WORKSPACE_STAT','UNOBSERVED_CARRIER'))]
assert prepare_exact_row_radix((base/'out/normal_composition/provider/target_numeric/provider.c').read_text())==(w/'candidate/target_numeric/provider.c').read_text()
assert (base/'out/normal_composition/provider/target_numeric/provider.o').read_bytes()==(base/'out/sparse_dyadic_group/control/target_numeric/provider.o').read_bytes()
lib=ctypes.CDLL(str(native/'numeric/provider.so'));lib.group_provider_workspace_bytes.restype=ctypes.c_size_t;capacity=lib.group_provider_workspace_bytes();assert capacity==124061504
lib.group_provider_rhs_owner_bytes.restype=ctypes.c_size_t;owner=lib.group_provider_rhs_owner_bytes();assert owner==30048912
proof=json.load(open(target/'qualification.json'));assert proof['comparisons']==589824 and proof['modes']==5
audit=audit_elf((w/'candidate/model.elf').read_bytes());assert audit['status']=='pass'
a=json.load(open(attrdir/'attribution.json'));functions=collections.Counter()
for row in a['rows']:functions[row['function']]+=row['instructions']
hist={}
for line in (w/'strict/stderr').read_text().splitlines():
 t=line.split()
 if len(t)==2:
  try:hist[int(t[0],16)]=int(t[1])
  except ValueError:pass
nm=subprocess.check_output(['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-nm','--defined-only',str(w/'candidate/model.elf')],text=True);calls={}
for line in nm.splitlines():
 t=line.split()
 if len(t)==3 and '_products_' in t[2]:calls[t[2]]=hist.get(int(t[0],16),0)
assert sum(calls.values())==480
files={p for folder in (w,native,target,attrdir)for p in folder.rglob('*')if p.is_file()}
files.update([control,base/'docs/perf_records/sparse_dyadic_complete_group_negative.json',core/'src/merlin/llvmlower/exact_row_radix_pack.py',core/'src/merlin/llvmlower/fused_encoded_witness.py',core/'merlin/tests/runtime/test_exact_row_radix_pack.py',core/'merlin/tests/runtime/test_exact_coefficient_widen.py'])
for name in ('build_exact_row_group','qualify_exact_row_native','qualify_exact_row_target','attribute_exact_row','seal_exact_row'):files.add(base/f'experiments/attention_projection_frontier/{name}.py')
for name in ('exact_row_group_driver.py','exact_row_native_driver.py','exact_row_group.log','exact_row_native.log','exact_row_target.log','exact_row_attribution.log'):files.add(base/'out'/name)
for manifest in (q,json.load(open(w/'candidate/build.json')),proof):
 for cmd in manifest['commands']:
  for value in cmd:
   p=Path(value)
   if p.is_file():files.add(p.resolve())
record=dict(status='QUALIFIED_INSTRUCTION_POSITIVE_HELD_NO_HARDWARE',control_instructions=old,candidate_instructions=roi,change_percent=(roi/old-1)*100,control_elf_sha=sha(base/'out/sparse_dyadic_group/control/model.elf'),candidate_elf_sha=sha(w/'candidate/model.elf'),matched_scope='Fresh normal owner-capable source called without owner, original complete first12-head group including allocation. Not byteidentical historical2072.',native=dict(groups=48,preparations=12,products=23040,fallback=0,original1600_bit_mismatches=0),workspace_bytes=capacity,owner_payload_bytes=owner,tests=49,target_comparisons=589824,target_modes=5,audit=audit,callback_entries=calls,callback_count=sum(calls.values()),numerical_contract='Source-exact canonical packing/widening transformation. Existing explicit approximate RMS4 source-roundoff policy and original whole gate unchanged; no new observation permission or point proof from estimates.',theorem='Mandatory finite BF16 scan proves nonzero normal row exponent span <=7*digits-8; every signed coefficient is exact/unsaturated and fits<=21 bits. Direct source widening equals original int->F32*power-of-two->F64; remaining rows use original encoder, including zeros/subnormals/uncertain endpoints.',attribution=dict(identity={'object':a['object_identity'],'elf':a['elf_identity']},exclusive_function_bodies=dict(functions.most_common()),scope='Whole histogram includes post-ROI source consumer. Use accepted2072 callsite exclusion; no shared helper or cycle assignment.'),limitations=['No whole target candidate built from this change. Native normal model uses same unchanged workspace/source call ABI with alternative numeric provider.','No hardware cycles or default promotion; root review/release required.','No whole5B forecast from the group instruction gain.'],pins={str(p):sha(p)for p in sorted(files)})
(base/'docs/perf_records/exact_row_radix_complete_group_qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(len(files),record['change_percent']);print(functions['encode_operand'])
