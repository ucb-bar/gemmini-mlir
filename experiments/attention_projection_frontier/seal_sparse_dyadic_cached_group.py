"""Retain the first complete exact sparse implementation as a cost negative."""
from pathlib import Path
import collections,hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];w=base/'out/sparse_dyadic_cached_group';core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def result(folder):
 terminal=json.loads((folder/'terminal.json').read_text());assert terminal['returncode']==0
 text=(folder/'stdout').read_text();assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
 rows=[line.split() for line in text.splitlines()];roi=int(next(r[1] for r in rows if r and r[0]=='WORKSPACE_GROUP_INSTRUCTIONS'));stats=[int(r[2]) for r in rows if r and r[0]=='WORKSPACE_STAT'];return dict(instructions=roi,stats=stats,terminal=terminal)
a,b=result(base/'out/sparse_dyadic_group/control_strict'),result(w/'strict');assert a['stats']==b['stats']
audits={name:audit_elf(((base/'out/sparse_dyadic_group/control/model.elf') if name=='control' else w/name/'model.elf').read_bytes()) for name in ('control','candidate')};assert all(x['status']=='pass' for x in audits.values())
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
attr=json.loads((base/'out/sparse_dyadic_cached_attribution/attribution.json').read_text());functions=collections.Counter()
for row in attr['rows']:functions[row['function']]+=row['instructions']
files={p.resolve() for p in w.rglob('*') if p.is_file()};files.update(p.resolve() for p in (base/'out/sparse_dyadic_cached_attribution').rglob('*') if p.is_file())
for name in ('adapt_sparse_dyadic','build_sparse_dyadic_group','qualify_sparse_dyadic_control','attribute_sparse_dyadic','seal_sparse_dyadic_group'):files.add((base/f'experiments/attention_projection_frontier/{name}.py').resolve())
for name in ('sparse_dyadic_products','sparse_dyadic_rms','fused_encoded_sparse_count','fused_encoded_witness','source_roundoff_policy'):
 files.add(core/f'src/merlin/llvmlower/{name}.py')
for name in ('sparse_dyadic_products','sparse_dyadic_rms','fused_encoded_sparse_count'):files.add(core/f'merlin/tests/runtime/test_{name}.py')
files.add(base/'docs/perf_records/sparse_dyadic_source_admission.json')
files.add(core/'src/merlin/llvmlower/sparse_dyadic_cached.py')
files.add(core/'merlin/tests/runtime/test_sparse_dyadic_cached.py')
files.add(base/'docs/perf_records/sparse_dyadic_complete_group_negative.json')
for name in ('build_sparse_dyadic_cached_group','attribute_sparse_dyadic_cached','seal_sparse_dyadic_cached_group'):files.add(base/f'experiments/attention_projection_frontier/{name}.py')
for name in ('candidate',):
 for cmd in json.loads((w/name/'build.json').read_text())['commands']:
  for word in cmd:
   p=Path(word)
   if p.is_file():files.add(p.resolve())
record=dict(scope='Complete original first12-head group, normal owner-capable source called with absent owner. Matched fresh unchanged provider object is byteidentical the normal composition, not the older2072 object; all allocation/metadata/duplicate sparse preparation/offset/readout/fallback work is in ROI.',status='NEGATIVE_NO_FULL48_CANDIDATE_OR_HARDWARE',control=a,candidate=b,change_percent=(b['instructions']/a['instructions']-1)*100,workspace_bytes={'control':124061504,'candidate':127041344},product_entry_counts=calls,actual_callbacks=sum(calls.values()),source_policy='Unchanged explicit approximate RMS4 source rounding; exact corrected mathematical source products do not prove ordered-F32 equality.',mechanism='Cached IEEE powers, integer BF16 exponent/RNE extraction replace libm without representation change. 196608 BF16 word comparisons passed. Mandatory canonical encoder adds a fixed two-byte residual count. One correction/output operation cap screens before extra preparation. Admitted operands use offset-two-byte degrees0..2, source-derived exact sparse correction and separate producer-issued corrected-product witness. Rejected batches retain original3digit route. Prepared RHS original representation remains reusable; sparse preparation is currently duplicated and fully counted.',private_product_descriptor='Ordinals0..4 complete canonical3digit,5..7 offset2byte degrees0..2. OOT dispatch binds explicit signed128 catalog; it does not assert unchanged old product semantics.',no_fsm_audits=audits,attribution={'identity':{'object':attr['object_identity'],'elf':attr['elf_identity']},'whole_executable_functions':dict(functions.most_common()),'scope':'Shared callees are whole-executable counts and not automatically ROI/exclusive source costs.'},tests=57,limitations=['No candidate full48 numerical claim; preceding95-pin packet validates exact corrected centers on original all48 operands independently.','No calibrated hardware ranking; instructions are not cycles.','Old2072 is a historical comparator only; matched control is1747436465.'],pins={str(p):sha(p) for p in sorted(files)})
(base/'docs/perf_records/sparse_dyadic_cached_complete_group_negative.json').write_text(json.dumps(record,indent=2)+'\n');print('PINS',len(files));print('CHANGE',record['change_percent']);print('CALLBACKS',sum(calls.values()))
