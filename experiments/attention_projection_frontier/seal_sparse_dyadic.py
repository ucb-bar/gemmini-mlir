"""Freeze exact sparse admission and actual-source functional evidence."""
from pathlib import Path
import collections, hashlib, json, subprocess
base=Path(__file__).resolve().parents[2]
core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007')
a=base/'out/observation_frontier/sparse_dyadic_admission'
b=base/'out/observation_frontier/sparse_dyadic_source_run'
c=base/'out/observation_frontier/sparse_dyadic_source'
x=json.loads((a/'qualification.json').read_text());y=json.loads((b/'qualification.json').read_text())
assert not x['errors'] and not y['errors'] and x['bit_mismatches']==y['bit_mismatches']==0
summary=collections.defaultdict(collections.Counter)
for row in x['exponent_census']:
 s=summary[row['k']];s['calls']+=1;s['prefix_admitted']+=row['exact_f64_prefix_admitted'];s['correction_terms']+=row['sparse_correction_terms'];s['maximum_prefix_bits']=max(s['maximum_prefix_bits'],row['prefix_bits'])
for row in y['exact_sparse_source_census']:summary[row['k']]['actual_c_admitted']+=row['accepted']
files=set()
for folder in (a,b,c):files.update(p.resolve() for p in folder.rglob('*') if p.is_file())
files.update((base/p).resolve() for p in ('experiments/attention_projection_frontier/screen_sparse_dyadic_admission.py','experiments/attention_projection_frontier/check_sparse_dyadic_source.py','experiments/attention_projection_frontier/seal_sparse_dyadic.py','experiments/attention_projection_frontier/screen_exponent_partition.py','experiments/attention_projection_frontier/capture_initial_bounds.py','out/sparse_dyadic_admission_driver.py','out/sparse_dyadic_source_driver.py','out/sparse_dyadic_admission.log','out/sparse_dyadic_source.log','out/normal_composition/qualification_source_rebound.json'))
files.update((core/p).resolve() for p in ('src/merlin/llvmlower/sparse_dyadic_products.py','merlin/tests/runtime/test_sparse_dyadic_products.py'))
for r in (x,y):
 for command in r['commands']:
  for item in command:
   p=Path(item)
   if p.is_file():files.add(p.resolve())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(scope='Read-only admission and additional generic exact-product calculations on every actual source operand set. Original provider remains selected. No changed whole candidate, target timing or hardware observation.',core_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=core,text=True).strip(),summary=dict(summary),native_source={'groups':48,'source_product_calls':4608,'original_degree_callbacks':23040,'bit_mismatches':0,'additional_three_degree_calls':y['actual_sparse_product_calls'],'additional_exact_source_cells':y['actual_checked_cells'],'diagnostic_errors':y['errors']},proof='For per-row common dyadic grid, K*(QA*QB + EA*SB + QA*EB) <= 2^53 bounds each dense-plus-correction prefix. Offset limb reconstruction has a separate exact integer prefix bound. BF16 exponent range makes exact dyadic scales finite and normal in binary64. Result is the mathematical source dot, not ordered-F32 accumulation.',resource_admission='Explicit one sparse correction term per output cell. This is a fixed experimental operation-count cap, not an estimated cycle equivalence or automatic production policy; no shape/name/source-ID selects it.',preparation_storage='Current reference producer allocates separate selected doubles, two signed-byte planes, worst-case size_t sparse indices and per-row metadata, plus three readout planes and private F64 output. Dual representation and prepared-owner reuse costs remain to be implemented and measured.',requirements=['distinct corrected-source-product witness, never fake equality flags for approximate operand arrays','existing source-FMA RMS4 permission and original replay remain','original three-digit checked fallback for resource/range/refusal','source noescape/immutable owner spans through synchronous complete callbacks','all preparation, metadata, offset arithmetic, allocation, readouts and fallback in complete cost'],qualification={'focused_native_refusal_ubsan_tests':26,'related_tests_prior_final_readonly_alias':51,'format':'PASS','no_regex':'PASS'},limitations=['No target execution or performance candidate measured.','Diagnostic get_counts array includes legacy opaque epoch slots; it is not the eight numerical provider statistics.','Dense validation callback is native exact-integer simulation; target product correctness requires the already frozen signed-byte catalog or fresh equivalent qualification.'],pins={str(p):sha(p) for p in sorted(files)})
path=base/'docs/perf_records/sparse_dyadic_source_admission.json';path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(summary,indent=2));print('PINS',len(files))
