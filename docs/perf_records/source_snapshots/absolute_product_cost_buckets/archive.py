from pathlib import Path
import json,hashlib,subprocess,shutil,re,datetime
w=Path(__file__).resolve().parent;r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-absolute-product-bounds-20261006');original=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/root-absolute-product-bounds-20261006');pins={}
def pin(p):p=Path(p).resolve();pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
snap=r/'docs/perf_records/source_snapshots/absolute_product_cost_buckets';snap.mkdir(parents=True,exist_ok=True)
for name in ['src/merlin/llvmlower/exact_absolute_products.py','merlin/runtime/c/exact_absolute_dot_bounds.h','src/merlin/llvmlower/source_attention_frontier.py','merlin/tests/runtime/test_exact_absolute_dot_bounds.py']:
 data=subprocess.check_output(['git','show','2f45db902:'+name],cwd=core);p=snap/Path(name).name;p.write_bytes(data);pin(p)
for p in w.glob('*.py'):shutil.copyfile(p,snap/p.name);pin(snap/p.name)
for p in w.rglob('*'):
 if p.is_file():pin(p)
results=[]
for bucket in ('high_readback','low_readback'):
 c=w/bucket
 for name in ('numeric_frozen','native_numeric_frozen'):
  for command in json.loads((c/name/'compile.json').read_text())['commands']:pin(command[0])
 for d in c.rglob('*.d'):
  for p in d.read_text().replace('\\\n',' ').split(':',1)[1].split():pin(p)
 for p in json.loads((c/'candidate/build.json').read_text())['link']:
  if Path(p).is_file():pin(p)
 v=json.loads((c/'candidate/spike_receipt.json').read_text());assert v['status']=='pass' and v['returncode']==0
 log=(c/'candidate/spike.log').read_text();assert 'ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in log
 count=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',log)[1]);results.append({'bucket':bucket,'instructions':count,'increase_percent':100*(count/2482366133-1),'elf_sha256':v['elf_sha256'],'stats':dict(re.findall(r'WORKSPACE_STAT (\d+) (\d+)',log)),'cost_policy':json.loads((c/'cost_policy.json').read_text())})
 nv=json.loads((c/'native/validation.json').read_text())
 for command in nv['commands']:
  for p in command:
   if Path(p).is_file():pin(p)
for sub in ('control','candidate'):
 for name in ('model.elf','build.json','spike_receipt.json','spike.log'):pin(original/sub/name)
for name in ('provider.c','provider.o','compile.json'):pin(original/'numeric_frozen'/name)
closure=json.loads((w/'native_pair_closure.json').read_text());assert closure['status']=='pass'
q={'schema':'absolute_product_cost_partition_review_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_proof_commit':'2f45db902','scope':'Two explicit diagnostic readback-intensity partitions of exact absolute source-product bounds. No automatic policy, model-name or golden selector. Complete group includes allocation, plane mutation, extra device calls/readbacks/reconstruction and changed source replay. Hardware cycles unknown.','control_instructions':2482366133,'all_stage_instructions':2675826572,'results':results,'native_whole':closure,'independent_proof_tests':42,'review':{'absolute_digits':'Canonical sign-coherent seven-bit magnitude digits; abs planes encode absolute reconstructed operand. Exact source/point-envelope equality required first.','integer_and_f64':'Original canonical degree/prefix range proof remains conservative for all-positive digits; row/column steps are exact positive binary powers.','prefix_overflow':'Every exact real prefix magnitude is at most total absolute product sum; global maximum plus outward gamma/subnormal radius below FLT_MAX admits finite source prefixes and endpoints.','lifetime':'Signed planes are mutated only after last signed product; center, source, reconstructed operands and envelopes stay separate. Readout/i64 scratch is reused; added absolute_center owns1MiB. Refusal uses original checked norm path with dead mutated planes.','eligibility':'No inference from group0 samples: full48 high1152/1152, low965/3456 actually admitted; unknown/nonpoint source retains checked bounds.','cost_model_limit':'20/(9K) includes only five i32 readbacks/nine cross-products, logical=physical only on the retained fully tiled shapes. Other costs intentionally excluded from partition but included in complete measured ROI.','decision':'Both partitions instruction-negative; no normal integration or hardware request. Whole native gates use an explicitly enlarged diagnostic workspace, not unchanged production ABI.'},'first_build_refusal':'Ambiguous text anchor refused before compilation; original failed build/log retained separately.','token_usage_available':False,'pins':pins};out=r/'docs/perf_records/absolute_product_cost_partition_review.json';out.write_text(json.dumps(q,indent=2)+'\n');print(out,len(pins))
