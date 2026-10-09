from pathlib import Path
import hashlib,json,subprocess,shutil,re,datetime
w=Path(__file__).resolve().parent;r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-coarse-absolute-upper-20261006');pins={}
def pin(p):
 p=Path(p).resolve();pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
snap=r/'docs/perf_records/source_snapshots/coarse_absolute_upper';snap.mkdir(parents=True,exist_ok=True)
for name in ['src/merlin/llvmlower/coarse_absolute_upper.py','src/merlin/llvmlower/exact_absolute_products.py','src/merlin/llvmlower/source_attention_frontier.py','merlin/runtime/c/absolute_upper_dot_bounds.h','merlin/tests/runtime/test_coarse_absolute_upper.py','docs/runtime/coarse_absolute_upper.md']:
 p=snap/Path(name).name;p.write_bytes(subprocess.check_output(['git','show','d80c90082:'+name],cwd=core));pin(p)
for p in w.glob('*.py'):shutil.copyfile(p,snap/p.name);pin(snap/p.name)
for p in w.rglob('*'):
 if p.is_file():pin(p)
for name in ['control_numeric','numeric_frozen','native_numeric_frozen']:
 for cmd in json.loads((w/name/'compile.json').read_text())['commands']:pin(cmd[0])
 for d in (w/name).glob('*.d'):
  for p in d.read_text().replace('\\\n',' ').split(':',1)[1].split():pin(p)
for arm in ['control','candidate']:
 for p in json.loads((w/arm/'build.json').read_text())['link']:
  if Path(p).is_file():pin(p)
n=json.loads((w/'native/validation.json').read_text());assert n['calls'][:3]==[48,0,0] and n['bitwise_mismatches']==0 and n['allclose'] and n['product_calls']==25157
for cmd in n['commands']:
 for p in cmd:
  if Path(p).is_file():pin(p)
v=json.loads((w/'candidate/spike_receipt.json').read_text());assert v['status']=='pass'
log=(w/'candidate/spike.log').read_text();count=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',log)[1]);assert count==2525442576
original=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/root-absolute-product-bounds-20261006')
for f in ['control/model.elf','control/spike.log','control/spike_receipt.json','candidate/spike.log']:pin(original/f)
assert hashlib.sha256((w/'control/model.elf').read_bytes()).hexdigest()==hashlib.sha256((original/'control/model.elf').read_bytes()).hexdigest()
q={'schema':'coarse_absolute_upper_qualification_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_commit':'8fed8bd44','independent_tests_commit':'d80c90082','tests_passed':52,'precision_bits':6,'shift':15,'workspace_bytes':124061504,'scope':'One explicit coarse positive absolute upper representation; control encoded-row source provider reproduced byte-identically. Not automatic cost selection or whole target integration.','original_group':v,'instructions':{'control':2482366133,'candidate':count,'increase_percent':100*(count/2482366133-1)},'stats':dict(re.findall(r'WORKSPACE_STAT (\d+) (\d+)',log)),'native_whole':n,'native_scope_correction':'Inherited validation scope string says diagnostic cost partition; this run instead uses coarse six-bit absolute upper for every eligible contraction. Explicit guarded enlarged caller pool is diagnostic-only, not ordinary workspace ABI integration. First nohup attempt made no execution progress; detached Popen completed once.','costs':{'extra_callback_per_eligible_contraction':1,'extra_readback_bytes_per_contraction':'4*m*n','extra_scalar_mac_per_contraction':'m*n*k','upper_conversion':'Read all three signed planes per operand element; write one ceil magnitude byte after final signed use.','reconstruction':'One i32-to-f64 conversion and two power-of-two scalings per output; no extra i64 reduction.','full48_extra_callbacks':2117,'full48_possible_contractions':4608,'full_group_max_extra_callbacks':96,'full_group_max_extra_readback_bytes':17301504,'full_group_max_extra_scalar_mac':402653184,'limits':'Group-specific eligibility and saved-host-cycle split were not instrumented; upper counts are analytical maximums, not measured traffic. Full ROI includes allocation/init/conversion/callback/readback/scaling/proof/fallback/replay. Native callback total is measured with exact integer stand-ins, not hardware.'},'decision':'Original group consumer and full1600 native exact, but complete target instructions +1.735%; retain negative, no hardware request or promotion. Hardware cycles unknown.','token_usage_available':False,'pins':pins}
(r/'docs/perf_records/coarse_absolute_upper_qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(len(pins))
