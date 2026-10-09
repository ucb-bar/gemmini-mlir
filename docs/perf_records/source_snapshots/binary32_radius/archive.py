from pathlib import Path
import json,hashlib,subprocess,shutil,re,datetime
w=Path(__file__).resolve().parent;r=w.parents[1];core=Path('/scratch/agustin/tmp/merlin-binary32-source-radius-20261006');old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed');pins={}
def pin(p):p=Path(p).resolve();pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
snap=r/'docs/perf_records/source_snapshots/binary32_radius';snap.mkdir(parents=True,exist_ok=True)
for n in ['merlin/runtime/c/binary32_separable_radius.h','src/merlin/llvmlower/binary32_separable_radius.py','merlin/tests/runtime/test_binary32_separable_radius.py']:
 p=snap/Path(n).name;p.write_bytes(subprocess.check_output(['git','show','d65f4bf19:'+n],cwd=core));pin(p)
p=snap/'host_binary32_radius.py';p.write_bytes(subprocess.check_output(['git','show','c71968f:mlir_oot/host_binary32_radius.py'],cwd=r));pin(p)
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
n=json.loads((w/'native/validation.json').read_text());assert n['calls'][:3]==[48,0,0] and n['bitwise_mismatches']==0 and n['allclose'] and n['product_calls']==23040
for cmd in n['commands']:
 for p in cmd:
  if Path(p).is_file():pin(p)
v=json.loads((w/'candidate/spike_receipt.json').read_text());assert v['status']=='pass'
for p in ['candidate/model.elf','candidate/spike_receipt.json','candidate/spike.log','numeric_frozen/provider.o']:pin(old/p)
log=(w/'candidate/spike.log').read_text();count=int(re.search(r'WORKSPACE_GROUP_INSTRUCTIONS (\d+)',log)[1]);assert count==2348442902
q={'schema':'binary32_radius_negative_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_commit':'d65f4bf19','oot_capability_commit':'c71968f','scope':'Explicit conservative binary32 separable radius atop composed2003 provider; fresh actual control object and final09ca ELF byte-identical. Native product stand-ins use unchanged workspace descriptor ABI, not fresh normal production reseal.','independent_native_tests':30,'proof':'Upward f32 factor and eta enclose admitted nonnegative source gamma*L1 and underflow error. Directed FMA with upward column maximum bounds their real product-plus-eta. Directed center casts and directed additions enclose exact source center +/-radius. Nonfinite widened endpoints retain complete prior checked bound. Source equality/point domains and finite source prefix proofs remain unchanged.','candidate':v,'instructions':{'control':2288221154,'candidate':count,'increase_percent':100*(count/2288221154-1)},'stats':dict(re.findall(r'WORKSPACE_STAT (\d+) (\d+)',log)),'native_whole':n,'assembly':'Actual provider.o has fmadd.s rup0x4e58, fadd.s rdn0x4e64, fadd.s rup0x4e70; static rm does not mutate ambient FRM. Old radius used fmul.d rup, three directed fadd.d and two directed casts. New stored-column conversion and finite checks add overhead.','limits':'Independent exhaustive target five-FRM/corner capability proof not completed: stopped optional qualification after negative complete cost. Complete original target consumer and full native gate pass; no claim of production capability release or hardware performance. Local FPU defaults sfma3/dfma4 are not bitstream-derived timing evidence.','decision':'Rejected on +2.632% complete instructions and increased replay; no hardware request.','token_usage_available':False,'pins':pins}
(r/'docs/perf_records/binary32_radius_negative.json').write_text(json.dumps(q,indent=2)+'\n');print(len(pins))
