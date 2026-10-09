"""Immutable source, complete consumer and explicit clock pair qualification."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants';C=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
whole=json.load(open(W/'normal_native/validation.json'));assert whole['elements']==1600 and whole['bitwise_mismatches']==0 and whole['allclose'] and whole['calls'][:3]==[48,0,0] and whole['calls'][4:7]==[12,0,0] and whole['product_calls']==23040
paths=[]
for sub in ('candidate','independent','target_independent','strict','normal_native','clock_pair'):
 paths.extend(p for p in (W/sub).rglob('*') if p.is_file())
for name in ('build_polynomial_constants.py','bind_polynomial_constants.py','check_polynomial_constants_native.py','check_polynomial_constants_target.py','qualify_polynomial_constants_group.py','qualify_polynomial_constants_whole.py','build_polynomial_constants_clock_pair.py','seal_polynomial_constants_group.py','qualify_monotone_target.py'):
 paths.append(B/'experiments/attention_projection_frontier'/name)
paths.extend([C/'src/merlin/llvmlower/prepared_polynomial_constants.py',C/'src/merlin/llvmlower/rounded_polynomial_monotonicity.py',C/'src/merlin/llvmlower/source_numeric_capability.py',C/'merlin/tests/ir/test_prepared_polynomial_constants.py'])
# Bind original full-source semantic/native objects and complete group link leaves.
paths.extend([B/'out/exact_row_normal/build_v1/lower/model.ll',B/'out/exact_row_group/candidate/model.elf',B/'out/exact_row_group/candidate/target_numeric/provider.o',B/'out/exact_row_group/candidate/target_numeric/provider.c',B/'out/exact_row_group/strict/stdout',Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/golden.npy')])
for arm in ('control','candidate'):
 record=json.load(open(W/'clock_pair'/arm/'build.json'))
 paths.extend(Path(x)for x in record['link']if Path(x).is_file())
for sub in ('native_numeric','target_numeric'):
 dep=(W/'candidate'/sub/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
 paths.extend(Path(x)for x in dep if Path(x).is_file())
# Exhaustive theorem is a retained independent prerequisite; flatten every pin.
theorem=B/'docs/perf_records/rounded_source_polynomial_monotonicity_qualification.json';paths.append(theorem)
for name,digest in json.load(open(theorem))['pins'].items():
 p=Path(name);assert sha(p)==digest,name;paths.append(p)
newstats=[4391,0,20866,29436,1883904,29,1656,282624];oldstats=[4391,0,42210,63537,4066368,63,1848,315392]
clocks={}
for arm,stats in [('control',oldstats),('candidate',newstats)]:
 d=W/'clock_pair'/arm;text=(d/'stdout').read_text();assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS'in text
 assert all(f'WORKSPACE_STAT {i} {value}\n'in text for i,value in enumerate(stats));assert json.load(open(d/'terminal.json'))['returncode']==0
 clocks[arm]=dict(elf=str((d/'model.elf').resolve()),sha256=sha(d/'model.elf'),stats=stats,unobserved_carrier_differences=3 if arm=='control'else 2,strict_proxy=int(next(x.split()[1]for x in text.splitlines()if x.startswith('WORKSPACE_GROUP_CYCLES '))))
address=json.load(open(W/'clock_pair/full_address_comparison.json'));assert address['identical']
r=dict(schema='prepared_polynomial_constants_group_qualification_v1',status='PASS; stock and fresh normal target execution UNKNOWN',core_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),core_source=str(C/'src/merlin/llvmlower/prepared_polynomial_constants.py'),core_tests='30 PASS: 17 context/refusal/native UBSan plus 13 existing theorem tests; no-regex PASS. Ruff unavailable in this interpreter.',native_independent=json.load(open(W/'independent/qualification.json'))['stdout'],target_independent=json.load(open(W/'target_independent/qualification.json'))['stdout'],native_whole=whole,complete_group=dict(control_retired=1713849505,candidate_retired=1560212849,reduction_fraction=1-1560212849/1713849505,callbacks=480,consumer_i8=786432,consumer_bf16_scales=1024,scope='Complete original first12head group only; composition of exact rounded monotonicity and immutable constants; no isolated constant-only gain claim.'),clock_pair=clocks,shared_addresses=address,witness_obligations=dict(context='Produced once after source-domain and complete immutable active-span admission, exact prepared-plan words, zero implementation budget and RNE. Copied plan remains private and unmodified; helper returns before owner scope ends.',inputs='Existing span producer/fallback proves finite ordered endpoints. Positive scaled maximum check gives scores<=0; original clamp retains below-cutoff zero. Per-input valid/upper fallback remains.',rounding='Same clamped scale, floor/signedzero subtraction, three source Horner FMAs, subtraction and final encoding FMA/conversion; no FMA reassociation.',uncertainty='Zero word budget removes only separately proved polynomial implementation expansion. Distinct x.lo/x.hi remain distinct inputs. RMS4 score estimates are still approximate, not exact source bounds; ambiguous BF16/quant observations still refine source.',outputs='Monotonic finite positive exact endpoint words permit literal valid interval; original F32 denominator lane/tree/alpha order and escaping BF16scale/i8 consumer retained.',effects='Stable RNE, nontrapping, flags/errno/interposition unobserved; private disjoint local xs/ys and immutable plan. Unsupported contexts use original helper.'),pins={str(p.resolve()):sha(p)for p in paths})
(W/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');(B/'docs/perf_records/prepared_polynomial_constants_group_qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('PINS',len(r['pins']));print(json.dumps(clocks,indent=2))
