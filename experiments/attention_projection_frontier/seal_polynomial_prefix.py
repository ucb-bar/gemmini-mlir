"""Preserve a losing fixed prefix partition without whole or hardware promotion."""
from pathlib import Path
import collections,hashlib,json,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/source-polynomial-prefix-table';C=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007');OLD=B/'out/artifacts/probes/prepared-polynomial-constants'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
text=(W/'strict/stdout').read_text();terminal=json.load(open(W/'strict/terminal.json'));assert terminal['returncode']==0
assert 'WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS' in text
roi=int(next(x.split()[1] for x in text.splitlines() if x.startswith('WORKSPACE_GROUP_INSTRUCTIONS ')))
stats=[int(x.split()[2]) for x in text.splitlines() if x.startswith('WORKSPACE_STAT ')];assert stats==[4391,0,96207,85120,5447680,86,1956,333824]
old_stats=[4391,0,20866,29436,1883904,29,1656,282624]
replay=lambda a:64*(a[0]+a[2])+a[4]+a[7]
functions=collections.Counter();headers=collections.Counter()
for row in json.load(open(W/'attribution/attribution.json'))['rows']:
 if row['function'] in ('group_provider_with_rhs','encode_operand','evaluate_products','certify_frontier') or row['function'].startswith(('qk_products_','pv192_products_','pv128_products_')):
  functions[row['function']]+=row['instructions'];headers[(row['function'],Path(row['file']).name)]+=row['instructions']
functions['unassigned_ROI_shared_or_runtime']=roi-sum(functions.values());assert sum(functions.values())==roi
paths=[]
for sub in ('candidate','generation','independent','independent_v2','independent_v3','target_independent','strict','attribution'):
 paths.extend(p for p in (W/sub).rglob('*') if p.is_file())
paths.extend(p for p in W.iterdir() if p.is_file() and p.name not in ("qualification.json","seal.log"))
paths.extend((B/'experiments/attention_projection_frontier').glob('*polynomial_prefix*.py'))
paths.extend([C/'src/merlin/llvmlower/polynomial_prefix_table.py',C/'src/merlin/llvmlower/prepared_polynomial_constants.py',C/'src/merlin/llvmlower/rounded_polynomial_monotonicity.py',C/'src/merlin/llvmlower/source_numeric_capability.py',C/'merlin/tests/ir/test_polynomial_prefix_table.py'])
paths.extend((C/'out/artifacts/probes/polynomial-prefix-table').glob('*'))
for role in ('native_numeric','target_numeric'):
 dep=(W/'candidate'/role/'provider.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
 paths.extend(Path(x)for x in dep if Path(x).is_file())
for path in json.load(open(W/'candidate/target_build.json'))['link']:
 if Path(path).is_file():paths.append(Path(path))
for suffix in ('candidate/model.elf','candidate/target_numeric/provider.o','candidate/target_numeric/provider.c','candidate/target_build.json','strict/stdout','strict/terminal.json','attribution/attribution.json'):
 paths.append(OLD/suffix)
theorem=B/'docs/perf_records/rounded_source_polynomial_monotonicity_qualification.json';paths.append(theorem)
for name,digest in json.load(open(theorem))['pins'].items():
 p=Path(name);assert sha(p)==digest,name;paths.append(p)
r=dict(schema='source_polynomial_prefix_table_negative_v1',status='NEGATIVE complete retired cost; no whole48/hardware campaign',core_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),partition=dict(low_bits=8,budget_bytes=4194304,actual_bytes=3339740,entries=834935,selection='One fixed explicit partition, not a golden-selected sweep. Constant leading prefix proved from equal endpoint words and complete monotonicity.'),source_permission='Unchanged explicit RMS4 policy. Exact lookup encloses the polynomial for each input endpoint; this does not make RMS4 input intervals rigorous source certificates. Original F32 denominator lane/tree and escaping i8/BF16scale observations remain.',checks=dict(core='42 tests PASS including 12 prefix synthesis/refusal tests',native=json.load(open(W/'independent_v3/qualification.json'))['stdout'],target=json.load(open(W/'target_independent/qualification.json'))['stdout'],table_source_reproduction='PASS exact generated header bytes; source function/table matching is rederived before emission',consumer_i8=786432,consumer_bf16_scales=1024,callbacks=480,unobserved_carrier_differences=10,final_noFSM=json.load(open(W/'candidate/target_build.json'))['audit'],whole_native='NOT RUN after complete group loss',stock='NOT RUN'),cost=dict(control_retired=1560212849,candidate_retired=roi,increase_fraction=roi/1560212849-1,scope='Complete original12-head group instruction ROI. No cycle/whole projection; static read-only table included in ELF, no runtime allocation or preprocessing. Target read traffic cache cost remains unknown.',old_stats=old_stats,new_stats=stats,old_replay_fmas=replay(old_stats),new_replay_fmas=replay(stats),control_polynomial_header_retired=163625999,candidate_table_header_retired=headers[('group_provider_with_rhs','polynomial_prefix_table.h')],functions=dict(functions),exclusive_function_headers=[dict(function=f,header=h,instructions=n)for(f,h),n in headers.most_common()]),retained_failures=['GCC native test compilation refused unavailable elementwise-FMA compiler capability; independent_v2 switched to Clang+UBSan.','independent_v2 test oracle omitted the original below-cutoff zero branch; production table correctly returned zero. independent_v3 fixed the oracle and passed. Neither failure caused a production source change.'],pins={str(p.resolve()):sha(p) for p in paths if p.is_file()})
(W/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');(B/'docs/perf_records/source_polynomial_prefix_table_negative.json').write_text(json.dumps(r,indent=2)+'\n');print('PINS',len(r['pins']));print(json.dumps(r['cost'],indent=2))
