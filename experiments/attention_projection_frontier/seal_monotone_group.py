"""Immutable rounded-source monotonicity screen; no hardware admission."""
from pathlib import Path
import hashlib,json,shutil,subprocess
base=Path(__file__).resolve().parents[2];w=base/'out/rounded_monotone_group';b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')
core=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
control=w/'control';control.mkdir(exist_ok=False);src=b/'target_numeric';dest=control/'target_numeric';dest.mkdir()
for p in src.iterdir():
 if p.suffix in ('.h','.c'):shutil.copyfile(p,dest/p.name)
commands=[]
for old in json.loads((src/'compile.json').read_text())['commands']:
 cmd=[x.replace(str(src),str(dest)) for x in old];subprocess.run(cmd,check=True);commands.append(cmd)
assert sha(src/'provider.o')==sha(dest/'provider.o')
link=json.loads((b/'build.json').read_text())['link'];link=[str(control/'model.elf') if x==str(b/'model.elf') else str(dest/'provider.o') if x==str(src/'provider.o') else x for x in link];subprocess.run(link,check=True,capture_output=True);commands.append(link)
assert sha(control/'model.elf')==sha(b/'model.elf')
(control/'reproduction.json').write_text(json.dumps(dict(commands=commands,object_byte_exact=True,elf_byte_exact=True,baseline=str(b/'model.elf'),sha256=sha(b/'model.elf')),indent=2)+'\n')
freeze=w/'source_snapshot';freeze.mkdir()
for name in ('rounded_polynomial_monotonicity.py','source_numeric_capability.py'):
 shutil.copyfile(core/'src/merlin/llvmlower'/name,freeze/name)
shutil.copyfile(core/'merlin/tests/runtime/test_rounded_polynomial_monotonicity.py',freeze/'test_rounded_polynomial_monotonicity.py')
shutil.copyfile(core/'out/qualification/rounded_monotonicity_tests.txt',freeze/'tests.txt')
roots=[w,base/'out/observation_frontier/rounded_monotonicity',base/'out/observation_frontier/rounded_monotone_native',base/'out/rounded_monotone_target_v3']
files={p.resolve() for root in roots for p in root.rglob('*') if p.is_file()}
files.update((base/'experiments/attention_projection_frontier').glob('*monoton*.py'))
files.update((base/'experiments/attention_projection_frontier').glob('*monotone*.py'))
for name in ('/tmp/rounded_monotone_target.log','/tmp/rounded_monotone_target_v2.log'):
 p=Path(name);q=freeze/p.name;shutil.copyfile(p,q);files.add(q)
# Bind all unchanged final-link inputs as well as the modified provider.
for x in link:
 p=Path(x)
 if p.is_file():files.add(p)
native=json.loads((base/'out/observation_frontier/rounded_monotone_native/qualification.json').read_text())
record=dict(schema='rounded_source_polynomial_monotonicity_screen_v1',core_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=core,text=True).strip(),scope='Exact source-exp rounded-DAG enclosure tightening within unchanged explicit RMS4 provider. No whole hardware or default promotion.',source_fact=dict(domain='all binary32 negative words cutoff through negative zero; positive zero separately checked; below cutoff sourcezero',checked=1118743633,independent_compilers=['clang','gcc'],trace_digest='00660be4cb43c119',rational_checks=18098,rounding='RNE',gradual_underflow=True,flags='unobserved/nontrapping'),native=dict(outputs=1600,bit_mismatches=native['bit_mismatches'],groups=native['snapshots'],product_calls=native['product_calls'],fallbacks=native['counts'][1],original_gate=native['allclose']),target=dict(source_word_checks=1031,rounding_modes=5,unsupported_modes='prepared fast path refuses',plan_domain_refusals=2,final_no_fsm=True),cost=dict(scope='complete original12-head group, allocation/consumer included',control_instructions=1734429991,candidate_instructions=1689599774,reduction_percent=100*(1-1689599774/1734429991),control_qk_replay_fmas=4066368,candidate_qk_replay_fmas=1883904,consumer='original786432i8/1024BF16scales and guards PASS',unobserved_carrier_differences=2,hardware_cycles=None),historical_failures='Two initial standalone proof links lacked malloc HTIF definitions; no target execution. Third uses original minimal crt/syscalls, passes; logs preserved.',pins={str(p):sha(p) for p in sorted(files)})
out=base/'docs/perf_records/rounded_source_polynomial_monotonicity_qualification.json';out.write_text(json.dumps(record,indent=2)+'\n');print(len(record['pins']),out)
