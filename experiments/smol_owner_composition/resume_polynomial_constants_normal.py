"""Select the original pinned capability snapshot after unbound registry refusal."""
from pathlib import Path
import hashlib,json,os,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants/normal';snapshot=B/'out/normal_composition/capability_snapshot';receipt=json.load(open(B/'out/exact_row_normal/qualification.json'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env=os.environ.copy();pins={}
for variable,name in [('MERLIN_RTL_FACTS','facts.json'),('MERLIN_TARGET_CONTRACT','target_contract.yaml')]:
 p=snapshot/name;assert sha(p)==receipt['pins'][str(p)],p;env[variable]=str(p);pins[str(p)]=sha(p)
(W/'explicit_capability_resume.json').write_text(json.dumps(dict(reason='First fresh attempt had no explicit target facts and refused before source routing; failed build_v1 and log preserved.',environment={k:env[k]for k in ('MERLIN_RTL_FACTS','MERLIN_TARGET_CONTRACT')},pins=pins),indent=2)+'\n')
for role in ('build_normal.py','validate_native.py'):
 original=W/'drivers'/role;selected=W/'drivers'/(role.removesuffix('.py')+'_explicit.py');selected.write_text(original.read_text().replace("'build_v1'","'build_v2'").replace("'build_v1/","'build_v2/").replace("'native_v1'","'native_v2'"))
 with (W/(selected.name+'.log')).open('w')as log:subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(selected)],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
print('POLYNOMIAL_NORMAL_EXPLICIT_FACTS_PASS',flush=True)
