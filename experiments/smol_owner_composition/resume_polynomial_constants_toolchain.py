"""Select the original pinned capability snapshot after unbound registry refusal."""
from pathlib import Path
import hashlib,json,os,subprocess
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants/normal';snapshot=B/'out/normal_composition/capability_snapshot';receipt=json.load(open(B/'out/exact_row_normal/qualification.json'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env=os.environ.copy();env['MERLIN_CHIPYARD']='/scratch2/agustin/chipyard';pins={}
assert Path(env['MERLIN_CHIPYARD']+'/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc').is_file()
for variable,name in [('MERLIN_RTL_FACTS','facts.json'),('MERLIN_TARGET_CONTRACT','target_contract.yaml')]:
 p=snapshot/name;assert sha(p)==receipt['pins'][str(p)],p;env[variable]=str(p);pins[str(p)]=sha(p)
(W/'explicit_toolchain_resume.json').write_text(json.dumps(dict(reason='First attempt lacked target facts; second routed all297 then refused missing MERLIN_CHIPYARD toolchain. Both failed builds/logs preserved; restore all five original recipe environment fields.',environment={k:env[k]for k in ('MERLIN_CHIPYARD','MERLIN_RTL_FACTS','MERLIN_TARGET_CONTRACT','MERLIN_COMPILER_PYTHON','MERLIN_MLIR_INSTALL')},pins=pins),indent=2)+'\n')
for role in ('build_normal.py','validate_native.py'):
 original=W/'drivers'/role;selected=W/'drivers'/(role.removesuffix('.py')+'_toolchain.py');selected.write_text(original.read_text().replace("'build_v1'","'build_v3'").replace("'build_v1/","'build_v3/").replace("'native_v1'","'native_v3'"))
 with (W/(selected.name+'.log')).open('w')as log:subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(selected)],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
print('POLYNOMIAL_NORMAL_EXPLICIT_FACTS_PASS',flush=True)
