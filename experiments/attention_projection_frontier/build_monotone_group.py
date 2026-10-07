"""Matched2072 source-bound rounded-DAG endpoint enclosure screen."""
from pathlib import Path
import hashlib,json,shutil,subprocess
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity,consume_rounded_polynomial_monotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
base=Path(__file__).resolve().parents[2]
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')
w=base/'out/rounded_monotone_group';d=w/'candidate';d.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mon=base/'out/observation_frontier/rounded_monotonicity'
for role in ('target_numeric','native_numeric'):
 src=b/role;dest=d/role;dest.mkdir()
 manifest=json.loads((src/'compile.json').read_text())
 for name,expected in manifest['pins'].items():
  assert sha(Path(name))==expected,name
 for p in src.iterdir():
  if p.suffix in ('.h','.c'):shutil.copyfile(p,dest/p.name)
 h=dest/'monotone_bit_polynomial.h'
 proof=RoundedPolynomialMonotonicity((0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000),1118743633,sha(mon/'independent_qualification.json'),sha(mon/'independent.c'),sha(h),True,True,True)
 contract=SourceNumericContract(*([True]*7),standard_floor_values=True,floor_interposition_unobserved=True)
 h.write_text(consume_rounded_polynomial_monotonicity(h.read_text(),proof,contract))
 commands=[]
 for old in manifest['commands']:
  cmd=[s.replace(str(src),str(dest)) for s in old];subprocess.run(cmd,check=True);commands.append(cmd)
 (dest/'compile.json').write_text(json.dumps(dict(commands=commands,proof=vars(proof),contract=vars(contract),pins={str(p):sha(p) for p in dest.iterdir() if p.is_file()}),indent=2)+'\n')
