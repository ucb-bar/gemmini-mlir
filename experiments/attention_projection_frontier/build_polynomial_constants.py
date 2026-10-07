"""Source-bound constant-context specialization of the admitted row provider."""
from pathlib import Path
import hashlib,json,shutil,subprocess
from merlin.llvmlower.exact_row_radix_pack import prepare_exact_row_radix
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity,consume_rounded_polynomial_monotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
from merlin.llvmlower.prepared_polynomial_constants import c_header
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/prepared-polynomial-constants';D=W/'candidate';D.mkdir(parents=True,exist_ok=False)
N=B/'out/normal_composition/provider';R=B/'out/exact_row_group/candidate/target_numeric';M=B/'out/observation_frontier/rounded_monotonicity'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=SourceNumericContract(*([True]*7),standard_floor_values=True,floor_interposition_unobserved=True)
roles={}
for role in ('native_numeric','target_numeric'):
 src=N/role;dest=D/role;dest.mkdir()
 for p in src.iterdir():
  if p.suffix in ('.c','.h'):shutil.copyfile(p,dest/p.name)
 source=prepare_exact_row_radix((dest/'provider.c').read_text());assert source==(R/'provider.c').read_text()
 h=dest/'monotone_bit_polynomial.h'
 proof=RoundedPolynomialMonotonicity((0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000),1118743633,sha(M/'independent_qualification.json'),sha(M/'independent.c'),sha(h),True,True,True)
 original=h.read_text();(dest/'prepared_polynomial_constants.h').write_text(c_header(original,proof,contract));h.write_text(consume_rounded_polynomial_monotonicity(original,proof,contract))
 anchor='static int soft_details(const float*q,'
 assert source.count(anchor)==1
 source=source.replace(anchor,'#include "prepared_polynomial_constants.h"\n'+anchor)
 anchor='  return soft_details_checked(q,k,mask,lo,hi,p,pl,ph,dl,dh,alpha,rows,counts,yl,yh,maxima);'
 assert source.count(anchor)==1
 source=source.replace(anchor,anchor+'\n const int polynomial_constants_admitted=merlin_polynomial_constants_admit(&root_prepared);')
 anchor='    merlin_polynomial_words_four(xs,&root_prepared,ys);'
 assert source.count(anchor)==1
 source=source.replace(anchor,'    merlin_polynomial_constants_four(xs,&root_prepared,polynomial_constants_admitted,ys);')
 (dest/'provider.c').write_text(source)
 commands=[]
 for old in json.load(open(N/'build.json'))['roles'][role]['commands']:
  cmd=[s.replace('out/normal_composition/provider/'+role,str(dest)) for s in old];subprocess.run(cmd,check=True);commands.append(cmd)
 roles[role]=dict(commands=commands,proof=vars(proof),contract=vars(contract),pins={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
(D/'build.json').write_text(json.dumps(dict(roles=roles),indent=2)+'\n')
print(D,flush=True)
