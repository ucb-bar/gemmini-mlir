"""One fixed source-function prefix partition; no captured values select cells."""
from pathlib import Path
import ctypes as C
import hashlib,json,shutil,subprocess
from merlin.llvmlower.polynomial_prefix_table import prepare,c_header
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/source-polynomial-prefix-table';D=W/'candidate';D.mkdir(exist_ok=False)
N=B/'out/artifacts/probes/prepared-polynomial-constants/candidate';G=W/'generation';G.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Use the exact source implementation whose full rounded monotonicity was proved.
source=(W/'generate.c').read_text().split('int main(')[0].replace('static uint32_t evaluate(', 'uint32_t evaluate(')
(G/'evaluator.c').write_text(source)
cmd=['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang','-O2','-fno-builtin','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-include',str(N/'native_numeric/numeric_capability.h'),'-I',str(N/'native_numeric'),str(G/'evaluator.c'),'-lm','-o',str(G/'evaluator.so')]
subprocess.run(cmd,check=True);lib=C.CDLL(str(G/'evaluator.so'));f=lib.evaluate;f.argtypes=[C.c_uint32];f.restype=C.c_uint32
prior=json.load(open(N/'build.json'))
record=prior['roles']['native_numeric'];pv=record['proof'];pv['plan_words']=tuple(pv['plan_words']);proof=RoundedPolynomialMonotonicity(**pv);contract=SourceNumericContract(**record['contract'])
original=B/'out/normal_composition/provider/native_numeric/monotone_bit_polynomial.h'
table=prepare(original.read_text(),proof,contract,lambda word:int(f(word)),low_bits=8,maximum_bytes=4*1024*1024)
header=c_header(table,original.read_text(),proof,contract,lambda word:int(f(word)));(G/'prefix_table.h').write_text(header)
metadata={k:v for k,v in vars(table).items() if k!='words'};metadata.update(entries=len(table.words),bytes=4*len(table.words),header_sha256=sha(G/'prefix_table.h'),evaluator_source_sha256=sha(G/'evaluator.c'),evaluator_so_sha256=sha(G/'evaluator.so'),command=cmd)
(G/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
roles={}
for role in ('native_numeric','target_numeric'):
 dest=D/role;dest.mkdir()
 for p in (N/role).iterdir():
  if p.suffix in ('.c','.h'):shutil.copyfile(p,dest/p.name)
 (dest/'polynomial_prefix_table.h').write_text(header)
 source=(dest/'provider.c').read_text()
 for old,new in [('#include "prepared_polynomial_constants.h"','#include "polynomial_prefix_table.h"'),('merlin_polynomial_constants_admit(&root_prepared)','merlin_polynomial_prefix_admit(&root_prepared)'),('merlin_polynomial_constants_four(xs,','merlin_polynomial_prefix_four(xs,')]:
  assert source.count(old)==1,old;source=source.replace(old,new)
 (dest/'provider.c').write_text(source)
 commands=[]
 for old in prior['roles'][role]['commands']:
  command=[x.replace(str(N/role),str(dest)) for x in old];subprocess.run(command,check=True);commands.append(command)
 roles[role]={'commands':commands,'pins':{str(p):sha(p) for p in dest.iterdir() if p.is_file()}}
(D/'build.json').write_text(json.dumps({'roles':roles,'source_control':str(N),'metadata':metadata},indent=2)+'\n')
print(D,flush=True)
