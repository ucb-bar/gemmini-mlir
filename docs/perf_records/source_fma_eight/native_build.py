from pathlib import Path
import shutil,json,subprocess
w=Path(__file__).resolve().parent
old=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed')
n=w/'native_numeric_frozen';n.mkdir(exist_ok=False)
for p in(old/'native_numeric_frozen').glob('*.h'):shutil.copyfile(p,n/p.name)
shutil.copyfile('/scratch/agustin/tmp/merlin-source-fma-eight-20261006/merlin/runtime/c/prepared_polynomial_batch.h',n/'prepared_polynomial_batch.h')
shutil.copyfile(old/'native_numeric_frozen/provider.c',n/'provider.c')
from merlin.llvmlower.source_fma_batch import SourceFmaBatchContract,emit_source_fma_batch_permission
(n/'native_fma_batch.h').write_text('static inline void native_source_fma_eight(const float fraction[8],float product[8],float coefficient){for(int i=0;i<8;i++)product[i]=__builtin_elementwise_fma(fraction[i],product[i],coefficient);}\n#define MERLIN_SOURCE_F32_FMA_EIGHT native_source_fma_eight\n'+emit_source_fma_batch_permission(SourceFmaBatchContract(8,*([True]*9))))
commands=[]
for argv in json.loads((old/'native_numeric_frozen/compile.json').read_text())['commands']:
 argv=[v.replace(str(old/'native_numeric_frozen'),str(n)) for v in argv];argv[1:1]=['-include',str(n/'native_fma_batch.h')];subprocess.run(argv,check=True);commands.append(argv)
(n/'compile.json').write_text(json.dumps({'commands':commands},indent=2)+'\n')
# Freeze the inherited normal native harness with only new candidate output paths.
source=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/docs/perf_records/source_snapshots/frontier_composed/validate_native.py').read_text()
source=source.replace(str(old/'native'),str(w/'native')).replace(str(old/'native_numeric_frozen'),str(n))
# The prior fixture checked its manifest pins; freeze this new compiler closure.
import hashlib
r=json.loads((n/'compile.json').read_text());r['pins']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in n.iterdir()if p.is_file() and p.name!='compile.json'};(n/'compile.json').write_text(json.dumps(r,indent=2)+'\n')
(w/'validate_native.py').write_text(source)
subprocess.run(['/scratch/agustin/projects/oscar-merlin/.venv/bin/python',str(w/'validate_native.py')],check=True)
