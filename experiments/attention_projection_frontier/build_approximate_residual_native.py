"""Fixed explicit residual-statistical policy, fresh native original-source screen."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
from merlin.llvmlower.approximate_residual_rms import ApproximateResidualRMSPolicy,c_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups
from merlin.llvmlower.radix_integer_reconstruct import c_fused_header
B=Path(__file__).resolve().parents[2]
W=B/'out/artifacts/probes/approximate-residual-rms'
W.mkdir(parents=True,exist_ok=False)
D=W/'native_numeric';D.mkdir()
C=B/'out/artifacts/probes/prepared-polynomial-constants/candidate'
for p in (C/'native_numeric').iterdir():
 if p.suffix in ('.c','.h'):shutil.copyfile(p,D/p.name)
oldcore=Path('/scratch/agustin/tmp/merlin-attention-projection-frontier-20261007/src/merlin/llvmlower/two_signed_byte_block_float.py')
spec=importlib.util.spec_from_file_location('frozen_two_byte',oldcore);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
(D/'two_signed_byte_block_float.h').write_text(mod.c_header())
(D/'approximate_residual_rms.h').write_text(c_header(ApproximateResidualRMSPolicy(*([True]*8))))
s=(D/'provider.c').read_text()
start=s.index('#ifndef MERLIN_RADIX_FUSED_INTEGER_RECONSTRUCT_H');end=s.index('\n#endif',start)+len('\n#endif');end=s.index('\n#endif',end)+len('\n#endif')
s=s[:start]+c_fused_header(plan_radix_product_groups(radix_bits=8,digits=2,reduction_length=192))+s[end:]
s=s.replace('#include "bf16_radix_pack.h"','#include "bf16_radix_pack.h"\n#include "two_signed_byte_block_float.h"')
old='''merlin_bf16_radix_row_widen(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row)'''
new='''merlin_two_signed_byte_row(&environment,source+row*k,k,rd+row*k,
    planes+(transpose?row:row*k),m*k,transpose?m:1,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row)'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('#include "source_rms_point_products.h"','#include "source_rms_point_products.h"\n#include "approximate_residual_rms.h"')
s=s.replace('double uncertainty[CHUNK];','double uncertainty[CHUNK];double selected_norm[CHUNK],residual_norm[CHUNK];')
s=s.replace('merlin_source_rms4_point_product_estimates(', 'merlin_source_residual_rms4_estimates(')
s=s.replace('scratch->uncertainty,CHUNK))return 1;', 'scratch->uncertainty,scratch->selected_norm,scratch->residual_norm,CHUNK))return 1;')
(D/'provider.c').write_text(s)
commands=[]
for old in json.loads((C/'build.json').read_text())['roles']['native_numeric']['commands']:
 cmd=[x.replace(str(C/'native_numeric'),str(D)) for x in old]
 subprocess.run(cmd,check=True);commands.append(cmd)
(W/'build.json').write_text(json.dumps({'commands':commands,'policy':'explicit statistical representation residual RMS4; no enclosure or equality certificate','fixed_representation_source':str(oldcore)},indent=2))
h=W/'native';h.mkdir()
s=(C.parent/'normal_native/run.py').read_text().replace(str(C.parent/'normal_native'),str(h)).replace(str(C/'native_numeric'),str(D))
s=s.replace('shape=(3*m*k,)).reshape(3,m,k)','shape=(2*m*k,)).reshape(2,m,k)').replace('shape=(3*k*n,)).reshape(3,k,n)','shape=(2*k*n,)).reshape(2,k,n)').replace('range(3):','range(2):').replace('0<=bd<3','0<=bd<2').replace('%480','%288').replace('//480','//288').replace('48*480','48*288')
s=s.replace("and r['bitwise_mismatches']==0",'')
s=s.replace('explicit approximate_source_roundoff_rms4; no exact observer theorem','explicit source-and-representation residual RMS4; heuristic only')
(h/'run.py').write_text(s)
print(h/'run.py',flush=True)
