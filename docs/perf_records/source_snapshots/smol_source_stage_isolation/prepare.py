"""Diagnostic source sensitivity; no compiler workload rule or performance claim."""
from pathlib import Path
import hashlib,json,subprocess
w=Path(__file__).resolve().parent
root=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005')
source=w/'ordered_dot.c'
source.write_text('''#include <math.h>
#include <stddef.h>
void ordered_source_dot(const float*a,const float*b,float*out,int m,int n,int k){
 for(int r=0;r<m;r++)for(int j=0;j<n;j++){
  float acc=0.0f;
  for(int z=0;z<k;z++)acc=fmaf(a[(size_t)r*k+z],b[(size_t)j*k+z],acc);
  out[(size_t)r*n+j]=acc;
 }
}
''')
command=['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang','-O2','-march=native','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(source),'-lm','-o',str(w/'ordered_dot.so')]
subprocess.run(command,check=True)
original=(root/'validate_endpoint_center_native.py').read_text()
injection='''
# Explicit diagnostic stage isolation; actual source contraction and scalar DAG
# stay unchanged. This grants no production approximate numeric contract.
ordered=C.CDLL(str(Path(__file__).parent/'ordered_dot.so'))
array=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
ordered.ordered_source_dot.argtypes=[array,array,array,C.c_int,C.c_int,C.c_int]
class StageIsolated(numeric.NativeCenterEndpointEvaluator):
    def __init__(self,contract):
        super().__init__(contract);self.contraction_index=0
    def _center(self,a,b):
        stage=self.contraction_index%7;self.contraction_index+=1
        exact=(stage==0)if os.environ['SOURCE_EXACT_STAGE']=='qk'else(stage!=0)
        if not exact:return numeric.NativeCenterEndpointEvaluator._center(a,b)
        a=np.ascontiguousarray(a,np.float32);b=np.ascontiguousarray(b,np.float32)
        assert a.shape[1]==b.shape[1]
        out=np.empty((a.shape[0],b.shape[0]),np.float32)
        ordered.ordered_source_dot(a,b,out,a.shape[0],b.shape[0],a.shape[1])
        return out
assert os.environ['SOURCE_EXACT_STAGE']in('qk','pv')
evaluator=StageIsolated(numeric.SOURCE_CONTRACT)
'''
before='evaluator = numeric.NativeCenterEndpointEvaluator(numeric.SOURCE_CONTRACT)'
assert original.count(before)==1
script=original.replace('w = Path(__file__).resolve().parent',f'w = Path({str(root)!r})').replace(before,injection)
script=script.replace('radix3_center_only_source_dag; original whole gate decides admission','diagnostic_source_exact_stage='+"' + os.environ['SOURCE_EXACT_STAGE'] + '")
# Replace the receipt policy through a direct statement instead of interpolation
# inside source strings. Preserve the original gate's failure assertion.
script=script.replace('diagnostic_source_exact_stage=\' + os.environ[\'SOURCE_EXACT_STAGE\'] + \'','radix3_center_only_source_dag; original whole gate decides admission')
needle='(run_out / "native_validation.json").write_text(json.dumps(receipt, indent=2) + "\\n")'
assert script.count(needle)==1
script=script.replace(needle,"receipt['diagnostic_source_exact_stage']=os.environ['SOURCE_EXACT_STAGE']\nreceipt['ordered_dot_source_sha256']=sha(Path(__file__).parent/'ordered_dot.c')\nreceipt['ordered_dot_shared_sha256']=sha(Path(__file__).parent/'ordered_dot.so')\nreceipt['scope']='Full original48-source-group native stage-isolation diagnostic. Source rounding retained in selected QK/PV stage; other stage uses held center approximation. Original whole elementwise gate unchanged. No target, performance or production numeric-contract claim.'\n"+needle)
(w/'validate_stage.py').write_text(script)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(w/'preparation.json').write_text(json.dumps({'command':command,'ordered_dot_source_sha256':sha(source),'ordered_dot_shared_sha256':sha(w/'ordered_dot.so'),'original_script':str(root/'validate_endpoint_center_native.py'),'original_script_sha256':sha(root/'validate_endpoint_center_native.py'),'new_script_sha256':sha(w/'validate_stage.py'),'compiler_sha256':sha(command[0]),'compiler_promotion':False},indent=2)+'\n')
print('STAGE_DIAGNOSTIC_READY',flush=True)
