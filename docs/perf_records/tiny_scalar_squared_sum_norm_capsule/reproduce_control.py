"""Default-off code bytes match frozen pre-change compiler on actual source."""
from pathlib import Path
import hashlib,json
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.broadcast_math_hoist import FEATURE as HOIST
from merlin.llvmlower.scalar_pointwise_packet import FEATURE as PACKET,MULTIPLY_FOUR_FEATURE,TWO_MULTIPLY_FOUR_FEATURE
from merlin.llvmlower.bufferized_result_identity import FEATURE as IDENTITY
from merlin.llvmlower.late_quant_rne import rewrite
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
features={HOIST,MULTIPLY_FOUR_FEATURE,TWO_MULTIPLY_FOUR_FEATURE,IDENTITY,PACKET,'fuse_quantize_round_convert','fuse_activation_polynomial_fma'}
raw=lower_to_llvm_ir((P/'source.mlir').read_text(),workdir=P/'control_reproduced',features=features).replace('forward','control').replace('dealloc_helper','control_dealloc_helper')
record={'schema':'generic_default_off_actual_source_reproduction_v1','before_core_commit':'9aa9a16d3','after_core_commit':'397ef9614','features':sorted(features),'normal_default_executable_IR_byteidentical':True,'scope':'Actual same original firstnorm source and explicit currentnorm policies; source/math/runtime unchanged. No scalar_squared_sum_accumulator selected.','pins':{str(P/'source.mlir'):sha(P/'source.mlir')},'token_usage_available':False}
for mode,isa in [('native','portable'),('target','rv64gc')]:
 text,proof=rewrite(raw,host_isa=isa,combine_clamp=mode=='target')
 original=P/'control'/(mode+'.ll');actual=P/'control_reproduced'/(mode+'.ll');actual.write_text(text)
 assert text==original.read_text(),mode
 record['pins'][str(original)]=sha(original);record['pins'][str(actual)]=sha(actual)
(P/'default_identity.json').write_text(json.dumps(record,indent=2)+'\n');print('PRECHANGE_CONTROL_DEFAULT_IR_BYTEEXACT',flush=True)
