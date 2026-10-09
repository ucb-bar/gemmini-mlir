"""Reclose current normal lowering against immutable1880 target/native LLVM."""
from pathlib import Path
import json,hashlib,subprocess
from merlin.llvmlower.lower import lower_model
from mlir_oot.late_quant_rne import merlin_host_llvm_transform

W=Path(__file__).resolve().parent
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
features={'respect_captured_quantization_scope','hoist_weight_invariant_quantize',
 'scalar_contraction_accumulator_8_outputs','unroll_scalar_contraction_reduction_by_2',
 'packet_scalar_pointwise_fma_division_2','fuse_quantize_round_convert','fuse_activation_polynomial_fma'}
C=W/'normal_control';C.mkdir(exist_ok=False)
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
source=B/'device_host_abi/model.mlir';pin=sha(source)
assert pin=='9b40fddfd8494ffc8f24a7d377609cd1a4cd5074b6183c7d8324c1bd4b089cbb'
r=lower_model(source.read_text(),workdir=C/'lower',targets=(),features=features)
assert sha(r.ll_path)==sha(B/'lower/model.ll')
selected=merlin_host_llvm_transform(LLVM,combine_clamp=True)(r.ll_path,C/'host_llvm')
records={}
for native in [False,True]:
 model=C/'host_llvm/model.native.ll' if native else selected
 bridge=B/'host_llvm/expanded_bridge.native.ll' if native else B/'host_llvm/expanded_bridge.ll'
 output=C/'host_llvm/expanded.native.ll' if native else C/'host_llvm/expanded.ll'
 expected=B/'host_llvm/model.native.ll' if native else B/'host_llvm/expanded.ll'
 argv=[str(LLVM/'llvm-link'),'-S',str(model),str(bridge),'-o',str(output)]
 subprocess.run(argv,check=True,capture_output=True)
 assert sha(output)==sha(expected),(native,sha(output),sha(expected))
 records['native' if native else 'target']={'path':str(output),'sha256':sha(output),'frozen_path':str(expected),'frozen_sha256':sha(expected),'bridge_sha256':sha(bridge),'argv':argv}
assert sha(source)==pin
record={'schema':'tiny_broadcast_packet_current_root_control_identity_v1','source_path':str(source),
 'source_sha256':pin,'features':sorted(features),'returned_raw_llvm_sha256':sha(r.ll_path),
 'raw_target_native_byteidentical':True,'selected_llvm':records,
 'core_head':subprocess.check_output(['git','-C','/scratch/agustin/tmp/merlin-golden-integration-20261004','rev-parse','HEAD'],text=True).strip(),
 'scope':'Current normal compiler with original feature choice returns immutable1880 raw/selected target/native bytes; no target cycle claim.'}
(C/'identity.json').write_text(json.dumps(record,indent=2)+'\n')
print('CURRENT_ROOT_NORMAL1880_LLVM_BYTE_EXACT',flush=True)
