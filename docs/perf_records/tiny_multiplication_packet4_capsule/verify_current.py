from pathlib import Path
import hashlib,json,subprocess
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.broadcast_math_hoist import FEATURE as HOIST
from merlin.llvmlower.scalar_pointwise_packet import MULTIPLY_FOUR_FEATURE
from merlin.llvmlower.late_quant_rne import rewrite
w=Path(__file__).resolve().parent;d=w/'installed_emission';d.mkdir(exist_ok=True)
source=w.parent/'tiny-broadcast-math-20261005/source.mlir';ll=lower_to_llvm_ir(source.read_text(),workdir=d/'lower',features={HOIST,MULTIPLY_FOUR_FEATURE}).replace('forward','multiply_packet').replace('dealloc_helper','multiply_packet_dealloc_helper')
target,_=rewrite(ll,host_isa='rv64gc',combine_clamp=True);native,_=rewrite(ll,host_isa='portable')
assert target==(w/'target.ll').read_text() and native==(w/'native.ll').read_text();(d/'target.ll').write_text(target)
clang=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang');flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
subprocess.run([str(clang),*flags,'-c',str(d/'target.ll'),'-o',str(d/'model.o')],check=True,capture_output=True);assert (d/'model.o').read_bytes()==(w/'model.o').read_bytes()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=dict(schema='installed_typed_multiplication_packet_emission_v1',source=sha(source),generic_module_path='/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/src/merlin/llvmlower/scalar_pointwise_packet.py',target_llvm_sha256=sha(d/'target.ll'),target_object_sha256=sha(d/'model.o'),native_llvm_sha256=hashlib.sha256(native.encode()).hexdigest(),current_typed_selector_regenerates_timed_artifact_byteexact=True,independent_strict_refusal_followup_changes_unsupported_selection_only=True,token_usage_available=False);r['generic_module_sha256']=sha(Path(r['generic_module_path']));(d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
