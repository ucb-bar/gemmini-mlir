"""Normal whole-source lowering and controlled verified norm baseline link."""
from pathlib import Path
import hashlib,json,subprocess,os
from datetime import datetime,timezone
from collections import Counter
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.broadcast_math_hoist import FEATURE as NORM_FEATURE
from merlin.llvmlower.scalar_pointwise_packet import FOUR_FEATURE
from merlin.llvmlower.constant_fma_packet import rewrite,validate_packet_source,analyze
from merlin.llvmlower.late_quant_rne import _tokens
from merlin.runtime.backends.spike_model import _transform_host_ir
from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.rv64gc_scalar_fma import RV64GCScalarFmaCapability
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
N=W.parent/'tiny-broadcast-math-20261005/whole'
F=W.parent/'fma-constant-lifetime-20261006'
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
now=lambda:datetime.now(timezone.utc).isoformat()
# Experimental admission is separate from general typed source/ISA policy.
capsule=json.loads((F/'efficient_timing/qualification.json').read_text())
price=json.loads((W.parents[3]/'docs/perf_records/tiny_fma_constant_lifetime_capsule_cycles.json').read_text())
assert price['finish_done'] and price['saved_percent']>0 and price['original_i8_words_exact']==11264
stage=json.loads((F/'full45056/qualification.json').read_text());assert stage['status']=='pass' and stage['original_i8_words']==45056
reclosure=json.loads((F/'promoted_object_reclosure.json').read_text());assert all(r['actual_object_byteidentical'] for r in reclosure['records'])
source_pins=json.loads((N/'normal_lower_recipe.json').read_text())['source_abi_pins']
for p,h in source_pins.items():assert sha(p)==h
old=json.loads((N/'controlled_link.json').read_text())
for p,h in old['candidate_objects'].items():assert sha(p)==h
assert sha(N/'model.elf')=='8dadfe71e3c224e5e16189cd3985101807d21e34490f73b119c083865e7d9c93'
# The actual original norm winner1926 is relinked before any candidate object exists.
base_argv=list(old['candidate_link_argv']);base_argv[-1]=str(W/'baseline_reproduced.elf')
subprocess.run(base_argv,check=True,capture_output=True);assert sha(W/'baseline_reproduced.elf')==sha(N/'model.elf')
print('NORM1926_BASELINE_RELINK_BYTE_EXACT',flush=True)
features={'respect_captured_quantization_scope','hoist_weight_invariant_quantize','scalar_contraction_accumulator_8_outputs','unroll_scalar_contraction_reduction_by_2','fuse_quantize_round_convert','fuse_activation_polynomial_fma',NORM_FEATURE,FOUR_FEATURE}
case=W/'whole';case.mkdir(exist_ok=False)
r=lower_model((B/'device_host_abi/model.mlir').read_text(),workdir=case/'lower',targets=(),features=features)
original_rne=merlin_host_llvm_transform(LLVM,combine_clamp=True)
capability=RV64GCScalarFmaCapability()
def callback(source,work):
 selected=original_rne(source,work/'rne')
 text=selected.read_text();packets=analyze(text,width=4,ordinary_nontrapping=True,exception_flags_unobserved=True)
 assert all(packet.source_sha256==hashlib.sha256(text.encode()).hexdigest() for packet in packets)
 validate_packet_source(text,packets[0])
 target,proof=rewrite(text,emitter=capability.emit,width=4,ordinary_nontrapping=True,exception_flags_unobserved=True)
 assert packets and len(proof['packets'])==len(packets)
 output=work/'model.ll';output.write_text(target)
 native=work/'model.native.ll';native.write_bytes((work/'rne/model.native.ll').read_bytes())
 for p in [output,native]:subprocess.run([str(LLVM/'llvm-as'),str(p),'-o',str(p.with_suffix('.bc'))],check=True,capture_output=True)
 proof.update(source_semantics='Every source llvm.fma.f32 operand position and single rounding retained; independent typed SSA calls only. Ordinary nontrapping IEEE gradual-underflow scope, exception flags unobserved. Separate actual five-rounding-mode/flags proofs are pinned.',ISA_capability={'isa':capability.isa,'abi':capability.abi,'ieee_f32_fma':capability.ieee_f32_fma,'ieee_gradual_underflow':capability.ieee_gradual_underflow},source_numeric_policy='Existing source policy; no reassociation or approximation.',native_policy='Portable original-source FMA and original portable RNE companion; target instruction packet independently gated by strict whole execution.',M2_paired_receipt_sha256=sha(W.parents[3]/'docs/perf_records/tiny_fma_constant_lifetime_capsule_cycles.json'),full45056_gate_sha256=sha(F/'full45056/qualification.json'),installed_object_reclosure_sha256=sha(F/'promoted_object_reclosure.json'))
 save(work/'constant_fma_receipt.json',proof)
 return output
selected,hook_receipt=_transform_host_ir(r.ll_path,case/'host_llvm',callback)
for native in [False,True]:
 input=case/'host_llvm/model.native.ll' if native else selected
 bridge=B/'host_llvm/expanded_bridge.native.ll' if native else B/'host_llvm/expanded_bridge.ll'
 output=case/'host_llvm/expanded.native.ll' if native else case/'host_llvm/expanded.ll'
 subprocess.run([str(LLVM/'llvm-link'),'-S',str(input),str(bridge),'-o',str(output)],check=True,capture_output=True)
 subprocess.run([str(LLVM/'llvm-as'),str(output),'-o',str(output.with_suffix('.bc'))],check=True,capture_output=True)
symbols=lambda p:Counter(t.text for t in _tokens(p.read_text()) if t.text.startswith('@'))
writer=json.loads((B/'device_host_abi/writer_contracts.json').read_text())
bound={'@'+name for route in writer['routes'] for name in [route['symbol'],route['borrowed_symbol'],route['symbol']+'__fresh_tensor_result']}
prev=symbols(N/'host_llvm/expanded.ll');current=symbols(case/'host_llvm/expanded.ll')
assert {n:prev[n] for n in bound}=={n:current[n] for n in bound};assert sum(route['calls'] for route in writer['routes'])==155
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
compile_argv=[str(LLVM/'clang'),*flags,'-c',str(case/'host_llvm/expanded.ll'),'-o',str(case/'model.o')]
proof_modules=[]
import merlin.llvmlower.constant_fma_packet as core_module
import mlir_oot.rv64gc_scalar_fma as oot_module
for mod in [core_module,oot_module]:proof_modules.append({'path':mod.__file__,'sha256':sha(mod.__file__)})
save(case/'normal_lower_recipe.json',{'schema':'tiny_source_constant_fma_norm_normal_lower_v1','source_seam':'Normal lower_model from complete frozen post-offload ABI source; replace explicit pointwise packet2 with packet4 and apply source-proven constant lifetime callback. Norm1926 feature and original RNE policy retained. Capture/preparation/catalog/weights/runtime/device unchanged.','core_commit':'0b613d5b1','OOT_commit':'69c86da','features':sorted(features),'source_abi_pins':source_pins,'normal_control_recipe_path':str(N/'normal_lower_recipe.json'),'normal_control_recipe_sha256':sha(N/'normal_lower_recipe.json'),'original_source_bound_calls':155,'all_original_source_bound_device_symbol_reference_counts_conserved':True,'candidate_compile_argv':compile_argv,'normal_lower_stats':r.stats,'raw_llvm_sha256':sha(r.ll_path),'selected_target_llvm_sha256':sha(case/'host_llvm/expanded.ll'),'selected_native_llvm_sha256':sha(case/'host_llvm/expanded.native.ll'),'host_llvm_transform_hook':hook_receipt,'actual_proof_modules':proof_modules,'token_usage_available':False})
subprocess.run(compile_argv,check=True,capture_output=True)
print('NORMAL_SOURCE_PACKET4_CONSTANT_FMA_OBJECT_BUILT',flush=True)
argv=list(old['candidate_link_argv']);argv=[str(case/'model.o') if value==str(N/'model.o') else value for value in argv];argv[-1]=str(case/'model.elf')
subprocess.run(argv,check=True,capture_output=True)
for p,h in old['candidate_objects'].items():assert sha(p)==h
for p,h in source_pins.items():assert sha(p)==h
selected_objects={p:h for p,h in old['candidate_objects'].items() if p!=str(N/'model.o')};selected_objects[str(case/'model.o')]=sha(case/'model.o')
audit=audit_elf((case/'model.elf').read_bytes());assert audit['status']=='pass';save(case/'model.nofsm_audit.json',audit)
save(case/'controlled_link.json',{'schema':'tiny_source_constant_fma_norm_controlled_link_v1','finished_utc':now(),'baseline_elf_sha256':sha(N/'model.elf'),'baseline_reproduced_sha256':sha(W/'baseline_reproduced.elf'),'baseline_byte_exact':True,'baseline_reproduction_argv':base_argv,'candidate_link_argv':argv,'baseline_objects':old['candidate_objects'],'candidate_objects':selected_objects,'only_changed_object':'model.o','elf_sha256':sha(case/'model.elf'),'inherited_marker':'37bdf9be0856','marker_contract':'Inherited historical1880 marker is nonunique; exact ELF/source/object pins identify candidate.','scope':'Normal typed source packet4/constant-lifetime FMA lowering composed with verified1926 norm feature, controlled link changing only norm1926 model.o. All original1880 host/runtime/device/startup/main/weights boundary objects frozen. No clamp, adjacentRNE or residentA composition.','before_verified_stock_job':1926,'before_verified_stock_cycles':461389700,'actual_hardware_cycles':None,'original_full_native_strict_gate':'pending','token_usage_available':False})
print('FULL_NORMAL_CANDIDATE_BUILD_PASS',sha(case/'model.elf'),flush=True)
