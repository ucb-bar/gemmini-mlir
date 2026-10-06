"""Normal typed host lowering and controlled full1880 boundary-object link."""
from pathlib import Path
import hashlib,json,subprocess,os
from datetime import datetime,timezone
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.scalar_pointwise_packet import FEATURE, BROADCAST_FEATURE
from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
SCRIPT=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/baremetal/spike/model_link.ld')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
save=lambda p,r:Path(p).write_text(json.dumps(r,indent=2)+'\n')
# Complete actual-source native/strict qualification passed. GSIM timed out
# after control only; stock timing remains an explicit experimental obligation.
capsule=json.loads((W/'capsule/qualification.json').read_text())
assert capsule['strict_RV64GC_original_pass'] and capsule['zero_FSM']
assert capsule['original_outputs']==45056
assert sha(Path(capsule['elf_path']))==capsule['elf_sha256']
local_timing=json.loads((W/'capsule/gsim_receipt.json').read_text())
assert local_timing['returncode']==-1 and local_timing['finish'] is None
features={'respect_captured_quantization_scope','hoist_weight_invariant_quantize','scalar_contraction_accumulator_8_outputs','unroll_scalar_contraction_reduction_by_2','packet_scalar_pointwise_fma_division_2','fuse_quantize_round_convert','fuse_activation_polynomial_fma'}
control=W/'normal_control';identity=json.loads((control/'identity.json').read_text());assert identity['raw_target_native_byteidentical']
assert sha(control/'lower/model.ll')==sha(B/'lower/model.ll')
assert sha(control/'host_llvm/expanded.ll')==sha(B/'host_llvm/expanded.ll')
assert sha(control/'host_llvm/expanded.native.ll')==sha(B/'host_llvm/model.native.ll')
candidate_features=(features-{FEATURE})|{BROADCAST_FEATURE}
order=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o','device_catalog/kernel.o','device/device_catalog_shim.o']
objects=[B/name for name in order];pins={str(p):sha(p) for p in objects}
link_flags=['-O2','-ffreestanding','-fno-builtin','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-nostdlib','-nostartfiles','-Wl,--defsym,MERLIN_WEIGHTS_BASE=0x84100000','-Wl,--defsym,MERLIN_STACK_BYTES=0x1000000','-T',str(SCRIPT)]
base_argv=[str(GCC),*link_flags,*map(str,objects),'-lm','-o',str(W/'baseline_reproduced.elf')]
subprocess.run(base_argv,check=True,capture_output=True);assert sha(W/'baseline_reproduced.elf')==sha(B/'model.elf')
(W/'baseline_reproduced.elf').unlink();os.link(B/'model.elf',W/'baseline_reproduced.elf');print('BASELINE_ACTUAL_RELINK_BYTE_EXACT',flush=True)
case=W/'whole';case.mkdir(exist_ok=False)
r=lower_model((B/'device_host_abi/model.mlir').read_text(),workdir=case/'lower',targets=(),features=candidate_features)
selected=merlin_host_llvm_transform(LLVM,combine_clamp=True)(r.ll_path,case/'host_llvm')
for native in [False,True]:
 original=case/'host_llvm/model.native.ll' if native else selected
 bridge=B/'host_llvm/expanded_bridge.native.ll' if native else B/'host_llvm/expanded_bridge.ll'
 output=case/'host_llvm/expanded.native.ll' if native else case/'host_llvm/expanded.ll'
 subprocess.run([str(LLVM/'llvm-link'),'-S',str(original),str(bridge),'-o',str(output)],check=True,capture_output=True)
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
compile_argv=[str(LLVM/'clang'),*flags,'-c',str(case/'host_llvm/expanded.ll'),'-o',str(case/'model.o')]
source_paths=[B/'model.prepared.mlir',B/'device_host_abi/model.mlir',B/'device_host_abi/borrowed.c',B/'device_host_abi/writer_contracts.json',B/'device_signatures.json',B/'device_catalog/device_catalog.json',B/'device/catalog_binding.json',B/'host_llvm/receipt.json',B/'host_llvm/expanded_bridge.ll',B/'host_llvm/expanded_bridge.native.ll',B/'quant_hoist_args.json']
source_pins={str(p):sha(p) for p in source_paths if p.exists()}
from merlin.llvmlower.late_quant_rne import _tokens
from collections import Counter
symbols=lambda p:Counter(t.text for t in _tokens(p.read_text()) if t.text.startswith('@'))
writer=json.loads((B/'device_host_abi/writer_contracts.json').read_text())
bound_symbols={'@'+name for route in writer['routes'] for name in [route['symbol'],route['borrowed_symbol'],route['symbol']+'__fresh_tensor_result']}
old_symbols=symbols(B/'host_llvm/expanded.ll');new_symbols=symbols(case/'host_llvm/expanded.ll')
assert {name:old_symbols[name] for name in bound_symbols}=={name:new_symbols[name] for name in bound_symbols}
assert sum(route['calls'] for route in writer['routes'])==155
save(case/'normal_lower_recipe.json',{'schema':'tiny_source_broadcast_packet_normal_lower_controlled1880_v1','core_commit':subprocess.check_output(['git','-C','/scratch/agustin/tmp/merlin-golden-integration-20261004','rev-parse','HEAD'],text=True).strip(),'source_seam':'Normal lower_model from immutable complete post-offload ABI source; capture/preparation/devicecatalog frozen. Original RNE policy and writer bridge remain explicit unchanged. Only original two-lane pointwise feature is replaced by optional two-lane broadcast-axis sharing.','features':sorted(candidate_features),'source_abi_pins':source_pins,'normal_control_llvm_byteidentical':True,'normal_control_selected_target_native_byteidentical':True,'control_model_object_byteidentical':True,'all_original_source_bound_device_symbol_reference_counts_conserved':True,'original_source_bound_calls':155,'candidate_compile_argv':compile_argv,'normal_lower_stats':r.stats,'raw_llvm_sha256':sha(r.ll_path),'selected_target_llvm_sha256':sha(case/'host_llvm/expanded.ll'),'selected_native_llvm_sha256':sha(case/'host_llvm/expanded.native.ll'),'token_usage_available':False})
subprocess.run(compile_argv,check=True,capture_output=True);print('BROADCAST_PACKET_WHOLE_COMPILED',flush=True)
selected_objects=[case/'model.o' if p==B/'model.o' else p for p in objects]
argv=[str(GCC),*link_flags,*map(str,selected_objects),'-lm','-o',str(case/'model.elf')]
subprocess.run(argv,check=True,capture_output=True)
for p,expected in pins.items():assert sha(p)==expected
for p,expected in source_pins.items():assert sha(p)==expected
audit=audit_elf((case/'model.elf').read_bytes());assert audit['status']=='pass';save(case/'model.nofsm_audit.json',audit)
save(case/'controlled_link.json',{'schema':'tiny_source_broadcast_packet_controlled1880_link_v1','finished_utc':datetime.now(timezone.utc).isoformat(),'baseline_elf_sha256':sha(B/'model.elf'),'baseline_reproduced_sha256':sha(W/'baseline_reproduced.elf'),'baseline_byte_exact':True,'baseline_reproduction_argv':base_argv,'candidate_link_argv':argv,'baseline_objects':pins,'candidate_objects':{str(p):sha(p) for p in selected_objects},'only_changed_object':'model.o','elf_sha256':sha(case/'model.elf'),'inherited_marker':'37bdf9be0856','marker_contract':'Inherited historical1880 marker is nonunique; exact ELF and all source/object hashes identify this candidate.','scope':'Normal typed source pointwise broadcast lowering, original RNE/writer composition, controlled link changing only model.o; all1880 host/runtime/device/startup/main/weights objects unchanged. No composition with1902,1911,1912,1920or1926. Local GSIM timed out after control only; candidate stock cycles are unknown.','actual_hardware_cycles':None,'original_full_native_strict_gate':'pending','token_usage_available':False})
print('BROADCAST_PACKET_WHOLE_BUILD_PASS',sha(case/'model.elf'),flush=True)
