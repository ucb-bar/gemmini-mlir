from pathlib import Path
import hashlib,json,subprocess,os
from datetime import datetime,timezone
from collections import Counter
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.broadcast_math_hoist import FEATURE as HOIST
from merlin.llvmlower.llvm_loop_outline import MERGE_FEATURE
from merlin.llvmlower.late_quant_rne import _tokens
from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent
N=W.parent/'tiny-broadcast-math-20261005'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc');SCRIPT=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/baremetal/spike/model_link.ld')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
cap=json.loads((W.parent/'tiny-pointwise-outline-20261006/capsule_result.json').read_text());assert cap['saved_percent']>0 and cap['DONE'] and cap['all5_frm_and_sticky_flags_match']
features={'respect_captured_quantization_scope','hoist_weight_invariant_quantize','scalar_contraction_accumulator_8_outputs','unroll_scalar_contraction_reduction_by_2','packet_scalar_pointwise_fma_division_2','fuse_quantize_round_convert','fuse_activation_polynomial_fma',HOIST}
source=B/'device_host_abi/model.mlir';sourcepaths=[source,B/'model.prepared.mlir',B/'device_host_abi/borrowed.c',B/'device_host_abi/writer_contracts.json',B/'device_signatures.json',B/'device_catalog/device_catalog.json',B/'device/catalog_binding.json',B/'host_llvm/receipt.json',B/'host_llvm/expanded_bridge.ll',B/'host_llvm/expanded_bridge.native.ll',B/'quant_hoist_args.json']
sourcepins={str(p):sha(p)for p in sourcepaths if p.exists()}
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
for arm,extra in [('control',set()),('whole',{MERGE_FEATURE})]:
 case=W/arm;case.mkdir(exist_ok=False)
 r=lower_model(source.read_text(),workdir=case/'lower',targets=(),features=features|extra)
 selected=merlin_host_llvm_transform(LLVM,combine_clamp=True)(r.ll_path,case/'host_llvm')
 for native in [False,True]:
  original=case/'host_llvm/model.native.ll'if native else selected;bridge=B/'host_llvm/expanded_bridge.native.ll'if native else B/'host_llvm/expanded_bridge.ll';output=case/'host_llvm/expanded.native.ll'if native else case/'host_llvm/expanded.ll'
  subprocess.run([str(LLVM/'llvm-link'),'-S',str(original),str(bridge),'-o',str(output)],check=True,capture_output=True)
 argv=[str(LLVM/'clang'),*flags,'-c',str(case/'host_llvm/expanded.ll'),'-o',str(case/'model.o')]
 subprocess.run(argv,check=True,capture_output=True)
 save(case/'normal_lower_recipe.json',dict(schema='typed_llvm_loop_organization_normal_source_lower_v1',core_commit='058a86465',features=sorted(features|extra),source_abi_pins=sourcepins,source_seam='Fresh normal lower_model on immutable original1880post-offload source. Same original pointwise2/RNE/writer bridge;1926 sourcehoist retained. Only explicit generic LLVM loop extraction/noinline/mergefunc added before original boundedRNE.',normal_lower_stats=r.stats,candidate_compile_argv=argv,raw_llvm_sha256=sha(r.ll_path),selected_target_llvm_sha256=sha(case/'host_llvm/expanded.ll'),selected_native_llvm_sha256=sha(case/'host_llvm/expanded.native.ll'),token_usage_available=False))
 print('NORMAL_WHOLE_COMPILED',arm,flush=True)
control=W/'control';case=W/'whole'
assert sha(control/'model.o')==sha(N/'whole/model.o')
assert sha(control/'host_llvm/expanded.ll')==sha(N/'whole/host_llvm/expanded.ll')
assert sha(control/'host_llvm/expanded.native.ll')==sha(N/'whole/host_llvm/expanded.native.ll')
order=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o','device_catalog/kernel.o','device/device_catalog_shim.o']
baseline=[control/'model.o'if name=='model.o'else B/name for name in order];selected=[case/'model.o'if name=='model.o'else B/name for name in order]
pins={str(p):sha(p)for p in baseline+selected}
linkflags=['-O2','-ffreestanding','-fno-builtin','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-nostdlib','-nostartfiles','-Wl,--defsym,MERLIN_WEIGHTS_BASE=0x84100000','-Wl,--defsym,MERLIN_STACK_BYTES=0x1000000','-T',str(SCRIPT)]
baseargv=[str(GCC),*linkflags,*map(str,baseline),'-lm','-o',str(W/'baseline_reproduced.elf')]
assert not (W/'baseline_reproduced.elf').exists();subprocess.run(baseargv,check=True,capture_output=True)
assert sha(W/'baseline_reproduced.elf')==sha(N/'whole/model.elf')=='8dadfe71e3c224e5e16189cd3985101807d21e34490f73b119c083865e7d9c93'
(W/'baseline_reproduced.elf').unlink();os.link(N/'whole/model.elf',W/'baseline_reproduced.elf');print('1926_BASELINE_RELINK_BYTEEXACT',flush=True)
argv=[str(GCC),*linkflags,*map(str,selected),'-lm','-o',str(case/'model.elf')];assert not (case/'model.elf').exists();subprocess.run(argv,check=True,capture_output=True)
for p,expected in pins.items():assert sha(p)==expected
for p,expected in sourcepins.items():assert sha(p)==expected
writer=json.loads((B/'device_host_abi/writer_contracts.json').read_text());bounds={'@'+name for route in writer['routes']for name in [route['symbol'],route['borrowed_symbol'],route['symbol']+'__fresh_tensor_result']}
symbols=lambda p:Counter(t.text for t in _tokens(p.read_text())if t.text.startswith('@'))
a=symbols(control/'host_llvm/expanded.ll');b=symbols(case/'host_llvm/expanded.ll');assert {name:a[name]for name in bounds}=={name:b[name]for name in bounds};assert sum(route['calls']for route in writer['routes'])==155
normal=json.loads((case/'normal_lower_recipe.json').read_text());normal.update(all_original_source_bound_device_symbol_reference_counts_conserved=True,original_source_bound_calls=155,normal_control_llvm_target_native_object_byteidentical1926=True);save(case/'normal_lower_recipe.json',normal)
audit=audit_elf((case/'model.elf').read_bytes());assert audit['status']=='pass';save(case/'model.nofsm_audit.json',audit)
save(case/'controlled_link.json',dict(schema='tiny_loop_organization_controlled1926_link_v1',finished_utc=datetime.now(timezone.utc).isoformat(),baseline_elf_sha256=sha(N/'whole/model.elf'),baseline_reproduced_sha256=sha(W/'baseline_reproduced.elf'),baseline_byte_exact=True,baseline_reproduction_argv=baseargv,candidate_link_argv=argv,baseline_objects={str(p):pins[str(p)]for p in baseline},candidate_objects={str(p):pins[str(p)]for p in selected},only_changed_object='model.o',elf_sha256=sha(case/'model.elf'),inherited_marker='37bdf9be0856',marker_contract='Inherited historical1880marker is nonunique; exactELF and source/object closure identifycandidate.',scope='Fresh normal generic exact LLVM loop organization; original1926hoist/FMApacket2/RNE/writer retained; onlymodel.o changed, all1880host/runtime/device/startup/main/weights objects frozen. No composition withrejected1954.',actual_hardware_cycles=None,original_full_native_strict_gate='pending',token_usage_available=False))
print('LOOP_ORGANIZATION_FULL_BUILD_PASS',sha(case/'model.elf'),flush=True)
