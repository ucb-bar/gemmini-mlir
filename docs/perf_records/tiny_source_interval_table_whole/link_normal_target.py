from pathlib import Path
import hashlib,json,subprocess,os
from datetime import datetime,timezone
from mlir_oot.no_fsm_audit import audit_elf
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';case=W/'target_v2';ORIGINAL=T.parent/'tiny-rectangular-whole-20261006/whole'
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc');SCRIPT=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/baremetal/spike/model_link.ld')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
native=json.loads((W/'host/validation.json').read_text());fallback=json.loads((W/'control_and_fallback.json').read_text());source=json.loads((case/'host_llvm/source_binding.json').read_text())
assert native['status']=='pass' and native['original_compiled_bits_exact'] and native['allclose'];assert fallback['control_normal_target_native_LLVM_expanded_objects_byteidentical2004'];assert source['all_original_device155_references_conserved']
for p,h in source['pins'].items():assert sha(p)==h
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
argv=[str(LLVM/'clang'),*flags,'-c',str(case/'host_llvm/expanded.ll'),'-o',str(case/'model.o')];subprocess.run(argv,check=True,capture_output=True)
order=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o','device_catalog/kernel.o','device/device_catalog_shim.o']
baseline=[W/'control/model.o'if name=='model.o'else B/name for name in order];selected=[case/'model.o'if name=='model.o'else B/name for name in order]
pins={str(p):sha(p)for p in baseline+selected}
linkflags=['-O2','-ffreestanding','-fno-builtin','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-nostdlib','-nostartfiles','-Wl,--defsym,MERLIN_WEIGHTS_BASE=0x84100000','-Wl,--defsym,MERLIN_STACK_BYTES=0x1000000','-T',str(SCRIPT)]
baseargv=[str(GCC),*linkflags,*map(str,baseline),'-lm','-o',str(W/'baseline_reproduced.elf')]
assert not (W/'baseline_reproduced.elf').exists();subprocess.run(baseargv,check=True,capture_output=True)
assert sha(W/'baseline_reproduced.elf')==sha(ORIGINAL/'model.elf')=='39c55f91736b5dffbb1d0460b5372891c492a669202988c722b468e6fe297d9a'
(W/'baseline_reproduced.elf').unlink();os.link(ORIGINAL/'model.elf',W/'baseline_reproduced.elf');print('2004_BASELINE_RELINK_BYTEEXACT',flush=True)
linkargv=[str(GCC),*linkflags,*map(str,selected),'-lm','-o',str(case/'model.elf')];assert not (case/'model.elf').exists();subprocess.run(linkargv,check=True,capture_output=True)
for p,h in pins.items():assert sha(p)==h
for p,h in source['pins'].items():assert sha(p)==h
audit=audit_elf((case/'model.elf').read_bytes());assert audit['status']=='pass';save(case/'model.nofsm_audit.json',audit)
save(case/'controlled_link.json',dict(schema='tiny_source_interval_controlled2004_link_v1',finished_utc=datetime.now(timezone.utc).isoformat(),baseline_elf_sha256=sha(ORIGINAL/'model.elf'),baseline_reproduced_sha256=sha(W/'baseline_reproduced.elf'),baseline_byte_exact=True,baseline_reproduction_argv=baseargv,candidate_compile_argv=argv,candidate_link_argv=linkargv,baseline_objects={str(p):pins[str(p)]for p in baseline},candidate_objects={str(p):pins[str(p)]for p in selected},only_changed_object='model.o',elf_sha256=sha(case/'model.elf'),inherited_marker='37bdf9be0856',marker_contract='Inherited1880marker nonunique; exactELF/source/object closure identifiescandidate.',scope='Normal _transform_host_ir on immutable actual2004normalLLVM with fresh22typed closedobservers/44scalarbindings, source callbacks and once/helper FRM guard; one8MiBreadonly sourcewide intervaltable shipped inside model.o. Existing complete upstream source/pipeline receipt remains frozen; no newcapture. All original155devicebindings and originalhost/runtime/startup/main/weights objects retained. Only hostmodelobject changed; nativefull256000 exact/Torch gate passes. No cycles forecast from localM2 gain.',actual_hardware_cycles=None,original_full_native_strict_gate='strict pending',token_usage_available=False))
print('NORMAL_SOURCE_INTERVAL_FULL_TARGET_BUILD_PASS',sha(case/'model.elf'),flush=True)
