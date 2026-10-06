from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
from mlir_oot.golden_device_profile import build
W=Path(__file__).resolve().parent
B=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
SOURCE=W.parent/'tiny-rectangular-whole-20261006/whole'
RUNTIME=Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime')
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
O=W.parents[3]
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
original=json.loads((SOURCE/'controlled_link.json').read_text())
native=json.loads((SOURCE/'host/validation.json').read_text())
strict=json.loads((SOURCE/'spike_validation.json').read_text())
assert native['status']==strict['status']=='pass' and native['original_compiled_bits_exact'] and native['allclose']
assert sha(SOURCE/'model.elf')==original['elf_sha256']==strict['elf_sha256']=='39c55f91736b5dffbb1d0460b5372891c492a669202988c722b468e6fe297d9a'
for p,h in original['candidate_objects'].items():assert sha(p)==h
(W/'overlay').mkdir(exist_ok=False)
units=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o']
links={}
for name in units+['model.elf','device_catalog/device_catalog.json','device_catalog/kernel.o','device/device_catalog_shim.o']:
 source=SOURCE/name if name in ['model.o','model.elf'] else B/name
 target=W/'overlay'/name;target.parent.mkdir(parents=True,exist_ok=True)
 os.link(source,target)
 assert source.stat().st_ino==target.stat().st_ino and sha(source)==sha(target)
 links[str(target)]={'source':str(source),'sha256':sha(source),'shared_inode':source.stat().st_ino,'size':source.stat().st_size}
clock=datetime.now(timezone.utc).isoformat()
r=build(W/'overlay',RUNTIME,GCC,LLVM,W/'profile')
assert r['expected_device_calls']==155 and len(r['boundaries'])==5 and sum(r['expected_symbol_calls'].values())==155
assert r['model_object_sha256']==sha(SOURCE/'model.o')
# The wrapper's complete compiler dependency closure is descriptive only; no rebuild of the pinned object.
compile_argv=[str(GCC),'-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-I',str(RUNTIME/'baremetal/spike'),'-I',str(RUNTIME/'c'),'-MD','-MF',str(W/'profile/device_profile.d'),'-fsyntax-only',str(W/'profile/device_profile.c')]
subprocess.run(compile_argv,check=True,capture_output=True)
import shlex
text=(W/'profile/device_profile.d').read_text().replace('\\\n',' ')
deps=[Path(p)for p in shlex.split(text.split(':',1)[1])]
files=set(deps+[Path(__file__),O/'mlir_oot/golden_device_profile.py',O/'mlir_oot/no_fsm_audit.py',GCC,LLVM/'llvm-nm',RUNTIME/'baremetal/spike/model_link.ld',SOURCE/'host/validation.json',SOURCE/'spike_validation.json',SOURCE/'reference_validation.json',SOURCE/'normal_lower_recipe.json',SOURCE/'controlled_link.json'])
receipt={'schema':'current2004_device_profile_overlay_v1','status':'built_not_qualified','started_utc':clock,'finished_utc':datetime.now(timezone.utc).isoformat(),'source_stock_job':2004,'source_stock_cycles':422018733,'source_elf_sha256':sha(SOURCE/'model.elf'),'profile_elf_sha256':r['elf_sha256'],'model_arithmetic_object_identical':True,'all_original_runtime_device_startup_main_weights_objects_identical':True,'expected_calls':155,'physical_added_elf_bytes':(W/'profile/model.elf').stat().st_size,'overlay_added_object_bytes':0,'semantic_scope':'Instrumentation only around155 frozen kernel calls and original whole runtime. Inside-call includes CPUcommand/address/control/fence/wait+accelerator/DMA; no physical accelerator-only fraction. Host gaps include wrapper bookkeeping. Native originalmodel qualification retained by exact object/ABI identity; fresh strictfull original256000/Torch+completecallorder/count/conservation required. No actualstockprofile yet.','marker_contract':'Inherited37bdf9be0856 nonunique; exactELF/SHA/object/source closure identifiesprofile.','token_usage_available':False,'overlay_links':links,'wrapper_dependency_check_argv':compile_argv,'pins':{str(p):sha(p)for p in sorted(files)},'build_manifest_path':str(W/'profile/profile_build.json'),'build_manifest_sha256':sha(W/'profile/profile_build.json')}
(W/'overlay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('CURRENT2004_PROFILE_BUILT',r['elf_sha256'],receipt['physical_added_elf_bytes'],flush=True)
