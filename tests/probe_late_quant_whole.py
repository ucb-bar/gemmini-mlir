"""Validate an already-built late-RNE ELF and its independent native oracle."""
import argparse,importlib.util,os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('build-dir','work','capture','llvm-bin','spike','source-cwd'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
base,work,capture,llvm,spike,cwd=(x.resolve() for x in (a.build_dir,a.work,a.capture,a.llvm_bin,a.spike,a.source_cwd))
build=work/'native_build';build.mkdir(exist_ok=True)
for original in base.iterdir():
    if original.name not in ('lower','model.elf') and not (build/original.name).exists():
        (build/original.name).symlink_to(original,target_is_directory=original.is_dir())
(build/'lower').mkdir(exist_ok=True)
if not (build/'lower/model.ll').exists():(build/'lower/model.ll').symlink_to(work/'model.native.ll')
if not (build/'model.elf').exists():(build/'model.elf').symlink_to(work/'model.elf')
spec=importlib.util.spec_from_file_location('whole_probe',Path(__file__).with_name('fused_whole_model_probe.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Catalog oracle paths are bound relative to the original build's working tree.
os.chdir(cwd)
print(module.native_validate(capture,build,work/'host',llvm),flush=True)
print(module.spike_validate(capture,build,spike,work),flush=True)
