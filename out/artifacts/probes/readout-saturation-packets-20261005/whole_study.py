"""Relink the pinned current control with only two exact CPU readouts changed."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.integer_readout import emit_readout
from mlir_oot.no_fsm_audit import audit_elf
from tests.fused_whole_model_probe import forward_args,quality,spike_validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
base=root/'out/whole_transfer_residual_composed'
build=base/'build_direct'
work=root/'out/artifacts/probes/readout-saturation-packets-20261005/whole'
work.mkdir(parents=True,exist_ok=True)
profile_path=Path('/scratch/agustin/tmp/gemmini-reference-parity-20261005/out/resnet1874_conserved_leaf_profile/profile_build.json')
profile=json.loads(profile_path.read_text())
assert sha(build/'model.elf')=='aabd9b15d947695ae20e37903fecc66bd63f200c0d6646b5f2797cb5a52631b2'
for path,digest in profile['objects'].items():assert sha(path)==digest
catalog=json.loads((build/'device_catalog/device_catalog.json').read_text())
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
replacements={};readout_rows=[];native_replacements={}
for route in catalog['fused_requantizations']:
    proof=route.get('integer_readout')
    if not proof:continue
    symbol=route['symbol']+'_readout'
    before=emit_readout(proof,symbol,fixedpoint=True)
    after=emit_readout(proof,symbol,fixedpoint=True,saturation_first=True,packet=8)
    argv=list(route['adapter_compilation']['compiler_argv'])
    old_object=Path(argv[argv.index('-o')+1]);source=old_object.with_suffix('.c')
    assert sha(old_object)==route['adapter_compilation']['object_sha256']
    text=source.read_text();assert text.startswith(before)
    directory=work/route['symbol'];directory.mkdir(exist_ok=True)
    new_source=directory/'adapter.c';new_source.write_text(after+text[len(before):])
    new_object=directory/'adapter.o'
    command=[str(new_source) if x==str(source) else str(new_object) if x==str(old_object) else x for x in argv]
    subprocess.run(command,check=True,capture_output=True)
    replacements[old_object]=new_object
    readout_rows.append(dict(symbol=symbol,proof=proof,old_source_sha256=sha(source),new_source_sha256=sha(new_source),old_object_sha256=sha(old_object),new_object_sha256=sha(new_object),compiler_argv=command))
    native_replacements[before]=after
assert len(readout_rows)==2
requant=next(x for x in profile['leaf_component_reproduction'] if Path(x['component']).name=='requant.o')
new_requant=work/'requant.o'
argv=[str(replacements.get(Path(x),Path(x))) if x.endswith('.o') else x for x in requant['argv']]
argv[argv.index('-o')+1]=str(new_requant)
subprocess.run(argv,check=True,capture_output=True)
replacements[Path(requant['component'])]=new_requant
nodes={}
def visit(value):
    if isinstance(value,dict):
        a=value.get('linker_argv')
        if a and '-r' in a:nodes[Path(a[a.index('-o')+1])]=(a,value['object_sha256'])
        for child in value.values():visit(child)
    elif isinstance(value,list):
        for child in value:visit(child)
visit(catalog['compilation'])
def rebuild(path):
    if path in replacements:return replacements[path]
    if path not in nodes:return path
    argv,digest=nodes[path];assert sha(path)==digest
    changed=list(argv)
    for index in range(argv.index('-r')+1,argv.index('-o')):
        changed[index]=str(rebuild(Path(argv[index])))
    target=work/path.name;changed[changed.index('-o')+1]=str(target)
    subprocess.run(changed,check=True,capture_output=True)
    return target
top=Path(catalog['compilation']['linker_argv'][-1]);assert top in nodes
new_catalog=rebuild(top)
old_argv=profile['linker_argv'];output_index=old_argv.index('-o')
flags=old_argv[:next(i for i,x in enumerate(old_argv) if x.endswith('.o'))]
units=[build/x for x in ['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o']]
shim=build/'device/device_catalog_shim.o'
control=work/'reproduced_control.elf'
command=flags+list(map(str,units))+[str(top),str(shim),'-lm','-o',str(control)]
subprocess.run(command,check=True,capture_output=True)
assert sha(control)==sha(build/'model.elf'),'control link must reproduce verified1874 exactly'
target_dir=work/'build';target_dir.mkdir(exist_ok=True)
elf=target_dir/'model.elf'
command[-1]=str(elf);command[command.index(str(top))]=str(new_catalog)
subprocess.run(command,check=True,capture_output=True)
audit=audit_elf(elf.read_bytes());assert audit['status']=='pass'
native_dir=work/'host';native_dir.mkdir(exist_ok=True)
native_sources=[]
for index,path in enumerate(catalog['native_oracle_sources']):
    text=Path(path).read_text();changed=False
    for before,after in native_replacements.items():
        if before in text:
            assert text.count(before)==1;text=text.replace(before,after);changed=True
    if changed:
        native_path=native_dir/f'oracle_{index}.c';native_path.write_text(text);native_sources.append(str(native_path))
    else:native_sources.append(path)
library=native_dir/'model.so'
native_command=['cc','-O3','-march=native','-fPIC','-shared',str(base/'host/model.o'),str(base/'host/reference.c'),*native_sources,str(build/'device/device_catalog_shim.c'),str(mlir_runtime_c()),'-lm','-o',str(library)]
subprocess.run(native_command,check=True,capture_output=True)
capture=base/'capture';reference=np.load(capture/'golden.npy');actual=np.zeros_like(reference)
args=forward_args(capture,build)
model=HostModel.load(str(library));model([(a.ctypes.data,a.shape) for a in args]+[(actual.ctypes.data,actual.shape)])
native=quality(actual,reference);assert native['exact_equal'];np.save(native_dir/'output.npy',actual)
(native_dir/'validation.json').write_text(json.dumps(native,indent=2)+'\n')
spike=spike_validate(capture,target_dir,Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'),work)
receipt=dict(schema='whole_exact_readout_schedule_qualification_v1',control_job=1874,control_cycles=39201279,
    control_elf_sha256=sha(control),elf=str(elf),elf_sha256=sha(elf),readout_options=dict(saturation_first=True,packet=8),
    changed_adapters=readout_rows,unchanged_control_object_pins={p:d for p,d in profile['objects'].items() if Path(p) not in replacements},
    native=native,spike=spike,no_fsm=audit,linker_argv=command,
    original_golden_sha256=sha(capture/'golden.npy'),baseline_profile_sha256=sha(profile_path),
    scope='Only exact generic host readout schedules changed; every device instruction schedule/host model/runtime/weight object preserved',
    full_model_hardware_measured=False)
(work/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(elf=str(elf),elf_sha256=sha(elf),native_exact=native['exact_equal'],spike_exact=spike['exact_equal'],metrics=spike['metrics']),indent=2),flush=True)
