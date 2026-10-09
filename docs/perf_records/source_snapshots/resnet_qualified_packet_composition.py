"""Combine qualified CPU packet and device/readout artifacts at exact ABI boundaries."""
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
study=root/'out/artifacts/probes/readout-saturation-packets-20261005'
base=Path('/scratch/agustin/tmp/gemmini-prestem-scheduled-rne-20261005/out/whole_quant_packet_live_transfer')
build=base/'build_direct'
work=study/'composed';work.mkdir(parents=True,exist_ok=True)
control_record=json.loads((root/'docs/perf_records/firesim1886_resnet_quant_packet_verified.json').read_text())
assert sha(build/'model.elf')==control_record['elf_sha256']
device_control=root/'out/whole_transfer_residual_composed/build_direct'
catalog=json.loads((build/'device_catalog/device_catalog.json').read_text())
device_catalog=json.loads((device_control/'device_catalog/device_catalog.json').read_text())
readout=json.loads((study/'whole/qualification.json').read_text())
assert catalog['source_sha256']==device_catalog['source_sha256']
for key in ('total_device_contractions','bindings','kernels','physical_layout','guarded_mean_additions','pooled_stem'):
    assert catalog[key]==device_catalog[key],key
for group in ('residual_additions','fused_requantizations'):
    assert len(catalog[group])==len(device_catalog[group])
    for old,new in zip(catalog[group],device_catalog[group]):
        for key in ('region','symbol','kernel','shape','proof','integer_readout','numeric_contract'):
            assert old.get(key)==new.get(key),(group,key)
profile=json.loads(Path('/scratch/agustin/tmp/gemmini-reference-parity-20261005/out/resnet1874_conserved_leaf_profile/profile_build.json').read_text())
argv=profile['linker_argv']
flags=argv[:next(i for i,x in enumerate(argv) if x.endswith('.o'))]
units=[build/x for x in ['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o']]
old_catalog=Path(catalog['compilation']['linker_argv'][-1])
assert sha(old_catalog)==catalog['compilation']['object_sha256']
control=work/'reproduced_control.elf'
command=flags+list(map(str,units))+[str(old_catalog),str(build/'device/device_catalog_shim.o'),'-lm','-o',str(control)]
subprocess.run(command,check=True,capture_output=True)
assert sha(control)==sha(build/'model.elf'),'must reproduce verified1886 control byte for byte'
new_catalog=study/'whole/guarded_mean_mixed.o'
# Close the selected object's identity against the independently qualified
# readout candidate, rather than assuming a path still contains that object.
readout_command=list(readout['linker_argv'])
readout_control=work/'reproduced_readout.elf'
readout_command[-1]=str(readout_control)
subprocess.run(readout_command,check=True,capture_output=True)
assert sha(readout_control)==readout['elf_sha256']
target_dir=work/'build';target_dir.mkdir(exist_ok=True)
elf=target_dir/'model.elf'
command[command.index(str(old_catalog))]=str(new_catalog);command[-1]=str(elf)
subprocess.run(command,check=True,capture_output=True)
audit=audit_elf(elf.read_bytes());assert audit['status']=='pass'
host=work/'host';host.mkdir(exist_ok=True)
sources=[];changes=0
for index,path in enumerate(catalog['native_oracle_sources']):
    text=Path(path).read_text();changed=False
    for row in catalog['fused_requantizations']:
        proof=row.get('integer_readout')
        if not proof:continue
        symbol=row['symbol']+'_readout'
        before=emit_readout(proof,symbol,fixedpoint=True)
        if before in text:
            assert text.count(before)==1
            text=text.replace(before,emit_readout(proof,symbol,fixedpoint=True,saturation_first=True,packet=8))
            changes+=1;changed=True
    if changed:
        source=host/f'oracle_{index}.c';source.write_text(text);sources.append(str(source))
    else:sources.append(path)
assert changes==2
library=host/'model.so'
subprocess.run(['cc','-O3','-march=native','-fPIC','-shared',str(base/'host/model.o'),str(base/'host/reference.c'),*sources,str(build/'device/device_catalog_shim.c'),str(mlir_runtime_c()),'-lm','-o',str(library)],check=True,capture_output=True)
capture=base/'capture';reference=np.load(capture/'golden.npy');actual=np.zeros_like(reference)
args=forward_args(capture,build);model=HostModel.load(str(library));model([(a.ctypes.data,a.shape) for a in args]+[(actual.ctypes.data,actual.shape)])
native=quality(actual,reference);assert native['exact_equal'];np.save(host/'output.npy',actual)
(host/'validation.json').write_text(json.dumps(native,indent=2)+'\n')
target=spike_validate(capture,target_dir,Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'),work)
record=dict(schema='source_bound_qualified_arm_composition_v1',control_job=1886,control_cycles=control_record['kernel_cycles'],
    elf=str(elf),elf_sha256=sha(elf),control_elf_reproduced_sha256=sha(control),
    catalog_source_sha256=catalog['source_sha256'],same_catalog_semantics_proofs_bindings_verified=True,
    unchanged_host_runtime_startup_weights_shim={str(p):sha(p) for p in [*units,build/'device/device_catalog_shim.o']},
    selected_composite_catalog_sha256=sha(new_catalog),readout_qualification_sha256=sha(study/'whole/qualification.json'),
    readout_candidate_reproduced_sha256=sha(readout_control),
    original_golden_sha256=sha(capture/'golden.npy'),native=native,spike=target,no_fsm=audit,linker_argv=command,
    scope='Compose1886genericquantCPU packets with1874bankedresidual and source-exact readout8; every original output rechecked; no additive gain inference',
    full_model_hardware_measured=False)
(work/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(elf=str(elf),elf_sha256=sha(elf),native_exact=native['exact_equal'],spike_exact=target['exact_equal'],metrics=target['metrics']),indent=2),flush=True)
