"""Actual default predecessor comparison and normal provider final-ELF audit."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from merlin.common.digest import sha256_file
from merlin.runtime.backends import spike_model
from mlir_oot.no_fsm_audit import audit_elf

CORE=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005')
WORK=Path(__file__).resolve().parent
flags=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin']

old_source=subprocess.check_output(['git','show','HEAD:src/merlin/runtime/backends/spike_model.py'],cwd=CORE,text=True)
snapshot=WORK/'predecessor_spike_model.py';snapshot.write_text(old_source)
legacy=types.ModuleType('merlin.runtime.backends._provider_predecessor')
legacy.__file__=str(snapshot);legacy.__package__='merlin.runtime.backends'
sys.modules[legacy.__name__]=legacy
exec(compile(old_source,str(snapshot),'exec'),legacy.__dict__)

bundle=WORK/'default_bundle';bundle.mkdir(exist_ok=True)
(bundle/'model.mlir').write_text('''module {
func.func @forward(%a: tensor<5xf32>) -> tensor<5xf32> {
 %init=tensor.empty():tensor<5xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
 ins(%a:tensor<5xf32>) outs(%init:tensor<5xf32>) {
 ^bb0(%v:f32,%unused:f32):
  %two=arith.constant 2.0:f32
  %sum=arith.addf %v,%two:f32
  linalg.yield %sum:f32
 }->tensor<5xf32>
 return %r:tensor<5xf32>
}
}''')
(bundle/'weights.safetensors.manifest.json').write_text('{"0":{"kind":"input","name":"values"}}')
values=np.array([-5,-.75,-0.,1.25,7],dtype=np.float32)
np.savez(bundle/'inputs.npz',in0=values)
old=legacy.build(bundle,WORK/'default_predecessor',arena_mb=1,cflags_override=flags)
new=spike_model.build(bundle,WORK/'default_current',arena_mb=1,cflags_override=flags)
assert old['elf'].read_bytes()==new['elf'].read_bytes()
assert old['build_hash']==new['build_hash']
assert not (WORK/'default_current/host_provider').exists()
assert 'host_provider' not in new

test_path=CORE/'merlin/tests/runtime/test_host_provider.py'
spec=importlib.util.spec_from_file_location('host_provider_test',test_path)
test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
provider=WORK/'normal_provider';provider.mkdir(exist_ok=True)
test.test_normal_upstream_model_links_ranked_provider_and_runs_stock_isa(provider)
elf=provider/'build/model.elf'
audit=audit_elf(elf.read_bytes());assert audit['status']=='pass'
run=spike_model.run(elf,mem_bytes=1<<34,isa='rv64gc',timeout=60)
np.testing.assert_array_equal(run['outputs'].view(np.uint32),(values+np.float32(2)).view(np.uint32))
(WORK/'normal_provider_uart.log').write_text(run['console'])
identities={}
for path in [snapshot,Path(__file__),CORE/'src/merlin/runtime/host_provider.py',
             CORE/'src/merlin/runtime/backends/spike_model.py',test_path,
             old['elf'],new['elf'],elf,provider/'build/compilation_recipe.json',
             provider/'build/host_provider/host_provider.json',WORK/'normal_provider_uart.log']:
    identities[str(path)]=sha256_file(path)
record=dict(scope='Actual minimal full normal upstream provider/fallback link, not full Smol target qualification',
            default_predecessor_elf_byte_identical=True,default_build_hash_identical=True,
            default_hook_inert=True,output_words=5,output_bit_mismatches=0,
            isa='rv64gc',memref_rank_mismatch=run['metrics']['memref_rank_mismatch'],
            strict_instructions=run['metrics']['cycles'],instruction_metric_is_hardware_cycles=False,
            nofsm=audit,pins=identities,token_usage_available=False)
(WORK/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:value for key,value in record.items() if key!='pins'},indent=2))
