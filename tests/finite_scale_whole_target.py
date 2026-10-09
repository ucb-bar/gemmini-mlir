"""Frozen2070-controlled final link and original whole target qualification.

Only the normal host model object changes. This prepares reviewable qualification;
it admits no hardware arm and makes no whole cycle prediction.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from mlir_oot.no_fsm_audit import audit_elf
from finite_scale_whole_prepare import OLD,HERE,OUT,LLVM,sha,save

GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
ORIGINAL=OLD/'out/artifacts/probes/closed-i8-interval-normal-2062-20261007/whole'
CURRENT_ELF_SHA='b8110c62eb1a771e8e05475bebe5fb0dc3448536bc19dbe865ada5536d53d0ed'
RAW='ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3'
GOLD=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle/golden.npy')
TARGET=OUT/'whole'


def main():
    seal=HERE/'docs/perf_records/finite_scale_guard_preparation.json';preparation=json.loads(seal.read_text())
    assert preparation['status']=='pass'
    for name,digest in preparation['pins'].items():assert sha(name)==digest,name
    native=json.loads((OUT/'qualification.json').read_text());assert all(e['all256000_original_words_exact'] and e['torch_gate'] for e in native['events'])
    original=json.loads((ORIGINAL/'build.json').read_text());old_model=ORIGINAL/'target/model.o';old_elf=ORIGINAL/'target/model.elf'
    assert sha(old_elf)==CURRENT_ELF_SHA and sha(OUT/'control/target/model.o')==sha(old_model)
    TARGET.mkdir(exist_ok=False);case=TARGET/'target';case.mkdir()
    os.link(OUT/'selected/target/model.o',case/'model.o');commands=[]
    def run(argv):
        argv=list(map(str,argv));commands.append(argv)
        result=subprocess.run(argv,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result
    baseline=[str(OUT/'control/target/model.o') if name==str(old_model) else name for name in original['candidate_link_argv']]
    baseline[-1]=str(TARGET/'baseline_reproduced.elf');run(baseline)
    assert sha(TARGET/'baseline_reproduced.elf')==CURRENT_ELF_SHA
    (TARGET/'baseline_reproduced.elf').unlink();os.link(old_elf,TARGET/'baseline_reproduced.elf')
    argv=[str(case/'model.o') if name==str(OUT/'control/target/model.o') else name for name in baseline];argv[-1]=str(case/'model.elf')
    objects={name:sha(name) for name in argv if name.endswith('.o')}
    unchanged={name:digest for name,digest in original['candidate_objects'].items() if name!=str(old_model)}
    assert len(unchanged)==11 and {name:digest for name,digest in objects.items() if name!=str(case/'model.o')}==unchanged
    run(argv);assert objects=={name:sha(name) for name in objects}
    elf=case/'model.elf';audit=audit_elf(elf.read_bytes());assert audit['status']=='pass';save(case/'model.nofsm_audit.json',audit)
    (case/'model.dump').write_text(run([GCC.with_name('riscv64-unknown-elf-objdump'),'-dr',case/'model.o']).stdout)
    output=OUT/'selected/native/host/output.npy';array=np.load(output)
    assert array.shape==(1,8,32000) and array.dtype==np.float32 and hashlib.sha256(array.astype('<f4').tobytes()).hexdigest()==RAW
    assert np.allclose(array,np.load(GOLD),atol=.03125,rtol=.02)
    recipe={'schema':'finite_scale_source_controlled2070_whole_link_v1','status':'pass','baseline_byte_exact':True,'control_job':2070,'control_cycles':394765577,'baseline_elf_sha256':CURRENT_ELF_SHA,'candidate_elf_sha256':sha(elf),'candidate_objects':objects,'candidate_link_argv':argv,'baseline_reproduction_argv':baseline,'unchanged_nonmodel_objects':unchanged,'only_changed_linked_object':'model.o','all155devicebindings':True,'source_producer_chains':44,'source_helpers':22,'whole_hardware_cycles':'UNKNOWN','pins':{str(p):sha(p) for p in [Path(__file__),seal,TARGET/'baseline_reproduced.elf',old_elf,ORIGINAL/'build.json',OUT/'qualification.json',*case.iterdir(),*map(Path,objects)] if p.is_file()},'commands':commands,'token_usage_available':False}
    save(TARGET/'build.json',recipe)
    started=datetime.now(timezone.utc).isoformat();cmd=[str(GCC.with_name('spike')),'--isa=rv64gc','--extension=gemmini',str(elf)]
    with (case/'spike.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,text=True)
    text=(case/'spike.log').read_text();metrics={}
    for line in text.splitlines():
        if line.startswith('METRIC '):
            _,name,value=line.split(maxsplit=2);assert name not in metrics;metrics[name]=value
    passed=result.returncode==0 and text.count('OUT_SHA256 f32le 256000 1024000 '+RAW)==1 and text.count('DONE')==1 and metrics.get('memref_rank_mismatch')=='0'
    record={'schema':'finite_scale_original_Tiny_whole_target_v1','status':'pass' if passed else 'fail','started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'argv':cmd,'elf_path':str(elf),'elf_sha256':sha(elf),'exit_code':result.returncode,'reference_path':str(output),'reference_sha256':sha(output),'torch_golden_path':str(GOLD),'torch_golden_sha256':sha(GOLD),'torch_atol':.03125,'torch_rtol':.02,'torch_allclose':True,'spike_console_path':str(case/'spike.log'),'spike_console_sha256':sha(case/'spike.log'),'spike_output_sha256':RAW if passed else None,'spike_full_output_match':passed,'metrics':metrics,'functional_instructions_not_hardware_cycles':int(metrics.get('cycles','0')),'normal_lower_recipe_path':str(OUT/'selected/target/host_llvm/source_binding.json'),'normal_lower_recipe_sha256':sha(OUT/'selected/target/host_llvm/source_binding.json'),'controlled_link_path':str(TARGET/'build.json'),'controlled_link_sha256':sha(TARGET/'build.json'),'no_fsm':True,'control_job':2070,'control_cycles':394765577,'whole_hardware_cycles':'UNKNOWN','hardware_submission':'NONE','token_usage_available':False}
    save(TARGET/'target_validation.json',record);assert passed
    print('FINITE_SCALE_ORIGINAL_WHOLE_STRICT_PASS',record['elf_sha256'],record['functional_instructions_not_hardware_cycles'],flush=True)

if __name__=='__main__':main()
