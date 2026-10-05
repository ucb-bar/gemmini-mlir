"""Optional packed stem before the exact fused/direct/dense mixed catalog."""
import json,subprocess,shutil
from pathlib import Path
from .fused_mixed_catalog import merlin_callbacks as fused_callbacks,sha
from .stem_bundle import build as build_stem
from .no_fsm_audit import audit_elf


def merlin_callbacks(llvm_bin,requant_bundle,*,contraction_calibrations=None):
    llvm_bin=Path(llvm_bin);prepare_fused,build_fused=fused_callbacks(llvm_bin,requant_bundle,contraction_calibrations=contraction_calibrations);state={}
    def prepare(source,work):
        directory=Path(work)/'packed_stem';manifest=build_stem(source,llvm_bin,directory)
        state.update(directory=directory,manifest=manifest)
        return prepare_fused(directory/'rewritten.mlir',Path(work)/'remaining')
    def build(source,work):
        work=Path(work);path,obj=build_fused(source,work);manifest=json.loads(path.read_text());stem=state['directory']/'stem.o'
        if sha(stem)!=state['manifest']['object_sha256']:raise ValueError('packed stem object identity changed')
        linker=llvm_bin/'ld.lld'
        if not linker.is_file():linker=Path(shutil.which('ld.lld'))
        combined=work/'stem_fused_mixed.o';command=[str(linker),'-r',str(obj),str(stem),'-o',str(combined)];subprocess.run(command,check=True,capture_output=True)
        audit=audit_elf(combined.read_bytes())
        if audit['status']!='pass':raise ValueError('packed stem composed no-FSM audit failed')
        manifest['compilation']={'schema':'gemmini_stem_fused_mixed_compile_v1','object_sha256':sha(combined),'object_nofsm_status':'pass','remaining_compilation':manifest['compilation'],'stem_manifest_sha256':sha(state['directory']/'stem.json'),'linker_argv':command,'linker_sha256':sha(linker)}
        manifest['packed_stem']=state['manifest'];manifest['total_device_contractions']+=1;manifest['native_oracle_sources'].append(str(state['directory']/'native_oracle.c'));path.write_text(json.dumps(manifest,indent=2)+'\n');return path,combined
    return prepare,build
