"""Compose pre-proved raw stem-pool fusion with remaining prepared catalogs."""
import json,shutil,subprocess
from pathlib import Path
from .fused_mixed_catalog import merlin_callbacks as fused_callbacks,sha
from .no_fsm_audit import audit_elf


def merlin_callbacks(llvm_bin,requant_bundle,pool_bundle):
    llvm_bin,pool_bundle=map(Path,(llvm_bin,pool_bundle));prepare_base,build_base=fused_callbacks(llvm_bin,requant_bundle);pool=json.loads((pool_bundle/'stem_pool.json').read_text());pin=sha(pool_bundle/'stem_pool.json')
    def prepare(source,work):
        from .frontend.parse import parse_module
        m=parse_module(Path(source).read_text());calls=[o for o in m.walk() if o.name=='func.call' and o.callee.root_reference.data==pool['symbol']]
        if len(calls)!=1:raise ValueError('exact pooled stem call missing')
        prepared=prepare_base(source,work)
        declarations=[o for o in parse_module(Path(prepared).read_text()).walk() if o.name=='func.func' and o.sym_name.data==pool['symbol']]
        if len(declarations)!=1 or [getattr(a.data.get('bufferization.access'),'data',None) for a in declarations[0].arg_attrs]!=['read','read','write']:
            raise ValueError('pooled stem declaration lost access contract')
        return prepared
    def build(source,work):
        if sha(pool_bundle/'stem_pool.json')!=pin or sha(pool_bundle/'stem_pool.o')!=pool['object_sha256']:raise ValueError('stem pool bundle changed')
        path,obj=build_base(source,work);manifest=json.loads(path.read_text());combined=Path(work)/'pool_fused_mixed.o';linker=llvm_bin/'ld.lld'
        if not linker.is_file():linker=Path(shutil.which('ld.lld'))
        command=[str(linker),'-r',str(obj),str(pool_bundle/'stem_pool.o'),'-o',str(combined)];subprocess.run(command,check=True,capture_output=True);audit=audit_elf(combined.read_bytes())
        if audit['status']!='pass':raise ValueError('pooled model object audit failed')
        manifest['compilation']={'schema':'gemmini_pool_fused_mixed_compile_v1','object_sha256':sha(combined),'object_nofsm_status':'pass','remaining_compilation':manifest['compilation'],'pool_manifest_sha256':pin,'linker_sha256':sha(linker),'linker_argv':command};manifest['pooled_stem']=pool;manifest['total_device_contractions']+=1;manifest['native_oracle_sources'].append(str(pool_bundle/'native_oracle.c'));path.write_text(json.dumps(manifest,indent=2)+'\n');return path,combined
    return prepare,build
