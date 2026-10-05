"""Compose an explicitly proved residual bundle with an existing catalog."""
import json
from pathlib import Path
import shutil
import subprocess
from .captured_residual_bundle import sha, binding_attributes
from .frontend.parse import parse_module
from .no_fsm_audit import audit_elf


def apply_capture(capture, bundle):
    capture,bundle=map(Path,(capture,bundle));record=json.loads((bundle/'residual.json').read_text())
    for path,key in ((capture/'model.mlir','source_sha256'),(capture/'capture_receipt.json','capture_receipt_sha256'),(capture/'weights.safetensors','weights_sha256'),(capture/'weights.safetensors.manifest.json','manifest_sha256'),(bundle/'rewritten.mlir','rewritten_sha256')):
        if sha(path)!=record[key]:raise ValueError('residual capture binding changed: '+str(path))
    if not record['routes']:return
    golden_sha=sha(capture/'golden.npy')
    receipt=json.loads((capture/'capture_receipt.json').read_text())
    shutil.copyfile(bundle/'rewritten.mlir',capture/'model.mlir')
    receipt['artifacts']['model.mlir']={'bytes':(capture/'model.mlir').stat().st_size,'sha256':record['rewritten_sha256']}
    receipt['residual_derivation']={'source_sha256':record['source_sha256'],'source_receipt_sha256':record['capture_receipt_sha256'],'bundle_manifest_sha256':sha(bundle/'residual.json'),'numeric_policy':record['numeric_policy'],'original_golden_sha256':golden_sha}
    (capture/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if sha(capture/'golden.npy')!=golden_sha:raise AssertionError('original golden changed')


def merlin_callbacks(llvm_bin,bundle,base_callbacks):
    llvm_bin,bundle=map(Path,(llvm_bin,bundle));record=json.loads((bundle/'residual.json').read_text());pin=sha(bundle/'residual.json')
    prepare_base,build_base=base_callbacks
    symbols={row['symbol']:row for row in record['routes']}
    if not symbols:return base_callbacks
    def check():
        if sha(bundle/'residual.json')!=pin or sha(bundle/'residual.o')!=record['object_sha256'] or sha(bundle/'native_oracle.c')!=record['native_oracle_sha256']:
            raise ValueError('residual artifact identity changed')
    def check_declarations(source):
        module=parse_module(Path(source).read_text())
        calls=[op.callee.root_reference.data for op in module.walk() if op.name=='func.call' and op.callee.root_reference.data in symbols]
        if sorted(calls)!=sorted(symbols):raise ValueError('residual call coverage changed')
        declarations={op.sym_name.data:op for op in module.walk() if op.name=='func.func' and op.sym_name.data in symbols}
        if set(declarations)!=set(symbols):raise ValueError('residual declaration coverage changed')
        for name,op in declarations.items():
            attrs=op.properties.get('arg_attrs')
            if attrs is None or [getattr(x.data.get('bufferization.access'),'data',None) for x in attrs]!=['read','read','write']:
                raise ValueError('residual bufferization access changed')
            for key,value in binding_attributes(symbols[name],record['source_sha256'],record['capture_receipt_sha256']).items():
                if op.attributes.get(key)!=value:raise ValueError('residual numerical/source contract changed: '+key)
    def prepare(source,work):
        check();check_declarations(source)
        prepared=prepare_base(source,work);check_declarations(prepared)
        return prepared
    def build(source,work):
        check();check_declarations(source)
        path,obj=build_base(source,work);manifest=json.loads(path.read_text());combined=Path(work)/'residual_mixed.o';linker=llvm_bin/'ld.lld'
        if not linker.is_file():linker=Path(shutil.which('ld.lld'))
        command=[str(linker),'-r',str(obj),str(bundle/'residual.o'),'-o',str(combined)]
        subprocess.run(command,check=True,capture_output=True);audit=audit_elf(combined.read_bytes())
        if audit['status']!='pass':raise ValueError('residual model object audit failed')
        manifest['compilation']={'schema':'gemmini_residual_mixed_compile_v1','object_sha256':sha(combined),'object_nofsm_status':'pass','remaining_compilation':manifest['compilation'],'residual_manifest_sha256':pin,'linker_sha256':sha(linker),'linker_argv':command}
        manifest['residual_additions']=record['routes'];manifest['residual_numeric_policy']=record['numeric_policy']
        manifest['total_device_contractions']+=len(symbols)
        manifest['native_oracle_sources'].append(str(bundle/'native_oracle.c'))
        path.write_text(json.dumps(manifest,indent=2)+'\n');return path,combined
    return prepare,build
