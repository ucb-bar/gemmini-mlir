"""Compose hash-bound captured epilogues, direct convolution, and dense catalogs."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from .direct_conv_bundle import build as build_direct
from .golden_device_catalog import compile_catalog
from .no_fsm_audit import audit_elf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_capture(capture:Path,bundle:Path,destination:Path):
    """Create derived capture preserving original parameter identities and provenance."""
    capture,bundle,destination=map(Path,(capture,bundle,destination))
    manifest=json.loads((bundle/'requant.json').read_text())
    for path,key in [(capture/'model.mlir','source_sha256'),(capture/'weights.safetensors','weights_sha256'),(capture/'weights.safetensors.manifest.json','manifest_sha256'),(bundle/'rewritten.mlir','rewritten_sha256'),(bundle/'requant.o','object_sha256')]:
        if sha(path)!=manifest[key]:raise ValueError('fused capture identity mismatch: '+str(path))
    destination.mkdir(parents=True,exist_ok=False)
    for source in capture.iterdir():
        if source.name not in ('model.mlir','capture_receipt.json'): (destination/source.name).symlink_to(source.resolve())
    shutil.copyfile(bundle/'rewritten.mlir',destination/'model.mlir')
    receipt=json.loads((capture/'capture_receipt.json').read_text())
    receipt['artifacts']['model.mlir']={'bytes':(destination/'model.mlir').stat().st_size,'sha256':manifest['rewritten_sha256']}
    policy=manifest.get('selected_max_output_lsb',0)
    receipt['derived_from']={'source_sha256':manifest['source_sha256'],'requant_manifest_sha256':sha(bundle/'requant.json'),'transform':'exact captured unary requantization' if policy==0 else 'captured unary requantization with explicitly selected local error policy','numeric_policy':{'unit':'quantized_output_lsb','local_limit':policy,'full_model_quality_established':False,'original_golden_preserved':True}}
    (destination/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return destination


def merlin_callbacks(llvm_bin:Path,requant_bundle:Path,*,flat_spatial=False,propagate_layout=False,reduction_channel_block=0,large_n=False,contraction_calibrations=None):
    """Callbacks for the derived capture; verify fused symbol set before compilation."""
    if type(reduction_channel_block) is not int or reduction_channel_block < 0:
        raise ValueError('reduction_channel_block must be a nonnegative integer')
    if reduction_channel_block and not propagate_layout:
        raise ValueError('reduction_channel_block requires propagate_layout')
    llvm_bin,requant_bundle=map(Path,(llvm_bin,requant_bundle));state={}
    requant=json.loads((requant_bundle/'requant.json').read_text());requant_sha=sha(requant_bundle/'requant.json')
    def check_requant():
        if sha(requant_bundle/'requant.json')!=requant_sha or sha(requant_bundle/'requant.o')!=requant['object_sha256']:raise ValueError('fused bundle changed')
    def prepare(source,work):
        from .frontend.parse import parse_module
        check_requant();module=parse_module(Path(source).read_text())
        from .paired_readout_binding import prepare_paired_scratch
        if prepare_paired_scratch(module, requant):
            from .direct_conv_binding import serialize
            rebound=Path(work)/'paired_scratch.mlir';rebound.parent.mkdir(parents=True,exist_ok=True)
            rebound.write_text(serialize(module, []));source=rebound
        symbols={x['symbol'] for x in requant['routes']};calls=[x.callee.root_reference.data for x in module.walk() if x.name=='func.call' and x.callee.root_reference.data in symbols]
        if len(calls)!=len(symbols) or set(calls)!=symbols:raise ValueError('prepared source does not contain exact fused call set')
        direct=Path(work)/'direct_conv';manifest=build_direct(Path(source),llvm_bin,direct,flat_spatial=flat_spatial,allow_empty=True)
        rewritten=direct/'rewritten.mlir'
        if sha(rewritten)!=manifest['rewritten_sha256']:raise ValueError('direct rewrite identity mismatch')
        reparsed=parse_module(rewritten.read_text())
        for declaration in reparsed.walk():
            if declaration.name=='func.func' and declaration.sym_name.data in symbols:
                attrs=declaration.properties.get('arg_attrs')
                access=[] if attrs is None else [x.data.get('bufferization.access') for x in attrs]
                route=next(x for x in requant['routes'] if x['symbol']==declaration.sym_name.data)
                readout=route.get('integer_readout')
                expected_access=['read','read','write','write'] if readout else ['read','read','write']
                if [getattr(x,'data',None) for x in access]!=expected_access:
                    raise ValueError('fused declaration lost its bufferization access contract')
                route=next(x for x in requant['routes'] if x['symbol']==declaration.sym_name.data)
                if readout:
                    attr=declaration.attributes.get('gemmini.integer_readout')
                    expected_sha=hashlib.sha256(json.dumps(readout,sort_keys=True).encode()).hexdigest()
                    schedule=route['schedule'];count=(schedule['h']-1)//schedule['stride']+1 if route['direct_conv'] else 0
                    elements=count*((schedule['w']-1)//schedule['stride']+1)*schedule['cout'] if route['direct_conv'] else schedule['m']*schedule['n']
                    paired=route.get('paired_readout',{}).get('applied',False)
                    if attr is None or attr.data['proof_sha256'].data!=expected_sha or attr.data['source_sha256'].data!=requant['source_sha256'] or attr.data['scratch_bytes'].value.data!=elements*(1 if paired else 4) or attr.data['scratch_ownership'].data!='caller_owned_unique':
                        raise ValueError('integer readout source/proof/scratch binding changed')
                    inputs=declaration.function_type.inputs.data
                    if len(inputs)!=4 or str(inputs[2].get_element_type())!=('i8' if paired else 'i32') or inputs[2].get_shape()!=inputs[3].get_shape():
                        raise ValueError('integer readout scratch geometry/type changed')
                if 'numeric_contract' in route:
                    expected=route['numeric_contract']
                    contract=declaration.attributes.get('merlin.numeric_contract')
                    if contract is None:
                        raise ValueError('fused declaration lost its numerical contract')
                    data=contract.data
                    if (getattr(data.get('unit'),'data',None)!='quantized_output_lsb'
                            or getattr(data.get('source_region'),'data',None)!=expected['source_region']
                            or data['max_abs_error'].value.data!=expected['max_output_lsb_error']
                            or data['selected_policy_limit'].value.data!=expected['selected_policy_limit']):
                        raise ValueError('fused declaration numerical contract differs from proof')
        from merlin.llvmlower.im2col_identity_view import rewrite_prepared_file
        identity=Path(work)/'identity_views';identity.mkdir(parents=True,exist_ok=True)
        prepared,report=rewrite_prepared_file(rewritten,identity)
        prepared=Path(prepared)
        (identity/'identity_view_report.json').write_text(json.dumps(report.to_dict(),indent=2)+'\n')
        if propagate_layout:
            from .physical_layout import rewrite_file
            prepared,layout_report=rewrite_file(prepared,Path(work)/'physical_layout',reduction_channel_block=reduction_channel_block)
            state['layout_report']=layout_report
        state.update(direct=direct,manifest=manifest,prepared_sha256=sha(prepared))
        return prepared
    def build(source,work):
        check_requant();source,work=Path(source),Path(work)
        if not state or sha(source)!=state['prepared_sha256']:raise ValueError('mixed catalog source identity mismatch')
        direct=state['direct']/'direct_conv.o'
        if sha(direct)!=state['manifest']['object_sha256']:raise ValueError('direct object identity mismatch')
        work.mkdir(parents=True,exist_ok=True)
        # Merlin offload rewrites its input in place. Preserve the exact bound bytes.
        shutil.copyfile(source,work/'catalog_source.mlir')
        manifest=compile_catalog(source,llvm_bin,work,large_n=large_n,contraction_calibrations=contraction_calibrations);linker=llvm_bin/'ld.lld'
        if not linker.is_file():
            found=shutil.which('ld.lld')
            if found is None:raise ValueError('ld.lld required')
            linker=Path(found)
        merged=work/'fused_mixed_kernel.o';command=[str(linker),'-r',str(work/'kernel.o'),str(direct),str(requant_bundle/'requant.o'),'-o',str(merged)]
        subprocess.run(command,check=True,capture_output=True);audit=audit_elf(merged.read_bytes())
        if audit['status']!='pass':raise ValueError('fused mixed object no-FSM audit failed')
        manifest['compilation']={'schema':'gemmini_fused_mixed_compile_v1','object_sha256':sha(merged),'object_nofsm_status':'pass','dense_compilation':manifest['compilation'],'direct_manifest_sha256':sha(state['direct']/'direct_conv.json'),'requant_manifest_sha256':requant_sha,'weights_sha256':requant['weights_sha256'],'linker_argv':command,'linker_sha256':sha(linker)}
        manifest['direct_convolutions']=state['manifest']['routes'];manifest['fused_requantizations']=requant['routes'];manifest['total_device_contractions']=manifest['covered_contractions']+len(manifest['direct_convolutions'])+len(requant['routes'])
        if propagate_layout:manifest['physical_layout']=state['layout_report']
        manifest['native_oracle_sources']=[str(state['direct']/'native_oracle.c'),str(requant_bundle/'native_oracle.c')]
        path=work/'device_catalog.json';path.write_text(json.dumps(manifest,indent=2)+'\n');(work/'fused_mixed_object_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        return path,merged
    return prepare,build
