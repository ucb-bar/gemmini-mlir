"""Bind normal golden model callbacks to Merlin's complete structural emission.

Target catalog ownership lives here. Outlining, exact cover, tensor boundaries,
IR preservation and package edit contracts use existing Merlin implementations.
An identity cover preserves the existing compiler decisions; it is not a solver
or a timing improvement. Final binary symbol closure is separate from arithmetic
qualification and downstream source-operation-to-machine-instruction accounting.
"""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from .contraction_patterns import match_integer_gemm
from .frontend.parse import parse_module
from .golden_compiler_export import export_inventory
from .no_fsm_audit import audit_elf


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row_sha(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _symbol_table(path):
    # Use Merlin's selected inspector, without adding another tool discovery rule.
    from merlin.llvmlower.device_build import _nm
    tool=_nm()
    if tool is None:raise ValueError('whole-model plan needs the selected symbol inspector')
    run=subprocess.run([str(tool),'--format=posix','--extern-only',str(path)],
                       check=True,capture_output=True,text=True,timeout=60)
    defined=Counter();unresolved=[]
    for line in run.stdout.splitlines():
        fields=line.split()
        if len(fields)<2:raise ValueError('unreadable compiler object symbol record')
        if fields[1].upper()=='U':unresolved.append(fields[0])
        elif fields[1].upper()=='T':defined[fields[0]]+=1
    return defined,unresolved


def _external_routes(catalog):
    routes={}
    for field in ('fused_requantizations','direct_convolutions','residual_additions','guarded_mean_additions'):
        for row in catalog.get(field,[]):
            symbol=row.get('symbol')
            if not symbol or symbol in routes:raise ValueError('ambiguous external catalog route')
            routes[symbol]=dict(family=field,route_sha256=_row_sha(row),
                                kernel=row.get('kernel'),region=row.get('region',row.get('source_region')))
    if catalog.get('pooled_stem'):
        row=catalog['pooled_stem'];symbol=row.get('symbol')
        if not symbol or symbol in routes:raise ValueError('ambiguous pooled catalog route')
        routes[symbol]=dict(family='pooled_stem',route_sha256=_row_sha(row),kernel=row.get('kernel'))
    return routes


class GoldenModelPlanBinding:
    """Target bridge around the ordinary prepared/catalog/final-image hooks."""

    def __init__(self,routing,*,package=None,original_source=None):
        if routing.catalog_builder is None:
            raise ValueError('whole-model plan binding requires a source-bound catalog builder')
        self.routing=routing
        self.package=Path(package) if package else Path(__file__).resolve().parents[1]
        self.original_source=Path(original_source) if original_source else None
        self.state=None

    def _write(self):
        if self.state is not None:
            (self.work/'compiler_plan.json').write_text(json.dumps(self.state,indent=2)+'\n')

    def prepare(self,source,work):
        from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program
        from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
        from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest,emit_global_plan
        from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
            OutlinedGlobalPlanEmitter,plan_dispatch_fusion,
        )
        if self.state is not None:raise ValueError('whole-model plan callbacks cannot be reused across builds')
        source,work=Path(source),Path(work)
        self.work=work/'global_plan';self.work.mkdir(parents=True,exist_ok=False)
        original_sha=None if self.original_source is None else _sha(self.original_source)
        before=_sha(source)
        prepared=Path(self.routing.prepared_transform(source,work)) if self.routing.prepared_transform else source
        payload=prepared.read_bytes();source_sha=hashlib.sha256(payload).hexdigest()
        snapshot=self.work/'prepared_source.mlir';snapshot.write_bytes(payload)
        module=parse_module(payload.decode())
        declarations={op.sym_name.data:op for op in module.body.block.ops
                      if op.name=='func.func' and not op.body.blocks}
        permitted=tuple(sorted(declarations))
        outlined,graph=lower_model_to_dispatch_program(module,prune=False,external_symbols=permitted)
        plan=plan_dispatch_fusion(graph,(),placement='existing_compiler_decision',
            representation=lambda name:ValueRepresentation('tensor_ssa','logical',graph.buffers[name].dtype))
        emitter=OutlinedGlobalPlanEmitter(outlined)
        emission=emit_global_plan(graph,plan,emitter)
        # Consume the checked identity emission at the actual compilation boundary,
        # preserving the original bytes consumed by the established compiler path.
        if not outlined.module.is_structurally_equivalent(emitter.module):
            raise ValueError('identity plan unexpectedly changed outlined source')
        if _sha(prepared)!=source_sha:
            raise ValueError('prepared source changed while its full graph was verified')
        if self.original_source is not None and _sha(self.original_source)!=original_sha:
            raise ValueError('original source changed during prepared plan binding')
        called=Counter(op.callee.string_value() for op in module.walk()
                       if op.name=='func.call' and op.callee.string_value() in declarations)
        self.state=dict(schema='golden_compiler_model_plan_v1',status='prepared_verified_catalog_pending',
            source_before_preparation_sha256=before,source_sha256=source_sha,
            original_source_sha256=original_sha,source_snapshot=str(snapshot.resolve()),
            original_rewrite_chain_equivalence='UNPROVEN by this structural adapter; retain existing source/numeric and whole-model qualification',
            logical_dispatch_digest=dispatch_digest(graph),logical_dispatch=graph.to_dict(),
            global_plan=plan.to_dict(),global_plan_emission=emission.receipt(),
            outlined_preservation_proof=emitter.proof,
            prepared_source_to_outlined='UNKNOWN: outlining may clone initializers; exact original prepared bytes are compiled unchanged',
            declared_external_abi={name:str(op.function_type) for name,op in declarations.items()},
            declared_external_entrypoints={name:('_mlir_ciface_'+name
                if 'llvm.emit_c_interface' in op.attributes else name) for name,op in declarations.items()},
            called_external_symbols=dict(called),prepared_ir_operation_cover_complete=True,
            selected_plan_controls_emission=True,selection_kind='identity preservation of existing compiler decisions',
            selection_controls_target_schedule=False,
            compiled_source_policy='preserve exact original prepared bytes after checked identity-emission gate',
            shared_solver_selected=False,cycles_status='UNKNOWN',occupancy_status='UNKNOWN',
            task_to_machine_instruction_accounting='UNKNOWN: ordinary host lowering has no complete source-task CFG receipt here',
            whole_model_correctness_verified=False,full_model_hardware_measured=False,
            package_export=export_inventory(self.package))
        self._write()
        self.module=module;self.prepared=prepared
        return prepared

    def compile(self,source,work):
        if self.state is None or _sha(source)!=self.state['source_sha256']:
            raise ValueError('catalog compilation does not match the verified prepared graph')
        manifest_path,obj=self.routing.catalog_builder(Path(source),Path(work))
        manifest_path,obj=Path(manifest_path),Path(obj)
        catalog=json.loads(manifest_path.read_text())
        if catalog.get('source_sha256')!=self.state['source_sha256']:
            raise ValueError('actual catalog source differs from the verified whole graph')
        if catalog.get('compilation',{}).get('object_sha256')!=_sha(obj):
            raise ValueError('actual catalog object differs from its compiler receipt')
        routes=_external_routes(catalog)
        called=self.state['called_external_symbols']
        if set(called)!=set(routes):
            raise ValueError(f'whole-model external catalog coverage differs: missing={sorted(set(called)-set(routes))}, unused={sorted(set(routes)-set(called))}')
        defined,unresolved=_symbol_table(obj)
        for name,row in routes.items():
            row['binary_entrypoint']=self.state['declared_external_entrypoints'][name]
        required={row['binary_entrypoint'] for row in routes.values()}|{
            row['symbol'] for row in catalog.get('kernels',[])}
        required.update(row['kernel'] for row in routes.values() if row.get('kernel'))
        if any(defined[symbol]!=1 for symbol in required) or unresolved:
            raise ValueError('compiled catalog lacks unique declared definitions or has unresolved references')
        # Use the target's existing exact matcher, not a region label, to close
        # dense source operations against the actual compiler's selected records.
        actual=[]
        for ordinal,op in enumerate(self.module.walk()):
            dims=match_integer_gemm(op)
            if dims is None:continue
            region=getattr(op.attributes.get('prov.region_id'),'data',None)
            actual.append(dict(source_operation_ordinal=ordinal,region=region,
                tensor_types=[str(v.type) for v in op.operands[:2]]+[str(op.results[0].type)],
                dimensions=dict(batch=dims.batch,m=dims.m,n=dims.n,k=dims.k)))
        bindings=catalog.get('bindings',[])
        if (not catalog.get('coverage_complete') or catalog.get('matched_contractions')!=len(actual)
                or catalog.get('covered_contractions')!=len(actual) or len(bindings)!=len(actual)):
            raise ValueError('actual dense catalog does not cover all matched source contractions')
        kernels={row['symbol']:row for row in catalog.get('kernels',[])}
        for source_row,binding in zip(actual,bindings,strict=True):
            if (any(binding.get(key)!=source_row[key] for key in ('source_operation_ordinal','region','tensor_types'))
                    or kernels.get(binding.get('symbol'),{}).get('dimensions')!=source_row['dimensions']):
                raise ValueError('dense implementation binding disagrees with the exact whole-model source')
            source_row['kernel_symbol']=binding['symbol']
        audit=audit_elf(obj.read_bytes())
        if audit['status']!='pass':raise ValueError('whole-model catalog contains forbidden target commands')
        if _sha(self.state['source_snapshot'])!=self.state['source_sha256']:
            raise ValueError('whole-model source snapshot changed after plan emission')
        self.state.update(status='prepared_and_catalog_verified_final_image_pending',
            catalog=dict(path=str(manifest_path.resolve()),manifest_sha256=_sha(manifest_path),
                         object_path=str(obj.resolve()),object_sha256=_sha(obj),
                         external_routes=routes,dense_bindings=actual,nofsm_audit=audit,
                         binary_symbol_closure_complete=True),
            catalog_numeric_equivalence='Existing independent source/numeric and original-model gates remain required')
        self.catalog=manifest_path;self.object=obj;self.required_symbols=required
        self._write()
        return manifest_path,obj

    def audit_final(self,elf):
        elf=Path(elf)
        if self.state is None or 'catalog' not in self.state:
            raise ValueError('final whole-model image has no bound catalog compilation')
        if self.routing.final_elf_audit is not None:self.routing.final_elf_audit(elf)
        if (_sha(self.catalog)!=self.state['catalog']['manifest_sha256']
                or _sha(self.object)!=self.state['catalog']['object_sha256']):
            raise ValueError('catalog artifacts changed before final image binding')
        if (_sha(self.state['source_snapshot'])!=self.state['source_sha256'] or
                (self.original_source is not None and
                 _sha(self.original_source)!=self.state['original_source_sha256'])):
            raise ValueError('whole-model source identity changed before final image binding')
        defined,unresolved=_symbol_table(elf)
        if any(defined[symbol]!=1 for symbol in self.required_symbols) or unresolved:
            raise ValueError('final ELF does not close its actual catalog symbols')
        audit=audit_elf(elf.read_bytes())
        if audit['status']!='pass':raise ValueError('final whole-model image contains forbidden target commands')
        binding_path=elf.parent/'device/catalog_binding.json'
        if not binding_path.is_file():raise ValueError('final image lacks the ordinary Merlin catalog ABI binding')
        binding=json.loads(binding_path.read_text())
        if (binding.get('source_sha256')!=self.state['source_sha256']
                or binding.get('manifest_sha256')!=self.state['catalog']['manifest_sha256']
                or binding.get('catalog_object_sha256')!=self.state['catalog']['object_sha256']):
            raise ValueError('ordinary catalog ABI binding differs from whole-model plan artifacts')
        shim=elf.parent/'device/device_catalog_shim.o'
        if _sha(shim)!=binding.get('shim_object_sha256'):
            raise ValueError('ordinary catalog ABI shim changed before final image binding')
        self.state.update(status='prepared_catalog_and_final_image_bound',
            final_image=dict(path=str(elf.resolve()),sha256=_sha(elf),nofsm_audit=audit,
                ordinary_catalog_binding=binding,ordinary_catalog_binding_sha256=_sha(binding_path),
                host_and_link_objects={p.name:_sha(p) for p in sorted(elf.parent.glob('*.o'))},
                binary_symbol_closure_complete=True))
        self._write()


def bind_device_routing(routing,*,package=None,original_source=None):
    """Enable full structural/accounting checks on the real model compiler hooks."""
    binding=GoldenModelPlanBinding(routing,package=package,original_source=original_source)
    result=replace(routing,prepared_transform=binding.prepare,catalog_builder=binding.compile,
                   final_elf_audit=binding.audit_final)
    return result,binding
