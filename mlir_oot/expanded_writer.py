"""Provider-owned bridge for explicitly contracted expanded-memref writers.

The caller supplies source-bound result identity and complete-write facts. This
module adapts the existing expanded C shim; it does not infer kernel semantics
from matching tensor shapes or automatically select catalog entries.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedWriterContract:
    symbol: str
    borrowed_symbol: str
    result_argument: int
    fully_written_arguments: tuple[int, ...]
    allocation_alignment: int


def transform(module, contracts):
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers

    contracts = tuple(contracts)
    for contract in contracts:
        for name in (contract.symbol, contract.borrowed_symbol):
            if not name or not (name[0].isascii() and (name[0].isalpha() or name[0] == '_')) or any(
                not (ch.isascii() and (ch.isalnum() or ch == '_')) for ch in name
            ):
                raise ValueError('C identifier required for expanded bridge')
    return rewrite_fresh_tensor_writers(module, [
        FreshTensorWriterContract(c.symbol, c.result_argument, c.fully_written_arguments,
                                  c.borrowed_symbol, c.allocation_alignment,
                                  declaration_abi='expanded_memref',
                                  wrapper_symbol=c.symbol + '__fresh_tensor_result',
                                  allow_initialized_writers=True) for c in contracts
    ])


def shim(routes):
    """Match the provider's intptr_t expanded C ABI, discarding its result."""
    ranks = sorted({rank for r in routes for rank in [*r['argument_ranks'], r['result_rank']]})
    if any(type(rank) is not int or rank <= 0 for rank in ranks):
        raise ValueError('positive ranked expanded descriptors required')
    code = '#include <stdint.h>\n'
    for rank in ranks:
        code += (f'typedef struct {{ void *allocated,*aligned; intptr_t offset; '
                 f'intptr_t sizes[{rank}],strides[{rank}]; }} writer_memref_{rank};\n')
    for route in routes:
        types = []
        values = []
        args = []
        for i, rank in enumerate(route['argument_ranks']):
            args.append(f'writer_memref_{rank}*a{i}')
            types.extend(['void*', 'void*', 'intptr_t'] + ['intptr_t'] * (2 * rank))
            values.extend([f'a{i}->allocated', f'a{i}->aligned', f'a{i}->offset'])
            values.extend(f'a{i}->sizes[{j}]' for j in range(rank))
            values.extend(f'a{i}->strides[{j}]' for j in range(rank))
        code += (f"extern writer_memref_{route['result_rank']} {route['symbol']}({','.join(types)});\n"
                 f"void _mlir_ciface_{route['borrowed_symbol']}({','.join(args)}) {{\n"
                 f" (void){route['symbol']}({','.join(values)});\n}}\n")
    return code


def merlin_callbacks(llvm_bin, base_host_transform, *, allocation_alignment, target_cflags):
    """Return explicit post-offload and LLVM-link callbacks for qualified catalogs."""
    from pathlib import Path
    import hashlib
    import json
    import subprocess
    from merlin.llvmlower.device_shim import emit_dense_translation_unit
    from merlin.xdsl_dialects._common import text
    from .frontend.parse import parse_module

    compiler = Path(llvm_bin)
    target_cflags = tuple(target_cflags)
    state = {}
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def prepare(prepared, sidecar, workdir):
        prepared, sidecar, workdir = Path(prepared), Path(sidecar), Path(workdir)
        routed = json.loads(sidecar.read_text())
        manifest = Path(routed['catalog_manifest'])
        catalog = json.loads(manifest.read_text())
        effects = catalog.get('abi', {}).get('writer_effects')
        expected = dict(schema='complete_output_writer_v1', fully_written_arguments=[2],
                        retains_arguments=False, frees_arguments=False)
        if effects != expected or catalog['source_sha256'] != routed['model_sha256']:
            raise ValueError('source-bound explicit complete-write catalog required')
        if digest(routed['catalog_object']) != catalog['compilation']['object_sha256']:
            raise ValueError('catalog object identity changed')
        bindings = {b['region']: b for b in catalog['bindings']}
        kernels, dtypes = {}, {}
        for row in routed['routed']:
            binding = bindings.get(row['source_region'])
            if binding is None or binding['tensor_types'] != row['tensor_types']:
                raise ValueError('routed tensor identity differs from catalog')
            symbol = row['symbol']
            if symbol in kernels and kernels[symbol] != binding['symbol']:
                raise ValueError('ambiguous source-bound kernel')
            kernels[symbol] = binding['symbol']
            dtypes[symbol] = tuple(row['dtypes'])
        if set(kernels) != set(routed['signatures']) or len(routed['routed']) != len(bindings):
            raise ValueError('source operation coverage differs')
        unit = emit_dense_translation_unit(routed['device'], routed['signatures'], dtypes,
                kernel_symbol_for=kernels.__getitem__, kernel_fully_written_arguments=(2,))
        if unit.skipped or len(unit.writer_contracts) != len(kernels):
            raise ValueError('adapter owner declined explicit writer contracts')
        contracts = [ExpandedWriterContract(c['symbol'],c['symbol']+'__borrowed_write',
                     c['result_argument'],tuple(c['fully_written_arguments']),allocation_alignment)
                     for c in unit.writer_contracts]
        module = parse_module(prepared.read_text())
        routes = transform(module,contracts)
        from merlin.llvmlower.passes_xdsl import add_c_interface
        add_c_interface(module)
        output = workdir/'model.mlir';output.write_text(text(module,generic=True))
        bridge = workdir/'borrowed.c';bridge.write_text(shim(routes))
        adapter = workdir/'expected_adapter.c';adapter.write_text(unit.text)
        (workdir/'writer_contracts.json').write_text(json.dumps({
            'source_sha256': catalog['source_sha256'], 'catalog_manifest_sha256':digest(manifest),
            'routing_sidecar_sha256':digest(sidecar),'adapter_source_sha256':digest(adapter),
            'borrowed_source_sha256':digest(bridge), 'contracts':unit.writer_contracts,
            'routes':routes, 'fully_written_kernel_arguments':effects,
        },indent=2)+'\n')
        state['bridge']=bridge
        return output

    def host_transform(source, directory):
        directory = Path(directory)
        selected = Path(base_host_transform(source,directory))
        for native in (False,True):
            original=directory/'model.native.ll' if native else selected
            bridge_ir=directory/('expanded_bridge.native.ll' if native else 'expanded_bridge.ll')
            command=[str(compiler/'clang'),'-O2','-S','-emit-llvm',str(state['bridge']),'-o',str(bridge_ir)]
            if not native:
                command[1:1]=list(target_cflags)
            subprocess.run(command,check=True,capture_output=True)
            linked=directory/('expanded.native.ll' if native else 'expanded.ll')
            subprocess.run([str(compiler/'llvm-link'),'-S',str(original),str(bridge_ir),'-o',str(linked)],check=True,capture_output=True)
            if native:(directory/'model.native.ll').write_bytes(linked.read_bytes())
        return directory/'expanded.ll'

    return prepare,host_transform
