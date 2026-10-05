"""Compose proven direct convolutions with the remaining dense GEMM catalog."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .direct_conv_bundle import build as build_direct
from .golden_device_catalog import compile_catalog, merlin_identity_view_transform
from .no_fsm_audit import audit_elf


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merlin_callbacks(llvm_bin: Path, *, host_roundeven: bool = False):
    """Return preparation/catalog callbacks sharing an exact, checked bundle.

    Instantiate per model build. The direct rewrite owns its source-bound
    manifest; the ordinary catalog binds the final prepared source and supplies
    only its remaining contraction adapters. No shape-only replacement occurs.
    """
    llvm_bin = Path(llvm_bin)
    bundle = {}

    def prepare(source: Path, work: Path) -> Path:
        direct = Path(work) / 'direct_conv'
        manifest = build_direct(Path(source), llvm_bin, direct)
        rewritten = direct / 'rewritten.mlir'
        if _sha(rewritten) != manifest['rewritten_sha256']:
            raise ValueError('direct convolution rewrite changed after compilation')
        prepared = merlin_identity_view_transform(rewritten, Path(work) / 'identity_views')
        if host_roundeven:
            from .frontend.parse import parse_module
            from .golden_host_math import rewrite_roundeven
            from merlin.xdsl_dialects._common import text

            module = parse_module(prepared.read_text())
            count = rewrite_roundeven(module)
            prepared = Path(work) / 'host_math.mlir'
            prepared.write_text(text(module))
            bundle['host_roundeven_calls'] = count
        bundle.update(directory=direct, manifest=manifest, prepared_sha256=_sha(prepared))
        return prepared

    def build(source: Path, work: Path) -> tuple[Path, Path]:
        source, work = Path(source), Path(work)
        if not bundle or _sha(source) != bundle['prepared_sha256']:
            raise ValueError('mixed catalog requires its exact prepared convolution bundle')
        direct = bundle['directory'] / 'direct_conv.o'
        if _sha(direct) != bundle['manifest']['object_sha256']:
            raise ValueError('direct convolution object changed after preparation')
        manifest = compile_catalog(source, llvm_bin, work)
        linker = llvm_bin / 'ld.lld'
        if not linker.is_file():
            found = shutil.which('ld.lld')
            if found is None:
                raise ValueError('mixed catalog requires ld.lld')
            linker = Path(found)
        merged = work / 'mixed_kernel.o'
        objects = [work / 'kernel.o', direct]
        host_receipt = None
        if bundle.get('host_roundeven_calls'):
            assembly = Path(__file__).resolve().parent.parent / 'runtime/roundevenf_rv64gc.S'
            host_object = work / 'host_roundeven.o'
            host_command = [str(llvm_bin / 'clang'), '--target=riscv64-unknown-elf',
                            '-march=rv64gc', '-mabi=lp64d', '-c', str(assembly), '-o', str(host_object)]
            subprocess.run(host_command, check=True, capture_output=True, text=True)
            objects.append(host_object)
            host_receipt = {'source_sha256': _sha(assembly), 'object_sha256': _sha(host_object),
                            'calls': bundle['host_roundeven_calls'], 'compiler_argv': host_command}
        command = [str(linker), '-r', *map(str, objects), '-o', str(merged)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        audit = audit_elf(merged.read_bytes())
        if audit['status'] != 'pass':
            raise ValueError('mixed catalog failed primitive-instruction audit')
        dense_receipt = manifest['compilation']
        manifest['compilation'] = {
            'schema': 'gemmini_golden_mixed_compile_v1',
            'object_sha256': _sha(merged), 'object_nofsm_status': audit['status'],
            'dense_compilation': dense_receipt,
            'direct_manifest_sha256': _sha(bundle['directory'] / 'direct_conv.json'),
            'direct_object_sha256': _sha(direct),
            'linker_sha256': _sha(linker), 'linker_argv': command,
            'host_roundeven': host_receipt,
        }
        manifest['direct_convolutions'] = bundle['manifest']['routes']
        manifest['total_device_contractions'] = len(manifest['direct_convolutions']) + manifest['covered_contractions']
        path = work / 'device_catalog.json'
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        (work / 'mixed_object_nofsm_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        return path, merged

    return prepare, build
