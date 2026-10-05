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


def merlin_callbacks(llvm_bin: Path):
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
        command = [str(linker), '-r', str(work / 'kernel.o'), str(direct), '-o', str(merged)]
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
        }
        manifest['direct_convolutions'] = bundle['manifest']['routes']
        manifest['total_device_contractions'] = len(manifest['direct_convolutions']) + manifest['covered_contractions']
        path = work / 'device_catalog.json'
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        (work / 'mixed_object_nofsm_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        return path, merged

    return prepare, build
