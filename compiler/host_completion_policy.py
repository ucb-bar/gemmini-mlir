"""Stock binding adapter for Merlin's static memref-copy expansion recipe.

The selected library recipe lowers static ranked copies through linalg's ordinary
load/store loops. No source arithmetic, synchronization, or runtime is authored
here. Dynamic/unranked copies retain their incumbent implementation and count.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from xdsl.dialects.builtin import MemRefType
from xdsl.dialects.linalg.ops import CopyOp
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import copy_expand
from host_numeric_policy import run_stock_pipeline as _incumbent


def selection():
    from merlin.compile.tuning_knobs import knob
    return knob('host_completion_optimization', default='off',
                values=('off', 'static_copy'), root=Path(__file__).resolve().parent)


def _static(typ):
    return (isinstance(typ, MemRefType) and len(typ.get_shape()) >= 1
            and all(extent >= 0 for extent in typ.get_shape()))


def expand_static_copies(module):
    """Supply copy_expand._expand_memref_copies's native binding operation API.

    Same admission: two ranked, nonzero-rank, static and identically shaped
    operands. This is copy transport, with exactly the original copy's alias
    requirements; arithmetic and floating-point operations are untouched.
    """
    expanded, skipped = [], []
    for operation in tuple(module.walk()):
        if operation.name != 'memref.copy':
            continue
        source, destination = operation.operands
        if (not _static(source.type) or not _static(destination.type)
                or source.type.get_shape() != destination.type.get_shape()):
            skipped.append({'source_type': str(source.type), 'destination_type': str(destination.type)})
            continue
        replacement = CopyOp([source], [destination], res=(), attributes=dict(operation.attributes))
        block = operation.parent_block()
        block.insert_op_before(replacement, operation)
        block.erase_op(operation)
        expanded.append({'source_type': str(source.type), 'destination_type': str(destination.type)})
    module.verify()
    return {'expanded': expanded, 'skipped': skipped}


def run_stock_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    """Drop-in stock pipeline; default off preserves its complete incumbent."""
    selected = selection()
    if selected == 'off':
        return _incumbent(source, lowered, passes, tool=tool, invoke=invoke,
                          print_module=print_module, report=report)
    locations = [i for i, name in enumerate(passes) if 'convert-linalg-to-loops' in name]
    if len(locations) != 1:
        raise ValueError('static copy expansion requires one actual linalg loop conversion boundary')
    split = locations[0]
    source, lowered = Path(source), Path(lowered)
    original = source.read_bytes()
    buffered = source.parent / 'completion-buffered.mlir'
    prior_report = {}
    _incumbent(source, buffered, passes[:split], tool=tool, invoke=invoke,
               print_module=print_module, report=prior_report)
    module = parse_mlir_text(buffered.read_text())
    observation = expand_static_copies(module)
    chosen = source.parent / 'completion-selected.mlir'
    chosen.write_text(print_module(module))
    invoke([tool, chosen, '--pass-pipeline=builtin.module(' + ','.join(passes[split:]) + ')',
            '--mlir-print-op-generic', '-o', lowered], stage='static copy loop lowering')
    if source.read_bytes() != original:
        raise ValueError('copy expansion modified original source')
    if report is not None:
        report.update(prior_report)
        report['completion_optimization'] = {
            'schema': 'merlin.stock_static_copy_expansion.v1', 'selection': selected,
            'mechanism': 'merlin.llvmlower.copy_expand.' + copy_expand.FEATURE,
            'library_sha256': hashlib.sha256(Path(copy_expand.__file__).read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(original).hexdigest(),
            'buffered_sha256': hashlib.sha256(buffered.read_bytes()).hexdigest(),
            'selected_sha256': hashlib.sha256(chosen.read_bytes()).hexdigest(),
            'lowered_sha256': hashlib.sha256(lowered.read_bytes()).hexdigest(),
            **observation, 'runtime_cycles': 'not_measured',
        }
