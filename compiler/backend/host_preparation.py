"""Exact tensor layout preparation using the selected stock MLIR pass.

Only unit extents are removed. Scalar regions, callback implementations and
invocation ownership remain with their existing compiler/runtime owners.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from io import StringIO


def prepare(source: str, work: Path, selection: str) -> tuple[str, dict | None]:
    if selection == 'baseline':
        return source, None
    if selection != 'unit_dimensions':
        raise ValueError('unknown exact host preparation: ' + selection)
    tool = Path(os.environ['MERLIN_MLIR_INSTALL']) / 'bin' / 'mlir-opt'
    work.mkdir(parents=True, exist_ok=True)
    original = work / 'host-preparation-source.mlir'
    transport = work / 'host-preparation-input.mlir'
    output = work / 'host-preparation-output.mlir'
    original.write_text(source)
    # Public source custom syntax can carry attributes before linalg.yield's
    # operands, accepted by its frontend but not upstream MLIR. Re-emit through
    # that same frontend in upstream generic syntax; preserve every attribute.
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from xdsl.printer import Printer
    module = parse_mlir_text(source)
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    transport.write_text(stream.getvalue())
    # This upstream layout pass folds only statically unit extents. It neither
    # replaces scalar arithmetic nor changes a reduction's non-unit ordering.
    # Generic printing is required by the downstream xDSL parser when source
    # provenance attributes are attached to linalg.yield operations.
    pipeline = 'builtin.module(func.func(linalg-fold-unit-extent-dims),canonicalize,cse)'
    argv = [str(tool), '--allow-unregistered-dialect', '--mlir-print-op-generic',
            '--pass-pipeline=' + pipeline, str(transport), '-o', str(output)]
    run = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise ValueError('exact host layout preparation failed: ' + run.stderr[-6000:])
    selected = output.read_text()
    parse_mlir_text(selected).verify()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    with tool.open('rb') as binary:
        tool_digest = hashlib.file_digest(binary, 'sha256').hexdigest()
    record = dict(schema='submission.exact_host_preparation.v1', selection=selection,
        binding_stage='caller_before_shared_host_lowering', pipeline=pipeline,
        command=argv, tool=dict(path=str(tool.resolve()), sha256=tool_digest),
        source=dict(path=str(original), sha256=digest(source.encode())),
        transport_input=dict(path=str(transport), sha256=digest(transport.read_bytes())),
        output=dict(path=str(output), sha256=digest(selected.encode())),
        numerical_contract='unchanged_scalar_regions_and_nonunit_iteration_order',
        invocation_scope='unchanged', hardware_cycles='not_measured')
    (work / 'host-preparation.json').write_text(json.dumps(record, indent=2))
    return selected, record
