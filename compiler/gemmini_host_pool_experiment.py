#!/usr/bin/env python3
"""Isolated experiment: expand bounded trailing source reduction loops in order.

Uses the existing host generic/scalar lowering and FnBuilder.for_range. This
does not select arithmetic recipes, alter placement or supply runtime permissions.
The default compiler never imports this development adapter.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'compiler_library')]


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


from gemmini_host_reductions import describe_ordered_generic as describe


def compile_ordered_source(source, *, evidence_sink=None):
    """Reuse the original graph compiler with a private host class binding.

    Function globals are copied, never mutated. Interface/device-only sources
    retain the original compile_source path and command-buffer schema exactly.
    """
    from types import FunctionType
    from gemmini_frontend import normalize_source
    from gemmini_backend import compile_source
    from gemmini_graph import GraphHost, compile_linalg
    from gemmini_host_reductions import lower_ordered_generic
    if normalize_source(source)['grammar'] != 'linalg':
        return compile_source(source, evidence_sink=evidence_sink)
    receipts = []

    class OrderedHost(GraphHost):
        def _t_linalg_generic(self, op):
            before = len(getattr(self, 'host_ordered_reduction_reports', []))
            result = lower_ordered_generic(self, op,
                ordinary_lowerer=super()._t_linalg_generic)
            receipts.extend(getattr(self, 'host_ordered_reduction_reports', [])[before:])
            return result

    private_globals = dict(compile_linalg.__globals__, GraphHost=OrderedHost)
    lower = FunctionType(compile_linalg.__code__, private_globals,
                         compile_linalg.__name__, compile_linalg.__defaults__,
                         compile_linalg.__closure__)
    cb, llvm = lower(source)
    if receipts:
        cb.setdefault('params', {})['host_pool_ordered_expansion'] = dict(
            source_sha256=sha(source.encode()), llvm_sha256=sha((llvm+'\n').encode()),
            realized_recipes=receipts,
            contract='original generic body/order/rounding/each-step store retained')
    if evidence_sink is not None:
        evidence_sink([], cb, llvm)
    return cb, llvm


def check(source, output):
    """Compare actual connected products and exact source pool-cone outputs."""
    import ctypes
    import numpy as np
    from xdsl.dialects import builtin, func
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from gemmini_backend import compile_source
    from gemmini_graph import compile_linalg
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = source.read_bytes()
    module = parse_mlir_text(payload.decode())
    block = next(o for o in module.walk() if o.name == 'func.func').body.block
    # Find by structural scalar reduction, never a source name or region ID.
    candidates = [o for o in block.ops if describe(o) is not None]
    pools = [o for o in candidates if [x.name for x in o.body.block.ops] ==
             ['arith.cmpf', 'arith.cmpf', 'arith.ori', 'arith.select', 'linalg.yield']]
    if len(pools) != 1:
        raise ValueError('expected one structurally admitted compare/select pool cone')
    pool = pools[0]
    # Clone its complete padding/init cone, exposing the original pad input.
    pad = pool.inputs[0].owner
    if pad.name != 'tensor.insert_slice':
        raise ValueError('pool input is not a source padding insertion')
    operations = list(block.ops)
    start = operations.index(pad.operands[1].owner.operands[0].owner)
    stop = operations.index(pool) + 1
    argument = pad.operands[0]
    fixture_block = Block(arg_types=[argument.type])
    mapping = {argument: fixture_block.args[0]}
    for op in operations[start:stop]:
        fixture_block.add_op(op.clone(value_mapper=mapping))
    result = mapping[pool.results[0]]
    fixture_block.add_op(func.ReturnOp(result))
    fixture = builtin.ModuleOp([func.FuncOp('forward', ([argument.type], [result.type]), Region(fixture_block))])
    fixture.verify()
    stream = io.StringIO(); Printer(stream=stream).print_op(fixture)
    fixture_text = stream.getvalue()
    (output / 'source-pool.mlir').write_text(fixture_text)
    tools = PACKAGE.parent / 'install/bin'
    records = {}
    for mode in ('off', 'on'):
        if mode == 'on':
            cb, ll = compile_ordered_source(payload.decode())
            fixture_cb, fixture_ll = compile_ordered_source(fixture_text)
        else:
            cb, ll = compile_source(payload.decode())
            fixture_cb, fixture_ll = compile_linalg(fixture_text)
        for name, text in (('connected', ll), ('fixture', fixture_ll)):
            base = output / (mode + '-' + name)
            base.with_suffix('.mlir').write_text(text + '\n')
            subprocess.run([str(tools / 'mlir-translate'), '--mlir-to-llvmir',
                str(base.with_suffix('.mlir')), '-o', str(base.with_suffix('.ll'))], check=True)
            subprocess.run([str(tools / 'clang'), '--target=riscv64-unknown-elf',
                '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-O2',
                '-ffreestanding', '-fno-builtin', '-c', str(base.with_suffix('.ll')),
                '-o', str(base.with_suffix('.o'))], check=True)
        native = (output / (mode + '-fixture.ll')).read_text()
        lines = []
        for line in native.splitlines():
            if 'call void asm sideeffect' in line:
                if '"fence"' not in line and '.insn r 0x7b, 3, 7, x0, x0, x0' not in line:
                    raise ValueError('fixture unexpectedly contains device arithmetic')
            else:
                lines.append(line)
        native_path = output / (mode + '-native.ll')
        native_path.write_text('\n'.join(lines) + '\n')
        shared = output / (mode + '.so')
        subprocess.run([str(tools / 'clang'), '-shared', '-fPIC', '-O2',
                        str(native_path), '-o', str(shared)], check=True)
        records[mode] = dict(llvm_sha256=sha((ll+'\n').encode()),
            object_sha256=sha((output / (mode+'-connected.o')).read_bytes()),
            kernel_abi=cb['kernel_abi'],
            realized_recipes=cb['params'].get('host_pool_ordered_expansion', {}).get('realized_recipes', []),
            fixture_realized_recipes=fixture_cb['params'].get('host_pool_ordered_expansion', {}).get('realized_recipes', []),
            connected_tasks=[(t['kind'], t['source_op_indices'])
                for t in cb['params']['global_program_plan']['tasks']])
    shape = tuple(int(n) for n in argument.type.get_shape())
    result_shape = tuple(int(n) for n in result.type.get_shape())
    rng = np.random.default_rng(247)
    values = rng.standard_normal(shape).astype(np.float32)
    # NaN payloads, infinities and signed zeros exercise original cmp/select.
    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                        0x7fc00001, 0x7fc01002, 0xffc02003], dtype=np.uint32)
    values.view(np.uint32).flat[::47] = np.resize(special, values.size // 47 + 1)[:len(values.flat[::47])]
    # Entire zero windows make the equal-value signed-zero selection observable.
    values[:, 0, :, :] = np.float32(-0.0)
    values[:, 0, ::2, ::2] = np.float32(0.0)
    actual = {}
    for mode in ('off', 'on'):
        lib = ctypes.CDLL(str(output / (mode + '.so')))
        fn = lib.kernel_entry; fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        actual[mode] = np.zeros(result_shape, dtype=np.float32)
        fn(values.ctypes.data, actual[mode].ctypes.data)
    np.testing.assert_array_equal(actual['off'].view(np.uint32), actual['on'].view(np.uint32))
    # Independently replay the exact source loop and select predicate in order.
    padded = np.pad(values, ((0,0),(0,0),(1,1),(1,1)), constant_values=-np.inf)
    expected = np.full(result_shape, -np.inf, dtype=np.float32)
    for y in range(3):
        for x in range(3):
            lane = padded[:,:,y:y+result_shape[2]*2:2,x:x+result_shape[3]*2:2]
            expected = np.where(np.isnan(lane) | (lane > expected), lane, expected)
    np.testing.assert_array_equal(expected.view(np.uint32), actual['on'].view(np.uint32))
    if records['off']['kernel_abi'] != records['on']['kernel_abi'] or records['off']['connected_tasks'] != records['on']['connected_tasks']:
        raise ValueError('source placement or publication changed')
    for record in records.values():
        del record['connected_tasks']
    report = dict(schema='merlin.gemmini.host_pool_experiment.v1',
        source_sha256=sha(payload), fixture_sha256=sha(fixture_text.encode()),
        adapter_sha256=sha(Path(__file__).read_bytes()), products=records,
        exact_bit_patterns_checked=int(expected.size), source_replay='pass',
        scalar_edge_cases=['positive and negative zero in equal-value windows',
                          'positive and negative infinity', 'positive and negative quiet NaN payloads'],
        llvm_translation_and_rv64gc_O2_objects='pass',
        object_compile_flags=['--target=riscv64-unknown-elf', '-march=rv64gc',
                              '-mabi=lp64d', '-mcmodel=medany', '-O2',
                              '-ffreestanding', '-fno-builtin'],
        object_scope='local kernel object; original broker link/ELF not requested',
        object_changed=records['off']['object_sha256'] != records['on']['object_sha256'],
        timing='UNKNOWN; no broker request; no performance promotion')
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(output / 'report.json')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check(args.source, args.output)
