"""Exercise live graph-contract consumption and stock whole-source lowering."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

_SOURCE = '''builtin.module {
func.func @forward(%x: tensor<2x3xi8>) -> tensor<2x3xi8> {
%e = tensor.empty() : tensor<3x2xi8>
%t = linalg.transpose ins(%x : tensor<2x3xi8>) outs(%e : tensor<3x2xi8>) permutation = [1, 0]
%f = tensor.empty() : tensor<2x3xi8>
%u = linalg.transpose ins(%t : tensor<3x2xi8>) outs(%f : tensor<2x3xi8>) permutation = [1, 0]
func.return %u : tensor<2x3xi8>
}}'''


def _evaluate(source, value):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module = parse_mlir_text(source)
    block = next(op for op in module.body.block.ops if op.name == 'func.func').body.block
    values = {block.args[0]: value}
    for op in block.ops:
        if op.name == 'tensor.empty':
            values[op.results[0]] = None
        elif op.name == 'linalg.transpose':
            values[op.results[0]] = values[op.operands[0]].transpose(tuple(op.permutation.get_values()))
        elif op.name == 'func.return':
            return values[op.operands[0]]
        else:
            raise ValueError('unsupported fixture operation: ' + op.name)
    raise ValueError('fixture lacks return')


def main():
    import numpy as np
    from merlin.llvmlower.graph_optimizations import GraphOptimizationContract
    from .host_layout import prepare_host_layout
    from .host_runtime import lower_whole_program

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = GraphOptimizationContract(hashlib.sha256(_SOURCE.encode()).hexdigest(), 'forward',
        propagate_layout=True, independent_element_order_unobserved=True, physical_layout=True)

    def discovery(module):
        assert not any(op.name == 'linalg.transpose' for op in module.walk())
        return ()

    def device(_):
        raise AssertionError('layout fixture has no device arithmetic')

    result = lower_whole_program(_SOURCE, args.out / 'whole-program', device, regions=discovery,
        lower_options={'graph_optimizations': contract})
    receipt = result.stats['adapter_host_graph_preparation']
    selected = Path(receipt['output']['path']).read_text()
    for offset in range(256):
        value = ((np.arange(6, dtype=np.int64) * 19 + offset) % 256 - 128).astype(np.int8).reshape(2, 3)
        assert np.array_equal(_evaluate(_SOURCE, value), _evaluate(selected, value))
    refusals = {}
    for name, proposal, source in (
        ('stale_source', contract, _SOURCE + '\n'),
        ('missing_permission', replace(contract, independent_element_order_unobserved=False), _SOURCE),
        ('saved_report', receipt, _SOURCE),
    ):
        try:
            prepare_host_layout(source, proposal, args.out / ('refusal-' + name))
        except (ValueError, TypeError) as error:
            refusals[name] = str(error)
        else:
            raise AssertionError('expected refusal: ' + name)
    report = dict(exact_cases=256, negative_cases=refusals, discovery_sees_prepared_source=True,
        stock_cli_lowering='passed', preparation=receipt,
        scope='fixture source equivalence and actual stock LLVM lowering; no execution or cycle claim')
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(exact_cases=256, negative_cases=len(refusals), stock_cli_lowering='passed')))


if __name__ == '__main__':
    main()
