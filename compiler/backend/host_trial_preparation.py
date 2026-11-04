"""Check exact slice preparation sharing and retain source-bound observations.

Run: python3 -m submission.backend.host_trial_preparation --out DIR
This exercises source semantics and checked sharing; it reports no hardware cycles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import tensor_provenance_cse as cse

_SOURCE = '''builtin.module {
  func.func @forward(%x: tensor<16xi8>, %y: tensor<16xi8>) -> i8 {
    %a = tensor.extract_slice %x[2] [6] [2] : tensor<16xi8> to tensor<6xi8>
    %b = tensor.extract_slice %a[1] [2] [2] : tensor<6xi8> to tensor<2xi8>
    %c = tensor.extract_slice %x[4] [2] [4] : tensor<16xi8> to tensor<2xi8>
    %d = tensor.extract_slice %y[4] [2] [4] : tensor<16xi8> to tensor<2xi8>
    %i = arith.constant 1 : index
    %p = tensor.extract %b[%i] : tensor<2xi8>
    %q = tensor.extract %c[%i] : tensor<2xi8>
    %r = tensor.extract %d[%i] : tensor<2xi8>
    %s = arith.addi %p, %q : i8
    %t = arith.addi %s, %r : i8
    func.return %t : i8
  }
}'''


def _evaluate(module, inputs):
    """Independent literal interpreter for the small checked fixture only."""
    import numpy as np
    block = next(op for op in module.body.block.ops if op.name == 'func.func').body.block
    values = dict(zip(block.args, inputs, strict=True))
    for op in block.ops:
        if op.name == 'tensor.extract_slice':
            source = values[op.operands[0]]
            offsets = tuple(op.static_offsets.get_values())
            sizes = tuple(op.static_sizes.get_values())
            strides = tuple(op.static_strides.get_values())
            result = source[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides, strict=True))]
        elif op.name == 'arith.constant':
            result = op.value.value.data
        elif op.name == 'tensor.extract':
            result = values[op.operands[0]][tuple(int(values[v]) for v in op.operands[1:])]
        elif op.name == 'arith.addi':
            result = np.int8((int(values[op.operands[0]]) + int(values[op.operands[1]]) + 128) % 256 - 128)
        elif op.name == 'func.return':
            return tuple(int(values[v]) for v in op.operands)
        else:
            raise ValueError('fixture interpreter does not support ' + op.name)
        values[op.results[0]] = result
    raise ValueError('fixture lacks a return')


def main():
    import numpy as np
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('public_sources/JC_resnet50_prepared_connected.mlir'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = parse_mlir_text(_SOURCE)
    baseline, baseline_rewrites = cse.rewrite_text(_SOURCE)
    selected, rewrites = cse.rewrite_text(_SOURCE, canonical_slices=True)
    assert not baseline_rewrites, baseline_rewrites
    assert rewrites == {'tensor.extract_slice.canonical': 1}, rewrites
    rewritten = parse_mlir_text(selected)
    # All signed i8 values appear in both roots; different SSA roots retain
    # distinct preparations. Negative and overflow results exercise exact wraps.
    for offset in range(256):
        x = ((np.arange(16, dtype=np.int64) * 19 + offset) % 256 - 128).astype(np.int8)
        y = ((np.arange(16, dtype=np.int64) * 31 - offset) % 256 - 128).astype(np.int8)
        assert _evaluate(original, (x, y)) == _evaluate(rewritten, (x, y))
    public_source = args.source.read_text()
    incumbent, incumbent_counts = cse.rewrite_text(public_source)
    prepared, prepared_counts = cse.rewrite_text(public_source, canonical_slices=True)
    (args.out / 'selected.mlir').write_text(prepared)
    def sha(value):
        return hashlib.sha256(value).hexdigest()
    report = dict(schema='submission.host_preparation_trial.v1',
        mechanism='shared canonical static tensor read views; original SSA roots',
        library=dict(path=str(Path(cse.__file__).resolve()), sha256=sha(Path(cse.__file__).read_bytes())),
        fixture=dict(source_sha256=sha(_SOURCE.encode()), cases=256, numerics='bit_exact',
            baseline_rewrites=baseline_rewrites, selected_rewrites=rewrites),
        public_source=dict(path=str(args.source.resolve()), source_sha256=sha(public_source.encode()),
            baseline_sha256=sha(incumbent.encode()), selected_sha256=sha(prepared.encode()),
            baseline_rewrites=incumbent_counts, selected_rewrites=prepared_counts),
        default='baseline retained', hardware_cycles=None,
        limitations=['Fixture source numerics only; whole connected execution must be checked separately.',
            'No rank-reducing or dynamic slices; returned preparation cones remain independent.'])
    (args.out / 'trial.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
