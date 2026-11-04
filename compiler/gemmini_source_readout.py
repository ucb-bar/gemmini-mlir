"""Source-bound scalar integer-to-i8 readout admission.

The shared output-fusion owner reads the original source chain. A second proof
checks the ordered binary32 multiplies on the entire i32 domain. This module
never deletes source operations: a graph owner must retain every observer and
explicitly consume the admitted members and destination. Refusals leave that
owner's original chain intact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from merlin.compile.analysis.contraction_shapes import contraction_extents, zero_initialised
from merlin.llvmlower.requantization import f32, prove_scale
from merlin.targetgen.output_path import validate
from merlin.xdsl_dialects.lowering import output_fusion as OF, compute_groups as CG


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _fingerprint(op):
    return _sha(str(op).encode())


def _scalar_product(op):
    """The shared SCALE class includes division; this readout absorbs multiplication.

    Keep the scalar binary32 rounding boundary explicit instead of interpreting
    a divisor, a reversed divide, or a larger expression as a multiplier.
    """
    if op.name != 'linalg.generic' or len(op.regions) != 1:
        return False
    body = op.regions[0].block
    operations = list(body.ops)
    return (len(body.args) == 3 and len(operations) == 2
            and operations[0].name == 'arith.mulf'
            and set(operations[0].operands) == set(body.args[:2])
            and operations[1].name == 'linalg.yield'
            and list(operations[1].operands) == list(operations[0].results))


def _signed_float_convert(op):
    if op.name != 'linalg.generic' or len(op.regions) != 1:
        return False
    body = op.regions[0].block
    operations = list(body.ops)
    return (len(body.args) == 2 and len(operations) == 2
            and operations[0].name == 'arith.sitofp'
            and list(operations[0].operands) == [body.args[0]]
            and str(body.args[0].type) == 'i32'
            and str(operations[0].results[0].type) == 'f32'
            and operations[1].name == 'linalg.yield'
            and list(operations[1].operands) == list(operations[0].results))


def selected_path():
    """Read the selected public path; do not infer a pipeline from output width."""
    import yaml
    root = Path(__file__).resolve().parent
    contract = (root / 'data/target_contract.yaml').read_bytes()
    facts = (root / 'data/rtl_facts.json').read_bytes()
    path = validate(yaml.safe_load(contract)['output_epilogue']['output_path'])
    # This is the path the common terminal's narrowing ACC selector implements.
    expected = [
        {'domain': 'accumulator', 'function': 'relu', 'kind': 'activation'},
        {'kind': 'convert', 'rounding': 'half_even', 'to': 'f32'},
        {'columns_per_scale': 16, 'dtype': 'f32', 'granularity': 'per_command',
         'kind': 'scale', 'rounding': 'half_even'},
        {'clamp': [-128, 127], 'kind': 'requant', 'rounding': 'half_even', 'to': 'i8'},
    ]
    if (path['accumulator'] != {'dtype': 'i32'} or not path['exits'].get('requant')
            or [s for s in path['readout'] if s['kind'] != 'pool'] != expected):
        raise ValueError('selected output path differs from common scalar ACC readout')
    return path, dict(contract_sha256=_sha(contract), hardware_sha256=_sha(facts),
        output_path_sha256=_sha(json.dumps(path, sort_keys=True).encode()))


@dataclass(frozen=True)
class SourceReadout:
    root: object
    members: tuple
    destination: object
    scale: float
    relu: bool
    receipt: dict
    _module: object
    _source: str

    def revalidate(self):
        if _fingerprint(self._module) != self._source:
            raise ValueError('source readout proof is stale after source mutation')
        fresh, receipt = plan(self.root, self._module)
        if fresh is None or receipt != self.receipt:
            raise ValueError('source or selected readout proof changed')
        return self

    def emit(self, fn, dst, local, rows, cols, *, output_stride,
             column_offset=0, row_offset=0):
        """Read an already completed original producer tile, at accumulator lane zero.

        The graph owner supplies the original tile's dependency completion and
        destination mapping. No source tensor arithmetic or reload is emitted.
        """
        self.revalidate()
        from gemmini_readout import emit_readout
        return emit_readout(fn, dst, local, rows, cols, output_stride=output_stride,
            output_dtype='i8', attrs={'acc_scale': self.scale, 'relu': self.relu},
            column_offset=column_offset, row_offset=row_offset)


def plan(root, module):
    """Return (checked scalar readout or None, explicit source-bound receipt).

    No captured runtime weights/bias values are accepted as constants. Full i32
    transition proof avoids borrowing an unproved source accumulator bound.
    """
    path, selected = selected_path()
    values = OF.Values()
    chain = OF.source_chain(root, values)
    try:
        decision = OF.decide(chain, path, values, contraction_extents(module))
    except FileNotFoundError as exc:
        # The common proof loads its declared format registry. A missing public
        # resource is not permission to invent format records or skip that proof.
        return None, dict(schema='gemmini.source_scalar_readout.v1', selected=False,
            source_sha256=_fingerprint(module), root_sha256=_fingerprint(root),
            selected_path=selected, source_stages=[s.kind for s in chain.stages],
            refusal=dict(code='proof_resource_unavailable', reason=str(exc)))
    receipt = dict(schema='gemmini.source_scalar_readout.v1', selected=False,
        source_sha256=_fingerprint(module), root_sha256=_fingerprint(root),
        selected_path=selected, shared_decision=decision.record(),
        source_stages=[s.kind for s in chain.stages],
        member_sha256=[_fingerprint(op) for op in chain.ops])

    def refuse(code, reason):
        receipt['refusal'] = dict(code=code, reason=reason)
        return None, receipt

    if decision.cut != 'requant':
        first = decision.first_refusal or dict(code='source_open', reason='no closed quantize boundary')
        return refuse(first['code'], first['reason'])
    dtypes = [str(v.type.element_type) for v in list(root.operands)[:2] + list(root.results)]
    if dtypes != ['i8', 'i8', 'i32'] or not zero_initialised(root):
        return refuse('producer_contract', 'requires original zero-seeded i8 x i8 -> i32 producer')
    kinds = [s.kind for s in chain.stages]
    if kinds.count('convert') != 1 or kinds[0] != 'convert' or chain.float_dtype != 'f32':
        return refuse('source_format', 'requires one initial i32-to-f32 conversion')
    if not _signed_float_convert(chain.stages[0].op):
        return refuse('source_convert_recipe', 'requires an explicit signed i32-to-f32 conversion')
    if any(k not in ('convert', 'scale', 'relu', 'quantize') for k in kinds):
        return refuse('source_stage', 'bias, residual, pool and padding require separate proofs')
    if any(stage.kind == 'scale' and not _scalar_product(stage.op) for stage in chain.stages):
        return refuse('source_scale_recipe', 'requires an explicit binary32 multiply of the two source inputs')
    # Every absorbed result except the final quantize must remain private.
    for op in (root, *chain.ops[:-1]):
        if len(op.results) != 1 or len(list(op.results[0].uses)) != 1:
            return refuse('source_observer', 'an intermediate has an additional observer')
    quant = chain.stages[-1]
    if (quant.detail.get('zero_point') != 0 or quant.detail.get('min') != -128
            or quant.detail.get('max') != 127 or OF._format_token(quant.detail.get('dtype')) != 'i8'):
        return refuse('source_quantize', 'requires signed i8 clamp and zero zero-point')
    import numpy as np
    scales = []
    for stage in chain.stages:
        if stage.kind in ('scale', 'quantize'):
            array = np.asarray(stage.numbers)
            if array.size != 1 and not np.all(array == array.flat[0]):
                return refuse('scale_granularity', 'source multiplier is not scalar')
            scalar = f32(float(array.flat[0]))
            scales.append(scalar if stage.kind == 'scale' else f32(1.0 / scalar))
    relu = 'relu' in kinds
    try:
        numeric = prove_scalar_readout(scales, relu=relu)
    except (ValueError, ZeroDivisionError) as exc:
        return refuse('inexact', str(exc))
    device_scales = decision.proof.get('device_scale', [])
    if not device_scales or any(f32(x) != numeric['scale'] for x in device_scales):
        return refuse('scale_disagreement', 'source and selected readout proofs chose different multipliers')
    receipt.update(selected=True, numeric_proof=numeric, preserves_original_placement=True,
        source_members_retained_until_owner_consumption=True)
    return SourceReadout(root, tuple(chain.ops), chain.ops[-1].results[0],
        numeric['scale'], relu, receipt, module, receipt['source_sha256']), receipt


def prove_scalar_readout(scales, *, relu=False):
    """Complete-domain numerical proof only; never source/physical authority.

    Ordered multiplies, not a Python product, define the source computation.
    ``plan`` additionally requires the common source/path proof before emission.
    """
    return prove_scale(tuple(scales), relu=relu)


def analyze(module):
    """Read all original integer producer chains; no source mutation or selection."""
    rows = []
    for op in module.walk():
        stage = CG.classify(op)
        if (stage is not None and stage.kind == CG.CONTRACTION and op.results
                and str(getattr(op.results[0].type, 'element_type', '')) == 'i32'):
            _, receipt = plan(op, module)
            rows.append(receipt)
    return rows


if __name__ == '__main__':
    import argparse
    from merlin.frontends.linalg_mlir import parse_mlir_text
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    module = parse_mlir_text(args.source.read_text())
    module.verify()
    args.output.write_text(json.dumps(dict(source_file_sha256=_sha(args.source.read_bytes()),
        decisions=analyze(module)), sort_keys=True, indent=2) + '\n')
