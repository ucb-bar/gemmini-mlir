"""Bank placement and executed command order, including paired-loop drains."""

import hashlib

import pytest
from xdsl.dialects import llvm

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_wide_resadd import ResidualPrefetchPlan, build
from mlir_oot.tables import isa, rtl_facts as F


def _executed(module):
    """Interpret only CPU address/control ops and record primitive device commands."""
    fn = module.body.block.first_op
    block = fn.body.blocks.first
    values = {arg: (i, 0) for i, arg in enumerate(block.args)}
    commands = []
    steps = 0
    while True:
        jump = False
        for op in block.ops:
            steps += 1
            assert steps < 2000000
            inputs = [values[v] for v in op.operands]
            if op.name == 'llvm.mlir.constant':
                values[op.results[0]] = op.value.value.data
            elif isinstance(op, llvm.AddOp):
                values[op.results[0]] = inputs[0] + inputs[1]
            elif isinstance(op, llvm.MulOp):
                values[op.results[0]] = inputs[0] * inputs[1]
            elif isinstance(op, llvm.GEPOp):
                values[op.results[0]] = (inputs[0][0], inputs[0][1] + inputs[1])
            elif isinstance(op, llvm.ICmpOp):
                assert op.predicate.value.data == llvm.ICmpPredicateFlag.SLT.to_int()
                values[op.results[0]] = inputs[0] < inputs[1]
            elif isinstance(op, llvm.BrOp):
                block = op.successors[0]
                values.update(zip(block.args, inputs))
                jump = True
                break
            elif isinstance(op, llvm.CondBrOp):
                assert len(inputs) == 1
                block = op.successors[0 if inputs[0] else 1]
                jump = True
                break
            elif isinstance(op, llvm.ReturnOp):
                return commands
            elif op.name.startswith('gemmini.'):
                commands.append((op, inputs))
            else:
                raise AssertionError(op.name)
        assert jump


def _arithmetic(commands, m, banked=False):
    """Resolve each executed compute to the input bytes and resident coefficient."""
    spad = {}
    result = []
    for op, inputs in commands:
        if op.name == 'gemmini.mvin':
            operand, offset = inputs[0]
            assert offset >= 0
            limit = 3 * F.DIM * F.DIM if operand == 3 else m * 4 * F.DIM
            assert offset + op.a('rows') * op.a('cols') <= limit
            for tile in range(op.a('cols') // F.DIM):
                spad[op.a('local') + tile * F.DIM] = (operand, offset + tile * F.DIM)
        elif op.name == 'gemmini.preload':
            weight = 'resident' if op.a('bd') == isa.GARBAGE_ADDR else spad[op.a('bd')]
            acc = op.a('c')
            if banked and acc & F.ACC_BANK_ROWS:
                acc -= F.ACC_BANK_ROWS
            result.append(('preload', weight, acc, op.a('bd_rows'), op.a('c_rows')))
        elif op.name == 'gemmini.compute':
            result.append(('compute', spad[op.a('a')], op.a('accumulate'), op.a('a_rows')))
        elif op.name == 'gemmini.mvout':
            assert inputs[0][0] == 2
            assert inputs[0][1] + op.a('rows') * op.a('cols') <= m * 4 * F.DIM
            acc = op.a('local')
            if banked and acc & F.ACC_BANK_ROWS:
                acc -= F.ACC_BANK_ROWS
            result.append(('store', inputs[0], acc, op.a('rows'), op.a('cols')))
    return result


@pytest.mark.parametrize('m', [16, 32, 48, 64, 80, 1024])
@pytest.mark.parametrize('p,q', [(1, 1), (127, 128), (254, 1), (2609, 2180), (32767, 32767)])
@pytest.mark.parametrize('banked', [False, True])
def test_prefetch_preserves_every_executed_arithmetic_command_and_tail(m, p, q, banked):
    serial = build(m, p, q, 1. / 65536)
    candidate = build(m, p, q, 1. / 65536, prefetch_m=True, banked_accumulators=banked)
    assert _arithmetic(_executed(candidate), m, banked) == _arithmetic(_executed(serial), m)
    if m == F.DIM:
        assert str(candidate) == str(serial)
    else:
        plan = ResidualPrefetchPlan(m, p, q, banked)
        extents = [(plan.operand_base(slot, 0), plan.operand_base(slot, 0) + 2 * plan.panel_rows)
                   for slot in (0, 1)]
        extents.append((plan.weight_base, plan.weight_base + 3 * F.DIM))
        for start, end in extents:
            assert start < end <= F.SPAD_ROWS
            assert start // F.SPAD_BANK_ROWS == (end - 1) // F.SPAD_BANK_ROWS
        for i, (start, end) in enumerate(extents):
            assert all(end <= other_start or other_end <= start
                       for other_start, other_end in extents[i + 1:])
        assert plan.attributes().data['chunks_per_panel'].value.data == (p + 126) // 127 + (q + 126) // 127
        assert candidate.body.block.first_op.attributes['gemmini.residual_m_prefetch'] == plan.attributes()
    lower(candidate)


def test_default_emission_matches_retained_serial_source():
    # Pre-change printer bytes, independently retained in the qualification receipt.
    previous_sha = '1ce7c23f4a149ee0ea93d9bfb9bb7ae6b9bf53682a430bbc90d5b96ee2ce3627'
    assert hashlib.sha256(str(build(48, 2609, 2180, .00037060913746245205)).encode()).hexdigest() == previous_sha


@pytest.mark.parametrize('fact,value,error', [
    ('SPAD_BANKS', 2, 'three proved'),
    ('SPAD_BANK_ROWS', 64, 'three proved'),
    ('ACC_ROWS', 32, 'footprint'),
    ('OPERAND_DTYPE', 'i16', 'signed-i8'),
])
def test_resource_proof_refuses_missing_capacity(monkeypatch, fact, value, error):
    monkeypatch.setattr(F, fact, value)
    with pytest.raises(ValueError, match=error):
        build(48, 1, 1, .001, prefetch_m=True)


def test_option_and_shape_refusals():
    for m in (0, 15, -16):
        with pytest.raises(ValueError):
            build(m, 1, 1, .001, prefetch_m=True)
    with pytest.raises(ValueError, match='boolean'):
        build(48, 1, 1, .001, prefetch_m=1)
    with pytest.raises(ValueError, match='require residual M prefetch'):
        build(48, 1, 1, .001, banked_accumulators=True)


def test_banked_accumulator_capacity_refusal(monkeypatch):
    monkeypatch.setattr(F, 'ACC_BANKS', 1)
    with pytest.raises(ValueError, match='two proved banks'):
        build(48, 1, 1, .001, prefetch_m=True, banked_accumulators=True)
