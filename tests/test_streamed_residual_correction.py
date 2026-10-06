"""Completed output ownership and exact primitive order across CPU correction."""

import copy
import importlib.util
from pathlib import Path

import pytest
from merlin.llvmlower.quantized_affine_pair import derive
from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)
from xdsl.dialects import llvm

from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_streamed_resadd import build
from mlir_oot.golden_wide_resadd import build as original

spec = importlib.util.spec_from_file_location(
    "control_proof", Path(__file__).with_name("test_wide_residual_prefetch.py")
)
control_proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control_proof)


def certificate(predictor=None):
    if predictor is None:
        predictor = {
            "p": 73,
            "q": 61,
            "scale": 0.013245166279375553,
        }
    return derive(
        0.011258588172495365,
        0.00940733402967453,
        0.011643771082162857,
        **predictor,
        relu=True,
    )


def prove(module, m):
    fn = next(
        op
        for op in module.body.block.ops
        if isinstance(op, llvm.FuncOp) and op.body.blocks
    )
    args = [StaticPointer(i, StaticInt(0, 64)) for i in range(4)]
    commands, pending, completed, corrected = [], set(), set(), set()
    for step in trace_static_function(
        fn,
        args,
        observe=lambda op: (
            op.name.startswith("gemmini.") or isinstance(op, llvm.CallOp)
        ),
        pointer_index_bits=64,
    ):
        op, values = step.operation, step.inputs
        if isinstance(op, llvm.CallOp):
            assert op.callee.root_reference.data == "merlin_correct"
            assert len(values) == 4 and not op.results
            assert [p.base for p in values[:3]] == [0, 1, 2]
            offsets = [p.offset.value for p in values[:3]]
            assert offsets == [offsets[0]] * 3 and offsets[0] % 1024 == 0
            assert values[3].value == 1024
            extent = set(range(offsets[0], offsets[0] + values[3].value))
            assert extent <= completed and not extent & pending
            assert not extent & corrected
            corrected.update(extent)
        else:
            inputs = [(p.base, p.offset.value) for p in values]
            commands.append((op, inputs))
            if op.name == "gemmini.mvout":
                assert values[0].base == 2
                extent = set(
                    range(values[0].offset.value, values[0].offset.value + 1024)
                )
                assert not extent & completed and not extent & corrected
                pending.update(extent)
            elif op.name == "gemmini.fence":
                completed.update(pending)
                pending.clear()
    assert not pending and corrected == completed == set(range(m * 64))
    return commands


@pytest.mark.parametrize("m", [16, 32, 48, 64, 80, 128])
@pytest.mark.parametrize(
    "predictor",
    [
        {"p": 73, "q": 61, "scale": 0.013245166279375553},
        {"p": 298, "q": 249, "scale": 0.0032446938566863537},
    ],
)
def test_completed_panels_and_original_arithmetic(m, predictor):
    proof = certificate(predictor)
    candidate, code, tables = build(m, proof, correction_symbol="merlin_correct")
    trace = prove(candidate, m)
    control = original(
        m,
        **proof["predictor"],
        relu=proof["source"]["relu"],
        prefetch_m=True,
        banked_accumulators=True,
    )
    assert control_proof._arithmetic(trace, m, True) == control_proof._arithmetic(
        control_proof._executed(control), m, True
    )
    assert len(tables) == 768 and "volatile float lhs" in code
    assert sum(op.name == "gemmini.fence" for op, _ in trace) == m // 16 + 2
    lower(candidate).verify()


def test_certificate_mutation_missing_identity_and_resource_refuse(monkeypatch):
    from mlir_oot.tables import rtl_facts as F

    for supplied in [None, {}, copy.deepcopy(certificate())]:
        if isinstance(supplied, dict) and "pairs" in supplied:
            supplied["pairs"] = 65535
        with pytest.raises(ValueError, match="certificate"):
            build(48, supplied, correction_symbol="merlin_correct")
    for name in ["bad-name", "gemmini_golden_wide_resadd", 12]:
        with pytest.raises(ValueError, match="identifier"):
            build(48, certificate(), correction_symbol=name)
    monkeypatch.setattr(F, "ACC_BANKS", 1)
    with pytest.raises(ValueError, match="two proved banks"):
        build(48, certificate(), correction_symbol="merlin_correct")
