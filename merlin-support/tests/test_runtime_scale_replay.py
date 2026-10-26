"""Symbolic replay identity, byte transport and repeated-call configuration."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from mlir_oot.core_aten.runtime_scale import RuntimeScaleRef

from mlir_oot.core_aten import runtime_scale_replay as R
from mlir_oot.core_aten.contraction_proof import _Check
from mlir_oot.core_aten.stream import ACC, Arg, Stream, simulate
from mlir_oot.core_aten.stream_proof import ProofError, Replay, decode, execute


def stream():
    # Three stores of the same seeded row exercise narrow/raw/narrow state changes.
    s = Stream(
        [
            Arg("seed", "i32", (1, 4), "input"),
            Arg("scale", "f32", (1,), "input"),
            Arg("other", "f32", (1,), "input"),
            Arg("q0", "i8", (1, 4), "output"),
            Arg("raw", "i32", (1, 4), "output"),
            Arg("q1", "i8", (1, 4), "output"),
        ]
    )
    s.config_ex()
    s.config_ld(0, 16)
    s.mvin((0, 0), ACC(0), 4, 1)
    s.config_st(4, scale_operand=RuntimeScaleRef(1))
    s.mvout((3, 0), ACC(0), 4, 1)
    s.fence()
    s.config_st(16)
    s.mvout((4, 0), ACC(0, full=True), 4, 1)
    s.fence()
    s.config_st(4, scale_operand=RuntimeScaleRef(1))
    s.mvout((5, 0), ACC(0), 4, 1)
    s.fence()
    return s


def buffers(scale):
    return {
        0: np.array([[-5, -3, 3, 5]], np.int32),
        1: np.array([scale], np.float32),
        2: np.array([scale], np.float32),
        3: np.zeros((1, 4), np.int8),
        4: np.zeros((1, 4), np.int32),
        5: np.zeros((1, 4), np.int8),
    }


def check_readout(token, *, full=False):
    expected = SimpleNamespace(
        narrow=True,
        relu=False,
        scale_operand=RuntimeScaleRef(1),
        bits=None,
        out_shape=(1, 4),
    )
    checker = object.__new__(_Check)
    checker.expected = expected
    store = SimpleNamespace(
        accumulator=True,
        full=full,
        config={"activation": 0, "acc_scale": token, "stride": 4},
    )
    checker._readout(store, 0, 4)


def test_decoded_runtime_store_retains_operand_identity():
    s = stream()
    ins = next(i for i in s.ins if i.a("scale_operand") is not None)
    command = decode(ins)
    assert command.fields["acc_scale"] == RuntimeScaleRef(1)
    assert command.arg is None and command.off == 0
    assert command.fields["stride"] == 4
    resolved = R.resolve_config_instruction(ins, s.args, buffers(0.25))
    assert decode(resolved).fields["acc_scale"] == 0x3E800000
    assert (
        ins.a("scale_operand") == RuntimeScaleRef(1)
        and ins.a("multiplier_bits") is None
    )
    with pytest.raises(ValueError, match="unknown"):
        R.resolve_config_instruction(
            ins._replace(attrs=ins.attrs + (("unlicensed", 0),)), s.args, buffers(0.25)
        )
    check_readout(RuntimeScaleRef(1))
    for wrong in (0x3F800000, RuntimeScaleRef(2)):
        with pytest.raises(ProofError, match="scale"):
            check_readout(wrong)
    replay = Replay(s).run()
    assert [store.config["acc_scale"] for store in replay.stores] == [
        RuntimeScaleRef(1),
        0x3F800000,
        RuntimeScaleRef(1),
    ]


def test_replay_reloads_scale_across_invocations_and_switches():
    s = stream()
    original = list(s.ins)
    for scale, expected in [
        (0, [0, 0, 0, 0]),
        (0.25, [-1, -1, 1, 1]),
        (0.5, [-2, -2, 2, 2]),
        (0.3, [-2, -1, 1, 2]),
    ]:
        values = buffers(scale)
        execute(s, values)
        assert values[3].tolist() == values[5].tolist() == [expected]
        assert np.array_equal(values[4], values[0])
        assert s.ins == original
        machine = simulate(
            s, {s.args[i].name: b.copy() for i, b in buffers(scale).items()}
        )
        assert machine["q0"].tolist() == machine["q1"].tolist() == [expected]
        # Machine's buffer ABI is byte-oriented, tested independently through simulate.
    config_indices = [i for i, ins in enumerate(s.ins) if ins.kind == "config_st"]
    stale = replace(
        s, ins=[ins for i, ins in enumerate(s.ins) if i != config_indices[-1]]
    )
    assert Replay(stale).run().stores[-1].config["acc_scale"] == 0x3F800000
    with pytest.raises(ProofError, match="scale"):
        check_readout(Replay(stale).run().stores[-1].config["acc_scale"])
    with pytest.raises(ProofError, match="width"):
        check_readout(RuntimeScaleRef(1), full=True)
    wrong_stride = replace(
        s,
        ins=[
            ins._replace(
                attrs=tuple((k, 8 if k == "stride" else v) for k, v in ins.attrs)
            )
            if i == config_indices[0]
            else ins
            for i, ins in enumerate(s.ins)
        ],
    )
    with pytest.raises(ProofError, match="stride"):
        checker = object.__new__(_Check)
        checker.expected = SimpleNamespace(
            narrow=True,
            relu=False,
            scale_operand=RuntimeScaleRef(1),
            bits=None,
            out_shape=(1, 4),
        )
        checker._readout(Replay(wrong_stride).run().stores[0], 0, 4)


@pytest.mark.parametrize(
    "bits", [0, 0x80000000, 1, 0x80000001, 0x3E99999A, 0x7F7FFFFF, 0xFF7FFFFF]
)
def test_scale_storage_bits_survive_transport(bits):
    s = stream()
    value = np.array([bits], np.uint32).view(np.float32)
    assert R.runtime_scale_bits(RuntimeScaleRef(1), s.args, {1: value}) == bits
    assert (
        R.runtime_scale_bits(RuntimeScaleRef(1), s.args, {1: value.view(np.uint8)})
        == bits
    )
    assert (
        int(
            np.asarray(
                R.runtime_scale_value(RuntimeScaleRef(1), s.args, {1: value})
            ).view(np.uint32)
        )
        == bits
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.5], np.float64),
        np.array([1], np.int32),
        np.array([0.5, 0.5], np.float32),
        np.zeros(3, np.uint8),
        np.array([np.nan], np.float32),
        np.array([np.inf], np.float32),
    ],
)
def test_invalid_runtime_storage_refuses(bad):
    with pytest.raises(ValueError):
        R.runtime_scale_bits(RuntimeScaleRef(1), stream().args, {1: bad})


def test_runtime_scale_semantic_gate_uses_f32_trials():
    from mlir_oot.core_aten.semantic_gate import _buffers

    s = stream()
    s.args.extend([Arg("a", "i8", (2, 2), "input"), Arg("w", "i8", (2, 2), "input")])
    seen = []
    for trial in range(len(R.SCALE_TRIAL_BITS)):
        values, inputs = _buffers(
            s,
            {
                "seed": (-(2**31), 2**31 - 1),
                "other": (-128, 127),
                "a": (-128, 127),
                "w": (-128, 127),
            },
            trial,
            np.random.default_rng(1),
        )
        assert values[1].dtype == np.float32 and np.isfinite(values[1]).all()
        seen.append(R.runtime_scale_bits(RuntimeScaleRef(1), s.args, values))
        assert inputs["seed"].dtype == np.int32
        if trial < 2:
            assert all(
                np.all(inputs[name] == (-128, 127)[trial]) for name in ("a", "w")
            )
    assert tuple(seen) == R.SCALE_TRIAL_BITS
    with pytest.raises(ValueError):
        R.runtime_scale_trial(-1)


def test_source_scale_binding_rejects_same_typed_argument_substitution():
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering.runtime_readout import scale_operand_record

    module = mq.parse("""builtin.module {
      func.func @forward(%first: tensor<1xf32>, %other: tensor<1xf32>) { func.return }
    }""")
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    first, other = function.body.block.args
    record = scale_operand_record(first)
    args = [Arg("scale", "f32", (1,), "input")]
    assert R.source_scale_ref(
        record, module, [("scale", first)], args
    ) == RuntimeScaleRef(0)
    with pytest.raises(ValueError, match="exact kernel input"):
        R.source_scale_ref(record, module, [("scale", other)], args)
    wrong = {**record, "source": {"func_arg": True, "element_index": 0}}
    with pytest.raises(ValueError, match="integers"):
        R.source_scale_ref(wrong, module, [("scale", other)], args)
