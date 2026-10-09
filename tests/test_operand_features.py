"""Feature transport refuses incomplete counts and implicit alias attribution."""

import copy
import json

import pytest
import test_spike_operand_telemetry as probe

from mlir_oot.operand_features import read_telemetry, summarize


@pytest.fixture(scope="module")
def observer(tmp_path_factory):
    return probe.observer.__wrapped__(tmp_path_factory)


@pytest.fixture
def evidence(observer, tmp_path):
    result, data = probe.run(
        observer, tmp_path, scopes="7 0x100 0x200 0x100\n", aggregate=True
    )
    assert result.returncode == 0
    return data


def read(tmp_path, data):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return read_telemetry(path)


def test_actual_invocation_payloads_and_unknown_dma_remain_separate(tmp_path, evidence):
    data = read(tmp_path, evidence)
    first = summarize(data, scope_id=7, invocation=1)
    second = summarize(data, scope_id=7, invocation=2)
    assert first["features"]["requested_dma_bytes"] is None
    assert first["features"]["array_work"] == 4096
    assert second["features"]["requested_dma_bytes"] == 1
    assert second["features"]["primitive_commands"]["LOAD_CMD"] == 1
    assert second["features"]["primitive_commands"]["CONFIG_CMD"] == 1
    assert second["features"]["primitive_commands"]["COMPUTE_AND_FLIP_CMD"] == 0
    assert summarize(data)["features"]["primitive_commands"]["LOAD_CMD"] == 5
    with pytest.raises(ValueError, match="explicit scope"):
        summarize(data, invocation=1)
    with pytest.raises(ValueError, match="no commands"):
        summarize(data, scope_id=8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("commands", 0),
        ("commands", True),
        ("requested_load_bytes", -1),
        ("funct", 8),
        ("funct", 25),
        ("a_rows", 65536),
        ("last_event", 99),
        ("padded_mac_slots", 1),
        ("scope_id", -1),
        ("invocation", 99),
        ("pc", 2),
    ],
)
def test_invalid_row_or_unknown_invocation_refused(tmp_path, evidence, field, value):
    data = copy.deepcopy(evidence)
    data["rows"][0][field] = value
    with pytest.raises(ValueError):
        read(tmp_path, data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dim", 8),
        ("elem_bytes", 2),
        ("acc_bytes", 8),
        ("total_commands", 12),
        ("pc_aggregation", "guessed"),
        ("schema", "other"),
        ("entries", None),
    ],
)
def test_regime_or_count_mismatch_refused(tmp_path, evidence, field, value):
    data = copy.deepcopy(evidence)
    data[field] = value
    with pytest.raises((ValueError, TypeError)):
        read(tmp_path, data)


def test_entry_sequence_must_be_actual_and_complete(tmp_path, evidence):
    for change in ("reordered", "missing", "duplicate_invocation"):
        data = copy.deepcopy(evidence)
        if change == "reordered":
            data["entries"].reverse()
        if change == "missing":
            data["entries"].pop()
        if change == "duplicate_invocation":
            data["entries"][1]["invocation"] = 1
        with pytest.raises(ValueError):
            read(tmp_path, data)
