"""The selected RoCC contract keeps its numeric ABI after JSON freezing."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend import rocc_semantics


def _selected_isa(monkeypatch, encoding):
    from merlin.targetgen.rtl import facts
    from merlin.targetgen import target_experiment

    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda _target: SimpleNamespace(encoding=encoding))
    monkeypatch.setattr(
        facts,
        "load_facts",
        lambda _target: {
            "facts": {
                "arrays": [{"name": "mesh", "rows": 16}],
                "interfaces": [
                    {"name": "funct_decode_table", "width": 7, "custom_opcode": 123, "funct3": 3},
                    {
                        "name": "register_bundle_layouts",
                        "bundles": {"ConfigMvoutRs1": {"fields": {"cmd_type": {"offset": 0, "width": 2}}}},
                    },
                ],
            }
        },
    )
    return rocc_semantics.isa_constants("selected-target")


def _contract_encoding():
    contract = Path(__file__).resolve().parents[1] / "contracts/target_contract.yaml"
    return yaml.safe_load(contract.read_text(encoding="utf-8"))["encoding"]


def _operand(raw):
    return {"raw": raw, "kind": "const", "arg_index": None, "offset": None}


def test_json_frozen_contract_decodes_same_classes_as_authored_yaml(monkeypatch):
    authored = _contract_encoding()
    frozen = json.loads(json.dumps(authored))
    original = _selected_isa(monkeypatch, authored)
    selected = _selected_isa(monkeypatch, frozen)
    assert selected["FUNCT_CLASS"] == original["FUNCT_CLASS"]
    assert selected["CONFIG_SUBTYPE"] == original["CONFIG_SUBTYPE"]
    assert all(type(key) is int for key in selected["FUNCT_CLASS"])
    assert all(type(key) is int for key in selected["CONFIG_SUBTYPE"])
    assert rocc_semantics.decode_instruction(7, _operand(0), _operand(0), selected) == ("FLUSH", {})
    assert rocc_semantics.decode_instruction(0, _operand(1), _operand(0), selected)[0] == "CONFIG_LD"
    assert rocc_semantics.instruction_funct("FLUSH", 0, selected) == 7


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("semantic_class", {True: "FLUSH"}),
        ("semantic_class", {"07": "FLUSH"}),
        ("semantic_class", {"+7": "FLUSH"}),
        ("semantic_class", {"128": "FLUSH"}),
        ("semantic_class", {7: "FLUSH", "7": "CONFIG"}),
        ("semantic_class", {"7": None}),
        ("config_subtype", {False: "CONFIG_EX"}),
        ("config_subtype", {"01": "CONFIG_LD"}),
        ("config_subtype", {"4": "CONFIG_LD"}),
        ("config_subtype", {1: "CONFIG_LD", "1": "CONFIG_ST"}),
    ],
)
def test_numeric_abi_rejects_invalid_or_ambiguous_codes(monkeypatch, field, replacement):
    encoding = copy.deepcopy(_contract_encoding())
    encoding[field] = replacement
    with pytest.raises(ValueError, match=field):
        _selected_isa(monkeypatch, encoding)
