"""The selected header and legal RTL table expand the OOT schedule vocabulary."""

from pathlib import Path

from backend import gemmini_sched as schedule


def test_real_header_derives_only_legal_single_instructions_with_declared_pointer_kinds(monkeypatch):
    include = Path(__file__).resolve().parents[1] / "resources/gemmini-rocc-tests/include"
    header = str(include / "gemmini.h")
    params = str(include / "gemmini_params.h")
    facts = schedule.schedule_facts(header, params, target=None)
    monkeypatch.setattr(schedule, "schedule_facts", lambda *_args, **_kwargs: facts)
    monkeypatch.setattr(schedule, "_legal_functs", lambda _target: (2, 3, 7))
    monkeypatch.setattr(schedule, "_semantic_classes", lambda _target: {2: "MVIN", 3: "MVOUT", 7: "FLUSH"})
    schedule.instruction_set.cache_clear()
    try:
        isa = schedule.instruction_set(header, params, target="gemmini")
    finally:
        schedule.instruction_set.cache_clear()
    assert "gemmini_extended_mvin" in isa.instrs
    assert "gemmini_extended_mvout" in isa.instrs
    assert "gemmini_flush" in isa.instrs
    assert "gemmini_extended_mvin2" not in isa.instrs  # its funct was not in the legal table
    mvin = {operand.name: operand.kind for operand in isa.instrs["gemmini_extended_mvin"].operands}
    assert mvin == {"dram_addr": "ptr", "spad_addr": "int", "cols": "int", "rows": "int"}
    flush = {operand.name: operand.kind for operand in isa.instrs["gemmini_flush"].operands}
    assert flush == {"skip": "int"}
