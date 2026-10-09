"""Independent strict protocol and unknown-attribution cases."""

from copy import deepcopy

import pytest
from paired_pointwise_report import parse_report


def fixture():
    pointers = {
        "a": "80000000",
        "out": "81000000",
        "arena": "82000000",
        "table": "83000000",
    }
    initial = [
        {"id": str(i), "digest": "123", "issues": "0" if not i else "11"}
        for i in range(3)
    ]
    rows = [
        {
            "id": str(i),
            "sample": str(n),
            "cycles": str(100 + n),
            "instructions": str(80 + i),
            "digest": "123",
            "issues": "0" if not i else "11",
        }
        for n, i in enumerate((0, 1, 2, 2, 1, 0))
    ]
    expected = {
        "pointers": pointers,
        "initial": initial,
        "strict_rows": rows,
        "scope": "independent complete fixture",
    }
    text = "\n".join(
        [
            "PAIRED_SOURCE_GATE original45056i8 all5frm sticky7 guards sourcecontinuation",
            *[
                "PAIRED_INITIAL " + " ".join(f"{k}={v}" for k, v in row.items())
                for row in initial
            ],
            "PAIRED_POINTERS " + " ".join(f"{k}={v}" for k, v in pointers.items()),
            *[
                "PAIRED_ROW " + " ".join(f"{k}={v}" for k, v in row.items())
                for row in rows
            ],
            "PAIRED_PASS original90112i32 original45056i8 allguards immutableinputs rank0 DONE",
        ]
    )
    return text, expected


def test_cycles_are_labels_but_instructions_and_complete_abi_stay_pinned():
    text, expected = fixture()
    actual = text.replace("cycles=102", "cycles=202")
    report = parse_report(actual, expected)
    assert report["mean_cycles"]["original_full"] == 102.5
    assert report["mean_cycles"]["overlap_tiles"] == 152.5
    assert report["whole_cycles"] == "UNKNOWN"
    assert report["pure_array_or_physical_memory_attribution"] == "UNKNOWN"


@pytest.mark.parametrize(
    "mutation",
    ["pointer", "digest", "instructions", "extra", "missing", "fail", "gate", "zero"],
)
def test_refuse_storage_or_source_or_complete_window_mutation(mutation):
    text, expected = fixture()
    if mutation == "pointer":
        text = text.replace("arena=82000000", "arena=82000040")
    elif mutation == "digest":
        text = text.replace("digest=123", "digest=124", 1)
    elif mutation == "instructions":
        text = text.replace("instructions=82", "instructions=83", 1)
    elif mutation == "extra":
        text += "\n" + next(
            line for line in text.splitlines() if line.startswith("PAIRED_ROW")
        )
    elif mutation == "missing":
        text = "\n".join(line for line in text.splitlines() if "sample=2 " not in line)
    elif mutation == "fail":
        text += "\nPAIRED_FAIL lateguard"
    elif mutation == "gate":
        text = text.replace("all5frm", "all4frm")
    else:
        text = text.replace("cycles=102", "cycles=0")
    with pytest.raises(ValueError):
        parse_report(text, deepcopy(expected))
