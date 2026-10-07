"""Strict protocol for an explicitly bound complete scalar-prefix experiment."""

import re


def digest(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def parse(
    console,
    *,
    arm,
    input_bytes,
    quant_elements,
    padded_elements,
    guard_bytes,
    expected_digest,
):
    pattern = (
        r"^PREFIX_COUNTER arm=(\d+) repeat=(\d+) cycles=(\d+) instructions=(\d+) "
        r"frm=(\d+) flags=(\d+) input=(\d+) output=(\d+) elements=(\d+) padded=(\d+) digest=([0-9a-f]{16})$"
    )
    rows = []
    for match in re.finditer(pattern, console.replace("\r", ""), re.MULTILINE):
        values = match.groups()
        rows.append(
            dict(
                zip(
                    (
                        "arm",
                        "repeat",
                        "cycles",
                        "instructions",
                        "frm",
                        "flags",
                        "input",
                        "output",
                        "elements",
                        "padded",
                        "digest",
                    ),
                    [*(int(x) for x in values[:-1]), values[-1]],
                    strict=True,
                )
            )
        )
    if len(rows) != 2 or [row["repeat"] for row in rows] != [0, 1]:
        raise ValueError("complete ordered counter windows missing")
    for row in rows:
        if (
            row["arm"] != arm
            or row["frm"] != 0
            or not 0 <= row["flags"] <= 31
            or row["elements"] != quant_elements
            or row["padded"] != padded_elements
            or row["digest"] != expected_digest
            or row["cycles"] <= 0
            or row["instructions"] <= 0
        ):
            raise ValueError("counter source/arm/numeric binding differs")
        if (
            row["input"] <= 0
            or row["output"] <= 0
            or not (
                row["input"] + input_bytes <= row["output"]
                or row["output"] + padded_elements <= row["input"]
            )
        ):
            raise ValueError(
                "declared immutable input/private output address intervals overlap"
            )
    if len({(row["input"], row["output"]) for row in rows}) != 1:
        raise ValueError("common addresses changed across repeats")
    modes = []
    for match in re.finditer(
        r"^PREFIX_MODE frm=(\d+) sticky=(\d+) control_flags=(\d+) candidate_flags=(\d+) exact=(\d+)$",
        console.replace("\r", ""),
        re.MULTILINE,
    ):
        mode, sticky, left, right, count = map(int, match.groups())
        if (
            count != padded_elements
            or not 0 <= left <= 31
            or not 0 <= right <= 31
            or (sticky and (left != 31 or right != 31))
        ):
            raise ValueError("mode/output/sticky contract differs")
        modes.append((mode, sticky, left, right))
    if [(row[0], row[1]) for row in modes] != [
        (mode, sticky) for mode in range(5) for sticky in range(2)
    ]:
        raise ValueError("complete mode coverage missing")
    marker = f"PREFIX_PASS quant={quant_elements} padded={padded_elements} guards={guard_bytes} immutable={input_bytes} modes=10"
    if console.replace("\r", "").splitlines().count(marker) != 1:
        raise ValueError("complete numerical/input/guard marker missing")
    return {
        "rows": rows,
        "modes": modes,
        "effect_scope": "flags recorded; equality is evidence only, existing flags-unobservable policy remains required",
    }
