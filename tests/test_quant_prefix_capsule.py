import pytest

from mlir_oot.quant_prefix_capsule import parse


def protocol():
    rows = [
        f"PREFIX_COUNTER arm=1 repeat={repeat} cycles=30 instructions=20 frm=0 flags=1 input=100 output=1000 elements=4 padded=8 digest=1111111111111111"
        for repeat in range(2)
    ]
    rows += [
        f"PREFIX_MODE frm={mode} sticky={sticky} control_flags={31 if sticky else 1} candidate_flags={31 if sticky else 0} exact=8"
        for mode in range(5)
        for sticky in range(2)
    ]
    rows += ["PREFIX_PASS quant=4 padded=8 guards=16 immutable=16 modes=10"]
    return "\n".join(rows)


def read(text):
    return parse(
        text,
        arm=1,
        input_bytes=16,
        quant_elements=4,
        padded_elements=8,
        guard_bytes=16,
        expected_digest="1111111111111111",
    )


def test_protocol_flags_observation_is_not_an_effect_permission():
    result = read(protocol())
    assert len(result["rows"]) == 2 and len(result["modes"]) == 10
    assert result["modes"][0][2:] == (1, 0)


@pytest.mark.parametrize(
    "change", ["truncated", "arm", "digest", "overlap", "mode", "marker", "flags"]
)
def test_wrong_or_incomplete_source_protocol_refuses(change):
    value = protocol()
    if change == "truncated":
        value = value.splitlines()[0]
    elif change == "arm":
        value = value.replace("arm=1", "arm=0")
    elif change == "digest":
        value = value.replace("digest=1111111111111111", "digest=0000000000000000")
    elif change == "overlap":
        value = value.replace("output=1000", "output=104")
    elif change == "mode":
        value = value.replace("frm=4 sticky=1", "frm=3 sticky=1")
    elif change == "marker":
        value = value.replace("PREFIX_PASS", "MISSING")
    else:
        value = value.replace("flags=1", "flags=32")
    with pytest.raises(ValueError):
        read(value)
