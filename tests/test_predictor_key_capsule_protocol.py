import pytest
from predictor_key_capsule_protocol import parse


def valid():
    return (
        "RECTIFIED_RANKED_COUNTER arm=1 cycles=100 instructions=42 before=0 after=0\n"
        "RECTIFIED_RANKED_PASS arm1 all802816 guards4096 inputs1605632 descriptors flags\n"
    )


def test_complete_original_protocol_and_crlf():
    assert parse(valid().replace("\n", "\r\n"), arm=1, elements=802816)["passed"]


@pytest.mark.parametrize(
    "text",
    [
        valid().splitlines()[0],
        valid() + valid(),
        valid().replace("guards4096", "guards128"),
        valid().replace("after=0", "after=1"),
        valid().replace("cycles=100", "cycles=0"),
        valid() + "RECTIFIED_RANKED_FAIL timed\n",
    ],
)
def test_partial_duplicate_wrong_guard_flags_zero_window_and_failure_refuse(text):
    with pytest.raises(ValueError):
        parse(text, arm=1, elements=802816)


def test_wrong_arm_and_shape_refuse():
    with pytest.raises(ValueError):
        parse(valid(), arm=0, elements=802816)
    with pytest.raises(ValueError):
        parse(valid(), arm=1, elements=65536)
