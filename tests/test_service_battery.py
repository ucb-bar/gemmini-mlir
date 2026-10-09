"""Independent directed checks of the service-report admission gate."""

import pytest

from mlir_oot.service_battery import parse_service_report

MANIFEST = {
    "cases": [
        {"id": 0, "family": "div", "expected_words": [0x3F800000], "expected_fflags": 1}
    ],
    "repetitions": 1,
    "empty_windows": 1,
    "expected_checksum": "0123456789abcdef",
}
REPORT = """SERVICE_ROW id=-1 repeat=0 cycles=10 instructions=4 frm=0 flags_before=0 flags_after=0
SERVICE_FP id=0 repeat=0 words=3f800000
SERVICE_ROW id=0 repeat=0 cycles=100 instructions=20 frm=0 flags_before=0 flags_after=1
SERVICE_PASS cases=1 repeats=1 checksum=0123456789abcdef
"""


def test_complete_report():
    assert parse_service_report(REPORT, MANIFEST)["complete"]


@pytest.mark.parametrize(
    "changed",
    [
        REPORT.rsplit("SERVICE_PASS", 1)[0],
        REPORT + REPORT,
        REPORT.replace("3f800000", "3f800001"),
        REPORT.replace("flags_after=1", "flags_after=0"),
        REPORT.replace("checksum=0123456789abcdef", "checksum=0123456789abcdee"),
        REPORT.replace("cycles=100", "cycles=0"),
    ],
)
def test_reject_incomplete_or_wrong_report(changed):
    with pytest.raises(ValueError):
        parse_service_report(changed, MANIFEST)
