"""Pure reader for the frozen complete-group candidate's stdout contract.

ELF, hardware, actual staging, rank and exit status must be checked separately.
The two carrier differences are explicitly unobserved diagnostic values; original
compiled consumer arrays and guards are checked inside the released executable.
"""

EXPECTED_STATS = (4391, 0, 20866, 29436, 1883904, 29, 1656, 282624)
EXPECTED_CARRIERS = 2
PASS_LINE = "WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS"


def _unsigned(token):
    if not token or not token.isascii() or not token.isdecimal():
        raise ValueError("unsigned decimal protocol value required")
    return int(token)


def parse_report(text, *, expected_stats=EXPECTED_STATS, expected_carriers=EXPECTED_CARRIERS):
    if not isinstance(text, str) or len(expected_stats) != 8:
        raise ValueError("complete group text and eight expected statistics required")
    cycles = []
    stats = {}
    totals = []
    differences = []
    passes = 0
    for line in text.splitlines():
        if line == PASS_LINE:
            passes += 1
            continue
        words = line.split()
        if not words:
            continue
        if words[0] == "WORKSPACE_GROUP_CYCLES":
            if len(words) != 2:
                raise ValueError("malformed cycle row")
            cycles.append(_unsigned(words[1]))
        elif words[0] == "WORKSPACE_STAT":
            if len(words) != 3:
                raise ValueError("malformed statistic row")
            index, value = map(_unsigned, words[1:])
            if index in stats or index >= len(expected_stats):
                raise ValueError("duplicate or unknown statistic")
            stats[index] = value
        elif words[0] == "UNOBSERVED_CARRIER_DIFFERENCES":
            if len(words) != 2:
                raise ValueError("malformed carrier total")
            totals.append(_unsigned(words[1]))
        elif words[0] == "UNOBSERVED_CARRIER_DIFF":
            if len(words) != 4:
                raise ValueError("malformed carrier observation")
            item = tuple(map(_unsigned, words[1:]))
            if item[1] == item[2] or any(old[0] == item[0] for old in differences):
                raise ValueError("invalid or duplicate carrier observation")
            differences.append(item)
        elif words[0].startswith(("WORKSPACE_", "UNOBSERVED_CARRIER")):
            raise ValueError("unknown or failed group protocol row")
    if len(cycles) != 1 or cycles[0] <= 0 or passes != 1:
        raise ValueError("one positive cycle count and one complete consumer PASS required")
    if stats != dict(enumerate(expected_stats)):
        raise ValueError("complete candidate statistics do not match")
    if totals != [expected_carriers] or len(differences) != expected_carriers:
        raise ValueError("carrier diagnostic closure does not match")
    return dict(cycles=cycles[0], stats=[stats[i] for i in range(8)],
                carrier_differences=[list(item) for item in differences],
                carrier_total=totals[0], original_compiled_consumer_and_guards="PASS")
