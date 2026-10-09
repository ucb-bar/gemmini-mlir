"""Validate the original complete Smol source-consumer capsule protocol."""
import re

STATS = [4391, 0, 42210, 63537, 4066368, 63, 1848, 315392]
DIFFERENCES = [(24151, 48661, 48660), (34746, 15546, 15547), (104653, 16009, 16008)]

def parse(text):
    text = text.replace('\r', '')
    assert not re.search(r'FAIL|REFUSAL|TRAP|mismatch|assert', text, re.I)
    cycles = re.findall(r'^WORKSPACE_GROUP_CYCLES ([0-9]+)$', text, re.M)
    assert len(cycles) == 1 and int(cycles[0]) > 0
    stats = [(int(a), int(b)) for a, b in re.findall(r'^WORKSPACE_STAT ([0-9]+) ([0-9]+)$', text, re.M)]
    assert stats == list(enumerate(STATS)), stats
    differences = [tuple(map(int, row)) for row in re.findall(r'^UNOBSERVED_CARRIER_DIFF ([0-9]+) ([0-9]+) ([0-9]+)$', text, re.M)]
    assert differences == DIFFERENCES, differences
    assert re.findall(r'^UNOBSERVED_CARRIER_DIFFERENCES ([0-9]+)$', text, re.M) == ['3']
    assert text.count('WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS') == 1
    return dict(cycles=int(cycles[0]), stats=stats, carrier_differences=differences,
                original_compiled_consumer_and_guards_pass=True)
