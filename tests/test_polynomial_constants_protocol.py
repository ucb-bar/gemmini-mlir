from pathlib import Path
import pytest
from experiments.attention_projection_frontier.polynomial_constants_protocol import parse_report,EXPECTED_STATS,PASS_LINE


def valid():
    return '\n'.join(['WORKSPACE_GROUP_CYCLES 123',
        'UNOBSERVED_CARRIER_DIFF 2 100 101','UNOBSERVED_CARRIER_DIFF 8 91 92',
        'UNOBSERVED_CARRIER_DIFFERENCES 2',
        *(f'WORKSPACE_STAT {i} {n}' for i,n in enumerate(EXPECTED_STATS)),PASS_LINE])


def test_reads_complete_protocol_without_external_actions():
    r=parse_report(valid());assert r['cycles']==123 and r['stats']==list(EXPECTED_STATS)


@pytest.mark.parametrize('mutate',[
 lambda s:s+'\nWORKSPACE_GROUP_CYCLES 9',
 lambda s:s+'\n'+PASS_LINE,
 lambda s:s.replace('WORKSPACE_STAT 0 4391','WORKSPACE_STAT 0 0'),
 lambda s:s.replace('UNOBSERVED_CARRIER_DIFFERENCES 2','UNOBSERVED_CARRIER_DIFFERENCES 3'),
 lambda s:s.replace('WORKSPACE_GROUP_CYCLES','WORKSPACE_GROUP_INSTRUCTIONS'),
 lambda s:s.replace('WORKSPACE_GROUP_CYCLES 123','WORKSPACE_GROUP_CYCLES -1'),
 lambda s:s.replace(PASS_LINE,'WORKSPACE_GROUP FAIL'),
 lambda s:s.replace('UNOBSERVED_CARRIER_DIFF 2 100 101',''),
 lambda s:s+'\nWORKSPACE_STAT 0 4391',
])
def test_refuses_partial_changed_or_duplicate_protocol(mutate):
    with pytest.raises(ValueError):parse_report(mutate(valid()))


def test_frozen_actual_strict_candidate():
    source=Path(__file__).resolve().parent/'fixtures/prepared_polynomial_constants_candidate.stdout'
    assert parse_report(source.read_text())['cycles']==1560212855
