from collections import Counter
from dataclasses import replace
import pytest
from mlir_oot.golden_conv import ConvShape,GoldenConv
from mlir_oot.golden_flat_conv import GoldenFlatConv,command_counts as flat_counts
from mlir_oot.golden_resident_conv import GoldenResidentConv,choose_compact_resident,command_counts as resident_counts,select_flat_resident_planes
from mlir_oot.conv_schedule import select_kernel
from mlir_oot.conv_family_frontier import choose_compact_frontier,footprint,ConvFootprintError
from test_resident_conv_command_loops import executed_commands

SHAPES=[ConvShape(14,14,256,256,bn=4,output_dtype='i8',scale=.001),
 ConvShape(7,7,512,512,bn=16,output_dtype='i8',scale=.001),
 ConvShape(5,7,32,67,bn=4,output_dtype='i8',scale=.01,relu=True),
 ConvShape(3,1,16,17,bn=2,output_dtype='i32'),
 ConvShape(9,5,80,19,bn=2,output_dtype='i8',scale=.001),
 ConvShape(5,5,48,33,bn=3,output_dtype='i32')]

@pytest.mark.parametrize('s',SHAPES)
def test_complete_resident_trace_matches_independent_formula_and_typed_bounds(s):
 g,o=choose_compact_resident(s)
 f=footprint(g);expected=resident_counts(s,rows_per_tile=o.rows_per_tile)
 for name in ('preload','compute','mvin_b','mvout'):
  assert f['command_counts'][name]==expected[name]
 assert f['command_counts']['mvin_a']+f['command_counts']['mvin_zero']==expected['mvin_a']
 assert f['command_counts']['fence']==2 and f['command_counts']['flush']==1
 assert f['requested_payload_bytes']['0']==s.h*s.w*s.cin
 assert f['requested_payload_bytes']['1']==9*s.cin*s.cout
 assert f['requested_payload_bytes']['2']==s.oh*s.ow*s.cout*(4 if s.output_dtype=='i32' else 1)
 assert f['unique_requested_bytes']==f['typed_storage_bytes']
 assert f['performance']=='UNKNOWN' and not f['timing_claim']
 # Constructor and caller IR are unchanged by a complete trace.
 assert executed_commands(g.build())==executed_commands(g.with_emission_options().build())

@pytest.mark.parametrize('s',SHAPES)
def test_flat_complete_counts_and_independent_DMA_traffic_formula(s):
 g,_=select_kernel(s,flat_spatial=True,virtual_padding=True)
 assert type(g) is GoldenFlatConv
 f=footprint(g);expected=flat_counts(g.conv,wide_a=g.wide_a,band_rows=g.band_rows,virtual_padding=True)
 for name in ('preload','compute','mvin_b','mvout'):
  assert f['command_counts'][name]==expected[name]
 assert f['command_counts']['mvin_a']+f['command_counts']['mvin_zero']==expected['mvin_a']
 assert f['requested_payload_bytes']['1']==expected['weight_bytes']
 assert f['unique_requested_bytes']==f['typed_storage_bytes']

@pytest.mark.parametrize('s',SHAPES)
def test_cross_family_tradeoff_retains_control_without_invented_cycle_score(s):
 c,_=select_kernel(s,flat_spatial=True,virtual_padding=True)
 c,_=select_flat_resident_planes(c)
 result,d=choose_compact_frontier(c,source_virtual_padding=True)
 assert result is c and not d['applied']
 assert d['comparison']=='UNKNOWN_TRADEOFF' and d['higher_components'] and d['lower_components']
 assert d['performance']=='UNKNOWN' and not d['timing_claim']
 assert all('score' not in x for x in (d,d['control'],d['candidate']))

@pytest.mark.parametrize('s',[
 ConvShape(14,14,256,256,stride=2,bn=4),ConvShape(7,7,32,19,stride=2,bn=2),
 ConvShape(14,15,64,17,bn=2),ConvShape(5,5,31,19,bn=2),
 ConvShape(14,14,1024,256,bn=4)])
def test_resources_stride_and_K_tail_refuse_exactly(s):
 c,_=select_kernel(s,flat_spatial=True,virtual_padding=True)
 result,d=choose_compact_frontier(c,source_virtual_padding=True)
 assert result is c and not d['applied'] and d['refusal']
 assert 'candidate' not in d

@pytest.mark.parametrize('proof',[False,True])
def test_unknown_selected_family_or_missing_source_proof_is_not_assumed(proof):
 c=GoldenConv(ConvShape(5,5,16,17,bn=2))
 result,d=choose_compact_frontier(c,source_virtual_padding=proof)
 assert result is c and not d['applied'] and d['refusal']

@pytest.mark.parametrize('keyword,value',[('source_virtual_padding',1),('prefetch_b',1),('compact_commands',1)])
def test_policy_boolean_contracts(keyword,value):
 c=GoldenFlatConv(ConvShape(3,3,16,17,bn=2),virtual_padding=True)
 with pytest.raises(ValueError):choose_compact_frontier(c,**{keyword:value})

def test_partial_CFG_is_not_a_complete_footprint():
 c=GoldenFlatConv(ConvShape(5,5,32,19,bn=2),virtual_padding=True)
 with pytest.raises(ValueError):footprint(c,max_steps=10)
 r,d=choose_compact_frontier(c,source_virtual_padding=True,max_steps=10)
 assert r is c and not d['applied'] and d['refusal'] and 'candidate' not in d

def test_packet_and_compact_options_are_compared_after_actual_emission_choices():
 c=GoldenFlatConv(ConvShape(14,14,64,67,bn=4,output_dtype='i8',scale=.01),wide_a=True,separate_b_bank=True,virtual_padding=True)
 r,d=choose_compact_frontier(c,source_virtual_padding=True,compact_commands=True,weight_issue_tiles=2)
 assert r is c and not d['applied']
 assert d['candidate_options']['weight_issue_tiles']==2 and d['candidate_options']['compact_commands']
 assert d['candidate']['complete'] and d['control']['complete']

@pytest.mark.parametrize('dtype,scale',[('i32',1.),('i8',.001)])
def test_explicit_command_repartition_emits_a_legal_unpriced_candidate(dtype,scale):
 c=GoldenFlatConv(ConvShape(14,14,256,256,bn=4,output_dtype=dtype,scale=scale),wide_a=True,separate_b_bank=True,virtual_padding=True)
 strict,baseline=choose_compact_frontier(c,source_virtual_padding=True,compact_commands=True,weight_issue_tiles=2)
 selected,d=choose_compact_frontier(c,source_virtual_padding=True,compact_commands=True,weight_issue_tiles=2,allow_command_repartition=True)
 assert strict is c and not baseline['applied'] and baseline['comparison']=='UNKNOWN_TRADEOFF'
 assert type(selected) is GoldenResidentConv and d['applied']
 assert d['comparison']=='UNPRICED_COMMAND_REPARTITION_CANDIDATE'
 assert d['performance']=='UNKNOWN' and not d['timing_claim'] and not d['automatic_policy']
 p=d['command_repartition_permission']
 assert p['candidate'] and not p['protected_higher_components']
 assert {'commands_mvin','commands_mvout','commands_preload','commands_compute'}.intersection(p['unpriced_higher_components'])
 assert p['command_count_deltas']['mvin_b']>0
 assert p['unpriced_higher_command_counts']['mvin_b']==p['command_count_deltas']['mvin_b']
 assert 'paired measurement' in p['profitability']
 assert d['control']==baseline['control'] and d['candidate']==baseline['candidate']

@pytest.mark.parametrize('s',[ConvShape(7,7,512,512,bn=16,output_dtype='i8',scale=.001),ConvShape(5,7,32,67,bn=4,output_dtype='i8',scale=.01),ConvShape(5,5,48,33,bn=3,output_dtype='i32')])
def test_command_repartition_does_not_admit_more_arithmetic_or_input_feed(s):
 c,_=select_kernel(s,flat_spatial=True,virtual_padding=True)
 c,_=select_flat_resident_planes(c)
 selected,d=choose_compact_frontier(c,source_virtual_padding=True,compact_commands=True,weight_issue_tiles=2,allow_command_repartition=True)
 assert selected is c and not d['applied']
 assert set(d['command_repartition_permission']['protected_higher_components']).intersection({'issued_MAC','A_feed_rows'})

def test_command_repartition_permission_requires_a_boolean():
 c=GoldenFlatConv(ConvShape(3,3,16,17,bn=2),virtual_padding=True)
 with pytest.raises(ValueError):choose_compact_frontier(c,allow_command_repartition=1)

@pytest.mark.parametrize('w',[1,3,7,14])
def test_independent_single_row_componentwise_dominance(w):
 s=ConvShape(1,w,32,19,bn=2,output_dtype='i8',scale=.01)
 c,_=select_kernel(s,flat_spatial=True,virtual_padding=True);c,_=select_flat_resident_planes(c)
 selected,d=choose_compact_frontier(c,source_virtual_padding=True)
 assert selected is not c and type(selected) is GoldenResidentConv and d['applied']
 assert d['comparison']=='STRUCTURAL_DOMINANCE' and not d['higher_components']
 assert d['control']['typed_storage_bytes']==d['candidate']['typed_storage_bytes']
 assert d['performance']=='UNKNOWN' and not d['automatic_policy']

@pytest.mark.parametrize('kwargs',[
 {'resident_input_policy':'compact_channel_planes_frontier'},
 {'resident_input_policy':'compact_channel_planes_frontier','flat_spatial':True,'virtual_padding':True,'resident_input_regions':('forbidden_identity',)},
])
def test_normal_bundle_option_validates_source_evidence_before_IO(tmp_path,kwargs):
 from mlir_oot.captured_requant_bundle import build
 with pytest.raises(ValueError):build(tmp_path/'absent',tmp_path/'tools',tmp_path/'out',**kwargs)
 assert not (tmp_path/'out').exists()
