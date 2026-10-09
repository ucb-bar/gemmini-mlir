"""Complete allocation/options survive independent residency policy ordering."""
from dataclasses import replace
import pytest
from mlir_oot.golden_gemm import GoldenGemm,Shape
from mlir_oot.dense_schedule import select_coalesced_resident_a,select_resident_a_output_blocks
from test_resident_a_output_blocks import source_tiles

@pytest.mark.parametrize('channels',[2,4])
@pytest.mark.parametrize('prefetch',[False,True])
@pytest.mark.parametrize('m,n,k',[(289,67,20),(529,73,65)])
def test_coalescing_commutes_with_output_block_selection_and_closes_all_source_cells(channels,prefetch,m,n,k):
 shape=Shape(m,n,k,bm=(m+15)//16,bn=1,cache_a=True,reuse_b=True,
             wide_b=True,output_dtype='i32',prefetch_b=prefetch)
 original=GoldenGemm(shape,prefetch_b_rows=(4096,8192) if prefetch else None)
 blocked,block_decision=select_resident_a_output_blocks(original,output_channel_tiles=channels)
 assert block_decision['applied'] and blocked.cached_a_output_blocks
 after,load_decision=select_coalesced_resident_a(blocked)
 assert load_decision['applied'] and after.cached_a_output_blocks and after.resident_a_load_tiles==4
 assert after.prefetch_b_rows==original.prefetch_b_rows
 assert load_decision['input_reserved_rows']==((m+15)//16)*((k+15)//16)*16
 if after.shape.bm<(m+15)//16:
  assert load_decision['input_reserved_rows']>after.shape.bm*((k+15)//16)*16
 first,first_load=select_coalesced_resident_a(original)
 before,before_block=select_resident_a_output_blocks(first,output_channel_tiles=channels)
 assert first_load['applied'] and before_block['applied']
 assert after.emission_options==before.emission_options and after.shape==before.shape
 assert str(after.with_emission_options().build())==str(before.with_emission_options().build())
 old,old_counts,old_bytes,old_config=source_tiles(blocked.with_emission_options())
 new,new_counts,new_bytes,new_config=source_tiles(after.with_emission_options())
 assert old==new and old_bytes==new_bytes and old_config==new_config
 assert old_counts['gemmini.compute']==new_counts['gemmini.compute']
 assert old_counts['gemmini.mvout']==new_counts['gemmini.mvout']
 assert old_counts['gemmini.mvin']-new_counts['gemmini.mvin']==load_decision['control_input_dma_commands']-load_decision['candidate_input_dma_commands']
 kept,again=select_coalesced_resident_a(after)
 assert kept is after and not again['applied']


def test_capacity_admitted_cached_B_family_declines_coalescing_without_losing_its_flags():
 shape=Shape(3,128,512,bm=1,bn=8,cache_b=True)
 original=GoldenGemm(shape,cached_b_resource_capacity=True)
 before=str(original.with_emission_options().build())
 kept,decision=select_coalesced_resident_a(original)
 assert kept is original and not decision['applied']
 assert decision['refusal']=='requires the complete resident A layout'
 assert str(kept.with_emission_options().build())==before


def test_plain_default_paths_and_illegal_resource_mutation_still_refuse():
 original=GoldenGemm(Shape(17,35,20,bm=2,bn=1,cache_a=True,reuse_b=True))
 before=str(original.with_emission_options().build())
 candidate,decision=select_coalesced_resident_a(original)
 assert decision['applied'] and candidate.resident_a_load_tiles==4
 assert str(original.with_emission_options().build())==before and not candidate.cached_a_output_blocks
 original.shape=replace(original.shape,m=8193)
 with pytest.raises(ValueError):select_coalesced_resident_a(original)
