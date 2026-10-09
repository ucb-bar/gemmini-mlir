from dataclasses import replace
import pytest
from mlir_oot.golden_conv import ConvShape
from mlir_oot.golden_resident_conv import GoldenResidentConv

def test_resident_resources_and_nonoverlap_admission():
 s=ConvShape(14,14,256,256,bn=4,output_dtype='i8',scale=.0012096002465113997,relu=True)
 g=GoldenResidentConv(s);assert g.plane==256 and g.bbase==8192
 # First4096 scratch rows hold16channel planes; weights start8192.
 for bad in [replace(s,w=15),replace(s,stride=2),replace(s,cin=255),replace(s,cin=1024),replace(s,bn=8),replace(s,explicit_halo=True)]:
  with pytest.raises(ValueError):GoldenResidentConv(bad)
 # Smaller odd widths/partial output channels are legal and fully verifiable.
 GoldenResidentConv(replace(s,h=3,w=5,cin=32,cout=19,bn=2)).build().verify()

def test_grouped_resident_rows_admit_final_stage_without_extra_mesh_tiles():
 from mlir_oot.golden_resident_conv import command_counts
 s=ConvShape(7,7,512,512,bn=16,output_dtype='i8',scale=.0015428082551807165,relu=True)
 # Two width7 rows are separated by the two halo lanes in a width9 plane.
 # Four mesh tiles cover49 valid output rows; the final tile has one row.
 g=GoldenResidentConv(s,rows_per_tile=2)
 assert g.row_tiles==((0,2,16),(2,2,16),(4,2,16),(6,1,7))
 g.build().verify()
 counts=command_counts(s,rows_per_tile=2)
 assert counts['padded_array_issue_cycles']==9*32*32*4*16
 assert counts['accumulator_rows']==1024 and counts['resident_input_rows']==2592
 # The old one-output-row schedule cannot hold this N panel in its accumulator.
 with pytest.raises(ValueError):GoldenResidentConv(s)

@pytest.mark.parametrize('rows',[0,3,True,1.5])
def test_grouped_resident_rows_refuse_unencodable_or_untyped_group(rows):
 s=ConvShape(7,7,512,512,bn=4)
 with pytest.raises(ValueError):GoldenResidentConv(s,rows_per_tile=rows)

def test_channel_loop_proves_tail_reads_within_resident_input():
 from mlir_oot.ir.gemmini_dialect import ComputeOp
 from mlir_oot.golden_device_lower import lower
 s=ConvShape(5,5,32,19,bn=2,output_dtype='i32')
 module=GoldenResidentConv(s,rows_per_tile=2,loop_channels=True).build()
 dynamic=[op for op in module.walk() if isinstance(op,ComputeOp) and len(op.operands_)==1]
 assert dynamic and all(op.a('a_max')+op.a('a_rows')<=op.a('a_reserved_rows') for op in dynamic)
 assert any(op.a('a_max')+op.a('a_rows')==98 for op in dynamic)
 lower(module).verify()

def test_resident_options_refuse_unselected_or_untyped_settings_before_output(tmp_path):
 from mlir_oot.captured_requant_bundle import build
 from mlir_oot.golden_resident_conv import ResidentConvOptions
 for options in ({'absent':ResidentConvOptions(loop_channels=True)}, {'r':{'loop_channels':True}}):
  with pytest.raises(ValueError,match='resident options'):
   build(tmp_path/'no_capture',tmp_path/'no_tools',tmp_path/'output',
       resident_input_regions=('r',),resident_input_options=options)
  assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('shape,rows', [
 (ConvShape(14,14,256,256,bn=4),1),
 (ConvShape(7,7,512,512,bn=16),2),
 (ConvShape(5,5,32,19,bn=2,output_dtype='i32'),2),
 (ConvShape(3,1,16,17,bn=2,output_dtype='i32'),3),
])
def test_compiler_policy_derives_rows_without_source_identity(shape,rows):
 from mlir_oot.golden_resident_conv import choose_compact_resident
 generator,options=choose_compact_resident(shape)
 assert options.rows_per_tile==rows and options.loop_channels
 generator.build().verify()
 # Including every valid row exactly once is independent of halo/mesh lanes.
 assert [y+r for y,count,_ in generator.row_tiles for r in range(count)]==list(range(shape.h))


@pytest.mark.parametrize('shape', [
 ConvShape(14,15,256,256,bn=4),
 ConvShape(14,14,256,256,stride=2,bn=4),
 ConvShape(14,14,1024,256,bn=4),
 ConvShape(14,14,256,256,bn=8),
])
def test_compiler_policy_refuses_unsupported_shape_or_resources(shape):
 from mlir_oot.golden_resident_conv import choose_compact_resident
 with pytest.raises(ValueError):choose_compact_resident(shape)


def test_compiler_policy_requires_source_padding_proof_and_no_id_selection(tmp_path):
 from mlir_oot.captured_requant_bundle import build
 for kwargs in (
  {'resident_input_policy':'unknown'},
  {'resident_input_policy':'compact_channel_planes'},
  {'resident_input_policy':'compact_channel_planes','flat_spatial':True,
   'virtual_padding':True,'resident_input_regions':('r',)},
 ):
  with pytest.raises(ValueError):build(tmp_path/'no_capture',tmp_path/'no_tools',tmp_path/'output',**kwargs)
  assert not (tmp_path/'output').exists()
