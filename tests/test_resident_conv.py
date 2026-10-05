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
