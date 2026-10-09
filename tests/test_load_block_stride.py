import pytest
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.utils.exceptions import VerifyException
from mlir_oot.ir.gemmini_dialect import ConfigLdOp
from mlir_oot.golden_device_lower import _encoded


def config(**attrs):
 op=ConfigLdOp(operands=[[]],result_types=[[]])
 op.attributes.update({k:IntegerAttr(v,i64) for k,v in attrs.items()})
 return op


def test_optional_load_layout_survives_encoding_and_defaults_stay():
 # Regression: explicit block_stride used to be silently discarded.
 op=config(stride=256,load_id=0,block_stride=256,pixel_repeats=3,shrunk=1)
 op.verify();_,rs1,rs2=_encoded(op)
 assert (rs1>>16)&0xffff==256 and (rs1>>8)&255==3
 assert (rs1>>2)&1==1 and rs2==256
 _,default,_=_encoded(config(stride=256,load_id=0))
 assert (default>>16)&0xffff==16 and (default>>8)&255==1


@pytest.mark.parametrize('field,value',[('block_stride',0),('block_stride',65536),('pixel_repeats',0),('pixel_repeats',256)])
def test_load_layout_fields_cannot_truncate(field,value):
 with pytest.raises(VerifyException):config(stride=64,load_id=0,**{field:value}).verify()

