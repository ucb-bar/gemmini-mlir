from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_product_sum import GoldenProductSum
from mlir_oot.golden_zero_product_sum import GoldenZeroProductSum
from mlir_oot.golden_device_lower import lower
from mlir_oot.golden_zero_product_sum import command_census
from merlin.llvmlower.encoded_i8_zeros import EncodedI8Nonzero
import pytest


def test_complete_tail_ir_and_explicit_summary_abi():
    args=dict(pairs=((0,1),(1,0)),lhs_planes=2,rhs_planes=2,absolute_bound=2*65*128**2)
    s=Shape(17,19,65,'i32',bm=2,bn=2,reuse_b=True,wide_b=True)
    dense=str(GoldenProductSum(s,**args).build())
    module=GoldenZeroProductSum(s,**args).build();module.verify()
    fn=next(iter(module.ops))
    assert len(fn.body.blocks[0].args)==5
    text=str(module)
    assert 'gemmini.encoded_i8_nonzero' in text
    assert 'llvm.cond_br' in text
    assert 'load_id = 2' in text
    lower(module).verify()
    assert str(GoldenProductSum(s,**args).build())==dense


def test_census_preserves_allzero_output_ownership_and_refuses_mismatch():
    s=Shape(17,19,65,'i32',bm=2,bn=2,reuse_b=True,wide_b=True)
    a=EncodedI8Nonzero(2,17,65,16,(False,)*4)
    b=EncodedI8Nonzero(2,19,65,16,(False,)*4)
    counts=command_census(s,((0,1),(1,0)),a,b)
    assert counts['mvout']==4 and counts['mvin_zero_acc']==4
    assert counts.get('compute',0)==0 and counts.get('mvin_b',0)==0
    assert counts['omitted_pair_blocks']==2
    with pytest.raises(ValueError):command_census(s,((0,0),),EncodedI8Nonzero(1,17,64,16,(True,True)),b)
