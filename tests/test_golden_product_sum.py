import pytest
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_product_sum import GoldenProductSum


def test_tail_module_and_explicit_domain():
    m=GoldenProductSum(Shape(17,19,65,'i32',wide_b=True,reuse_b=True),pairs=((0,2),(1,1),(2,0)),lhs_planes=3,rhs_planes=3,absolute_bound=3*65*127**2).build()
    m.verify()

@pytest.mark.parametrize('kwargs',[
 {'pairs':((3,0),)}, {'absolute_bound':1}, {'lhs_magnitude_bound':129},
 {'absolute_bound':1<<31}, {'pairs':()},
])
def test_refuse_invalid_spans_or_numeric_contract(kwargs):
    values=dict(pairs=((0,0),),lhs_planes=1,rhs_planes=1,absolute_bound=65*127**2)
    values.update(kwargs)
    with pytest.raises(ValueError):GoldenProductSum(Shape(17,19,65,'i32'),**values)
