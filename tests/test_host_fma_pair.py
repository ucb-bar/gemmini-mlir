import pytest
from mlir_oot.host_fma_pair import SourceFmaPairCapability


def test_explicit_capability_and_early_clobber():
    header=SourceFmaPairCapability('rv64gc','lp64d',True,True,True,True).header()
    assert header.count('fmadd.s')==2
    assert ': "=&f"(l),"=&f"(h)' in header
    assert 'csr' not in header


@pytest.mark.parametrize('position,value',[(0,'rv32gc'),(1,'lp64'),(2,False),(3,False),(4,False),(5,False),(2,1)])
def test_unknown_contract_refused(position,value):
    args=['rv64gc','lp64d',True,True,True,True];args[position]=value
    with pytest.raises(ValueError):SourceFmaPairCapability(*args).header()
