import ctypes
import subprocess
import numpy as np
import pytest
from mlir_oot.golden_wide_resadd import build,tables
from mlir_oot.captured_residual_bundle import source_table,wide_oracle,wide_adapter,binding_attributes

@pytest.mark.parametrize('p,q',[(1,1),(127,127),(128,254),(2609,2180),(32767,32767)])
def test_chunk_kernel_verifies_for_coefficient_edges(p,q):
    module=build(16,p,q,1./65536);module.verify()
    table=np.frombuffer(tables(p,q),np.int8).reshape(3,16,16)
    for t,v in zip(table,(127,p%127,q%127)):assert np.array_equal(t,np.eye(16,dtype=np.int8)*v)

@pytest.mark.parametrize('args',[(0,1,1,1.),(15,1,1,1.),(16,0,1,1.),(16,32768,1,1.),(16,1,1,float('nan'))])
def test_kernel_refuses_unsupported_contract(args):
    with pytest.raises(ValueError):build(*args)


def test_wide_native_oracle_matches_full_original_source_domain(tmp_path):
    q=dict(lhs_scale=.011258588172495365,rhs_scale=.00940733402967453,output_scale=.011643771082162857,relu=True)
    coeff=dict(p=2609,q=2180,scale=.00037060913746245205)
    route={'m':1024,'proof':{'source':q,'coefficients':coeff,'primitive':{'readout':coeff['scale']}},'numeric_policy':{'kind':'exact_source','max_output_lsb':0},'physical_layout':{'test':'binding'}}
    attrs=binding_attributes(route,'a'*64,'b'*64)
    assert attrs['gemmini.wide_integer_coefficients'].data['p'].value.data==2609
    assert 'gemmini.residual_layout' in attrs
    (tmp_path/'oracle.c').write_text(wide_adapter(route,'adapter','kernel')+wide_oracle(route,'kernel'))
    subprocess.run(['cc','-O2','-ffp-contract=off','-shared','-fPIC',str(tmp_path/'oracle.c'),'-lm','-o',str(tmp_path/'oracle.so')],check=True)
    lib=ctypes.CDLL(str(tmp_path/'oracle.so'));fn=lib.kernel;fn.argtypes=[ctypes.c_void_p]*4
    a=np.repeat(np.arange(-128,128,dtype=np.int8),256);b=np.tile(np.arange(-128,128,dtype=np.int8),256);out=np.full(65552,-77,dtype=np.int8)
    fn(a.ctypes.data,b.ctypes.data,out.ctypes.data,None)
    assert np.array_equal(out[:65536],source_table(q).ravel())
    assert np.all(out[65536:]==-77)
