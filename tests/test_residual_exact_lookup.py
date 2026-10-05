import ctypes
import subprocess
import numpy as np
from mlir_oot.captured_residual_bundle import lookup_kernel, source_table


def test_lookup_full_domain_against_independent_compiled_source_chain(tmp_path):
    q=dict(lhs_scale=.010870203375816345,rhs_scale=.0059822131879627705,output_scale=.01697084680199623,relu=True)
    route={'m':1024,'proof':{'source':q}}
    source=lookup_kernel(route,'lookup')+f'''
#include <math.h>
void reference(const int8_t*a,const int8_t*b,int8_t*c) {{
 for(int i=0;i<65536;i++) {{
 volatile float x=(float)a[i]*{float(q['lhs_scale']).hex()}f;
 volatile float y=(float)b[i]*{float(q['rhs_scale']).hex()}f;
 volatile float z=x+y;if(z<0)z=0;
 volatile float v=z*{float(np.float32(1./q['output_scale'])).hex()}f;
 float r=nearbyintf(v);if(r<0)r=0;if(r>127)r=127;c[i]=(int8_t)r;
 }}
}}
'''
    (tmp_path/'test.c').write_text(source)
    subprocess.run(['cc','-O2','-ffp-contract=off','-fPIC','-shared',str(tmp_path/'test.c'),'-lm','-o',str(tmp_path/'test.so')],check=True)
    lib=ctypes.CDLL(str(tmp_path/'test.so'));lib.lookup.argtypes=[ctypes.c_void_p]*5;lib.reference.argtypes=[ctypes.c_void_p]*3
    a=np.repeat(np.arange(-128,128,dtype=np.int8),256);b=np.tile(np.arange(-128,128,dtype=np.int8),256)
    actual=np.full(65552,-77,dtype=np.int8);expected=np.empty(65536,dtype=np.int8)
    lib.lookup(a.ctypes.data,b.ctypes.data,actual.ctypes.data,None,None)
    lib.reference(a.ctypes.data,b.ctypes.data,expected.ctypes.data)
    assert np.array_equal(actual[:65536],expected)
    assert np.all(actual[65536:]==-77)
    assert np.array_equal(source_table(q).ravel(),expected)


def test_lookup_rewrite_is_exact_for_separately_rounded_load_counterexample():
    from pathlib import Path
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.golden_resadd_proof import op_name
    from mlir_oot.captured_residual_bundle import inspect,rewrite,serialize
    import pytest
    text=Path(__file__).with_name('fixtures').joinpath('captured_resadd.mlir').read_text()
    text=text.replace('%scale = tensor.splat %one : tensor<f32>','%scale = tensor.splat %one : tensor<f32>\n    %half = arith.constant 0.5 : f32\n    %input_scale = tensor.splat %half : tensor<f32>')
    text=text.replace('(%a, %scale, %zp)','(%a, %input_scale, %zp)').replace('(%b, %scale, %zp)','(%b, %input_scale, %zp)')
    module=parse_module(text);q=next(x for x in module.walk() if op_name(x)=='quant_ext.quantize_per_tensor')
    with pytest.raises(ValueError):inspect(q)
    route=inspect(q,implementation='cpu_lut')
    assert route['proof']['exact'] and route['proof']['table_sha256']
    assert route['proof']['primitive']=={}
    declaration=rewrite(q,route,'exact_lut','a'*64,'b'*64)
    parsed=parse_module(serialize(module,[declaration]));parsed.verify()
    assert route['numeric_policy']['kind']=='exact_source'
    with pytest.raises(ValueError,match='exact policy'):inspect(q,1,implementation='cpu_lut')
