import ctypes
import subprocess

import numpy as np
import pytest

from mlir_oot.guarded_quantized_mean import derive, emit_kernel


def run_case(tmp_path, count, scales, inputs):
    proof = derive(count, *scales)
    source = emit_kernel(proof, 'candidate', len(inputs)) + f'''
#include <math.h>
void original(const int8_t *a,int8_t *b) {{
 for(int c=0;c<{len(inputs)};++c) {{
  float sum=0.0f;
  for(int k=0;k<{count};++k) {{
   float value=(float)a[c*{count}+k]*{proof['input_scale'].hex()}f;
   sum=value+sum;
  }}
  float mean=sum/{float(count).hex()}f;
  float code=nearbyintf(mean*{proof['reciprocal'].hex()}f);
  if(code<-128)code=-128;if(code>127)code=127;b[c]=(int8_t)code;
 }}
}}
'''
    path = tmp_path/'mean.c'; path.write_text(source)
    subprocess.run(['cc', '-O2', '-ffp-contract=off', '-shared', '-fPIC', str(path),
                    '-lm', '-o', str(tmp_path/'mean.so')], check=True)
    lib = ctypes.CDLL(str(tmp_path/'mean.so'))
    results = []
    for name in ('candidate', 'original'):
        output = np.full(len(inputs)+32, -77, dtype=np.int8)
        fn = getattr(lib, name); fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        fn(inputs.ctypes.data, output.ctypes.data)
        assert np.all(output[len(inputs):] == -77)
        results.append(output[:len(inputs)])
    assert np.array_equal(*results)
    return proof


@pytest.mark.parametrize('scales', [(0.5, 1.0), (1.7433754205703735, 1.3525974750518799)])
def test_all_65536_two_element_arrays(tmp_path, scales):
    a,b = np.meshgrid(np.arange(-128,128,dtype=np.int8),
                      np.arange(-128,128,dtype=np.int8),indexing='ij')
    inputs = np.stack([a.ravel(),b.ravel()],axis=1)
    run_case(tmp_path,2,scales,inputs)


def test_all_reachable_sums_and_adversarial_orders(tmp_path):
    count=49
    totals=np.arange(-128*count,127*count+1,dtype=np.int32)
    # Begin with all-128 and redistribute the sum. Interior sums deliberately
    # combine large opposing values. Reverse order stresses serial rounding.
    shifted=totals+128*count
    values=np.clip(shifted[:,None]-255*np.arange(count),0,255)-128
    inputs=np.concatenate([values,values[:,::-1]]).astype(np.int8)
    proof=run_case(tmp_path,count,(1.7433754205703735,1.3525974750518799),inputs)
    assert proof['ambiguous_sums']==[-4581,-2300,-2262,-19,19,2262,2300,4581]
    assert len(proof['table'])==255*count+1


@pytest.mark.parametrize('arguments', [(0,1.,1.),(129,1.,1.),(2,0.,1.),
    (2,float('nan'),1.),(2,1.,float('inf')),(2,1e38,1.)])
def test_unsupported_arithmetic_refuses(arguments):
    with pytest.raises(ValueError):derive(*arguments)


def test_certificate_tampering_refuses():
    proof=derive(2,.5,1.);proof['table'][0]=7
    with pytest.raises(ValueError,match='certificate changed'):emit_kernel(proof,'bad',1)
