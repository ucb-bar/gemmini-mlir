import ctypes
import subprocess
import numpy as np
import pytest
from mlir_oot.exact_integer_readout import derive,evaluate,fixedpoint_candidate,emit_readout,emit_conv_wrapper
from mlir_oot.golden_requant import quantized

@pytest.mark.parametrize('scales,relu', [([.5,.5],False),([.1525018811225891,.0011392629239708185,4.235067367553711],True),([1.7959643602371216,.00084189377957955,.45207950472831726],True)])
def test_integer_readout_against_compiled_original_source(tmp_path,scales,relu):
    proof=derive(scales,relu=relu)
    candidate=fixedpoint_candidate(proof);assert candidate and candidate['max_estimate_output_error']<=1
    source=emit_readout(proof,'binary')+emit_readout(proof,'fixed',fixedpoint=True)
    body='volatile float value=(float)x;'
    for s in scales:body+=f'value=value*{float(s).hex()}f;'
    low=0 if relu else -128
    source+=f'''\n#include <math.h>
void original(const int32_t*a,int8_t*b,size_t n){{for(size_t i=0;i<n;i++){{int32_t x=a[i];{body}
float y=nearbyintf(value);if(y<{low})y={low};if(y>127)y=127;b[i]=(int8_t)y;}}}}
'''
    (tmp_path/'readout.c').write_text(source)
    subprocess.run(['cc','-O2','-ffp-contract=off','-fPIC','-shared',str(tmp_path/'readout.c'),'-lm','-o',str(tmp_path/'readout.so')],check=True)
    boundaries={-(1<<31),(1<<31)-1,0}
    for t in proof['thresholds']:
        boundaries.update(x for x in (t-1,t,t+1) if -(1<<31)<=x<(1<<31))
    # Exhaust all unsaturated transition intervals, plus full i32 extremes.
    interior=np.arange(max(-(1<<31),proof['thresholds'][0]-2),min((1<<31),proof['thresholds'][-1]+2),dtype=np.int32)
    a=np.unique(np.concatenate([interior,np.array(sorted(boundaries),dtype=np.int32)]))
    lib=ctypes.CDLL(str(tmp_path/'readout.so'));outputs=[]
    for name in ('binary','fixed','original'):
        fn=getattr(lib,name);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        output=np.full(a.size+8,-77,dtype=np.int8);fn(a.ctypes.data,output.ctypes.data,a.size)
        assert np.all(output[a.size:]==-77);outputs.append(output[:a.size])
    assert np.array_equal(outputs[0],outputs[2]);assert np.array_equal(outputs[1],outputs[2])


def test_narrow_domain_repeated_sentinel_and_proof_tampering():
    p=derive([.01],-1,1,False)
    assert [evaluate(x,p) for x in (-1,0,1)]==[0,0,0]
    with pytest.raises(ValueError):evaluate(2,p)
    p['thresholds'][0]+=1
    with pytest.raises(ValueError,match='changed'):emit_readout(p,'bad')


def test_explicit_scratch_wrapper_has_no_static_mutable_storage():
    source=emit_conv_wrapper(derive([.5]),'adapter','kernel',64)
    assert 'int32_t*scratch,size_t scratch_elements' in source
    assert 'kernel(a,b,scratch)' in source
    assert 'static int32_t' not in source

@pytest.mark.parametrize('scales', [[],[-1],[0],[float('inf')],[float('nan')]])
def test_refuse_nonmonotone_or_invalid_scales(scales):
    with pytest.raises(ValueError):derive(scales)
