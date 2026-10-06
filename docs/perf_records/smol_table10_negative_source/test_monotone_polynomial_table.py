"""Independent source samples for curvature-bounded integer interpolation."""
import ctypes
import random
import shutil
import struct
import subprocess

import pytest
from merlin.common.paths import merlin_dir

PLANS = [
    [-87.3365478515625,1.4426950216293335,-.079204238951206207,-.22433836758136749,.3035426139831543,.00010703434963943437,8388608.,1065353216.],
    [-40.,.5,.02,-.1,.2,.004,8388608.,1065353216.],
    [-32.,.75,-.01,-.02,.1,.001,4194304.,1065353216.],
]
SOURCE = r'''
#include "monotone_polynomial_table.h"
static merlin_monotone_bit_polynomial prepared;
static merlin_monotone_polynomial_table table;
static int32_t knots[65537+2];
int setup(const float *a,unsigned bits,size_t capacity){
 merlin_bit_polynomial_plan p={a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
 merlin_fma_bound env=merlin_fma_bound_begin();
 prepared=merlin_monotone_bit_polynomial_prepare(&env,&p,0);
 knots[0]=1234567;knots[65538]=-7654321;
 table=merlin_monotone_polynomial_table_prepare(&prepared,knots+1,capacity,bits);
 return table.valid;
}
void bound(float lo,float hi,uint32_t *out){
 merlin_f32_interval y=merlin_monotone_polynomial_table_apply(merlin_interval(lo,hi),&table);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);out[2]=y.valid;
}
void old_bound(float lo,float hi,uint32_t *out){
 merlin_f32_interval y=merlin_monotone_bit_polynomial_apply(merlin_interval(lo,hi),&prepared);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);out[2]=y.valid;
}
uint32_t original(float x){
 const merlin_bit_polynomial_plan *p=&prepared.checked.source;
 if(x<p->cutoff)return 0;
 float s=x*p->scale,f=s-floorf(s),poly=p->coefficients[0];
 for(int i=1;i<4;i++)poly=fmaf(f,poly,p->coefficients[i]);
 return(uint32_t)(int32_t)fmaf(p->bit_multiplier,s-poly,p->bit_bias);
}
int guards(void){return knots[0]==1234567&&knots[65538]==-7654321;}
'''

@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc: pytest.skip('native compiler required')
    work=tmp_path_factory.mktemp('polynomial-table')
    source=work/'test.c';source.write_text(SOURCE);so=work/'test.so'
    subprocess.run([cc,'-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC',
                    '-I',str(merlin_dir()/'runtime/c'),str(source),'-lm','-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so));f=ctypes.c_float;u=ctypes.POINTER(ctypes.c_uint32)
    lib.setup.argtypes=[ctypes.POINTER(f),ctypes.c_uint,ctypes.c_size_t]
    lib.bound.argtypes=lib.old_bound.argtypes=[f,f,u]
    lib.original.argtypes=[f];lib.original.restype=ctypes.c_uint32
    return lib

@pytest.mark.parametrize('plan',PLANS)
@pytest.mark.parametrize('bits',[8,10,12,16])
def test_source_enclosed_across_fraction_tables_and_floor_jumps(native,plan,bits):
    a=(ctypes.c_float*8)(*plan)
    assert native.setup(a,bits,(1<<bits)+1)
    rng=random.Random(991)
    points=[a[0],0.,-2.**-149,-2.**-126]
    points.extend(k/a[1] for k in range(int(a[0]*a[1]),1))
    points.extend((k+(j/(1<<bits)))/a[1]
                  for k in [-1,-2,-8] for j in [1,2,127,(1<<bits)-1])
    points.extend(rng.uniform(a[0],0) for _ in range(500))
    for point in points:
        point=ctypes.c_float(point).value
        word=struct.unpack('<I',struct.pack('<f',point))[0]
        neighbors=[point]
        if point<0:
            neighbors.extend(struct.unpack('<f',struct.pack('<I',word+d))[0] for d in [-1,1])
        for radius in [0.,1e-5,.01,2.]:
            lo=ctypes.c_float(min(neighbors)-radius).value
            hi=ctypes.c_float(min(0.,max(neighbors)+radius)).value
            out=(ctypes.c_uint32*3)();native.bound(lo,hi,out)
            assert out[2]
            samples=[lo,hi,*neighbors]
            samples.extend(rng.uniform(lo,hi) for _ in range(12))
            for x in samples:
                if lo<=x<=hi:
                    original=native.original(x)
                    assert out[0]<=original<=out[1],(plan,bits,lo,hi,x,list(out),original)
    assert native.guards()

@pytest.mark.parametrize('bits,capacity',[(7,65537),(17,65537),(10,1024)])
def test_invalid_table_retains_original_checked_result(native,bits,capacity):
    a=(ctypes.c_float*8)(*PLANS[0]);assert not native.setup(a,bits,capacity)
    selected=(ctypes.c_uint32*3)();old=(ctypes.c_uint32*3)()
    native.bound(-1.,-.5,selected);native.old_bound(-1.,-.5,old)
    assert list(selected)==list(old) and native.guards()
