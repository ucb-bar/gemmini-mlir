"""Exact finite endpoint narrowing, including signed zero and refused domains."""
import ctypes
from dataclasses import replace
from fractions import Fraction
import math
import random
import shutil
import struct
import subprocess
import pytest
from merlin.common.paths import merlin_dir
from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract, emit_exact_bound_conversion_permission

CONTRACT=ExactBoundConversionContract(True,True,True,True,True)
MAX=struct.unpack('<f',struct.pack('<I',0x7f7fffff))[0]
def frombits(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def doublebits(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def oracle(value):
    if not math.isfinite(value) or abs(value)>MAX:return None
    if value==0:
        z=0x80000000 if math.copysign(1,value)<0 else 0
        return z,z
    f=Fraction.from_float(abs(value));lo,hi=0,0x7f7fffff
    while lo<hi:
        mid=(lo+hi+1)//2
        if Fraction.from_float(frombits(mid))<=f:lo=mid
        else:hi=mid-1
    up=lo if Fraction.from_float(frombits(lo))==f else lo+1
    return (up|0x80000000,lo|0x80000000)if value<0 else(lo,up)
def cases():
    values=[0.,-0.,float('inf'),-float('inf'),float('nan'),2*MAX,-2*MAX,math.nextafter(MAX,math.inf),MAX,-MAX]
    rng=random.Random(4117)
    words=[1,2,0x7fffff,0x800000,0x800001,0x7f7fffff]+[(e<<23)|m for e in range(1,255) for m in (0,1,0x7fffff)]+[rng.randrange(1,0x7f800000)for _ in range(256)]
    for word in words:
        x=float(frombits(word));y=float(frombits(word+1))if word<0x7f7fffff else x
        for v in (x,math.nextafter(x,0.),math.nextafter(x,math.inf),(x+y)/2):values.extend((v,-v))
    return [(doublebits(x),oracle(x))for x in values]

C=r'''
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static unsigned bits(float x){unsigned b;memcpy(&b,&x,4);return b;}
static float raw(unsigned x){float f;memcpy(&f,&x,4);return f;}
static float down_neighbor(float x){unsigned b=bits(x);if((b&0x7fffffff)==0)return raw(0x80000001);return raw((b>>31)?b+1:b-1);}
static float up_neighbor(float x){unsigned b=bits(x);if((b&0x7fffffff)==0)return raw(1);return raw((b>>31)?b-1:b+1);}
#ifdef __riscv
static unsigned getmode(void){unsigned x;__asm__ volatile("frrm %0":"=r"(x));return x;}
static void setmode(unsigned x){__asm__ volatile("fsrm %0"::"r"(x):"memory");}
#else
static unsigned getmode(void){return fegetround();}
static void setmode(unsigned x){fesetround(x);}
static float floor_provider(double x){unsigned old=getmode();setmode(FE_DOWNWARD);volatile double d=x;volatile float f=(float)d;setmode(old);return f;}
static float ceil_provider(double x){unsigned old=getmode();setmode(FE_UPWARD);volatile double d=x;volatile float f=(float)d;setmode(old);return f;}
#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) floor_provider(x)
#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) ceil_provider(x)
#endif
int probe(uint64_t rawvalue,unsigned expectedlo,unsigned expectedhi,int valid,unsigned mode){
 double x;memcpy(&x,&rawvalue,8);unsigned old=getmode();setmode(mode);
 int admitted=x>=-(double)FLT_MAX&&x<=(double)FLT_MAX;
 if(!admitted){setmode(old);return !valid;}
 if(!valid){setmode(old);return 0;}
 volatile double source=x;float lo=(float)source,hi=lo;
 if((double)lo>x)lo=down_neighbor(lo);if((double)hi<x)hi=up_neighbor(hi);
 float a=MERLIN_F32_EXACT_FLOOR_FROM_F64(x),b=MERLIN_F32_EXACT_CEIL_FROM_F64(x);
 int pass=bits(a)==expectedlo&&bits(b)==expectedhi&&bits(a)==bits(lo)&&bits(b)==bits(hi)&&getmode()==mode;
 setmode(old);return pass;
}
int native_mode(int i){int modes[]={FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};return modes[i];}
'''
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cc=shutil.which('clang')or shutil.which('cc')
    if not cc:pytest.skip('C compiler needed')
    w=tmp_path_factory.mktemp('exact-bound');(w/'test.c').write_text(C)
    subprocess.run([cc,'-O2','-frounding-math','-ffp-contract=off','-shared','-fPIC',str(w/'test.c'),'-lm','-o',str(w/'test.so')],check=True)
    lib=ctypes.CDLL(str(w/'test.so'));lib.probe.argtypes=[ctypes.c_uint64,ctypes.c_uint,ctypes.c_uint,ctypes.c_int,ctypes.c_uint];return lib
@pytest.mark.parametrize('mode',range(4))
def test_fraction_endpoint_bits_and_environment(native,mode):
    for raw,result in cases():
        assert native.probe(raw,*(result or (0,0)),result is not None,native.native_mode(mode))==1,(hex(raw),result,mode)
@pytest.mark.parametrize('field',list(CONTRACT.__dict__))
def test_required_obligations(field):
    with pytest.raises(ValueError,match='complete exact'):
        emit_exact_bound_conversion_permission(replace(CONTRACT,**{field:False}))
@pytest.mark.parametrize('bad',[True,None,{},'yes'])
def test_typed_permission(bad):
    with pytest.raises(ValueError,match='typed exact'):
        emit_exact_bound_conversion_permission(bad)
