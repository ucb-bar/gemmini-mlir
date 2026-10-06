"""Exact rational coverage of positive uncertainty plane encoding."""
import ctypes as C
from fractions import Fraction
import random
import shutil
import struct
import subprocess

import pytest
from merlin.common.paths import merlin_dir


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    cc=shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    w=tmp_path_factory.mktemp("positive-uncertainty")
    (w/"proof.c").write_text('''
#include "positive_uncertainty_product.h"
int pack(float*a,float*l,float*h,float*b,int m,int n,int k,int8_t*u,int8_t*v,double*x,double*y){
 int active=0;
 return merlin_uncertainty_pack(a,l,h,m,k,u,x,&active)&&merlin_absolute_rhs_pack(b,n,k,v,y);
}
double finish(int32_t n,double a,double b){return merlin_uncertainty_product_upper(n,a,b);}
int refused_rounding(void){float x=1;double s;int8_t p;int active;int old=fegetround();fesetround(FE_DOWNWARD);int ok=merlin_uncertainty_pack(&x,&x,&x,1,1,&p,&s,&active);fesetround(old);return !ok;}
''')
    subprocess.run([cc,"-O2","-shared","-fPIC","-ffp-contract=off","-fno-fast-math","-I",str(merlin_dir()/"runtime/c"),str(w/"proof.c"),"-lm","-o",str(w/"proof.so")],check=True)
    lib=C.CDLL(str(w/"proof.so"))
    lib.pack.argtypes=[C.POINTER(C.c_float)]*4+[C.c_int]*3+[C.POINTER(C.c_int8)]*2+[C.POINTER(C.c_double)]*2
    lib.finish.argtypes=[C.c_int32,C.c_double,C.c_double];lib.finish.restype=C.c_double
    return lib


def f32(word):
    return struct.unpack("f",struct.pack("I",word))[0]


@pytest.mark.parametrize("k",[1,3,17,193])
def test_complete_rational_dot_and_guards(library,k):
    rng=random.Random(k);m,n=3,5
    def value():
        return f32(rng.randrange(0x7f800000)|(rng.randrange(2)<<31))
    a=[value() for _ in range(m*k)];a[0]=-0.0
    low=[];high=[]
    for x in a:
        low.append(min(x,0.0));high.append(max(x,0.0))
    b=[value() for _ in range(n*k)]
    b[0]=f32(1)
    arrays=[(C.c_float*len(x))(*x) for x in [a,low,high,b]]
    before=[bytes(x) for x in arrays]
    u=(C.c_int8*(m*k+8))(*([-77]*(m*k+8)));v=(C.c_int8*(n*k+8))(*([-77]*(n*k+8)))
    xs=(C.c_double*m)();ys=(C.c_double*n)()
    assert library.pack(*arrays,m,n,k,u,v,xs,ys)==1
    assert [bytes(x) for x in arrays]==before
    assert list(u)[m*k:]==[-77]*8 and list(v)[n*k:]==[-77]*8
    for r in range(m):
        for c in range(n):
            exact=sum(max(abs(Fraction(low[r*k+z])-Fraction(a[r*k+z])),abs(Fraction(high[r*k+z])-Fraction(a[r*k+z])))*abs(Fraction(b[c*k+z])) for z in range(k))
            integer=sum(u[r*k+z]*v[z*n+c] for z in range(k))
            assert 0<=integer<=k*4096
            assert exact<=integer*Fraction(xs[r])*Fraction(ys[c])
            assert exact<=Fraction(library.finish(integer,xs[r],ys[c]))


def test_refuses_unknown_domain(library):
    assert library.refused_rounding()==1
    f=(C.c_float*1)(1);bad=(C.c_float*1)(float("nan"));u=(C.c_int8*1)();d=(C.c_double*1)()
    assert library.pack(f,bad,f,f,1,1,1,u,u,d,d)==0
    assert library.pack(f,f,f,f,1,1,524288,u,u,d,d)==0
    assert library.pack(f,f,bad,f,1,1,1,u,u,d,d)==0
