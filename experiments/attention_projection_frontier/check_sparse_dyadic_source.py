"""Execute generic exact sparse producer/finish on unchanged attention operands."""
from pathlib import Path
import subprocess
from merlin.llvmlower.sparse_dyadic_products import c_header, plan_sparse_dyadic, SparseDyadicEffects
base=Path(__file__).resolve().parents[2]
h=base/'out/observation_frontier/sparse_dyadic_source';h.mkdir(parents=True,exist_ok=False)
(h/'helper.h').write_text(c_header(plan_sparse_dyadic(192),SparseDyadicEffects(*([True]*6))))
(h/'check.c').write_text(r'''
#include "helper.h"
#include <stdlib.h>
typedef void(*product_fn)(const int8_t*,const int8_t*,int32_t*,int,int,int,int);
typedef void(*check_fn)(const double*,int,int);
int check(const float*as,const float*bs,size_t m,size_t n,size_t k,
 product_fn product,check_fn compare){
 double*a=calloc(m*k,sizeof(double)),*b=calloc(n*k,sizeof(double)),*out=calloc(m*n,sizeof(double));
 int8_t*ap=calloc(2*m*k,1),*bp=calloc(2*n*k,1);
 size_t*ai=calloc(m*k,sizeof(size_t)),*bi=calloc(n*k,sizeof(size_t));
 merlin_dyadic_row*ar=calloc(m,sizeof(*ar)),*br=calloc(n,sizeof(*br));
 merlin_dyadic_owner ao={0},bo={0};int epoch=0;
 int32_t*g[3];for(int j=0;j<3;j++)g[j]=calloc(m*n,sizeof(int32_t));
 int result=-1;
 if(!a||!b||!out||!ap||!bp||!ai||!bi||!ar||!br||!g[0]||!g[1]||!g[2])goto done;
 merlin_dyadic_lease pa=merlin_dyadic_prepare(as,m,k,0,a,ap,ai,ar,&epoch,&ao);
 merlin_dyadic_lease pb=merlin_dyadic_prepare(bs,n,k,1,b,bp,bi,br,&epoch,&bo);
 result=0;
 if(!pa.owner||!pb.owner||!merlin_dyadic_admit(&pa,&pb,as,bs,m,n,k,&epoch,m*n))goto done;
 for(int d=0;d<3;d++)product(ap,bp,g[d],m,n,k,d);
 result=merlin_dyadic_finish(out,g[0],g[1],g[2],&pa,&pb,as,bs,m,n,k,&epoch,m*n)?1:-2;
 if(result==1)compare(out,m,n);
done:
 free(a);free(b);free(out);free(ap);free(bp);free(ai);free(bi);free(ar);free(br);
 for(int j=0;j<3;j++)free(g[j]);return result;
}
''')
cc='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
cmd=[cc,'-O2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',str(h/'check.c'),'-lm','-o',str(h/'check.so')]
subprocess.run(cmd,check=True)
s=(base/'out/sparse_dyadic_admission_driver.py').read_text().replace("h=base/'out/observation_frontier/sparse_dyadic_admission'", "h=base/'out/observation_frontier/sparse_dyadic_source_run'")
start=s.index('def row_metadata(x):');end=s.index('lib.diagnostic_set_product.argtypes',start)
s=s[:start]+'''
helper=C.CDLL(str(base/'out/observation_frontier/sparse_dyadic_source/check.so'))
PAIR=C.CFUNCTYPE(None,C.POINTER(C.c_int8),C.POINTER(C.c_int8),C.POINTER(C.c_int32),C.c_int,C.c_int,C.c_int,C.c_int)
CHECK=C.CFUNCTYPE(None,C.POINTER(C.c_double),C.c_int,C.c_int)
helper.check.argtypes=[C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_size_t,C.c_size_t,C.c_size_t,PAIR,CHECK]
actual_product_calls=0;actual_checked_cells=0;expected_center=None
@PAIR
def pair(a,b,c,m,n,k,d):
 global actual_product_calls
 try:
  av=np.ctypeslib.as_array(a,shape=(2*m*k,)).reshape(2,m,k)
  bv=np.ctypeslib.as_array(b,shape=(2*k*n,)).reshape(2,k,n)
  out=np.zeros((m,n),np.float64)
  for x in range(2):
   y=d-x
   if 0<=y<2:out+=av[x].astype(np.float64)@bv[y].astype(np.float64)
  assert np.all(out==np.rint(out)) and np.all(abs(out)<=2**31-1)
  np.ctypeslib.as_array(c,shape=(m*n,))[:]=out.astype(np.int32).ravel()
  actual_product_calls+=1
 except BaseException as e:errors.append(repr(e))
@CHECK
def compare(p,m,n):
 global actual_checked_cells
 try:
  out=np.ctypeslib.as_array(p,shape=(m*n,)).reshape(m,n)
  assert np.array_equal(out,expected_center)
  actual_checked_cells+=m*n
 except BaseException as e:errors.append(repr(e))
@PROD
def product_census(a,b,m,n,k):
 global expected_center
 try:
  av=np.ctypeslib.as_array(a,shape=(m*k,)).reshape(m,k)
  bv=np.ctypeslib.as_array(b,shape=(n*k,)).reshape(n,k)
  expected_center=av.astype(np.float64)@bv.astype(np.float64).T
  result=helper.check(a,b,m,n,k,pair,compare)
  assert result>=0
  census.append(dict(index=len(census),m=m,n=n,k=k,accepted=bool(result),resource_cap=m*n))
 except BaseException as e:errors.append(repr(e))
''' +s[end:]
s=s.replace('record=dict(exponent_census=census,', 'record=dict(exact_sparse_source_census=census,actual_sparse_product_calls=actual_product_calls,actual_checked_cells=actual_checked_cells,')
s=s.replace('Conservative exact common-grid prefix admission for offset two-byte plus sparse corrections. No device route is selected; preparation, offsets, correction and fallback costs remain unpriced.', 'Diagnostic generic C preparation and exact sparse finish, one correction per output resource cap, all allocation included in diagnostic wall time only. Original provider outputs unchanged. Dense degree products use independent native integer simulation, no target or performance claim.')
(base/'out/sparse_dyadic_source_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
