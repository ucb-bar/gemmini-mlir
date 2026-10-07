"""Read-only exact-grid admission census; no alternative output is selected."""
from pathlib import Path
base = Path(__file__).resolve().parents[2]
s = (base/'experiments/attention_projection_frontier/screen_exponent_partition.py').read_text()
s = s.replace("exponent_partition'", "sparse_dyadic_admission'")
start = s.index("def row_metadata(x):")
end = s.index("lib.diagnostic_set_product.argtypes", start)
s = s[:start] + '''def row_metadata(x):
 x=x.astype(np.float64)
 largest=np.max(np.abs(x),axis=1)
 ex=np.frexp(largest)[1];scale_exp=np.where(largest==0,0,ex-15)
 q=np.rint(np.ldexp(x,-scale_exp[:,None]))
 assert np.all(np.abs(q)<=32640)
 selected=np.ldexp(q,scale_exp[:,None]); residual=x-selected
 frac,xe=np.frexp(np.abs(x));sig=np.rint(frac*256).astype(np.int64)
 # BF16 significands contain <=8 bits. Trailing zero removal gives exact grid.
 tz=np.zeros(257,dtype=np.int64)
 for j in range(1,257):tz[j]=(j&-j).bit_length()-1
 source_exp=xe-8+tz[sig]
 source_exp=np.where(x==0,10000,source_exp)
 qi=np.abs(q).astype(np.int64)
 qlow=qi&-qi
 qtz=np.frexp(qlow)[1]-1
 selected_exp=np.where(qi==0,10000,scale_exp[:,None]+qtz)
 grid=np.minimum(source_exp.min(axis=1),selected_exp.min(axis=1))
 grid=np.where(grid==10000,0,grid)
 su=np.ldexp(largest,-grid)
 qu=np.ldexp(np.max(np.abs(selected),axis=1),-grid)
 eu=np.ldexp(np.max(np.abs(residual),axis=1),-grid)
 # These are exact integer powers-of-two scalings of BF16/dyadic quantities.
 assert np.all(su==np.rint(su)) and np.all(qu==np.rint(qu)) and np.all(eu==np.rint(eu))
 return dict(source=max(map(int,su)),selected=max(map(int,qu)),error=max(map(int,eu)),nnz=int(np.count_nonzero(residual)),grid_min=int(grid.min()),grid_max=int(grid.max()),scale_min=int(scale_exp.min()),scale_max=int(scale_exp.max()))
@PROD
def product_census(a,b,m,n,k):
 try:
  av=np.ctypeslib.as_array(a,shape=(m*k,)).reshape(m,k)
  bv=np.ctypeslib.as_array(b,shape=(n*k,)).reshape(n,k)
  assert np.isfinite(av).all() and np.isfinite(bv).all()
  assert not np.any(av.view(np.uint32)&65535) and not np.any(bv.view(np.uint32)&65535)
  am=row_metadata(av);bm=row_metadata(bv)
  prefix=k*(am['selected']*bm['selected']+am['error']*bm['source']+am['selected']*bm['error'])
  terms=am['nnz']*n+bm['nnz']*m
  # A sufficient whole-contraction admission; not a tuned cost policy.
  admitted=prefix<=2**53
  census.append(dict(index=len(census),m=m,n=n,k=k,a=am,b=bm,prefix_bound=str(prefix),prefix_bits=prefix.bit_length(),exact_f64_prefix_admitted=admitted,sparse_correction_terms=terms,source_words=(m+n)*k,readback_bytes=3*m*n*4,offset_finish_cells=m*n,dense_products=4*m*n*k,old_products=9*m*n*k))
 except BaseException as e:errors.append(repr(e))
''' + s[end:]
s=s.replace('out/exponent_partition_driver.py','out/sparse_dyadic_admission_driver.py')
s=s.replace('Fixed16word K partitions are an analytical dispatch screen; irregular packing/readout/host reduction costs remain unpriced, no target selection.', 'Conservative exact common-grid prefix admission for offset two-byte plus sparse corrections. No device route is selected; preparation, offsets, correction and fallback costs remain unpriced.')
exec(compile(s,str(__file__),'exec'))
