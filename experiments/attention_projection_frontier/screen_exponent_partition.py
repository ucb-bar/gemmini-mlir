"""Read-only exponent/dispatch census on actual unchanged48 attention calls."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/capture_initial_bounds.py').read_text().replace("h=base/'out/observation_frontier/initial_bounds'","h=base/'out/observation_frontier/exponent_partition'")
needle="(dest/'provider.c').write_text(s)"
replacement='''anchor='static int dot_bounds(const float*a,'
assert s.count(anchor)==1
s=s.replace(anchor,'typedef void(*diagnostic_product_fn)(const float*,const float*,int,int,int);\\nstatic diagnostic_product_fn diagnostic_product;\\nvoid diagnostic_set_product(diagnostic_product_fn f){diagnostic_product=f;}\\n'+anchor)
anchor=' merlin_fma_bound rms_environment=merlin_fma_bound_begin();'
assert s.count(anchor)==1;s=s.replace(anchor,' if(diagnostic_product)diagnostic_product(a,b,m,n,k);\\n'+anchor)
(dest/'provider.c').write_text(s)
'''
s=s.replace(needle,replacement)
needle='CALL=C.CFUNCTYPE(None,C.POINTER(C.c_int8)'
insert='''census=[]
PROD=C.CFUNCTYPE(None,C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_int,C.c_int,C.c_int)
def row_metadata(x):
 maximum=np.max(np.abs(x),axis=1);exponent=np.frexp(maximum)[1]
 scaled=np.ldexp(x.astype(np.float64),15-exponent[:,None])
 exact=(scaled==np.rint(scaled))&(scaled>=-32768)&(scaled<=32767)
 return np.all(exact,axis=1),int(np.count_nonzero(~exact))
@PROD
def product_census(a,b,m,n,k):
 try:
  av=np.ctypeslib.as_array(a,shape=(m*k,)).reshape(m,k)
  bv=np.ctypeslib.as_array(b,shape=(n*k,)).reshape(n,k)
  assert np.isfinite(av).all() and np.isfinite(bv).all()
  assert not np.any(av.view(np.uint32)&65535) and not np.any(bv.view(np.uint32)&65535)
  ae,ao=row_metadata(av);be,bo=row_metadata(bv)
  hot=int(ae.sum())*int(be.sum())
  block_hot_macs=block_readbytes=block_packwords=0;all_hot_blocks=0
  for begin in range(0,k,16):
   length=min(16,k-begin);ar,_=row_metadata(av[:,begin:begin+length]);br,_=row_metadata(bv[:,begin:begin+length])
   cells=int(ar.sum())*int(br.sum());block_hot_macs+=cells*length
   block_readbytes+=cells*3*4
   block_packwords+=(int(ar.sum())+int(br.sum()))*length
   all_hot_blocks+=int(bool(ar.all() and br.all()))
  census.append(dict(index=len(census),m=m,n=n,k=k,a_words=m*k,b_words=n*k,a_exact_rows=int(ae.sum()),b_exact_rows=int(be.sum()),a_noninteger_words=ao,b_noninteger_words=bo,full_hot_macs=hot*k,full_cold_macs=(m*n-hot)*k,sparse_correction_terms=ao*n+bo*m,block16_hot_macs=block_hot_macs,block16_cold_macs=m*n*k-block_hot_macs,block16_min_readbytes=block_readbytes,block16_packwords=block_packwords,block16_all_hot_blocks=all_hot_blocks,blocks=(k+15)//16))
 except BaseException as e:errors.append(repr(e))
lib.diagnostic_set_product.argtypes=[PROD];lib.diagnostic_set_product(product_census)
'''
s=s.replace(needle,insert+needle)
s=s.replace("np.save(h/f'group_{snapshots:02d}.npy',np.stack(heads,axis=1))","pass  # No duplicate endpoint snapshots needed for this read-only census.")
s=s.replace("record=dict(scope=", "record=dict(exponent_census=census,scope=")
s=s.replace("Diagnostic snapshots before eager final quantizer refinement; original refinement remains active. RMS4 intervals are approximate, not rigorous source certificates.","Read-only exact BF16 representation census on unchanged source operands. Fixed16word K partitions are an analytical dispatch screen; irregular packing/readout/host reduction costs remain unpriced, no target selection.")
s += "\nprint('CENSUS_CALLS',len(census),flush=True)\n"
(base/'out/exponent_partition_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
