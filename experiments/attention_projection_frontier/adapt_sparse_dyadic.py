"""Source-bound private ABI experiment; generic arithmetic delegates to Merlin."""
from merlin.llvmlower.sparse_dyadic_products import c_header, plan_sparse_dyadic, SparseDyadicEffects
from merlin.llvmlower.sparse_dyadic_rms import c_header as rms_header
from merlin.llvmlower.fused_encoded_sparse_count import c_header as count_header
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy


def adapt(source, destination):
    (destination/'sparse_dyadic_products.h').write_text(c_header(plan_sparse_dyadic(192),SparseDyadicEffects(*([True]*6))))
    (destination/'sparse_dyadic_rms.h').write_text(rms_header(ApproximateSourceRoundoffPolicy(*([True]*8))))
    (destination/'fused_encoded_sparse_count.h').write_text(count_header())
    s=source
    # New explicit private callback descriptor: ordinals0..4 retain the complete
    # three-digit catalog;5..7 denote offset-two-byte degrees0..2, bound128.
    s=s.replace('struct attention_workspace {','#include "sparse_dyadic_rms.h"\n#include "fused_encoded_sparse_count.h"\nstruct attention_workspace {\n double sparse_a[ROWS*CHUNK],sparse_b[CHUNK*DEPTH];\n int8_t sparse_ap[2*ROWS*CHUNK],sparse_bp[2*CHUNK*DEPTH];\n size_t sparse_ai[ROWS*CHUNK],sparse_bi[CHUNK*DEPTH];\n merlin_dyadic_row sparse_am[ROWS],sparse_bm[CHUNK];')
    old='unsigned char *flags,merlin_encoded_row_equality *proof){'
    assert s.count(old)==1;s=s.replace(old,'unsigned char *flags,merlin_encoded_row_equality *proof,size_t *sparse_count){')
    old=' if(m<=0||k<=0||!proof||!flags||(lower==0)!=(upper==0))return 0;'
    assert s.count(old)==1;s=s.replace(old,old+'\n if(!sparse_count)return 0;*sparse_count=0;')
    old='float step;if(!merlin_bf16_radix_row_widen('
    assert s.count(old)==1;s=s.replace(old,'float step;size_t count;if(!merlin_bf16_radix_row_widen_sparse_count(')
    old='lower?lower+row*k:0,upper?upper+row*k:0,flags+row))return 0;'
    assert s.count(old)==1;s=s.replace(old,'lower?lower+row*k:0,upper?upper+row*k:0,flags+row,&count))return 0;\n  *sparse_count+=count;')
    s=s.replace('int rows,length;merlin_encoded_row_equality proof;','int rows,length;merlin_encoded_row_equality proof;size_t sparse_count;')
    s=s.replace('x->flags,&x->proof))','x->flags,&x->proof,&x->sparse_count))')
    s=s.replace(' merlin_encoded_row_equality aproof,bproof;',' merlin_encoded_row_equality aproof,bproof;size_t an=0,bn=0;')
    s=s.replace('w->encoded_a_exact,&aproof))','w->encoded_a_exact,&aproof,&an))')
    s=s.replace('bstep=rhs->steps;bproof=rhs->proof;','bstep=rhs->steps;bproof=rhs->proof;bn=rhs->sparse_count;')
    s=s.replace('w->encoded_b_exact,&bproof))','w->encoded_b_exact,&bproof,&bn))')
    anchor=' const int32_t *planes[MERLIN_RADIX_FUSED_INTEGER_GROUPS];'
    assert s.count(anchor)==1
    block=r'''
 /* Mandatory encoder count is only an early cost screen. Exact preparation
  * and the dyadic prefix/range witness are checked independently below. */
 if(an<=SIZE_MAX/(size_t)n && bn<=SIZE_MAX/(size_t)m &&
    an*(size_t)n<=SIZE_MAX-bn*(size_t)m && an*(size_t)n+bn*(size_t)m<=(size_t)m*n){
  merlin_dyadic_owner ao={0},bo={0};
  merlin_dyadic_lease pa=merlin_dyadic_prepare(w->a,m,k,0,w->sparse_a,w->sparse_ap,
    w->sparse_ai,w->sparse_am,epoch,&ao);
  merlin_dyadic_lease pb=merlin_dyadic_prepare(bsource,n,k,1,w->sparse_b,w->sparse_bp,
    w->sparse_bi,w->sparse_bm,epoch,&bo);
  if(merlin_dyadic_admit(&pa,&pb,w->a,bsource,m,n,k,epoch,(size_t)m*n)){
   for(int d=0;d<3;d++)if(!product(opaque,w->sparse_ap,w->sparse_bp,w->readout[d],m,n,k,5+d))return 0;
   merlin_dyadic_product exact=merlin_dyadic_finish_product(w->center,
    w->readout[0],w->readout[1],w->readout[2],&pa,&pb,w->a,bsource,m,n,k,epoch,(size_t)m*n);
   merlin_fma_bound env=merlin_fma_bound_begin();
   if(exact.valid && merlin_source_rms4_corrected_product_estimates(&env,&exact,w->a,bsource,
      w->center,w->lower,w->upper,m,n,k,w->norms.uncertainty,CHUNK,epoch))return 1;
  }
 }
'''
    s=s.replace(anchor,block+anchor)
    return s
