"""Optional exact absolute-product producer for private source FMA bounds.

The existing canonical encoder uses three signed seven-bit magnitude digits,
all sharing each coefficient's sign. Their absolute values therefore encode
the exact absolute reconstructed operand without carry recoding. Source and
point-envelope equality are independently admitted for every row first.
"""


def prepare_exact_absolute_products(text: str) -> str:
    """Refuse changed producer, last-use or scratch grammar before mutation."""
    original = ''' for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->center[r*n+c]*=w->astep[r]*w->bstep[c];
 return dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms,&aproof,&bproof);'''
    required = (
        ' merlin_encoded_row_equality aproof,bproof;',
        'w->arf,w->ar,w->ap,w->astep,w->al,w->ah,w->encoded_a_exact,&aproof)',
        'w->brf,w->br,w->bp,w->bstep,0,0,w->encoded_b_exact,&bproof)',
        'merlin_radix_integer_finish_exact_f64(w->center,w->integer_center,(size_t)m*n);',
        ' int64_t integer_center[ROWS*CHUNK];',
    )
    if text.count(original) != 1 or any(text.count(x) != 1 for x in required):
        raise ValueError('exact absolute product producer or last-use grammar changed')
    replacement = ''' for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->center[r*n+c]*=w->astep[r]*w->bstep[c];
 int exact=1;
 for(int r=0;r<m;r++)exact=exact&&merlin_encoded_row_matches(&aproof,r,w->a+r*k,w->ar+r*k,w->al+r*k,w->ah+r*k,k);
 for(int c=0;c<n;c++)exact=exact&&merlin_encoded_row_matches(&bproof,c,w->b+c*k,w->br+c*k,0,0,k);
 if(exact){
  /* Signed planes' last signed use is complete. Readout/integer scratch is
   * reusable; the original real center and every source/envelope are preserved.
   * Canonical seven-bit magnitude digits share each coefficient's sign. */
  for(int i=0;i<3*m*k;i++){int v=w->ap[i];if(v==INT8_MIN)return 0;w->ap[i]=(int8_t)(v<0?-v:v);}
  for(int i=0;i<3*n*k;i++){int v=w->bp[i];if(v==INT8_MIN)return 0;w->bp[i]=(int8_t)(v<0?-v:v);}
  for(int degree=0;degree<5;degree++){
   if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,degree))return 0;
   if(degree==0)merlin_radix_integer_begin_from_first_group_exact_i64(w->integer_center,w->readout,(size_t)m*n);
   else merlin_radix_integer_accumulate_exact_i64(w->integer_center,w->readout,(size_t)m*n,(unsigned)degree);
  }
  merlin_radix_integer_finish_exact_f64(w->absolute_center,w->integer_center,(size_t)m*n);
  for(int r=0;r<m;r++)for(int c=0;c<n;c++)w->absolute_center[r*n+c]*=w->astep[r]*w->bstep[c];
  merlin_fma_bound env=merlin_fma_bound_begin();
  merlin_fma_zero_gamma_plan admitted=merlin_fma_zero_gamma_prepare(&env,k);
  merlin_fma_zero_gamma_batch gamma=merlin_fma_zero_gamma_batch_prepare(&admitted);
  merlin_exact_absolute_dot_bounds bounds=merlin_exact_absolute_dot_prepare(&gamma,w->center,w->absolute_center,(size_t)m*n,k);
  if(bounds.valid){
   for(int i=0;i<m*n;i++)if(!merlin_exact_absolute_dot_apply(&bounds,i,&w->lower[i],&w->upper[i]))return 0;
   return 1;
  }
 }
 /* Unknown or nonpoint representation keeps every original checked norm.
  * The private planes are dead on this path and fully reinitialized next use. */
 return dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms,&aproof,&bproof);'''
    return '#include "exact_absolute_dot_bounds.h"\n' + text.replace(
        original, replacement).replace(
            ' int64_t integer_center[ROWS*CHUNK];',
            ' int64_t integer_center[ROWS*CHUNK];\n double absolute_center[ROWS*CHUNK];')
