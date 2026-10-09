"""Defer exact probability bins through the owned rigorous PV uncertainty path.

This transforms the compiler-owned executor only. Complete source/consumer
matching, immutable input ownership and retained source fallback remain required.
The initial PV product bounds enclose every original probability in its BF16
interval. Refinement narrows those intervals, so the prior bounds remain valid.
Before original PV replay, every probability in the selected head/query row is
made source-exact. No center is admitted as an exact source value by this policy.
"""


def defer_probability_certification(text: str) -> str:
    replay = """    if(mask[off+j] && bins.low_bits != bins.high_bits) {
      float exact=source_dot(q+row*DEPTH,k+(tile*CHUNK+j)*DEPTH,DEPTH,1);
      if(!(lo[off+j]<=exact && exact<=hi[off+j]))return 0;
      y=merlin_interval_point(source_poly(exact*SCORE_SCALE-mx)); bins=merlin_bf16_interval_prepare(y); counts[2]++;
    }
"""
    state = "uint8_t den_exact[ROWS],cell_exact[ROWS*DEPTH];"
    initialize = "memset(h->den_exact,0,sizeof(h->den_exact));memset(h->cell_exact,0,sizeof(h->cell_exact));return 1;"
    marker = "static int certify_frontier(struct attention_workspace *w){"
    refine = """    struct attention_head *h=&w->heads[head];
    if(pass&&!h->den_exact[row]){"""
    required = (
        "w->a[r*length+z]=h->p[ix];w->al[r*length+z]=h->plo[ix];w->ah[r*length+z]=h->phi[ix];",
        "w->astep,w->al,w->ah,w->encoded_a_exact,&aproof)",
        "dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms,&aproof,&bproof)",
        "double l=alo?alo[t]:a[t],h=ahi?ahi[t]:a[t];",
        "if(l!=a[t]||h!=a[t]){positions[used]=z;",
        "if(!exact_source_partials(h->p,h->v,h->partlo,h->parthi,ROWS,row,d,w->refinement))return 0;",
    )
    if text.count(replay) != 2 or any(text.count(s) != 1 for s in (state, initialize, marker, refine, *required)):
        raise ValueError("complete checked source probability/uncertainty/refinement grammar required")
    if "merlin_bf16_exact_point_finish" in text:
        raise ValueError("deferred probabilities cannot use eager point-span admission")
    text = text.replace(replay, "")
    text = text.replace(state, state + "uint8_t probability_exact[ROWS];")
    text = text.replace(initialize, initialize.replace("return 1;", "memset(h->probability_exact,0,sizeof(h->probability_exact));return 1;"))
    helper = """static int refine_probability_row(struct attention_head *h,int row,unsigned long long *counts){
 if(h->probability_exact[row])return 1;
 for(int tile=0;tile<2;tile++)for(int j=0;j<CHUNK;j++){
  int ix=row*KEYS+tile*CHUNK+j;
  if(merlin_interval_bits(h->plo[ix])!=merlin_interval_bits(h->phi[ix])){
   if(!h->mask[ix])return 0;
   float score=source_dot(h->q+row*DEPTH,h->k+(tile*CHUNK+j)*DEPTH,DEPTH,1);
   if(!(h->qlo[ix]<=score&&score<=h->qhi[ix]))return 0;
   float y=source_poly(score*SCORE_SCALE-h->maxima[row*2+tile]);
   if(!MERLIN_SOURCE_ISFINITE(y)||!(h->ylo[ix]<=y&&y<=h->yhi[ix]))return 0;
   float point=merlin_interval_bf16(y);
   if(!MERLIN_SOURCE_ISFINITE(point)||!(h->plo[ix]<=point&&point<=h->phi[ix]))return 0;
   h->p[ix]=h->plo[ix]=h->phi[ix]=point;h->ylo[ix]=h->yhi[ix]=y;counts[2]++;
  }
 }
 h->probability_exact[row]=1;return 1;
}
"""
    text = text.replace(marker, helper + marker)
    return text.replace(refine, refine.replace("    if(pass", "    if(!refine_probability_row(h,row,w->softcounts))return 0;\n    if(pass"))
