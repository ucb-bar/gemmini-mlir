"""Preserve source cell and denominator order around four independent polynomials."""


def prepare_polynomial_batch_four(text: str) -> str:
    start = text.index("static int soft_details(")
    end = text.index("\nstatic int endpoint_intervals(", start)
    selected = text[start:end]
    before = """   for(int j=0;j<CHUNK;j++){
    merlin_f32_interval y=merlin_interval_point(0);
    if(mask[off+j]){if(!(hi[off+j]*SCORE_SCALE<=mx))return 0;merlin_soft_interval score=merlin_softmax_score(&domain,lo[off+j],hi[off+j],mx);merlin_f32_interval x={score.lo,score.hi,1};y=merlin_monotone_bit_polynomial_apply_words(x,&root_prepared);if(!y.valid)return 0;}
"""
    after = """   for(int base=0;base<CHUNK;base+=4){
    int active=CHUNK-base<4?CHUNK-base:4;
    merlin_f32_interval xs[4],ys[4];
    for(int cell=0;cell<4;cell++){
     int j=base+cell;xs[cell]=merlin_interval_point(0);
     if(cell<active&&mask[off+j]){
      if(!(hi[off+j]*SCORE_SCALE<=mx))return 0;
      merlin_soft_interval score=merlin_softmax_score(&domain,lo[off+j],hi[off+j],mx);
      xs[cell]=(merlin_f32_interval){score.lo,score.hi,1};
     }
    }
    merlin_polynomial_words_four(xs,&root_prepared,ys);
    for(int cell=0;cell<active;cell++){
    int j=base+cell;
    merlin_f32_interval y=mask[off+j]?ys[cell]:merlin_interval_point(0);
    if(!y.valid)return 0;
"""
    tail = "   }\n   for(int n=LANES/2;"
    if selected.count(before) != 1 or selected.count(tail) != 1:
        raise ValueError("prepared source cell schedule changed; batching refused")
    selected = selected.replace(before, after).replace(tail, "   }}\n   for(int n=LANES/2;")
    return '#include "prepared_polynomial_batch.h"\n' + text[:start] + selected + text[end:]
