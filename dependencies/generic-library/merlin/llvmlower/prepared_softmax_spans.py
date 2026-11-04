"""Carry private successful producer coverage to one softmax consumer.

This transform recognizes the emitter's complete mask gather / bound production
/ tile copy / immediate consumption region. It does not admit caller witnesses,
unknown product callbacks, mutable aliases, or arbitrary precomputed intervals.
The callback's existing exact complete-product/private-output contract remains
required; source bound success is not inferred from measured operand values.
"""


def _has_direct_output_index(text: str) -> bool:
    """Inspect identifier/punctuation tokens, never substring or regex matches.

    The emitter owns this fixed C body. Unknown token contexts conservatively
    refuse; a lo/hi subscript is supported only as an address passed to one of
    the separately counted successful bound producers.
    """
    tokens = []
    position = 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        end = position + 1
        if char.isalpha() or char == "_":
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
        tokens.append(text[position:end])
        position = end
    return any(
        token in ("lo", "hi")
        and index + 1 < len(tokens)
        and tokens[index + 1] == "["
        and (index == 0 or tokens[index - 1] != "&")
        for index, token in enumerate(tokens)
    )


def prepare_softmax_produced_spans(text: str) -> str:
    """Bind fresh single-use evidence; refuse changed producer/use grammar."""
    # The generated producer body is checked in full, rather than recognizing
    # only the successful call and ignoring possible writes inserted around it.
    region = """ for(int head=0;head<HEADS;head++){
  struct attention_head *h=&w->heads[head];if(!gather_head(h,inputs,head))return 0;
  for(int tile=0;tile<2;tile++){
   memcpy(w->a,h->q,sizeof(h->q));memcpy(w->al,h->q,sizeof(h->q));memcpy(w->ah,h->q,sizeof(h->q));
   memcpy(w->b,h->k+tile*CHUNK*DEPTH,CHUNK*DEPTH*sizeof(float));
   if(!evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque))return 0;
   for(int r=0;r<ROWS;r++)for(int j=0;j<CHUNK;j++){
    h->qlo[r*KEYS+tile*CHUNK+j]=w->lower[r*CHUNK+j];h->qhi[r*KEYS+tile*CHUNK+j]=w->upper[r*CHUNK+j];
   }
  }
  if(!soft_details(h->q,h->k,h->mask,h->qlo,h->qhi,h->p,h->plo,h->phi,h->denlo,h->denhi,h->alpha,ROWS,w->softcounts,h->ylo,h->yhi,h->maxima))return 0;"""
    mask = """  for(int r=0;r<ROWS;r++)for(int k=0;k<CHUNK;k++){
   unsigned char value=((const unsigned char*)in[mi].data)[physical(&in[mi],in[mi].sizes[1]==1?0:head,r,k)];
   if(value>1)return 0;h->mask[r*KEYS+tile*CHUNK+k]=value;
  }"""
    # All success paths in the emitted dot writer are the certified radius,
    # private product-row, or checked zero-gamma paths. Their contracts include
    # finite ordered endpoints and overflow refusal. Header implementations are
    # part of the compiler/provider dependency seal, as with other capabilities.
    bound_paths = (
        "if(radius_plan.valid){if(!merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;continue;}",
        "if(!(product_row.valid?merlin_fma_product_row_apply(&product_row,j,chunk.absolute_upper,chunk.representation_error_upper,&lo[r*n+j],&hi[r*n+j]):merlin_fma_zero_gamma_batch_apply(&gamma,chunk,&lo[r*n+j],&hi[r*n+j])))return 0;",
    )
    if text.count(region) != 1 or text.count(mask) != 1 or any(text.count(x) != 1 for x in bound_paths):
        raise ValueError("softmax producer coverage or successful bound grammar changed")
    start_bound = text.index("static int dot_bounds(")
    end_bound = text.index("\n}\n", start_bound) + 2
    bounds = text[start_bound:end_bound]
    # No additional success exit or direct interval write may bypass one of
    # the three admitted producers. Norm preparation may differ by its own
    # separately proved option; the output writer grammar stays fixed.
    if bounds.count("return 1;") != 1 or not bounds.endswith(" }return 1;\n}"):
        raise ValueError("successful bound exit coverage changed")
    if bounds.count("&lo[r*n+j]") != 3 or bounds.count("&hi[r*n+j]") != 3:
        raise ValueError("successful bound output coverage changed")
    if _has_direct_output_index(bounds):
        raise ValueError("unproved direct bound output write/read refused")
    generated = region.replace(
        "  for(int tile=0;tile<2;tile++){",
        "  unsigned char span_epoch;\n"
        "  merlin_softmax_produced_spans spans=merlin_softmax_spans_begin(h->qlo,h->qhi,h->mask,ROWS,CHUNK,&span_epoch);\n"
        "  for(int tile=0;tile<2;tile++){",
        1,
    )
    generated = generated.replace(
        "   }\n  }\n  if(!soft_details(",
        "   }\n   if(!merlin_softmax_spans_record_tile(&spans,h->qlo,h->qhi,h->mask,ROWS,CHUNK,tile,&span_epoch))return 0;\n  }\n  if(!soft_details(",
        1,
    )
    generated = generated.replace("h->yhi,h->maxima))", "h->yhi,h->maxima,&spans,&span_epoch))")
    text = text.replace(region, generated)
    start = text.index("static int soft_details(")
    end = text.index("\nstatic int endpoint_intervals(", start)
    soft = text[start:end]
    signature = "float*yl,float*yh,float*maxima){"
    admission = "!merlin_softmax_admit_active_spans(&domain,lo,hi,mask,(size_t)rows*KEYS)"
    if soft.count(signature) != 1 or soft.count(admission) != 1:
        raise ValueError("softmax single consumer admission grammar changed")
    soft = soft.replace(
        signature, "float*yl,float*yh,float*maxima,merlin_softmax_produced_spans *spans,const void *span_epoch){"
    )
    soft = soft.replace(
        admission,
        "(!merlin_softmax_spans_consume(spans,lo,hi,mask,(size_t)rows*KEYS,span_epoch)&&!merlin_softmax_admit_active_spans(&domain,lo,hi,mask,(size_t)rows*KEYS))",
    )
    return '#include "prepared_softmax_spans.h"\n' + text[:start] + soft + text[end:]
