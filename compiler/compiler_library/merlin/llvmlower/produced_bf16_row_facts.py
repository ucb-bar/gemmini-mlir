"""Explicit owned BF16 producer facts for unchanged packing/norm consumers.

The original source matcher, lifetime, fallback and numerical policy are
separate caller obligations. This modifier never selects by workload identity.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProducedBF16RowFactsContract:
    complete_finite_bf16_producer_writes: bool
    complete_source_and_consumer_use_closure: bool
    private_disjoint_metadata_storage: bool
    source_and_metadata_immutable_until_consumption: bool
    prepared_rhs_epoch_and_consumer_quota_proven: bool
    original_encoded_equality_retained: bool
    original_scalar_dag_and_fallback_retained: bool
    standard_unobserved_representation_copies: bool
    stable_rne: bool
    gradual_underflow: bool
    nontrapping: bool
    exception_flags_unobserved: bool

    def validate(self) -> None:
        if any(type(value) is not bool or not value for value in vars(self).values()):
            raise ValueError("complete owned BF16 producer/source/effects/lifetime contract required")


def _forward_packing_facts(source: str) -> str:
    # The exact producer and consumer symbols are source experiment witnesses.
    # A compiler implementation must prove their typed write/use closure.
    start = source.index("static inline int merlin_bf16_radix_row_widen(")
    body = source.index("{", start)
    end = body + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    helper = source[start:end]
    helper = helper.replace("merlin_bf16_radix_row_widen(", "merlin_bf16_radix_row_widen_produced(")
    helper = helper.replace("unsigned char *exact) {", "unsigned char *exact, uint32_t maximum, uint32_t minimum) {")
    scan_start = helper.index("  uint32_t maximum = 0")
    scan_end = helper.index("  int exponent = 0;", scan_start)
    helper = (
        helper[:scan_start]
        + (
            "  if ((maximum & UINT32_C(0xffff)) || (minimum & UINT32_C(0xffff)) || maximu"
            "m >= UINT32_C(0x7f800000) || minimum > maximum) return 0;\n"
        )
        + helper[scan_end:]
    )
    source = source[:end] + "\n" + helper + "\n" + source[end:]

    def change(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError("owned BF16 producer/consumer grammar changed")
        source = source.replace(old, new)

    # Union storage is already private and dead in the complete component.
    # Keep the allocation size unchanged; no model allocation gain is claimed.
    change(
        "float p[ROWS*KEYS],plo[ROWS*KEYS],phi[ROWS*KEYS],ylo[ROWS*KEYS],yhi[ROWS*KEYS];",
        (
            "float p[ROWS*KEYS]; union {float plo[ROWS*KEYS];uint32_t pmeta[ROWS*2*PARTS*"
            "2];}; union {float phi[ROWS*KEYS];uint32_t qmeta[ROWS*2];};float ylo[ROWS*KE"
            "YS],yhi[ROWS*KEYS];"
        ),
    )
    change("float *source,*rf;double *rd,*steps;", "float *source;uint32_t *metadata;double *rd,*steps;")
    change(
        "float source[RHS_WORDS],rf[RHS_WORDS];double rd[RHS_WORDS],steps[RHS_ROWS];",
        (
            "float source[RHS_WORDS];union {float rf[RHS_WORDS];uint32_t metadata[2*RHS_R"
            "OWS];};double rd[RHS_WORDS],steps[RHS_ROWS];"
        ),
    )
    change(
        "x->source=o->source+words;x->rf=o->rf+words;x->rd=o->rd+words;",
        "x->source=o->source+words;x->metadata=o->metadata+2*rows;x->rd=o->rd+words;",
    )

    # The producer already admits each source word before publication. Metadata
    # is reset and completed within that same private immutable source epoch.
    change(
        " for(int r=0;r<ROWS;r++)for(int d=0;d<DEPTH;d++){\n  float value=load_bf16",
        (
            " for(int r=0;r<ROWS;r++){h->qmeta[2*r]=0;h->qmeta[2*r+1]=UINT32_C(0x7f800000"
            ");for(int d=0;d<DEPTH;d++){\n  float value=load_bf16"
        ),
    )
    change(
        "h->q[r*DEPTH+d]=value;\n }",
        (
            "h->q[r*DEPTH+d]=value;uint32_t mag=merlin_interval_bits(value)&UINT32_C(0x7f"
            "ffffff);if(mag>h->qmeta[2*r])h->qmeta[2*r]=mag;if(mag<h->qmeta[2*r+1])h->qme"
            "ta[2*r+1]=mag;\n }}"
        ),
    )
    change(
        "  for(int r=0;r<n;r++)for(int z=0;z<k;z++){\n   float value=load_bf16",
        (
            "  for(int r=0;r<n;r++){x->metadata[2*r]=0;x->metadata[2*r+1]=UINT32_C(0x7f80"
            "0000);for(int z=0;z<k;z++){\n   float value=load_bf16"
        ),
    )
    change(
        "x->source[r*k+z]=value;\n  }",
        (
            "x->source[r*k+z]=value;uint32_t mag=merlin_interval_bits(value)&UINT32_C(0x7"
            "fffffff);if(mag>x->metadata[2*r])x->metadata[2*r]=mag;if(mag<x->metadata[2*r"
            "+1])x->metadata[2*r+1]=mag;\n  }}"
        ),
    )

    change("float*maxima){", "float*maxima,uint32_t*metadata){")
    change(
        "float*maxima,merlin_softmax_produced_spans *spans",
        "float*maxima,uint32_t*metadata,merlin_softmax_produced_spans *spans",
    )
    change("alpha,rows,counts,yl,yh,maxima);", "alpha,rows,counts,yl,yh,maxima,metadata);")
    init = " for(int i=0;i<rows*2*PARTS;i++){metadata[2*i]=0;metadata[2*i+1]=UINT32_C(0x7f800000);}\n"
    change(
        " for(int row=0;row<rows;row++){\n  float old=", init + " for(int row=0;row<rows;row++){\n  float old=", count=2
    )
    change(
        "p[off+j]=point.value;",
        (
            "p[off+j]=point.value;size_t meta=2*(row*2*PARTS+tile*PARTS+j/SEGMENT);uint32"
            "_t mag=merlin_interval_bits(point.value)&UINT32_C(0x7fffffff);if(mag>metadat"
            "a[meta])metadata[meta]=mag;if(mag<metadata[meta+1])metadata[meta+1]=mag;"
        ),
        count=2,
    )
    change("h->ylo,h->yhi,h->maxima,&spans", "h->ylo,h->yhi,h->maxima,h->pmeta,&spans")

    change("merlin_encoded_row_equality *proof){", "merlin_encoded_row_equality *proof,const uint32_t *metadata){")
    old = """  float step;if(!merlin_bf16_radix_row_widen(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row))return 0;"""
    new = """  float step;int ok;
  if(metadata)ok=merlin_bf16_radix_row_widen_produced(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row,metadata[2*row],metadata[2*row+1]);
  else ok=merlin_bf16_radix_row_widen(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row);
  if(!ok)return 0;"""
    change(old, new)
    change(
        "encode_operand(x->source,n,k,1,x->rf,x->rd,x->planes,x->steps,0,0,x->flags,&x->proof)",
        "encode_operand(x->source,n,k,1,0,x->rd,x->planes,x->steps,0,0,x->flags,&x->proof,x->metadata)",
    )
    change("const struct prepared_rhs_entry *rhs){", "const struct prepared_rhs_entry *rhs,const uint32_t *metadata){")
    change("w->encoded_a_exact,&aproof)", "w->encoded_a_exact,&aproof,metadata)")
    change("w->encoded_b_exact,&bproof)", "w->encoded_b_exact,&bproof,0)")
    change("rhs?&rhs->entries[head][tile]:0)", "rhs?&rhs->entries[head][tile]:0,h->qmeta)")
    # Metadata is interleaved across six row segments, so copy the selected
    # block to existing dead scratch before the consumer; all cost is included.
    change(
        (
            "   unsigned char product_epoch;\n   merlin_source_point_span points={w->a,(si"
            "ze_t)ROWS*length,&product_epoch};"
        ),
        (
            "   uint32_t row_metadata[2*ROWS];for(int r=0;r<ROWS;r++){row_metadata[2*r]=h"
            "->pmeta[2*(r*2*PARTS+tile*PARTS+part)];row_metadata[2*r+1]=h->pmeta[2*(r*2*P"
            "ARTS+tile*PARTS+part)+1];}\n   unsigned char product_epoch;\n   merlin_source_"
            "point_span points={w->a,(size_t)ROWS*length,&product_epoch};"
        ),
    )
    change(
        "rhs?&rhs->entries[head][2+tile*PARTS+part]:0)", "rhs?&rhs->entries[head][2+tile*PARTS+part]:0,row_metadata)"
    )
    return source


def _forward_roundoff_maxima(source: str) -> str:
    signature = "const merlin_encoded_row_equality *bproof){"
    if source.count(signature) != 1:
        raise ValueError("owned encoded-row RMS4 bounds signature changed")
    source = source.replace(signature, "const merlin_encoded_row_equality *bproof,const uint32_t *bmetadata){")
    original = (
        "merlin_source_rms4_point_product_estimates(&rms_environment,aproof,bproof,a,"
        "alo,ahi,b,ar,br,center,lo,hi,m,n,k,scratch->uncertainty,CHUNK)"
    )
    selected = (
        (
            "(bmetadata?merlin_source_rms4_point_product_estimates_produced_max(&rms_envi"
            "ronment,aproof,bproof,a,alo,ahi,b,ar,br,center,lo,hi,m,n,k,scratch->uncertai"
            "nty,CHUNK,bmetadata):"
        )
        + original
        + ")"
    )
    if source.count(original) != 1:
        raise ValueError("owned original RMS4 maximum consumer changed")
    source = source.replace(original, selected)
    original = "&w->norms,&aproof,&bproof);"
    if source.count(original) != 1:
        raise ValueError("owned encoded-row bound invocation changed")
    source = source.replace(original, "&w->norms,&aproof,&bproof,rhs?rhs->metadata:0);")
    return source.replace('#include "source_rms_point_products.h"', '#include "source_rms_produced_maximum.h"')


def prepare_produced_bf16_row_facts(source: str, *, plan, contract: ProducedBF16RowFactsContract) -> str:
    """Forward complete producer facts inside the already admitted emitter.

    The mandatory BF16 producer finite checks and every original scalar stage,
    equality flag, RHS epoch/quota and fallback remain. Unknown source grammar
    refuses. The existing RMS4 consumer is specialized only if its independently
    permitted numerical policy is already present; this feature grants none.
    """
    from .source_attention_frontier import SourceAttentionFrontierPlan

    if not isinstance(contract, ProducedBF16RowFactsContract):
        raise ValueError("typed owned BF16 producer-facts contract required")
    contract.validate()
    if not isinstance(plan, SourceAttentionFrontierPlan):
        raise ValueError("typed complete source-attention plan required")
    plan.validate()
    # Each probability row has two chunks, three segments and two word facts.
    # Existing dead probability/RHS float storage must cover every fact byte.
    if 2 * 3 > plan.chunk or 2 * plan.chunk * plan.depth < 2 * plan.chunk + 6 * plan.depth:
        raise ValueError("dead private producer storage does not cover metadata")
    if "merlin_bf16_radix_row_widen_produced" in source:
        raise ValueError("owned BF16 producer facts already selected")
    if source.count("static inline int merlin_bf16_radix_row_widen(") != 1:
        raise ValueError("complete owned fused encoder required")
    producers = (
        ("if(!MERLIN_SOURCE_ISFINITE(value))return 0;h->q[r*DEPTH+d]=value;", 1),
        ("if(!MERLIN_SOURCE_ISFINITE(value))return 0;x->source[r*k+z]=value;", 1),
        ("merlin_bf16_exact_point_finish(y,bins);if(!point.valid)return 0;p[off+j]=point.value;", 2),
        ("w->a[r*length+z]=h->p[ix];", 1),
        ("merlin_source_point_span points={w->a,(size_t)ROWS*DEPTH,&product_epoch};", 1),
        ("merlin_source_point_span points={w->a,(size_t)ROWS*length,&product_epoch};", 1),
    )
    if any(source.count(text) != count for text, count in producers):
        raise ValueError("complete owned finite BF16 producer/copy epoch required")
    try:
        selected = _forward_packing_facts(source)
    except (IndexError, ValueError) as exc:
        raise ValueError("complete owned BF16 producer/consumer grammar required") from exc
    if '#include "source_rms_point_products.h"' in selected:
        selected = _forward_roundoff_maxima(selected)
    return selected
