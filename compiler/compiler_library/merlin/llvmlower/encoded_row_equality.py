"""Bind private encoder equality to unchanged row norm consumption."""


def prepare_encoded_row_equality(text: str) -> str:
    """Refuse unknown write/use grammar; no mutable pointer cache is emitted."""
    changes = (
        (
            "int m,int n,int k,struct product_scratch *scratch){",
            "int m,int n,int k,struct product_scratch *scratch,const merlin_encoded_row_equality *aproof,const merlin_encoded_row_equality *bproof){",
        ),
        (
            "  for(int z=0;z<k;z++){double bv=b[j*k+z];merlin_dot_norms_add(&bn[j],MERLIN_SOURCE_F64_ABS(bv));merlin_representation_error_add(&en[j],bv,br[j*k+z]);}",
            "  int exact=merlin_encoded_row_matches(bproof,j,b+j*k,br+j*k,0,0,k);\n"
            "  for(int z=0;z<k;z++){double bv=b[j*k+z];merlin_dot_norms_add(&bn[j],MERLIN_SOURCE_F64_ABS(bv));if(!exact)merlin_representation_error_add(&en[j],bv,br[j*k+z]);}",
        ),
        (
            "  for(int z=0;z<k;z++){\n   int t=r*k+z;",
            "  int exact=merlin_encoded_row_matches(aproof,r,a+r*k,ar+r*k,alo?alo+r*k:0,ahi?ahi+r*k:0,k);\n"
            "  if(exact){for(int z=0;z<k;z++)merlin_dot_norms_add(&an,MERLIN_SOURCE_F64_ABS((double)a[r*k+z]));}\n"
            "  else for(int z=0;z<k;z++){\n   int t=r*k+z;",
        ),
        (
            "  merlin_dot_norms_finish(&an);if(requirements.require_l2)merlin_dot_norms_finish(&rn);merlin_representation_error_finish(&ae);",
            "  merlin_dot_norms_finish(&an);if(exact){rn=an;rnl1=(merlin_l1_norm){an.l1,an.valid};}else if(requirements.require_l2)merlin_dot_norms_finish(&rn);merlin_representation_error_finish(&ae);",
        ),
        (
            "int8_t *planes,double *steps){",
            "int8_t *planes,double *steps,const float *lower,const float *upper,unsigned char *flags,merlin_encoded_row_equality *proof){",
        ),
        (
            " for(int i=0;i<m*k;i++)rd[i]=rf[i];return 1;",
            " *proof=merlin_encoded_rows_widen(&environment,source,rf,lower,upper,rd,flags,m,k);return proof->valid;",
        ),
        (
            "static int evaluate_products(struct attention_workspace *w,int m,int n,int k,merlin_attention_product product,void *opaque){",
            "static int evaluate_products(struct attention_workspace *w,int m,int n,int k,merlin_attention_product product,void *opaque){\n"
            " merlin_encoded_row_equality aproof,bproof;",
        ),
        ("w->arf,w->ar,w->ap,w->astep)", "w->arf,w->ar,w->ap,w->astep,w->al,w->ah,w->encoded_a_exact,&aproof)"),
        ("w->brf,w->br,w->bp,w->bstep)", "w->brf,w->br,w->bp,w->bstep,0,0,w->encoded_b_exact,&bproof)"),
        ("m,n,k,&w->norms);", "m,n,k,&w->norms,&aproof,&bproof);"),
        (
            " uint8_t row_certified[ROWS],pending[HEADS*DEPTH];",
            " uint8_t encoded_a_exact[ROWS],encoded_b_exact[CHUNK];\n uint8_t row_certified[ROWS],pending[HEADS*DEPTH];",
        ),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError("encoded equality producer/use grammar changed; refused")
        text = text.replace(before, after)
    return '#include "encoded_row_equality.h"\n' + text
