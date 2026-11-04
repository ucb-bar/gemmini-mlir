"""Explicit physical immutable RHS preparation for source-proved consumers.

The caller must bind PreparedOperandOwner to the actual source opportunity and
borrowed producer/consumer effects. This transform alone grants no reuse. The
extra owner allocation and initialization are explicit and must be timed.
"""

_STORAGE = r"""
struct prepared_rhs_entry {
 float *source,*rf;double *rd,*steps;int8_t *planes;unsigned char *flags;
 int rows,length;merlin_encoded_row_equality proof;
};
#define RHS_WORDS (2*HEADS*KEYS*DEPTH)
#define RHS_ROWS (HEADS*(KEYS+2*PARTS*DEPTH))
struct prepared_rhs_owner {
 float source[RHS_WORDS],rf[RHS_WORDS];double rd[RHS_WORDS],steps[RHS_ROWS];
 int8_t planes[3*RHS_WORDS];unsigned char flags[RHS_ROWS];
 struct prepared_rhs_entry entries[HEADS][2+2*PARTS];
 merlin_attention_view views[11];const void *epoch;size_t next,uses;int ready;
};
size_t @SYMBOL@_rhs_owner_bytes(void){return sizeof(struct prepared_rhs_owner);}
size_t @SYMBOL@_rhs_owner_alignment(void){return _Alignof(struct prepared_rhs_owner);}
/* Only the compiler-owned, source-disjoint pool may be invalidated. The normal
 * prepare bridge does this before checking descriptors, so refusal cannot expose
 * stale validity from the preceding epoch. No source/input memory is touched. */
int @SYMBOL@_rhs_owner_invalidate(void *storage,size_t bytes){
 if(!storage||bytes<sizeof(struct prepared_rhs_owner)||
    (uintptr_t)storage%_Alignof(struct prepared_rhs_owner))return 0;
 ((struct prepared_rhs_owner*)storage)->ready=0;return 1;
}
static int rhs_input_index(int entry){return entry<2?(entry?6:1):(entry-2<PARTS?3:8)+(entry-2)%PARTS;}
static int rhs_views_equal(const merlin_attention_view *a,const merlin_attention_view *b){
 if(a->data!=b->data||a->offset!=b->offset)return 0;
 for(int i=0;i<4;i++)if(a->sizes[i]!=b->sizes[i]||a->strides[i]!=b->strides[i])return 0;
 return 1;
}
static int rhs_disjoint_view(const merlin_attention_view *v,int element_bytes,const void *storage,size_t bytes){
 if(!v||!storage||!bytes||(uintptr_t)storage>UINTPTR_MAX-bytes)return 0;
 for(int i=0;i<4;i++)if(v->sizes[i]<=0)return 0;
 if(!valid_view(v,v->sizes,element_bytes))return 0;
 uint64_t maximum=(uint64_t)v->offset;
 for(int i=0;i<4;i++)maximum+=(uint64_t)(v->sizes[i]-1)*(uint64_t)v->strides[i];
 uintptr_t low=(uintptr_t)v->data+(uint64_t)v->offset*element_bytes;
 uintptr_t high=(uintptr_t)v->data+(maximum+1)*element_bytes;
 return (uintptr_t)storage+bytes<=low||high<=(uintptr_t)storage;
}
/* Caller-proved immutable source owners and complete typed consumer lifetime
 * are mandatory. The epoch is a capability, never a pointer cache key. */
int @SYMBOL@_rhs_prepare(const merlin_attention_view *inputs,void *storage,size_t bytes,
 const void *epoch,size_t uses){
 if(!inputs||!storage||!epoch||!uses||bytes<sizeof(struct prepared_rhs_owner)||
    (uintptr_t)storage%_Alignof(struct prepared_rhs_owner))return 0;
 for(int i=0;i<11;i++)if(!rhs_disjoint_view(&inputs[i],i==2||i==7?1:2,storage,bytes))return 0;
 struct prepared_rhs_owner *o=storage;o->ready=0;o->epoch=epoch;o->next=0;o->uses=uses;
 size_t words=0,rows=0;
 for(int head=0;head<HEADS;head++)for(int e=0;e<2+2*PARTS;e++){
  int part=(e-2)%PARTS,n=e<2?CHUNK:DEPTH,k=e<2?DEPTH:CHUNK-part*SEGMENT;
  if(e>=2&&k>SEGMENT)k=SEGMENT;
  int ix=rhs_input_index(e);int64_t shape[4]={1,HEADS,e<2?CHUNK:k,DEPTH};
  if(!valid_view(&inputs[ix],shape,2))return 0;
  struct prepared_rhs_entry *x=&o->entries[head][e];size_t count=(size_t)n*k;
  if(count>RHS_WORDS-words||(size_t)n>RHS_ROWS-rows)return 0;
  x->source=o->source+words;x->rf=o->rf+words;x->rd=o->rd+words;
  x->planes=o->planes+3*words;x->steps=o->steps+rows;x->flags=o->flags+rows;x->rows=n;x->length=k;
  for(int r=0;r<n;r++)for(int z=0;z<k;z++){
   float value=load_bf16(&inputs[ix],head,e<2?r:z,e<2?z:r);
   if(!MERLIN_SOURCE_ISFINITE(value))return 0;x->source[r*k+z]=value;
  }
  if(!encode_operand(x->source,n,k,1,x->rf,x->rd,x->planes,x->steps,0,0,x->flags,&x->proof))return 0;
  o->views[ix]=inputs[ix];words+=count;rows+=n;
 }
 if(words!=RHS_WORDS||rows!=RHS_ROWS)return 0;
 o->ready=1;return 1;
}
"""


def prepare_attention_rhs_owner(text: str) -> str:
    """Add explicit owner producer and consumption ABI; default emitter unchanged."""
    marker = "static int evaluate_products("
    if text.count(marker) != 1 or "merlin_encoded_row_equality aproof,bproof;" not in text:
        raise ValueError("prepared RHS requires the exact encoded-row producer seam")
    text = text.replace(marker, _STORAGE + "\n" + marker, 1)
    old_epoch = "const void *epoch){\n if(!merlin_source_point_span_matches"
    if text.count(old_epoch) != 1:
        raise ValueError("prepared RHS point producer seam changed")
    text = text.replace(
        old_epoch, "const void *epoch,const struct prepared_rhs_entry *rhs){\n if(!merlin_source_point_span_matches", 1
    )
    old = """ if(!encode_operand(w->a,m,k,0,w->arf,w->ar,w->ap,w->astep,0,0,w->encoded_a_exact,&aproof)||
    !encode_operand(w->b,n,k,1,w->brf,w->br,w->bp,w->bstep,0,0,w->encoded_b_exact,&bproof))return 0;"""
    new = """ if(!encode_operand(w->a,m,k,0,w->arf,w->ar,w->ap,w->astep,0,0,w->encoded_a_exact,&aproof))return 0;
 const float *bsource=w->b;const double *br=w->br,*bstep=w->bstep;const int8_t *bp=w->bp;
 if(rhs){
  if(rhs->rows!=n||rhs->length!=k||!rhs->proof.valid)return 0;
  bsource=rhs->source;br=rhs->rd;bp=rhs->planes;bstep=rhs->steps;bproof=rhs->proof;
 }else if(!encode_operand(w->b,n,k,1,w->brf,w->br,w->bp,w->bstep,0,0,w->encoded_b_exact,&bproof))return 0;"""
    if text.count(old) != 1:
        raise ValueError("prepared RHS encoding seam changed")
    text = text.replace(old, new)
    text = text.replace("product(opaque,w->ap,w->bp,w->readout", "product(opaque,w->ap,bp,w->readout")
    text = text.replace("w->astep[r]*w->bstep[c]", "w->astep[r]*bstep[c]")
    text = text.replace("dot_bounds(w->a,0,0,w->b,w->ar,w->br,", "dot_bounds(w->a,0,0,bsource,w->ar,br,")
    signature = """int @SYMBOL@(const merlin_attention_view *inputs,merlin_attention_view *output,
 void *workspace,size_t capacity,merlin_attention_product product,void *opaque){"""
    changed = """int @SYMBOL@_with_rhs(const merlin_attention_view *inputs,merlin_attention_view *output,
 void *workspace,size_t capacity,merlin_attention_product product,void *opaque,
 void *prepared,size_t prepared_bytes,const void *epoch,size_t consumer){
 if(!inputs||!output||!workspace||!product)return 0;
 struct prepared_rhs_owner *rhs=prepared;
 if(rhs){
  if(prepared_bytes<sizeof(*rhs)||(uintptr_t)rhs%_Alignof(struct prepared_rhs_owner)||!rhs->ready||rhs->epoch!=epoch||rhs->next!=consumer||consumer>=rhs->uses)return 0;
  if((uintptr_t)workspace>UINTPTR_MAX-capacity||(uintptr_t)rhs>UINTPTR_MAX-prepared_bytes||
     !((uintptr_t)workspace+capacity<=(uintptr_t)rhs||(uintptr_t)rhs+prepared_bytes<=(uintptr_t)workspace))return 0;
  if(!rhs_disjoint_view(output,2,rhs,prepared_bytes))return 0;
  for(int i=0;i<11;i++)if(!rhs_disjoint_view(&inputs[i],i==2||i==7?1:2,rhs,prepared_bytes))return 0;
  for(int e=0;e<2+2*PARTS;e++){int ix=rhs_input_index(e);if(!rhs_views_equal(&inputs[ix],&rhs->views[ix])){rhs->ready=0;return 0;}}
  rhs->next++;if(rhs->next==rhs->uses)rhs->ready=0;
 }"""
    if text.count(signature) != 1:
        raise ValueError("prepared RHS entry ABI changed")
    text = text.replace(signature, changed)
    if text.count("product,opaque,&points,&product_epoch))") != 2:
        raise ValueError("prepared RHS QK/PV consumer seam changed")
    text = text.replace(
        "product,opaque,&points,&product_epoch))",
        "product,opaque,&points,&product_epoch,rhs?&rhs->entries[head][tile]:0))",
        1,
    )
    text = text.replace(
        "product,opaque,&points,&product_epoch))",
        "product,opaque,&points,&product_epoch,rhs?&rhs->entries[head][2+tile*PARTS+part]:0))",
        1,
    )
    text += """\nint @SYMBOL@(const merlin_attention_view *inputs,merlin_attention_view *output,
 void *workspace,size_t capacity,merlin_attention_product product,void *opaque){
 return @SYMBOL@_with_rhs(inputs,output,workspace,capacity,product,opaque,0,0,0,0);
}\n"""
    return text
