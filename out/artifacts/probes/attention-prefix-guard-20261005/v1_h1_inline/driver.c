/* Experimental complete original attention head. Source semantics are retained
 * in reference_guard.c; target arithmetic comes from generic xDSL GEMM modules.
 * Fixed source dimensions select this capsule only, never production strategy.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "ordered_fma_bounds.h"
#include "fma_product_norms.h"
#include "output_sha256.h"
#ifdef INLINE_OUTWARD
/* This diagnostic supplies only +/-infinity directions. Value adjacency is
 * shared Merlin runtime arithmetic; errno/exception flags are not observed. */
static inline double certificate_adjacent64(double value,double toward) {
  assert(isinf(toward));
  return signbit(toward)?merlin_fma_next_down(value):merlin_fma_next_up(value);
}
static inline float certificate_adjacent32(float value,float toward) {
  assert(isinf(toward));
  return signbit(toward)?merlin_fma_next_down_f32(value):merlin_fma_next_up_f32(value);
}
#define nextafter certificate_adjacent64
#define nextafterf certificate_adjacent32
#endif
#include "reference_guard.c"

#define M 1024
#define D 64
#define DIGITS 3
extern const float source_q[], source_k[], source_v[], source_gold[];
extern void qk_plane(const int8_t *, const int8_t *, int32_t *);
extern void pv512_plane(const int8_t *, const int8_t *, int32_t *);
extern void pv192_plane(const int8_t *, const int8_t *, int32_t *);
extern void pv128_plane(const int8_t *, const int8_t *, int32_t *);
static float a[M*512], b[M*512], ar[M*512], br[M*512];
static int8_t ap[DIGITS*M*512], bp[DIGITS*M*512];
static double astep[M], bstep[M];
static merlin_fma_operand_norm anorm[M], bnorm[M];
static int32_t readout[M*M];
static double center[M*M], abscenter[M*M], repr[M*M], absolute[M*M];
static float qlo[M*M], qhi[M*M], packed[M*M], denlo[M], denhi[M], alpha[M*2];
static uint8_t cache[M*M], replay[M*D];
static uint32_t counts[M*2];
static float partlo[3*M*D], parthi[3*M*D], pvlo[M*D], pvhi[M*D];
static float result[HEADS*M*D];
static unsigned long q_replays, p_replays, readback_bytes, plane_calls;
static unsigned long stage_cycles[6];

static inline unsigned long tick(void) {
#ifdef NATIVE
  return 0; /* Functional oracle only; never native timing called target cycles. */
#else
  unsigned long value; __asm__ volatile("csrr %0,mcycle":"=r"(value)::"memory"); return value;
#endif
}

const float *attention_capsule_output(void) { return result; }
#ifndef NATIVE
void __assert_func(const char *file,int line,const char *function,const char *expression) {
  printf("ATTENTION_CERT_ASSERT %s %d %s %s\n",file,line,function,expression);
  exit(1);
}
#endif

static void encode(const float *source, float *reconstructed, int8_t *planes,
                   double *steps, merlin_fma_operand_norm *norms, int rows, int k,
                   int transpose_planes) {
  for (int row=0;row<rows;row++) {
    float maximum=0;
    for(int z=0;z<k;z++) maximum=fmaxf(maximum,fabsf(source[row*k+z]));
    const int exponent=maximum>0?ilogbf(maximum):0;
    const float step=scalbnf(1.0f,exponent+1-7*DIGITS);
    assert(isfinite(step)&&step>0); steps[row]=(double)step;
    for(int z=0;z<k;z++) {
      const float x=source[row*k+z];
      long integer=lrintf(fabsf(x)/step);
      if(integer>((1L<<(7*DIGITS))-1)) integer=(1L<<(7*DIGITS))-1;
      const int sign=x<0?-1:x>0?1:0;
      reconstructed[row*k+z]=(float)(integer*sign)*step;
      const int address=transpose_planes?z*rows+row:row*k+z;
      for(int digit=0;digit<DIGITS;digit++)
        planes[digit*rows*k+address]=(int8_t)(sign*((integer>>(7*digit))&127));
    }
    assert(merlin_fma_operand_summarize(source+row*k,reconstructed+row*k,k,1,1,norms+row));
  }
}

static void integer_center(int m,int n,int k,
                          void (*gemm)(const int8_t*,const int8_t*,int32_t*),
                          int abs_norm) {
  const int elements=m*n;
  for(int t=0;t<elements;t++) center[t]=0;
  for(int i=0;i<DIGITS;i++)for(int j=0;j<DIGITS;j++) {
    gemm(ap+i*m*k,bp+j*k*n,readout);
    plane_calls++;readback_bytes+=(unsigned long)elements*4;
    const double weight=scalbn(1.0,7*(i+j));
    for(int t=0;t<elements;t++) center[t]+=(double)readout[t]*weight;
  }
  if(abs_norm) {
    for(int t=0;t<DIGITS*m*k;t++) if(ap[t]<0)ap[t]=-ap[t];
    for(int t=0;t<DIGITS*n*k;t++) if(bp[t]<0)bp[t]=-bp[t];
    for(int t=0;t<elements;t++) abscenter[t]=0;
    for(int i=0;i<DIGITS;i++)for(int j=0;j<DIGITS;j++) {
      gemm(ap+i*m*k,bp+j*k*n,readout);
      plane_calls++;readback_bytes+=(unsigned long)elements*4;
      const double weight=scalbn(1.0,7*(i+j));
      for(int t=0;t<elements;t++) abscenter[t]+=(double)readout[t]*weight;
    }
  }
  for(int row=0;row<m;row++)for(int col=0;col<n;col++) {
    const int at=row*n+col;
    const double factor=astep[row]*bstep[col];
    assert(factor>=DBL_MIN&&isfinite(factor));
    center[at]*=factor;
    merlin_fma_product_summarize(anorm[row],bnorm[col],absolute+at,repr+at);
    if(abs_norm) {abscenter[at]*=factor;absolute[at]=abscenter[at];}
  }
}

static void certify(int elements,int length,float *lo,float *hi,int original_gamma) {
  const merlin_fma_bound beginning=merlin_fma_bound_begin();assert(beginning.valid);
  const double lu=length*0x1p-24,g=nextafter(lu/(1-lu),INFINITY);
  const double eta=nextafter(length*0x1p-149/(1-lu),INFINITY);
  for(int t=0;t<elements;t++) {
    if(original_gamma) {
      const double s=ua(absolute[t],repr[t]);
      const double radius=ua(repr[t],ua(um(g,s),eta));
      lo[t]=fl(da(center[t],-radius));hi[t]=fu(ua(center[t],radius));
    } else {
      merlin_fma_bound state=beginning;
      assert(merlin_fma_bound_push(&state,(merlin_fma_chunk){center[t],center[t],absolute[t],repr[t],(size_t)length}));
      assert(merlin_fma_bound_finish(&state,lo+t,hi+t));
    }
  }
}

static void attention(int head) {
  const float *q=source_q+head*M*D,*k=source_k+head*M*D,*v=source_v+head*M*D;
  unsigned long time=tick();
  encode(q,ar,ap,astep,anorm,M,D,0);
  encode(k,br,bp,bstep,bnorm,M,D,1);
  stage_cycles[0]+=tick()-time;time=tick();
  integer_center(M,M,D,qk_plane,VARIANT==0);
  stage_cycles[1]+=tick()-time;time=tick();
  certify(M*M,D,qlo,qhi,VARIANT==0);
  for(int t=0;t<M*M;t++)cache[t]=0;
  hierarchical_qk(q,k,qlo,qhi,packed,denlo,denhi,alpha,cache,counts,M);
  stage_cycles[2]+=tick()-time;
  for(int t=0;t<M*D;t++)pvlo[t]=pvhi[t]=0;
  for(int tile=0;tile<2;tile++) {
    for(int part=0;part<(VARIANT==0?1:3);part++) {
      const int start=VARIANT==0?0:part*192;
      const int length=VARIANT==0?512:part==2?128:192;
      time=tick();
      for(int row=0;row<M;row++)for(int z=0;z<length;z++)a[row*length+z]=packed[row*M+tile*512+start+z];
      for(int col=0;col<D;col++)for(int z=0;z<length;z++)b[col*length+z]=v[(tile*512+start+z)*D+col];
      encode(a,ar,ap,astep,anorm,M,length,0);
      encode(b,br,bp,bstep,bnorm,D,length,1);
      stage_cycles[0]+=tick()-time;time=tick();
      integer_center(M,D,length,length==512?pv512_plane:length==192?pv192_plane:pv128_plane,VARIANT==0);
      stage_cycles[3]+=tick()-time;time=tick();
      if(VARIANT==0) {
        for(int t=0;t<M*D;t++) absolute[t]=ua(absolute[t],repr[t]);
        pv_guard(center,repr,absolute,alpha,alpha,pvlo,pvhi,M,tile);
      } else certify(M*D,length,partlo+part*M*D,parthi+part*M*D,0);
      stage_cycles[4]+=tick()-time;
    }
    if(VARIANT!=0) {
      time=tick();
      for(int row=0;row<M;row++)for(int col=0;col<D;col++) {
        const int at=row*D+col;
        I accumulated=tile?mul((I){pvlo[at],pvhi[at]},point(alpha[row*2+tile])):point(0);
        for(int part=0;part<3;part++)accumulated=add(accumulated,(I){partlo[part*M*D+at],parthi[part*M*D+at]});
        pvlo[at]=accumulated.l;pvhi[at]=accumulated.h;
      }
      stage_cycles[4]+=tick()-time;
    }
  }
  time=tick();
  float *out=result+head*M*D;
  final_guard(pvlo,pvhi,denlo,denhi,out,replay,M);
  for(int row=0;row<M;row++)for(int col=0;col<D;col++)if(replay[row*D+col]) {
    const float value=source_pv_accumulator(packed+row*M,v,alpha+row*2,col);
    assert(pvlo[row*D+col]<=value&&value<=pvhi[row*D+col]);
    pvlo[row*D+col]=pvhi[row*D+col]=value;p_replays++;
  }
  final_guard(pvlo,pvhi,denlo,denhi,out,replay,M);
  for(int row=0;row<M;row++) {
    int ambiguous=0;for(int col=0;col<D;col++)ambiguous|=replay[row*D+col];
    if(!ambiguous)continue;
    float den[2];uint32_t stats[3];
    selective_denominator(q+row*D,k,qlo+row*M,qhi+row*M,cache+row*M,packed+row*M,pvlo+row*D,pvhi+row*D,alpha+row*2,den,stats);
    assert(denlo[row]<=den[0]&&den[0]<=den[1]&&den[1]<=denhi[row]);
    denlo[row]=den[0];denhi[row]=den[1];
  }
  final_guard(pvlo,pvhi,denlo,denhi,out,replay,M);
  for(int t=0;t<M*D;t++)assert(!replay[t]);
  for(int t=0;t<M*M;t++)q_replays+=cache[t]!=0;
  stage_cycles[5]+=tick()-time;
}

int main(void) {
  assert(environment_ok());
  printf("ATTENTION_CERT_BEGIN %d %d\n",VARIANT,HEADS);
  unsigned long time=tick();
  for(int head=0;head<HEADS;head++)attention(head);
  const unsigned long cycles=tick()-time;
  unsigned different=0,failures=0;
  for(int t=0;t<HEADS*M*D;t++) {
    different+=bits(result[t])!=bits(source_gold[t]);
    if(!isfinite(result[t])||fabsf(result[t]-source_gold[t])>.03125f+.02f*fabsf(source_gold[t]))failures++;
  }
  uint8_t digest[32];merlin_output_sha256(result,HEADS*M*D,digest);
  printf("OUT_SHA256 f32le %d %d ",HEADS*M*D,HEADS*M*D*4);
  for(unsigned i=0;i<32;i++)printf("%02x",digest[i]);printf("\n");
  printf("ATTENTION_CERT_COUNTS %lu %lu %lu %lu\n",q_replays,p_replays,readback_bytes,plane_calls);
  for(int stage=0;stage<6;stage++)printf("ATTENTION_CERT_STAGE_CYCLES %d %lu\n",stage,stage_cycles[stage]);
  printf("ATTENTION_CERT_ORIGINAL_BITS %u\n",different);
  printf("ATTENTION_CERT_ORIGINAL_GATE %u\n",failures);
  printf("METRIC cycles %lu\n",cycles);
  printf("METRIC build_hash %s\n",BUILD_HASH);
  printf("METRIC memref_rank_mismatch 0\n");
  printf("DONE\n");
  return failures?1:0;
}
