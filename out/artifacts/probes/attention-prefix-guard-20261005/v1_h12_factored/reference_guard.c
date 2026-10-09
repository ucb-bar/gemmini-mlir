#include <assert.h>
/* Fixed-fixture diagnostic. Strict IEEE binary32 RNE arithmetic, no FTZ/DAZ.
 * Every bound-producing binary64 op is widened with nextafter. */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
typedef struct {float l,h;} I;
static double ua(double a,double b){return nextafter(a+b,INFINITY);}
static double um(double a,double b){return nextafter(a*b,INFINITY);}
static double ud(double a,double b){return nextafter(a/b,INFINITY);}
static double da(double a,double b){return nextafter(a+b,-INFINITY);}
static float fl(double x){float y=(float)x;return (double)y>x?nextafterf(y,-INFINITY):y;}
static float fu(double x){float y=(float)x;return (double)y<x?nextafterf(y,INFINITY):y;}
static I point(float x){return (I){x,x};}
static I add(I a,I b){return (I){a.l+b.l,a.h+b.h};}
static I sub(I a,I b){return (I){a.l-b.h,a.h-b.l};}
static I mul(I a,I b){float x[4]={a.l*b.l,a.l*b.h,a.h*b.l,a.h*b.h};I c={x[0],x[0]};for(int i=1;i<4;i++){c.l=fminf(c.l,x[i]);c.h=fmaxf(c.h,x[i]);}return c;}
static I fm(I a,I b,I c){float low[4]={fmaf(a.l,b.l,c.l),fmaf(a.l,b.h,c.l),fmaf(a.h,b.l,c.l),fmaf(a.h,b.h,c.l)};float high[4]={fmaf(a.l,b.l,c.h),fmaf(a.l,b.h,c.h),fmaf(a.h,b.l,c.h),fmaf(a.h,b.h,c.h)};I d={low[0],high[0]};for(int i=1;i<4;i++){d.l=fminf(d.l,low[i]);d.h=fmaxf(d.h,high[i]);}return d;}
static uint32_t bits(float x){uint32_t v;memcpy(&v,&x,4);return v;}
static float unbits(uint32_t x){float v;memcpy(&v,&x,4);return v;}
static float bf(float x){uint32_t u=bits(x);return unbits((u+0x7fff+((u>>16)&1))&0xffff0000u);}
static float ex(float x){if(x<unbits(0xc2aeac50u))return 0;float s=x*unbits(0x3fb8aa3bu),f=s-floorf(s);float p=fmaf(f,-.079204240219773236f,-.22433836478672356f);p=fmaf(f,p,.30354260500649682f);p=fmaf(f,p,.00010703434948458272f);float t=fmaf(8388608.f,s-p,1065353216.f);return unbits((uint32_t)(int32_t)t);}
static I exp_from_s(I s,float floor_value){
 I f=sub(s,point(floor_value));
 I p=fm(f,point(-.079204240219773236f),point(-.22433836478672356f));p=fm(f,p,point(.30354260500649682f));p=fm(f,p,point(.00010703434948458272f));
 I t=fm(sub(s,p),point(8388608.f),point(1065353216.f));
 if(t.l<0 || t.h>=2147483648.f)return (I){0,INFINITY};
 return (I){unbits((uint32_t)(int32_t)t.l),unbits((uint32_t)(int32_t)t.h)};
}
/* Complete binary32 monotonicity enumeration is recorded separately.
 * This tighter evaluator is eligible only for the source's nonpositive domain. */
static I exi(I x){assert(x.l<=x.h && x.h<=0.f);return (I){ex(x.l),ex(x.h)};}
static I sum8(I *values){I s[8];for(int j=0;j<8;j++)s[j]=point(0);for(int i=0;i<512;i++)s[i%8]=add(s[i%8],values[i]);for(int k=4;k>=1;k/=2)for(int j=0;j<k;j++)s[j]=add(s[j],s[j+k]);return s[0];}
static int alphai(I arg,I *out,int cap,uint64_t *calls){
 if(bits(arg.l)==bits(arg.h)){*out=point(expf(arg.l));(*calls)++;return 1;}
 // Enumerate actual libm results: no monotonicity or correct-rounding axiom.
 float x=arg.l,low=INFINITY,high=-INFINITY;
 for(int count=0;count<cap;count++){float y=expf(x);(*calls)++;low=fminf(low,y);high=fmaxf(high,y);if(x==arg.h){*out=(I){low,high};return 1;}x=nextafterf(x,INFINITY);}
 *out=(I){0,1};return 0;
}
int environment_ok(void){volatile float subnormal=0x1p-149f;return subnormal+subnormal==0x1p-148f && fegetround()==FE_TONEAREST && FLT_RADIX==2 && FLT_MANT_DIG==24 && DBL_MANT_DIG==53;}
/* Norm bounds: original A/B versus digit reconstructions AR/BR. Row-L1 times
 * column-Linf is inexpensive and conservative; no hidden matrix product. */
void contraction_bounds(const float*a,const float*b,const float*ar,const float*br,const double*center,float*lo,float*hi,double*repr,double*absolute,int m,int n,int k,int source_length){
 double u=0x1p-24,g=ud(source_length*u,1-source_length*u),eta=ud(source_length*0x1p-149,1-source_length*u);
 for(int i=0;i<m;i++){
  double l1=0,r1=0,e1=0;
  for(int z=0;z<k;z++){l1=ua(l1,fabs((double)a[i*k+z]));r1=ua(r1,fabs((double)ar[i*k+z]));e1=ua(e1,nextafter(fabs((double)a[i*k+z]-ar[i*k+z]),INFINITY));}
  for(int j=0;j<n;j++){
   double bmax=0,emax=0;for(int z=0;z<k;z++){bmax=fmax(bmax,fabs((double)b[z*n+j]));emax=fmax(emax,nextafter(fabs((double)b[z*n+j]-br[z*n+j]),INFINITY));}
   int t=i*n+j;double e=ua(um(e1,bmax),um(r1,emax)),s=um(l1,bmax),radius=ua(e,ua(um(g,s),eta));repr[t]=e;absolute[t]=s;
   lo[t]=fl(da(center[t],-radius));hi[t]=fu(ua(center[t],radius));
  }
 }
}
/* Logit bounds are for unscaled QK dots. Packed output is valid only on rows
 * not flagged. Flagged rows are replaced by exact source QK immediately. */
void softmax_guard(const float*loglo,const float*loghi,float*packed,float*denlo,float*denhi,float*alo,float*ahi,uint8_t*replay,uint64_t*calls,uint32_t*ambiguous,int rows,int cap){
 for(int row=0;row<rows;row++){
  I old=point(-INFINITY),den=point(0);int bad=0;ambiguous[row]=0;
  for(int tile=0;tile<2;tile++){
   I scores[512],values[512],mx=old;
   for(int j=0;j<512;j++){int at=row*1024+tile*512+j;scores[j]=mul((I){loglo[at],loghi[at]},point(.125f));mx.l=fmaxf(mx.l,scores[j].l);mx.h=fmaxf(mx.h,scores[j].h);}
   for(int j=0;j<512;j++){I centered=sub(scores[j],mx);centered.h=fminf(centered.h,0);values[j]=exi(centered);float l=bf(values[j].l),h=bf(values[j].h);if(bits(l)!=bits(h)||!isfinite(values[j].h)){bad=1;ambiguous[row]++;}packed[row*1024+tile*512+j]=l;}
   I alpha=point(0);
   if(tile){if(old.l>=mx.h)alpha=point(1);else if(bad)alpha=(I){0,1};else if(!alphai(sub(old,mx),&alpha,cap,calls))bad=1;}
   alo[row*2+tile]=alpha.l;ahi[row*2+tile]=alpha.h;
   den=fm(alpha,den,sum8(values));old=mx;
  }
  denlo[row]=den.l;denhi[row]=den.h;replay[row]=bad||!(den.l>0)||!isfinite(den.h);
 }
}
void source_qrow(const float*q,const float*k,float*packed,float*den,float*alpha){
 float maximum=-INFINITY,sum=0;
 for(int tile=0;tile<2;tile++){
  float scores[512],mx=maximum;I values[512];
  for(int j=0;j<512;j++){float dot=0;for(int z=0;z<64;z++)dot=fmaf(q[z],k[(tile*512+j)*64+z],dot);scores[j]=dot*.125f;mx=fmaxf(mx,scores[j]);}
  for(int j=0;j<512;j++){float y=ex(scores[j]-mx);packed[tile*512+j]=bf(y);values[j]=point(y);}
  alpha[tile]=expf(maximum-mx);sum=fmaf(alpha[tile],sum,sum8(values).l);maximum=mx;
 }
 *den=sum;
}
void pv_guard(const double*center,const double*repr,const double*absolute,const float*alo,const float*ahi,float*lo,float*hi,int rows,int tile){
 double u=0x1p-24,g192=ud(192*u,1-192*u),g3=ud(3*u,1-3*u);
 double coefficient=ua(g192,um(g3,ua(1,g192))),eta=ud(600*0x1p-149,1-192*u);
 for(int row=0;row<rows;row++)for(int j=0;j<64;j++){
  int t=row*64+j;I seed=tile?mul((I){lo[t],hi[t]},(I){alo[row*2+tile],ahi[row*2+tile]}):point(0);
  double magnitude=fmax(fabs((double)seed.l),fabs((double)seed.h));double err=ua(repr[t],ua(um(coefficient,absolute[t]),ua(um(g3,magnitude),eta)));
  lo[t]=fl(da(da((double)seed.l,center[t]),-err));hi[t]=fu(ua(ua((double)seed.h,center[t]),err));
 }
}
void final_guard(const float*lo,const float*hi,const float*dl,const float*dh,float*out,uint8_t*replay,int rows){
 for(int r=0;r<rows;r++){I recip={1.f/dh[r],1.f/dl[r]};for(int j=0;j<64;j++){int t=r*64+j;I y=mul((I){lo[t],hi[t]},recip);float l=bf(y.l),h=bf(y.h);out[t]=l;replay[t]=bits(l)!=bits(h)||!isfinite(l)||!isfinite(h)||(y.l<=0 && y.h>=0);}}
}
float source_pv(const float*p,const float*v,const float*alpha,float denominator,int column){
 float out=0;
 for(int tile=0;tile<2;tile++){
  out*=alpha[tile];for(int start=0;start<512;start+=192){float partial=0;int end=start+192<512?start+192:512;for(int z=start;z<end;z++)partial=fmaf(p[tile*512+z],v[(tile*512+z)*64+column],partial);out+=partial;}
 }
 return bf(out*(1.f/denominator));
}

void source_qk(const float*q,const float*k,float*out,int rows){for(int r=0;r<rows;r++)for(int j=0;j<1024;j++){float s=0;for(int z=0;z<64;z++)s=fmaf(q[r*64+z],k[j*64+z],s);out[r*1024+j]=s;}}
void test_exp_intervals(const float*lo,const float*hi,const float*sample,float*outlo,float*outhi,float*exact,int count){for(int i=0;i<count;i++){I y=exi((I){lo[i],hi[i]});outlo[i]=y.l;outhi[i]=y.h;exact[i]=ex(sample[i]);}}


#include <assert.h>
static float cached_dot(const float*q,const float*k,float*lo,float*hi,uint8_t*cached,int j,uint32_t*new_calls){
 if(!cached[j]){float dot=0;for(int z=0;z<64;z++)dot=fmaf(q[z],k[j*64+z],dot);assert(dot>=lo[j] && dot<=hi[j]);lo[j]=hi[j]=dot;cached[j]=1;(*new_calls)++;}
 return lo[j];
}
/* Establish the running maximum by replaying only possible winners, then
 * certify each BF16 probability separately. Reuse every replayed logit. */
void hierarchical_qk(const float*q,const float*k,float*loglo,float*loghi,float*packed,float*denlo,float*denhi,float*alpha,uint8_t*cached,uint32_t*counts,int rows){
 for(int row=0;row<rows;row++){
  float maximum=-INFINITY;I denominator=point(0);float*lo=loglo+row*1024;float*hi=loghi+row*1024;uint8_t*cache=cached+row*1024;counts[row*2]=counts[row*2+1]=0;
  for(int tile=0;tile<2;tile++){
   int offset=tile*512;float lower=maximum;for(int j=0;j<512;j++)lower=fmaxf(lower,lo[offset+j]*.125f);
   float newmax=maximum;
   for(int j=0;j<512;j++)if(hi[offset+j]*.125f>=lower){float dot=cached_dot(q+row*64,k,lo,hi,cache,offset+j,counts+row*2);newmax=fmaxf(newmax,dot*.125f);}
   I values[512];
   for(int j=0;j<512;j++){
    int z=offset+j;I centered=sub(mul((I){lo[z],hi[z]},point(.125f)),point(newmax));centered.h=fminf(centered.h,0);I y=exi(centered);
    float a=bf(y.l),b=bf(y.h);
    if(bits(a)!=bits(b)||!isfinite(y.h)){
     float dot=cached_dot(q+row*64,k,lo,hi,cache,z,counts+row*2+1);y=point(ex(dot*.125f-newmax));a=bf(y.l);
    }
    packed[row*1024+z]=a;values[j]=y;
   }
   float scale=expf(maximum-newmax);alpha[row*2+tile]=scale;
   denominator=fm(point(scale),denominator,sum8(values));maximum=newmax;
  }
  denlo[row]=denominator.l;denhi[row]=denominator.h;assert(denominator.l>0 && isfinite(denominator.h));
 }
}
float source_pv_accumulator(const float*p,const float*v,const float*alpha,int column){
 float out=0;for(int tile=0;tile<2;tile++){out*=alpha[tile];for(int start=0;start<512;start+=192){float partial=0;int end=start+192<512?start+192:512;for(int z=start;z<end;z++)partial=fmaf(p[tile*512+z],v[(tile*512+z)*64+column],partial);out+=partial;}}return out;
}
uint32_t complete_qk_row(const float*q,const float*k,float*lo,float*hi,uint8_t*cached,const float*packed,float*den,const float*alpha){
 float maximum=-INFINITY,sum=0;uint32_t calls=0;
 for(int tile=0;tile<2;tile++){
  float scores[512],newmax=maximum;I values[512];
  for(int j=0;j<512;j++){scores[j]=cached_dot(q,k,lo,hi,cached,tile*512+j,&calls)*.125f;newmax=fmaxf(newmax,scores[j]);}
  for(int j=0;j<512;j++){float y=ex(scores[j]-newmax);assert(bits(bf(y))==bits(packed[tile*512+j]));values[j]=point(y);}
  float scale=expf(maximum-newmax);assert(bits(scale)==bits(alpha[tile]));sum=fmaf(scale,sum,sum8(values).l);maximum=newmax;
 }
 *den=sum;return calls;
}

/* Diagnostic refinement of existing source-ordered interval certificates.
 * Prioritize uncertainty contribution only; every acceptance still compares
 * exact BF16 endpoint bits. Original source dots are the only fallback. */

#include <stdlib.h>
typedef struct {double width;int index;} Candidate;
static int largest_first(const void*a,const void*b){
 const Candidate*x=a,*y=b;
 if(x->width>y->width)return -1;
 if(x->width<y->width)return 1;
 return (x->index>y->index)-(x->index<y->index);
}
static I denominator_interval(I*values,const float*alpha){
 return fm(point(alpha[1]),sum8(values),sum8(values+512));
}
static int certified_outputs(const float*pvlo,const float*pvhi,I denominator){
 float out[64];uint8_t pending[64];
 final_guard(pvlo,pvhi,&denominator.l,&denominator.h,out,pending,1);
 for(int j=0;j<64;j++)if(pending[j])return 0;
 return 1;
}
uint32_t selective_denominator(const float*q,const float*k,float*lo,float*hi,uint8_t*cached,const float*packed,const float*pvlo,const float*pvhi,const float*alpha,float*den,uint32_t*stats){
 I values[1024];float maxima[2];float maximum=-INFINITY;
 Candidate candidates[1024];int count=0;uint32_t calls=0,checks=0;
 for(int tile=0;tile<2;tile++){
  int offset=tile*512;float newmax=maximum;
  /* The earlier exact-winner pass cached every possible maximum. Therefore
   * max(lower bounds) equals the true source maximum; assert all upper bounds
   * are also below it before reusing that certificate. */
  for(int j=0;j<512;j++)newmax=fmaxf(newmax,lo[offset+j]*.125f);
  for(int j=0;j<512;j++)assert(hi[offset+j]*.125f<=newmax);
  assert(bits(expf(maximum-newmax))==bits(alpha[tile]));maxima[tile]=newmax;
  for(int j=0;j<512;j++){
   int z=offset+j;
   if(cached[z])values[z]=point(ex(lo[z]*.125f-newmax));
   else{
    I centered=sub(mul((I){lo[z],hi[z]},point(.125f)),point(newmax));centered.h=fminf(centered.h,0);values[z]=exi(centered);
    assert(bits(bf(values[z].l))==bits(packed[z]) && bits(bf(values[z].h))==bits(packed[z]));
    double width=(double)values[z].h-values[z].l;
    if(tile==0)width*=alpha[1];
    if(width>0)candidates[count++]=(Candidate){width,z};
   }
  }
  maximum=newmax;
 }
 qsort(candidates,count,sizeof(Candidate),largest_first);
 I bound=denominator_interval(values,alpha);int done=0;
 for(int i=0;i<=count;i++){
  /* Recompute in the original eight-lane source order. Batching refinement
   * affects its cost only, never the numeric certificate or replay decisions. */
  if(i%16==0||i==count){
   bound=denominator_interval(values,alpha);checks++;
   if(certified_outputs(pvlo,pvhi,bound)){done=1;break;}
  }
  if(i==count)break;
  int z=candidates[i].index;float dot=cached_dot(q,k,lo,hi,cached,z,&calls);
  float exact=ex(dot*.125f-maxima[z/512]);assert(values[z].l<=exact && exact<=values[z].h);assert(bits(bf(exact))==bits(packed[z]));values[z]=point(exact);
 }
 assert(done);den[0]=bound.l;den[1]=bound.h;stats[0]=checks;stats[1]=count;stats[2]=calls;return calls;
}

/* The absolute reconstructed digit dot is an additional integer-plane product.
 * Adding the same representation-error bound encloses sum(abs(A*B)) of the
 * original source operands, avoiding the loose row-L1 times column-Linf norm. */
void contraction_bounds_absolute(const float*a,const float*b,const float*ar,const float*br,const double*center,const double*abscenter,float*lo,float*hi,double*repr,double*absolute,int m,int n,int k,int source_length){
 double u=0x1p-24,g=ud(source_length*u,1-source_length*u),eta=ud(source_length*0x1p-149,1-source_length*u);
 for(int i=0;i<m;i++){
  double r1=0,e1=0;
  for(int z=0;z<k;z++){r1=ua(r1,fabs((double)ar[i*k+z]));e1=ua(e1,nextafter(fabs((double)a[i*k+z]-ar[i*k+z]),INFINITY));}
  for(int j=0;j<n;j++){
   double bmax=0,emax=0;for(int z=0;z<k;z++){bmax=fmax(bmax,fabs((double)b[z*n+j]));emax=fmax(emax,nextafter(fabs((double)b[z*n+j]-br[z*n+j]),INFINITY));}
   int t=i*n+j;assert(isfinite(abscenter[t])&&abscenter[t]>=0);double e=ua(um(e1,bmax),um(r1,emax)),s=ua(abscenter[t],e),radius=ua(e,ua(um(g,s),eta));repr[t]=e;absolute[t]=s;
   lo[t]=fl(da(center[t],-radius));hi[t]=fu(ua(center[t],radius));
  }
 }
}
