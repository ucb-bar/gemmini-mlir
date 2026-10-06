/* Experimental generic closed integer observer. Requires RNE, gradual underflow,
 * nontrapping and unobserved flags/errno; otherwise original source must run.
 * Every multiplication is rounded separately, matching source order. */
#include <stdint.h>
#include <string.h>
#include <math.h>
static inline uint32_t bits(float x) { uint32_t u;__builtin_memcpy(&u,&x,4);return u; }
static inline float asfloat(uint32_t u) {float x;__builtin_memcpy(&x,&u,4);return x;}
static inline int interior(float x) {unsigned e=(bits(x)>>23)&255;return e>=2&&e<=252;}
static inline float original(float d,float a,float b,float c){float r=1.0f/d;float p=a*r;float q=p*b;return q*c;}
__attribute__((always_inline)) inline float reciprocal_observer(float d,float a,float b,float c) {
 uint32_t u=bits(d);
 if(u<0x01800000u||u>=0x7d000000u) return original(d,a,b,c);
 float r=asfloat(0x7ef311c3u-u);
 r=r*__builtin_fmaf(-d,r,2.0f);
 r=r*__builtin_fmaf(-d,r,2.0f);
 float p=a*r;float q=p*b;float z=q*c;
 if(!interior(p)||!interior(q)||!interior(z))return original(d,a,b,c);
 float mag=asfloat(bits(z)&0x7fffffffu);
 /* Large outputs are in the same source [-128,127] saturation bin because
  * proven source/candidate ratio lies in (1-2^-15,1+2^-15). */
 if(mag>=256.0f)return z;
 float rn=(mag+0x1p23f)-0x1p23f;
 float distance=__builtin_fabsf(mag-rn);
 if(distance<0.5f-0x1p-7f)return z;
 return original(d,a,b,c);
}
__attribute__((always_inline)) float exported_observer(float d,float a,float b,float c){return reciprocal_observer(d,a,b,c);}
float exported_original(float d,float a,float b,float c){return original(d,a,b,c);}
int exported_fast(float d,float a,float b,float c) {
 uint32_t u=bits(d);if(u<0x01800000u||u>=0x7d000000u)return 0;
 float r=asfloat(0x7ef311c3u-u);r=r*__builtin_fmaf(-d,r,2);r=r*__builtin_fmaf(-d,r,2);
 float p=a*r,q=p*b,z=q*c;if(!interior(p)||!interior(q)||!interior(z))return 0;
 float mag=asfloat(bits(z)&0x7fffffff);if(mag>=256)return 1;
 float rn=(mag+0x1p23f)-0x1p23f;return __builtin_fabsf(mag-rn)<0.5f-0x1p-7f;
}
void vector_test(const float *d,const float *a,const float *b,const float *c,int *source,int *candidate,int *fast,unsigned long n) {
 for(unsigned long i=0;i<n;i++){
  float x=exported_original(d[i],a[i],b[i],c[i]);float y=exported_observer(d[i],a[i],b[i],c[i]);
  x=__builtin_fminf(__builtin_fmaxf(x,-128),127);y=__builtin_fminf(__builtin_fmaxf(y,-128),127);
  source[i]=(int)__builtin_roundevenf(x);candidate[i]=(int)__builtin_roundevenf(y);fast[i]=exported_fast(d[i],a[i],b[i],c[i]);
 }
}
