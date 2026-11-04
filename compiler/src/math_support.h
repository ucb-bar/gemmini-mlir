#pragma once
#include <string>
namespace mc {
inline std::string hostSupport() {return R"C(
#include <stdint.h>
#define HS_INLINE static __attribute__((always_inline)) inline
HS_INLINE float __host_f32bits(uint32_t x){union{uint32_t u;float f;}v;v.u=x;return v.f;}
HS_INLINE uint32_t __host_bitsf(float x){union{uint32_t u;float f;}v;v.f=x;return v.u;}
HS_INLINE double __host_f64bits(uint64_t x){union{uint64_t u;double f;}v;v.u=x;return v.f;}
HS_INLINE float __host_absf(float x){return __host_f32bits(__host_bitsf(x)&0x7fffffffu);}
HS_INLINE float __host_sqrtf(float x){float r;
#ifdef __riscv
 __asm__("fsqrt.s %0,%1":"=f"(r):"f"(x));return r;
#else
 return __builtin_sqrtf(x);
#endif
}
HS_INLINE float __host_fmaf(float x,float y,float z){float r;
#ifdef __riscv
 __asm__("fmadd.s %0,%1,%2,%3":"=f"(r):"f"(x),"f"(y),"f"(z));return r;
#else
 return __builtin_fmaf(x,y,z);
#endif
}
HS_INLINE float __host_rintf(float x){
 uint32_t ax=__host_bitsf(x)&0x7fffffffu;
 int small=ax<0x4b000000u;
 float bounded=small?x:0.0f;int r;
#ifdef __riscv
 __asm__("fcvt.w.s %0,%1,rne":"=r"(r):"f"(bounded));
#else
 int n=(int)bounded;double d=(double)bounded-n;if(d>0.5||(d==0.5&&(n&1)))n++;if(d< -0.5||(d== -0.5&&(n&1)))n--;r=n;
#endif
 float rounded=(float)r;
 rounded=r==0?bounded*0.0f:rounded;
 return small?rounded:x;
}
HS_INLINE float __host_floorf(float x){if(__host_absf(x)>=8388608.0f||x!=x)return x;int r=(int)x;if((float)r>x)--r;return r?(float)r:__host_f32bits(__host_bitsf(x)&0x80000000u);}
HS_INLINE float __host_ceilf(float x){if(__host_absf(x)>=8388608.0f||x!=x)return x;int r=(int)x;if((float)r<x)++r;return r?(float)r:__host_f32bits(__host_bitsf(x)&0x80000000u);}
HS_INLINE float __host_roundf(float x){if(__host_absf(x)>=8388608.0f||x!=x)return x;int r=(int)((double)x+(x<0?-0.5:0.5));return r?(float)r:__host_f32bits(__host_bitsf(x)&0x80000000u);}
/* Reduction in double avoids the loss in a float x - n*log(2). The
   remaining interval is [-log(2)/2,log(2)/2]; degree fourteen Taylor
   evaluation has error well below the precision of a float result. */
HS_INLINE double __host_exp(double x){
 if(x!=x)return x;if(x>89.0)return (double)__host_f32bits(0x7f800000u);if(x< -105.0)return 0.0;
 int n=(int)(x*1.44269504088896340736+(x<0?-0.5:0.5));
 double r=(x-n*0.69314718036912381649)-n*1.90821492927058770002e-10;
 double p=1.14707455977297247139e-11;
 p=1.60590438368216145994e-10+r*p;p=2.08767569878680989792e-9+r*p;
 p=2.50521083854417187751e-8+r*p;p=2.75573192239858925110e-7+r*p;
 p=2.75573192239858906526e-6+r*p;p=2.48015873015873015844e-5+r*p;
 p=1.98412698412698412530e-4+r*p;p=1.38888888888888894189e-3+r*p;
 p=8.33333333333333321769e-3+r*p;p=4.16666666666666643537e-2+r*p;
 p=1.66666666666666657415e-1+r*p;p=0.5+r*p;p=1.0+r*p;p=1.0+r*p;
 return p*__host_f64bits((uint64_t)(n+1023)<<52);
}
HS_INLINE float __host_expf(float x){return (float)__host_exp((double)x);}
HS_INLINE float __host_exp2f(float x){return (float)__host_exp((double)x*0.693147180559945309417);}
HS_INLINE double __host_log(float x){
 uint32_t u=__host_bitsf(x);if((u&0x7fffffffu)==0)return -(double)__host_f32bits(0x7f800000u);
 if(u&0x80000000u)return (double)__host_f32bits(0x7fc00000u);if(u>=0x7f800000u)return (double)x;
 int e=(int)(u>>23)-127;if(e== -127){x*=8388608.0f;u=__host_bitsf(x);e=(int)(u>>23)-150;}
 double m=(double)__host_f32bits((u&0x007fffffu)|0x3f800000u);if(m>1.41421356237309504880){m*=0.5;++e;}
 double t=(m-1.0)/(m+1.0),z=t*t;
 double p=1.0/19.0;p=1.0/17.0+z*p;p=1.0/15.0+z*p;p=1.0/13.0+z*p;p=1.0/11.0+z*p;p=1.0/9.0+z*p;p=1.0/7.0+z*p;p=1.0/5.0+z*p;p=1.0/3.0+z*p;p=1.0+z*p;
 return 2.0*t*p+e*0.693147180559945309417;
}
HS_INLINE float __host_logf(float x){return (float)__host_log(x);}
HS_INLINE float __host_log2f(float x){return (float)(__host_log(x)*1.44269504088896340736);}
HS_INLINE float __host_tanhf(float x){if(x!=x)return x;float a=__host_absf(x);if(a>=10.0f)return x<0?-1.0f:1.0f;if(a<0.0001f)return x;double p=__host_exp(2.0*(double)a);float t=(float)((p-1.0)/(p+1.0));return x<0?-t:t;}
/* Payne-Hanek reduction. The 256 fractional bits of 2/pi retain more
   than 120 useful reduction bits even at the largest finite f32. */
HS_INLINE double __host_trig_reduce(float x,int *quadrant){
 uint32_t u=__host_bitsf(x)&0x7fffffffu;if(u<=0x3f490fdau){*quadrant=0;return (double)x;}
 static const uint32_t invpi[8]={0xDEBBC561u,0xFE5163ABu,0x3C439041u,0xDB629599u,0xF534DDC0u,0xFC2757D1u,0x4E441529u,0xA2F9836Eu};
 uint32_t product[9];uint64_t carry=0;uint32_t mant=(u&0x7fffffu)|0x800000u;
 for(int i=0;i<8;++i){uint64_t v=(uint64_t)invpi[i]*mant+carry;product[i]=(uint32_t)v;carry=v>>32;}product[8]=(uint32_t)carry;
 int exponent=(int)(u>>23)-127;int shift=279-exponent;
 int q=(int)((product[shift>>5]>>(shift&31))&1u);int s=shift+1;q|=(int)((product[s>>5]>>(s&31))&1u)<<1;
 double frac=0.0,weight=0.5;for(int i=1;i<=60;++i){int k=shift-i;if((product[k>>5]>>(k&31))&1u)frac+=weight;weight*=0.5;}
 if(frac>=0.5){frac-=1.0;q=(q+1)&3;}if(__host_bitsf(x)>>31){frac=-frac;q=(-q)&3;}*quadrant=q;return frac*1.570796326794896619231;
}
HS_INLINE double __host_sin_poly(double r){double z=r*r;return r+r*z*(-1.66666666666666657415e-1+z*(8.33333333333333321769e-3+z*(-1.98412698412698412530e-4+z*(2.75573192239858925110e-6+z*(-2.50521083854417187751e-8+z*(1.60590438368216145994e-10+z*(-7.64716373181981647590e-13)))))));}
HS_INLINE double __host_cos_poly(double r){double z=r*r;return 1.0+z*(-0.5+z*(4.16666666666666643537e-2+z*(-1.38888888888888894189e-3+z*(2.48015873015873015844e-5+z*(-2.75573192239858906526e-7+z*(2.08767569878680989792e-9+z*(-1.14707455977297247139e-11+z*4.77947733238738529744e-14)))))));}
HS_INLINE float __host_sinf(float x){if((__host_bitsf(x)&0x7fffffffu)<0x39800000u)return x;if((__host_bitsf(x)&0x7fffffffu)>=0x7f800000u)return x-x;int q;double r=__host_trig_reduce(x,&q);double y=(q&1)?__host_cos_poly(r):__host_sin_poly(r);return (float)((q&2)?-y:y);}
HS_INLINE float __host_cosf(float x){if((__host_bitsf(x)&0x7fffffffu)>=0x7f800000u)return x-x;int q;double r=__host_trig_reduce(x,&q);double y=(q&1)?__host_sin_poly(r):__host_cos_poly(r);return (float)(((q+1)&2)?-y:y);}
HS_INLINE float __host_erff(float x){
 if(x!=x)return x;double a=(double)__host_absf(x),v;if(a>=6.0)return x<0?-1.0f:1.0f;
 double z=a*a;if(a<=2.0){double term=a,sum=a;for(int n=1;n<=36;++n){term*= -z/n;sum+=term/(2*n+1);}v=sum*1.12837916709551257390;}
 else{double b=z+0.5,c=1.0e100,d=1.0/b,h=d;for(int i=1;i<=64;++i){double an= -i*(i-0.5);b+=2.0;d=an*d+b;c=b+an/c;d=1.0/d;h*=d*c;}v=1.0-__host_exp(-z)*a*h*0.56418958354775628695;}
 return (float)((__host_bitsf(x)>>31)?-v:v);
}
HS_INLINE float __host_powf(float x,float y){
 uint32_t ax=__host_bitsf(x)&0x7fffffffu,ay=__host_bitsf(y)&0x7fffffffu;
 if(ay==0||x==1.0f)return 1.0f;if(x!=x||y!=y)return x+y;
 if(ay==0x7f800000u){float a=__host_absf(x);if(a==1.0f)return 1.0f;return ((a>1.0f)==(y>0))?__host_f32bits(0x7f800000u):0.0f;}
 int odd=0,integer=0;if(ay>=0x4b800000u)integer=1;else{int n=(int)y;integer=(float)n==y;odd=integer&&(n&1);}
 if(ax==0||ax==0x7f800000u){float v=((ax==0)==(y>0))?0.0f:__host_f32bits(0x7f800000u);return (__host_bitsf(x)>>31)&&odd?-v:v;}
 if(x<0&&!integer)return __host_f32bits(0x7fc00000u);
 float v=(float)__host_exp((double)y*__host_log(__host_absf(x)));return x<0&&odd?-v:v;
}
#define __builtin_expf __host_expf
#define __builtin_exp2f __host_exp2f
#define __builtin_logf __host_logf
#define __builtin_log2f __host_log2f
#define __builtin_tanhf __host_tanhf
#define __builtin_sinf __host_sinf
#define __builtin_cosf __host_cosf
#define __builtin_erff __host_erff
#define __builtin_powf __host_powf
#define __builtin_sqrtf __host_sqrtf
#define __builtin_fmaf __host_fmaf
#define __builtin_rintf __host_rintf
#define __builtin_floorf __host_floorf
#define __builtin_ceilf __host_ceilf
#define __builtin_roundf __host_roundf
#define __builtin_fabsf __host_absf
)C";}
}
