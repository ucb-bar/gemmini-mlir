#ifndef MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H
#define MERLIN_ADMITTED_SOURCE_NUMERIC_CAPABILITY_H
#ifdef MERLIN_SOURCE_F32_MATH_H
#error "source numeric capability must precede numeric headers"
#endif
#if !defined(__clang__) && !defined(__GNUC__)
#error "source numeric builtin capability unavailable"
#endif
#if !defined(__has_builtin)
#error "absolute-value compiler capability unavailable"
#elif !__has_builtin(__builtin_fabsf) || !__has_builtin(__builtin_fabs)
#error "absolute-value compiler capability unavailable"
#endif
#define MERLIN_SOURCE_F32_ABS(x) __builtin_fabsf((x))
#define MERLIN_SOURCE_F64_ABS(x) __builtin_fabs((x))
#endif

#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
#include <stdio.h>
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static uint32_t b32(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static uint64_t b64(double x){uint64_t n;__builtin_memcpy(&n,&x,8);return n;}
static float (*volatile source32)(float)=fabsf;
static double (*volatile source64)(double)=fabs;
static int check32(uint32_t n){
 uint32_t expected=n&UINT32_C(0x7fffffff);
 return b32(source32(f32(n)))==expected&&b32(MERLIN_SOURCE_F32_ABS(f32(n)))==expected;
}
static int check64(uint64_t n){
 uint64_t expected=n&UINT64_C(0x7fffffffffffffff);
 return b64(source64(f64(n)))==expected&&b64(MERLIN_SOURCE_F64_ABS(f64(n)))==expected;
}
static int seen;
static double once(void){seen++;return -1;}
int main(void){
 unsigned long original;__asm__ volatile("csrr %0,frm":"=r"(original));
 uint64_t rng=713;
 for(int mode=0;mode<5;mode++){
  __asm__ volatile("csrw frm,%0"::"r"((unsigned long)mode));
  for(uint32_t i=0;i<65536;i++)if(!check32(i<<16))return 2;
  uint32_t mf[]={0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff};
  uint64_t md[]={0,1,UINT64_C(0x7ffffffffffff),UINT64_C(0x8000000000000),UINT64_C(0xffffffffffffe),UINT64_C(0xfffffffffffff)};
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<256;e++)for(unsigned m=0;m<6;m++)
   if(!check32((s<<31)|(e<<23)|mf[m]))return 3;
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<2048;e++)for(unsigned m=0;m<6;m++)
   if(!check64(((uint64_t)s<<63)|((uint64_t)e<<52)|md[m]))return 4;
  for(unsigned i=0;i<10000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!check32((uint32_t)rng)||!check64(rng))return 5;
  }
  seen=0;if(MERLIN_SOURCE_F64_ABS(once())!=1||seen!=1)return 6;
  unsigned long current;__asm__ volatile("csrr %0,frm":"=r"(current));if(current!=(unsigned long)mode)return 7;
 }
 __asm__ volatile("csrw frm,%0"::"r"(original));
 printf("ABSOLUTE PASS 565925\n");
 return 0;
}
