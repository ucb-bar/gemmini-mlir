#include <math.h>
#include "builtin_minmax.h"
#include <stdint.h>
#include <stdio.h>
static uint32_t b32(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static uint64_t b64(double x){uint64_t n;__builtin_memcpy(&n,&x,8);return n;}
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static float(*volatile original_min32)(float,float)=fminf;
static float(*volatile original_max32)(float,float)=fmaxf;
static double(*volatile original_min64)(double,double)=fmin;
static double(*volatile original_max64)(double,double)=fmax;

static int equal32(float a,float b){
 uint32_t x=b32(a),y=b32(b);
 if((x&UINT32_C(0x7fffffff))==0&&(y&UINT32_C(0x7fffffff))==0)return 1;
 if((x&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000)&&
    (y&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000))return 1;
 return x==y;
}
static int equal64(double a,double b){
 uint64_t x=b64(a),y=b64(b);
 if((x&UINT64_C(0x7fffffffffffffff))==0&&(y&UINT64_C(0x7fffffffffffffff))==0)return 1;
 if((x&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000)&&
    (y&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000))return 1;
 return x==y;
}

static unsigned long checks;
static int pair32(uint32_t a,uint32_t b){
 float x=f32(a),y=f32(b);checks++;
 return equal32(original_min32(x,y),MERLIN_SOURCE_F32_MIN(x,y))&&
        equal32(original_max32(x,y),MERLIN_SOURCE_F32_MAX(x,y));
}
static int pair64(uint64_t a,uint64_t b){
 double x=f64(a),y=f64(b);checks++;
 return equal64(original_min64(x,y),MERLIN_SOURCE_F64_MIN(x,y))&&
        equal64(original_max64(x,y),MERLIN_SOURCE_F64_MAX(x,y));
}
static int observed;
static double once(void){observed++;return observed==1?3:4;}
int main(void){
 unsigned long previous;__asm__ volatile("csrr %0,frm":"=r"(previous));
 uint64_t rng=713;
 const uint32_t df[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff,0xffffffff};
 const uint64_t dd[]={0,UINT64_C(0x8000000000000000),1,UINT64_C(0x8000000000000001),UINT64_C(0x000fffffffffffff),UINT64_C(0x800fffffffffffff),UINT64_C(0x0010000000000000),UINT64_C(0x8010000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0xbff0000000000000),UINT64_C(0x7fefffffffffffff),UINT64_C(0xffefffffffffffff),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000000),UINT64_C(0xfff8000000000000),UINT64_C(0x7ff0000000000001),UINT64_C(0xfff0000000000001),UINT64_C(0x7fffffffffffffff),UINT64_C(0xffffffffffffffff)};
 for(unsigned long mode=0;mode<5;mode++){
  __asm__ volatile("csrw frm,%0"::"r"(mode));
  for(unsigned i=0;i<20;i++)for(unsigned j=0;j<20;j++){
   if(!pair32(df[i],df[j]))return 1;
   if(!pair64(dd[i],dd[j]))return 2;
  }
  for(uint32_t i=0;i<65536;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!pair32(i<<16,(uint32_t)rng))return 3;
   if(!pair32((uint32_t)rng,i<<16))return 4;
  }
  for(unsigned i=0;i<20000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;uint64_t a=rng;
   rng=rng*UINT64_C(6364136223846793005)+1;uint64_t b=rng;
   if(!pair32((uint32_t)a,(uint32_t)b))return 5;
   if(!pair64(a,b))return 6;
  }
  observed=0;if(MERLIN_SOURCE_F64_MIN(once(),once())!=3||observed!=2)return 7;
  observed=0;if(MERLIN_SOURCE_F64_MAX(once(),once())!=4||observed!=2)return 8;
  unsigned long current;__asm__ volatile("csrr %0,frm":"=r"(current));
  if(current!=mode)return 9;
 }
 __asm__ volatile("csrw frm,%0"::"r"(previous));
 printf("BUILTIN_MINMAX PASS %lu\n",checks);
 return 0;
}
