
#include "benchmark_buffer.h"
#include <assert.h>
static size_t oracle(const unsigned char *a,const unsigned char *b,size_t n){
 for(size_t i=0;i<n;i++)if(a[i]!=b[i])return i;
 return n;
}

#include <stdio.h>
static unsigned char cost_a[8192] __attribute__((aligned(__alignof__(uint64_t))));
static unsigned char cost_b[8192] __attribute__((aligned(__alignof__(uint64_t))));
static __attribute__((noinline)) size_t byte_difference(const void*a,const void*b,size_t n){
 const unsigned char*x=a,*y=b;
 for(size_t i=0;i<n;i++)if(x[i]!=y[i])return i;
 return n;
}
static __attribute__((noinline)) size_t word_difference(const void*a,const void*b,size_t n){
 return merlin_benchmark_first_difference(a,b,n);
}
static void measure(void){
 for(size_t i=0;i<8192;i++)cost_a[i]=cost_b[i]=(unsigned char)(i*73+19);
 for(size_t offset=0;offset<2;offset++){
  unsigned long start,end;
  __asm__ volatile("csrr %0,mcycle":"=r"(start)::"memory");
  for(int i=0;i<16;i++){__asm__ volatile("":::"memory");assert(byte_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);}
  __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
  printf("BENCH_BYTE offset%lu instructions%lu\n",(unsigned long)offset,end-start);
  __asm__ volatile("csrr %0,mcycle":"=r"(start)::"memory");
  for(int i=0;i<16;i++){__asm__ volatile("":::"memory");assert(word_difference(cost_a+offset,cost_b+offset,8192-offset)==8192-offset);}
  __asm__ volatile("csrr %0,mcycle":"=r"(end)::"memory");
  printf("BENCH_WORD offset%lu instructions%lu\n",(unsigned long)offset,end-start);
 }
 printf("BENCHMARK_BUFFER EXACT PASS\n");
}

int main(void){
 unsigned char a[320],b[320],saved[320];
 assert(merlin_benchmark_first_difference(0,0,0)==0);
 merlin_benchmark_fill(0,123,0);
 for(size_t ax=0;ax<16;ax++)for(size_t bx=0;bx<16;bx++){
  for(size_t i=0;i<320;i++)a[i]=(unsigned char)(i*73+19);
  for(size_t n=0;n<=257;n++){
   for(size_t i=0;i<320;i++)b[i]=0x5a;
   for(size_t i=0;i<n;i++)b[bx+i]=a[ax+i];
   memcpy(saved,b,sizeof(b));
   assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==n);
   assert(merlin_benchmark_first_difference(a+ax,a+ax,n)==n);
   assert(!memcmp(b,saved,sizeof(b)));
   for(size_t position=0;position<n;position++){
    b[bx+position]^=0x80;
    assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==position);
    assert(merlin_benchmark_first_difference(b+bx,a+ax,n)==position);
    assert(oracle(a+ax,b+bx,n)==position);
    if(position+1<n){
     b[bx+position+1]^=1;
     assert(merlin_benchmark_first_difference(a+ax,b+bx,n)==position);
     b[bx+position+1]^=1;
    }
    b[bx+position]^=0x80;
   }
  }
 }
 for(unsigned value=0;value<256;value++)for(size_t offset=0;offset<16;offset++){
  for(size_t n=0;n<=257;n++){
   for(size_t i=0;i<320;i++)b[i]=0xa5;
   merlin_benchmark_fill(b+offset,(unsigned char)value,n);
   for(size_t i=0;i<320;i++)
    assert(b[i]==(i>=offset&&i<offset+n?(unsigned char)value:0xa5));
  }
 }
 measure();
 return 0;
}
