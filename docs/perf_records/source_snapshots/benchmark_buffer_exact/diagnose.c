#include "benchmark_buffer.h"
#include <stdio.h>
static unsigned char a[64] __attribute__((aligned(__alignof__(uint64_t))));
static unsigned char b[64] __attribute__((aligned(__alignof__(uint64_t))));
int main(void){
 for(unsigned i=0;i<64;i++)a[i]=b[i]=(unsigned char)i;
 printf("BEFORE_ALIGNED\n");
 if(merlin_benchmark_first_difference(a,b,64)!=64)return 1;
 printf("BEFORE_UNALIGNED\n");
 if(merlin_benchmark_first_difference(a+1,b+1,63)!=63)return 2;
 printf("BEFORE_COUNTER\n");
 unsigned long count;__asm__ volatile("rdinstret %0":"=r"(count)::"memory");
 printf("AFTER_COUNTER %lu\n",count);
 return 0;
}
