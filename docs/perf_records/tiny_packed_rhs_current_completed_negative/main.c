#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],input_b[];extern const int32_t expected[];
extern const uint64_t chosen_arm,expected_b_xor,expected_a_xor;
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*);
extern void gemmini_golden_gemm(const int8_t*,const int8_t*,int32_t*);
static int32_t destination[8*5632+128] __attribute__((aligned(64)));
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t words_xor(const void *ptr,unsigned long n){const uint64_t*p=ptr;uint64_t x=0;for(unsigned long i=0;i<n/8;i++)x^=p[i];return x;}
static int check(void){for(unsigned i=0;i<8*5632;i++)if(destination[i]!=expected[i]){printf("VALUE_FAIL %u %d %d\n",i,destination[i],expected[i]);return 0;}for(unsigned i=8*5632;i<8*5632+128;i++)if(destination[i]!=0x4d4d4d4d){printf("GUARD_FAIL %u\n",i);return 0;}return 1;}
int main(void){void(*fn)(const int8_t*,const int8_t*,int32_t*)=chosen_arm?gemmini_golden_gemm:gemmini_golden_a5705ab56e324ba1;
if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_FAIL\n");return 1;}
for(unsigned repeat=0;repeat<3;repeat++){for(unsigned i=0;i<8*5632+128;i++)destination[i]=0x4d4d4d4d;uint64_t t=tick();fn(input_a,input_b,destination);t=tick()-t;if(!check())return 2;printf("PACKED_RHS_COMPLETE_CYCLES %lu %u %lu\n",chosen_arm,repeat,t);}
if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_MUTATED\n");return 3;}
printf("PACKED_RHS_COMPLETE PASS %lu 45056\n",chosen_arm);return 0;}
