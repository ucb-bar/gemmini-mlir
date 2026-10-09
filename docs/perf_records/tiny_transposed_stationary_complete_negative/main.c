#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t input_a[],input_b[];extern const int32_t expected[];
extern const uint64_t chosen_arm,expected_b_xor,expected_a_xor;
extern void gemmini_golden_a5705ab56e324ba1(const int8_t*,const int8_t*,int32_t*),gemmini_golden_gemm(const int8_t*,const int8_t*,int32_t*);
struct d2{void*a,*p;long off,size[2],stride[2];};
extern void _mlir_ciface_prepare_activation(struct d2*,struct d2*),_mlir_ciface_restore_output(struct d2*,struct d2*);
static int8_t activation_t[2048*8+128] __attribute__((aligned(64)));
static int32_t output_t[5632*8+128] __attribute__((aligned(64))),destination[8*5632+128] __attribute__((aligned(64)));
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t words_xor(const void*ptr,unsigned long n){const uint64_t*p=ptr;uint64_t x=0;for(unsigned long i=0;i<n/8;i++)x^=p[i];return x;}
static int check(void){for(unsigned i=0;i<8*5632;i++)if(destination[i]!=expected[i]){printf("VALUE_FAIL %u %d %d\n",i,destination[i],expected[i]);return 0;}for(unsigned i=8*5632;i<8*5632+128;i++)if(destination[i]!=0x4d4d4d4d||output_t[i]!=0x4d4d4d4d){printf("OUTPUT_GUARD_FAIL %u\n",i);return 0;}for(unsigned i=2048*8;i<2048*8+128;i++)if(activation_t[i]!=0x4d){printf("A_GUARD_FAIL %u\n",i);return 0;}return 1;}
int main(void){
 struct d2 a={(void*)input_a,(void*)input_a,0,{8,2048},{2048,1}},at={activation_t,activation_t,0,{2048,8},{8,1}},ct={output_t,output_t,0,{5632,8},{8,1}},c={destination,destination,0,{8,5632},{5632,1}};
 if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_FAIL\n");return 1;}
 for(unsigned repeat=0;repeat<3;repeat++){
 for(unsigned i=0;i<8*5632+128;i++)destination[i]=output_t[i]=0x4d4d4d4d;for(unsigned i=0;i<2048*8+128;i++)activation_t[i]=0x4d;
 uint64_t t=tick();
 if(chosen_arm){_mlir_ciface_prepare_activation(&a,&at);gemmini_golden_gemm(input_b,activation_t,output_t);_mlir_ciface_restore_output(&ct,&c);}else gemmini_golden_a5705ab56e324ba1(input_a,input_b,destination);
 t=tick()-t;
 if(!check())return 2;
 if(chosen_arm)for(unsigned i=0;i<2048;i++)for(unsigned j=0;j<8;j++)if(activation_t[i*8+j]!=input_a[j*2048+i]){printf("A_PERM_FAIL\n");return 3;}
 printf("TRANSPOSED_STATIONARY_COMPLETE_CYCLES %lu %u %lu\n",chosen_arm,repeat,t);
 }
 if(words_xor(input_a,8*2048)!=expected_a_xor||words_xor(input_b,2048*5632)!=expected_b_xor){printf("INPUT_MUTATED\n");return 4;}
 printf("TRANSPOSED_STATIONARY_COMPLETE PASS %lu 45056\n",chosen_arm);return 0;
}