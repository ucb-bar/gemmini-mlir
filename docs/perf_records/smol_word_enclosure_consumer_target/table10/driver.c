#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {void*allocated,*aligned;int64_t offset,size[4],stride[4];} D4;
typedef struct {void*allocated,*aligned;int64_t offset,size[1],stride[1];} D1;
extern const uint16_t raw_q[],raw_k[],raw_v[],policy_gold[];
extern const uint8_t raw_mask[];
extern void attention_frontier_writer(D4*,D4*,D4*,D4*,D4*,D4*,D4*,D4*,D4*,D4*,D4*,D4*,D1*);
extern int group_provider_statistics(void*,size_t,uint64_t*,size_t);
static _Alignas(64) unsigned char workspace[121963584+64];
static uint16_t output[196608+256];
/* Capsule refusal is fatal. Normal production links the actual retained source
 * function; this deliberate stub must never be used for production admission. */
void _mlir_ciface_source_attention_frontier_fallback(D4*a,D4*b,D4*c,D4*d,D4*e,D4*f,D4*g,D4*h,D4*i,D4*j,D4*k,D4*out){puts("CAPSULE UNEXPECTED SOURCE REFUSAL");exit(2);}
static D4 view(const void*p,int64_t offset,int h,int m,int n,int64_t hs,int64_t ms){D4 v={(void*)p,(void*)p,offset,{1,h,m,n},{h*hs,hs,ms,1}};return v;}
static uint64_t tick(void){uint64_t n;__asm__ volatile("csrr %0,mcycle":"=r"(n)::"memory");return n;}

#include <string.h>
#include "benchmark_buffer.h"
typedef struct {void*allocated,*aligned;int64_t offset,size[3],stride[3];} D3;
typedef struct {void*allocated,*aligned;int64_t offset,size[2],stride[2];} D2;
extern const uint16_t original_source[],original_scale[];
extern const int8_t original_i8[];
extern void _mlir_ciface_source_quant_frontier(D4*,D3*,D2*);
static _Alignas(64) uint16_t consumer_input[12*1024*64];
static _Alignas(64) int8_t consumer_output[1024*768+64];
static _Alignas(64) uint16_t consumer_scales[1024+32];
static _Alignas(64) uint16_t saved_q[12*256*64],saved_k[12*1024*64],saved_v[12*1024*64];
static _Alignas(64) uint8_t saved_mask[256*1024];
static int check_consumer(const uint16_t*carrier){
 for(int h=0;h<12;h++)for(int part=0;part<4;part++)
  memcpy(consumer_input+(h*1024+part*256)*64,carrier+h*256*64,256*64*2);
 merlin_benchmark_fill(consumer_output,0xa5,sizeof(consumer_output));
 merlin_benchmark_fill(consumer_scales,0x5a,sizeof(consumer_scales));
 D4 input=view(consumer_input,0,12,1024,64,1024*64,64);
 D3 q={consumer_output,consumer_output,0,{1,1024,768},{1024*768,768,1}};
 D2 scale={consumer_scales,consumer_scales,0,{1,1024},{1024,1}};
 _mlir_ciface_source_quant_frontier(&input,&q,&scale);
 if(q.allocated!=consumer_output||q.aligned!=consumer_output||q.offset||
    scale.allocated!=consumer_scales||scale.aligned!=consumer_scales||scale.offset)return 20;
 for(int part=0;part<4;part++){
  if(merlin_benchmark_first_difference(consumer_output+part*256*768,original_i8,256*768)!=256*768)return 21;
  if(merlin_benchmark_first_difference(consumer_scales+part*256,original_scale,256*2)!=256*2)return 22;
 }
 for(int i=1024*768;i<1024*768+64;i++)if((uint8_t)consumer_output[i]!=0xa5)return 23;
 for(int i=1024;i<1024+32;i++)if(consumer_scales[i]!=0x5a5a)return 24;
 return 0;
}


extern void printstr(const char*);
extern void tohost_exit(uintptr_t) __attribute__((noreturn));
void htif_putc(char c){char s[2]={c,0};printstr(s);}
void htif_puts(const char*s){printstr(s);}
void htif_putd(long v){printf("%ld",v);}
void htif_puthex(unsigned long long v){printf("0x%lx",(unsigned long)v);}
void htif_exit(int code){tohost_exit(code);}
uintptr_t handle_trap(uintptr_t cause,uintptr_t epc,uintptr_t regs[32]){
 uintptr_t value;__asm__ volatile("csrr %0,mtval":"=r"(value));
 printf("CONSUMER_TRAP cause=%lu pc=%lx value=%lx\n",cause,epc,value);
 tohost_exit(1337);
}

int main(void){
 memcpy(saved_q,raw_q,sizeof(saved_q));memcpy(saved_k,raw_k,sizeof(saved_k));memcpy(saved_v,raw_v,sizeof(saved_v));memcpy(saved_mask,raw_mask,sizeof(saved_mask));

 D4 a[11];a[0]=view(raw_q,0,12,256,64,256*64,64);
 for(int t=0;t<2;t++){
  int base=t?6:1;
  a[base]=view(raw_k,t*512*64,12,512,64,1024*64,64);
  a[base+1]=view(raw_mask,t*512,1,256,512,256*1024,1024);
  for(int s=0;s<3;s++)a[base+2+s]=view(raw_v,(t*512+s*192)*64,12,s==2?128:192,64,1024*64,64);
 }
 D4 out=view(output,0,12,256,64,256*64,64);D1 scratch={workspace,workspace,0,{121963584},{1}};
 for(int i=0;i<196608+256;i++)output[i]=0xa55a;
 for(int i=121963584;i<121963584+64;i++)workspace[i]=0x5a;
 uint64_t start=tick();attention_frontier_writer(&a[0],&a[1],&a[2],&a[3],&a[4],&a[5],&a[6],&a[7],&a[8],&a[9],&a[10],&out,&scratch);uint64_t elapsed=tick()-start;

 printf("WORKSPACE_GROUP_INSTRUCTIONS %lu\n",elapsed);
 unsigned differences=0;
 for(int i=0;i<196608;i++)if(output[i]!=policy_gold[i]){
  ++differences;printf("UNOBSERVED_CARRIER_DIFF %d %u %u\n",i,policy_gold[i],output[i]);
 }
 printf("UNOBSERVED_CARRIER_DIFFERENCES %u\n",differences);
 int reference_status=check_consumer(original_source);if(reference_status){printf("ORIGINAL_CONSUMER_FAIL %d\n",reference_status);return reference_status;}
 int candidate_status=check_consumer(output);if(candidate_status){printf("CANDIDATE_CONSUMER_FAIL %d\n",candidate_status);return candidate_status;}
 if(merlin_benchmark_first_difference(raw_q,saved_q,sizeof(saved_q))!=sizeof(saved_q)||
    merlin_benchmark_first_difference(raw_k,saved_k,sizeof(saved_k))!=sizeof(saved_k)||
    merlin_benchmark_first_difference(raw_v,saved_v,sizeof(saved_v))!=sizeof(saved_v)||
    merlin_benchmark_first_difference(raw_mask,saved_mask,sizeof(saved_mask))!=sizeof(saved_mask))return 25;

 for(int i=196608;i<196608+256;i++)if(output[i]!=0xa55a)return 4;
 for(int i=121963584;i<121963584+64;i++)if(workspace[i]!=0x5a)return 5;
 if(out.allocated!=output||out.aligned!=output||out.offset!=0)return 6;
 uint64_t stats[8];if(!group_provider_statistics(workspace,121963584,stats,8))return 7;
for(int i=0;i<8;i++)printf("WORKSPACE_STAT %d %lu\n",i,stats[i]);
 printf("%s\n","WORKSPACE_GROUP ORIGINAL_COMPILED_CONSUMER AND GUARDS PASS");return 0;
}
