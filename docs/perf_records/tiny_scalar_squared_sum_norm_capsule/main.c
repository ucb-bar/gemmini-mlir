#include <stdint.h>
extern int printf(const char *,...);
extern float weight[],activation[];extern int8_t expected[];
struct d1 {void*a,*p;long off,size,stride;};struct d3{void*a,*p;long off,size[3],stride[3];};
extern void _mlir_ciface_control(struct d1*,struct d3*,struct d3*),_mlir_ciface_accumulated(struct d1*,struct d3*,struct d3*);
static int8_t out[16384+128];static uint8_t arena[256*1024];static unsigned cursor;static int errno_value;
int *__errno(void){return &errno_value;}
extern void *memcpy(void*d,const void*s,unsigned long n);
extern void *memset(void*d,int v,unsigned long n);
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static void mode(unsigned m){asm volatile("csrw frm,%0"::"r"((uint64_t)m):"memory");}
static unsigned flags(void){unsigned long f;asm volatile("csrr %0,fflags":"=r"(f));return (unsigned)f;}
static void clear(void){asm volatile("csrw fflags,zero":::"memory");}
int main(void){struct d1 w={weight,weight,0,2048,1};struct d3 a={activation,activation,0,{1,8,2048},{16384,2048,1}},o={out,out,0,{1,8,2048},{16384,2048,1}};unsigned control_flags[5];static int8_t mode_reference[16384];
for(unsigned m=0;m<5;m++){mode(m);clear();cursor=0;_mlir_ciface_control(&w,&a,&o);control_flags[m]=flags();memcpy(mode_reference,out,16384);if(m==0)for(unsigned i=0;i<16384;i++)if(out[i]!=expected[i]){printf("FAIL_NATIVE %u\n",i);return 1;}
clear();cursor=0;_mlir_ciface_accumulated(&w,&a,&o);unsigned candidate_flags=flags();if(candidate_flags!=control_flags[m]){printf("FLAG_FAIL %u %u %u\n",m,control_flags[m],candidate_flags);return 2;}for(unsigned i=0;i<16384;i++)if(out[i]!=mode_reference[i]){printf("MODE_FAIL %u %u\n",m,i);return 3;}}
mode(0);for(unsigned r=0;r<2;r++)for(unsigned step=0;step<2;step++){unsigned arm=r==0?step:1-step;for(unsigned i=0;i<sizeof(out);i++)out[i]=73;cursor=0;uint64_t t=tick();if(arm==0)_mlir_ciface_control(&w,&a,&o);else _mlir_ciface_accumulated(&w,&a,&o);t=tick()-t;
for(unsigned i=0;i<16384;i++)if(out[i]!=expected[i]){printf("VALUE_FAIL %u %u\n",arm,i);return 4;}for(unsigned i=16384;i<sizeof(out);i++)if(out[i]!=73){printf("GUARD_FAIL\n");return 5;}printf("SQUARED_SUM_NORM_COMPLETE_CYCLES %u %u %lu\n",r,arm,t);}
printf("SOURCE_SQUARED_SUM_NORM_COMPLETE PASS\n");return 0;}
