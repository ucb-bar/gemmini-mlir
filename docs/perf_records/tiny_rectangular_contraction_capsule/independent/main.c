#include <stdint.h>
extern int printf(const char*,...);
struct d3{void*a,*p;long off,size[3],stride[3];};
extern void _mlir_ciface_case0_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case0_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case0_a[],case0_b[],case0_c[];
extern void _mlir_ciface_case1_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case1_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case1_a[],case1_b[],case1_c[];
extern void _mlir_ciface_case2_control(struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case2_candidate(struct d3*,struct d3*,struct d3*);
extern float case2_a[],case2_b[],case2_c[];
extern void _mlir_ciface_case3_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case3_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case3_a[],case3_b[],case3_c[];
extern void _mlir_ciface_case4_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case4_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case4_a[],case4_b[],case4_c[];
extern void _mlir_ciface_case5_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case5_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case5_a[],case5_b[],case5_c[];
extern void _mlir_ciface_case6_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case6_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case6_a[],case6_b[],case6_c[];
extern void _mlir_ciface_case7_control(struct d3*,struct d3*,struct d3*,struct d3*);
extern void _mlir_ciface_case7_candidate(struct d3*,struct d3*,struct d3*,struct d3*);
extern float case7_a[],case7_b[],case7_c[];
struct entry{unsigned m,n,k,trans,alias;float*a,*b,*c;void*control,*candidate;};
static struct entry cases[]={{4,12,5,0,0,case0_a,case0_b,case0_c,(void*)_mlir_ciface_case0_control,(void*)_mlir_ciface_case0_candidate},{2,8,7,1,0,case1_a,case1_b,case1_c,(void*)_mlir_ciface_case1_control,(void*)_mlir_ciface_case1_candidate},{4,4,4,0,1,case2_a,case2_b,case2_c,(void*)_mlir_ciface_case2_control,(void*)_mlir_ciface_case2_candidate},{2,8,0,0,0,case3_a,case3_b,case3_c,(void*)_mlir_ciface_case3_control,(void*)_mlir_ciface_case3_candidate},{2,8,3,0,0,case4_a,case4_b,case4_c,(void*)_mlir_ciface_case4_control,(void*)_mlir_ciface_case4_candidate},{3,4,5,0,0,case5_a,case5_b,case5_c,(void*)_mlir_ciface_case5_control,(void*)_mlir_ciface_case5_candidate},{0,8,3,0,0,case6_a,case6_b,case6_c,(void*)_mlir_ciface_case6_control,(void*)_mlir_ciface_case6_candidate},{2,0,3,0,0,case7_a,case7_b,case7_c,(void*)_mlir_ciface_case7_control,(void*)_mlir_ciface_case7_candidate}};
static union{float f;uint32_t w;} out[2048+128],reference[2048];
static uint8_t arena[1024*1024];static unsigned cursor;static int err;
int *__errno(void){return &err;}
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint32_t checksum(float*p,unsigned n){uint32_t s=0;for(unsigned i=0;i<n;i++){union{float f;uint32_t w;}v={p[i]};s^=v.w;}return s;}
int main(void){for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
struct entry*e=&cases[ci];unsigned an=2*e->m*e->k,bn=2*e->k*e->n,cn=2*e->m*e->n;
uint32_t ca=checksum(e->a,an),cb=checksum(e->b,bn),cc=checksum(e->c,cn);
struct d3 a={e->a,e->a,0,{2,e->m,e->k},{e->m*e->k,e->k,1}},b={e->b,e->b,0,{2,e->trans?e->n:e->k,e->trans?e->k:e->n},{e->k*e->n,e->trans?e->k:e->n,1}},c={e->c,e->c,0,{2,e->m,e->n},{e->m*e->n,e->n,1}},o={out,out,0,{2,e->m,e->n},{e->m*e->n,e->n,1}};
for(unsigned frm=0;frm<5;frm++){asm volatile("csrw frm,%0"::"r"((unsigned long)frm):"memory");unsigned baseflags=0;
for(unsigned arm=0;arm<2;arm++){
for(unsigned i=0;i<cn+128;i++)out[i].w=0x7fc12345;cursor=0;unsigned long seed=8;asm volatile("csrw fflags,%0"::"r"(seed):"memory");
void*call=arm?e->candidate:e->control;if(e->alias)((void(*)(struct d3*,struct d3*,struct d3*))call)(&a,&b,&o);else((void(*)(struct d3*,struct d3*,struct d3*,struct d3*))call)(&a,&b,&c,&o);
unsigned long flags;asm volatile("csrr %0,fflags":"=r"(flags));if(!arm)baseflags=flags;else if(baseflags!=flags){printf("FLAG_FAIL %u %u %u %lu\n",ci,frm,baseflags,flags);return 1;}
for(unsigned i=0;i<cn;i++){if(!arm)reference[i].w=out[i].w;else if(reference[i].w!=out[i].w){printf("VALUE_FAIL %u %u %u %x %x\n",ci,frm,i,reference[i].w,out[i].w);return 2;}}
for(unsigned i=cn;i<cn+128;i++)if(out[i].w!=0x7fc12345){printf("GUARD_FAIL %u\n",ci);return 3;}
if(ca!=checksum(e->a,an)||cb!=checksum(e->b,bn)||cc!=checksum(e->c,cn)){printf("INPUT_FAIL %u\n",ci);return 4;}
}}
printf("RECTANGULAR_INDEPENDENT_CASE PASS %u %u\n",ci,cn);
}asm volatile("csrw frm,zero":::"memory");printf("RECTANGULAR_INDEPENDENT_TARGET PASS\n");return 0;}
