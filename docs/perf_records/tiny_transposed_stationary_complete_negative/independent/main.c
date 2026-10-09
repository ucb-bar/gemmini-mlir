#include <stdint.h>
extern int printf(const char*,...);
extern const int8_t case0_a[];
extern const int8_t case0_b[];
extern const int8_t case0_bt[];
extern const int32_t case0_gold[];
extern void case0_dense(const int8_t*,const int8_t*,int32_t*);
extern void case0_transposed(const int8_t*,const int8_t*,int32_t*);
extern const int8_t case1_a[];
extern const int8_t case1_b[];
extern const int8_t case1_bt[];
extern const int32_t case1_gold[];
extern void case1_dense(const int8_t*,const int8_t*,int32_t*);
extern void case1_transposed(const int8_t*,const int8_t*,int32_t*);
extern const int8_t case2_a[];
extern const int8_t case2_b[];
extern const int8_t case2_bt[];
extern const int32_t case2_gold[];
extern void case2_dense(const int8_t*,const int8_t*,int32_t*);
extern void case2_transposed(const int8_t*,const int8_t*,int32_t*);
extern const int8_t case3_a[];
extern const int8_t case3_b[];
extern const int8_t case3_bt[];
extern const int32_t case3_gold[];
extern void case3_dense(const int8_t*,const int8_t*,int32_t*);
extern void case3_transposed(const int8_t*,const int8_t*,int32_t*);
struct entry{unsigned m,n,k;const int8_t*a,*b,*bt;const int32_t*gold;void(*dense)(const int8_t*,const int8_t*,int32_t*),(*transposed)(const int8_t*,const int8_t*,int32_t*);};
static struct entry cases[]={{3,35,32,case0_a,case0_b,case0_bt,case0_gold,case0_dense,case0_transposed},{7,73,64,case1_a,case1_b,case1_bt,case1_gold,case1_dense,case1_transposed},{17,65,32,case2_a,case2_b,case2_bt,case2_gold,case2_dense,case2_transposed},{16,64,32,case3_a,case3_b,case3_bt,case3_gold,case3_dense,case3_transposed}};
static int8_t at[4096+128] __attribute__((aligned(64)));
static int32_t ct[4096+128] __attribute__((aligned(64))),out[4096+128] __attribute__((aligned(64)));
static uint32_t sum8(const int8_t*p,unsigned n){uint32_t h=0;for(unsigned i=0;i<n;i++)h=h*31+(uint8_t)p[i];return h;}
int main(void){for(unsigned ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
struct entry*e=&cases[ci];unsigned cn=e->m*e->n,an=e->m*e->k,bn=e->k*e->n;uint32_t ah=sum8(e->a,an),bh=sum8(e->b,bn),bth=sum8(e->bt,bn);
for(unsigned arm=0;arm<2;arm++){
for(unsigned i=0;i<cn+128;i++)out[i]=ct[i]=0x4d4d4d4d;for(unsigned i=0;i<an+128;i++)at[i]=0x4d;
if(arm){for(unsigned r=0;r<e->m;r++)for(unsigned c=0;c<e->k;c++)at[c*e->m+r]=e->a[r*e->k+c];e->transposed(e->bt,at,ct);for(unsigned r=0;r<e->m;r++)for(unsigned c=0;c<e->n;c++)out[r*e->n+c]=ct[c*e->m+r];}else e->dense(e->a,e->b,out);
for(unsigned i=0;i<cn;i++)if(out[i]!=e->gold[i]){printf("VALUE_FAIL %u %u %u %d %d\n",ci,arm,i,out[i],e->gold[i]);return 1;}
for(unsigned i=cn;i<cn+128;i++)if(out[i]!=0x4d4d4d4d||ct[i]!=0x4d4d4d4d){printf("OUTPUT_GUARD_FAIL %u\n",ci);return 2;}
for(unsigned i=an;i<an+128;i++)if(at[i]!=0x4d){printf("A_GUARD_FAIL %u\n",ci);return 3;}
if(ah!=sum8(e->a,an)||bh!=sum8(e->b,bn)||bth!=sum8(e->bt,bn)){printf("INPUT_FAIL %u\n",ci);return 4;}
}printf("INDEPENDENT_TRANSPOSED_I8_CASE PASS %u %u\n",ci,cn);
}printf("INDEPENDENT_TRANSPOSED_STATIONARY PASS\n");return 0;}