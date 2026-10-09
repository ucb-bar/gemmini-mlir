
#include <stdint.h>
extern int printf(const char *,...);
extern int32_t a[],b[];extern float scale_a[],scale_b[];extern int8_t expected[];
struct d3 {void *alloc,*ptr;long off;long size[3],stride[3];};
struct d1 {void *alloc,*ptr;long off,size,stride;};
extern void _mlir_ciface_baseline(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
extern void _mlir_ciface_broadcast(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
static int8_t guarded[45056+128];
static uint8_t arena[4*45056+1024];static unsigned cursor;
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void *p=arena+cursor;cursor+=n;return p;}
void free(void *p){(void)p;}
static inline uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t hash(const void *ptr,unsigned size){const uint8_t *p=ptr;uint64_t h=1469598103934665603ul;for(unsigned i=0;i<size;i++){h^=p[i];h*=1099511628211ul;}return h;}
static int validate(unsigned arm){for(unsigned i=0;i<45056;i++)if(guarded[i+64]!=expected[i]){printf("FAIL %u %u %d %d\n",arm,i,guarded[i+64],expected[i]);return 1;}for(unsigned i=0;i<64;i++)if(guarded[i]!=73||guarded[45056+64+i]!=73){printf("GUARD_FAIL\n");return 2;}return 0;}
int main(void){
 struct d3 da={a,a,0,{1,8,5632},{45056,5632,1}},db={b,b,0,{1,8,5632},{45056,5632,1}};
 struct d1 sa={scale_a,scale_a,0,5632,1},sb={scale_b,scale_b,0,5632,1};
 struct d3 out={guarded,guarded+64,0,{1,8,5632},{45056,5632,1}};
 uint64_t input_hash[4]={hash(a,180224),hash(scale_a,22528),hash(b,180224),hash(scale_b,22528)};
 for(unsigned i=0;i<45184;i++)guarded[i]=73;
 for(unsigned arm=0;arm<2;arm++){cursor=0;if(arm==0)_mlir_ciface_baseline(&da,&sa,&db,&sb,&out);else _mlir_ciface_broadcast(&da,&sa,&db,&sb,&out);if(validate(arm))return 1;}
 const unsigned order[4]={0,1,1,0};
 for(unsigned i=0;i<4;i++){
  cursor=0;uint64_t t=tick();if(order[i]==0)_mlir_ciface_baseline(&da,&sa,&db,&sb,&out);else _mlir_ciface_broadcast(&da,&sa,&db,&sb,&out);t=tick()-t;
  if(validate(order[i]))return 2;printf("GATE_PACKET_CYCLES %u %u %lu\n",i,order[i],t);
 }
 if(hash(a,180224)!=input_hash[0]||hash(scale_a,22528)!=input_hash[1]||hash(b,180224)!=input_hash[2]||hash(scale_b,22528)!=input_hash[3]){printf("INPUT_CHANGED\n");return 3;}
 printf("ORIGINAL_GATE_PACKET PASS 45056 128\n");return 0;
}
