
#include <stdint.h>
extern int printf(const char *,...);
extern int32_t a[],b[];extern float scale_a[],scale_b[];extern int8_t expected[];
struct d3 {void *alloc,*ptr;long off;long size[3],stride[3];};
struct d1 {void *alloc,*ptr;long off,size,stride;};
extern void _mlir_ciface_control(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
extern void _mlir_ciface_materialized(struct d3*,struct d1*,struct d3*,struct d1*,struct d3*);
static int8_t guarded[45056+128];
static uint8_t arena[4*45056+1024];static unsigned cursor;
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void *p=arena+cursor;cursor+=n;return p;}
void free(void *p){(void)p;}
static inline uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static uint64_t hash(const void *ptr,unsigned size){const uint8_t *p=ptr;uint64_t h=1469598103934665603ul;unsigned i=0;for(;i+8<=size;i+=8){uint64_t word;__builtin_memcpy(&word,p+i,8);h^=word;h*=1099511628211ul;}for(;i<size;i++){h^=p[i];h*=1099511628211ul;}return h;}
static void poison(void){for(unsigned i=0;i<45184;i++)guarded[i]=73;for(unsigned i=0;i<45056;i++)guarded[i+64]=expected[i]==73?74:73;}
static int validate(unsigned arm){for(unsigned i=0;i<45056;i++)if(guarded[i+64]!=expected[i]){printf("FAIL %u %u %d %d\n",arm,i,guarded[i+64],expected[i]);return 1;}for(unsigned i=0;i<64;i++)if(guarded[i]!=73||guarded[45056+64+i]!=73){printf("GUARD_FAIL\n");return 2;}return 0;}
extern void source_probe0(float*,float*,float*);
extern void source_probe1(float*,float*,float*);
extern void source_probe2(float*,float*,float*);
extern void source_probe3(float*,float*,float*);
extern void source_probe4(float*,float*,float*);
extern void source_probe5(float*,float*,float*);
extern void source_probe6(float*,float*,float*);
extern void source_probe7(float*,float*,float*);
extern void selected_probe0(float*,float*,float*);
extern void selected_probe1(float*,float*,float*);
extern void selected_probe2(float*,float*,float*);
extern void selected_probe3(float*,float*,float*);
extern void selected_probe4(float*,float*,float*);
extern void selected_probe5(float*,float*,float*);
extern void selected_probe6(float*,float*,float*);
extern void selected_probe7(float*,float*,float*);

static int check_primitives(void){
 typedef void(*fn)(float*,float*,float*);
 fn source[8]={source_probe0,source_probe1,source_probe2,source_probe3,source_probe4,source_probe5,source_probe6,source_probe7};fn selected[8]={selected_probe0,selected_probe1,selected_probe2,selected_probe3,selected_probe4,selected_probe5,selected_probe6,selected_probe7};
 unsigned rng=0x6f48a293;unsigned sa[4],sb[4],outa[4],outb[4];float a[4],b[4],oa[4],ob[4];
 for(unsigned frm=0;frm<5;frm++)for(unsigned op=0;op<8;op++)for(unsigned trial=0;trial<512;trial++){
  for(unsigned j=0;j<4;j++){rng=rng*1664525+1013904223;sa[j]=rng;rng=rng*1664525+1013904223;sb[j]=rng;__builtin_memcpy(a+j,sa+j,4);__builtin_memcpy(b+j,sb+j,4);}
  unsigned f0,f1;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");source[op](a,b,oa);asm volatile("csrr %0,fflags":"=r"(f0)::"memory");
  asm volatile("csrw fflags,%0"::"r"(8):"memory");selected[op](a,b,ob);asm volatile("csrr %0,fflags":"=r"(f1)::"memory");
  for(unsigned j=0;j<4;j++){__builtin_memcpy(outa+j,oa+j,4);__builtin_memcpy(outb+j,ob+j,4);if(outa[j]!=outb[j]){printf("FMA_BITS_FAIL %u %u %u %u\n",frm,op,trial,j);return 1;}}
  if(f0!=f1){printf("FMA_FLAGS_FAIL %u %u %u %u %u\n",frm,op,trial,f0,f1);return 2;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("FMA_PRIMITIVE PASS 81920 fivefrm stickyflags\n");return 0;
}

int main(void){asm volatile("csrw fflags,zero":::"memory");unsigned frm;asm volatile("csrr %0,frm":"=r"(frm));if(frm!=0){printf("RNE_REFUSED\n");return 9;}
 struct d3 da={a,a,0,{1,8,5632},{45056,5632,1}},db={b,b,0,{1,8,5632},{45056,5632,1}};
 struct d1 sa={scale_a,scale_a,0,5632,1},sb={scale_b,scale_b,0,5632,1};
 struct d3 out={guarded,guarded+64,0,{1,8,5632},{45056,5632,1}};
 uint64_t input_hash[4]={hash(a,180224),hash(scale_a,22528),hash(b,180224),hash(scale_b,22528)};
 for(unsigned i=0;i<45184;i++)guarded[i]=73;
 for(unsigned arm=0;arm<2;arm++){poison();cursor=0;if(arm==0)_mlir_ciface_control(&da,&sa,&db,&sb,&out);else _mlir_ciface_materialized(&da,&sa,&db,&sb,&out);if(validate(arm))return 1;}
 const unsigned order[4]={0,1,1,0};
 for(unsigned i=0;i<4;i++){
  poison();cursor=0;uint64_t t=tick();if(order[i]==0)_mlir_ciface_control(&da,&sa,&db,&sb,&out);else _mlir_ciface_materialized(&da,&sa,&db,&sb,&out);t=tick()-t;
  if(validate(order[i]))return 2;printf("FMA_LIFETIME_CYCLES %u %u %lu\n",i,order[i],t);
 }
 if(hash(a,180224)!=input_hash[0]||hash(scale_a,22528)!=input_hash[1]||hash(b,180224)!=input_hash[2]||hash(scale_b,22528)!=input_hash[3]){printf("INPUT_CHANGED\n");return 3;}
 printf("ORIGINAL_FMA_LIFETIME PASS 45056 128\n");return 0;
}
