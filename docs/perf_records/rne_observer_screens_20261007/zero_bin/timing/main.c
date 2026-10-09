#include <stdint.h>
extern int printf(const char*,...);
typedef void(*F)(int32_t*,float*,int32_t*,float*,int8_t*);
extern void lazy_M8_b16_rne(int32_t*,float*,int32_t*,float*,int8_t*);
extern void control_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern void selected_M8_b16(int32_t*,float*,int32_t*,float*,int8_t*);

extern void lazy_M8_b16(int32_t*,float*,int32_t*,float*,int8_t*);
extern void integer_M8_b16(int32_t*,float*,int32_t*,float*,int8_t*);
extern void finite_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern void cells_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern int32_t a[],b[];extern float scale_a[],scale_b[];extern int8_t expected[];
static int8_t guarded[45184],source[45184],original[45184];
static void clear(int8_t*p){for(unsigned i=0;i<45184;i++)p[i]=73;}
static uint64_t hash(const void*p,unsigned n){const uint8_t*x=p;uint64_t h=1469598103934665603ul;for(unsigned i=0;i<n;i++){h^=x[i];h*=1099511628211ul;}return h;}
static unsigned flags(void){unsigned f;asm volatile("csrr %0,fflags":"=r"(f)::"memory");return f;}
static void mode(unsigned frm,unsigned flags){asm volatile("csrw frm,%0\ncsrw fflags,%1"::"r"(frm),"r"(flags):"memory");}
static int valid(void){for(unsigned i=0;i<45184;i++)if((i<64||i>=45120)?guarded[i]!=73:guarded[i]!=expected[i-64])return 1;return 0;}
static uint64_t ticks(void){uint64_t v;asm volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
static uint64_t instructions(void){uint64_t v;asm volatile("csrr %0,minstret":"=r"(v)::"memory");return v;}
static void fence(void){asm volatile("fence":::"memory");}
int main(void){
 uint64_t inputpins[4]={hash(a,180224),hash(scale_a,22528),hash(b,180224),hash(scale_b,22528)};
 const unsigned presets[]={0,1,2,4,8,16,31};
 clear(guarded);mode(0,0);cells_M8(a,scale_a,b,scale_b,guarded+64);
 if(valid()){printf("INTEGER_RESULT_FAIL initialRNE\n");return 1;}
 printf("INTEGER_RESULT_SOURCE_GATE initialRNE words=45056 original guards PASS\n");
 printf("INTEGER_RESULT_POINTERS a=%lx sa=%lx b=%lx sb=%lx out=%lx\n",(unsigned long)a,(unsigned long)scale_a,(unsigned long)b,(unsigned long)scale_b,(unsigned long)(guarded+64));
 F pair[2]={finite_M8,cells_M8};const unsigned order[4]={0,1,1,0};
 for(unsigned sample=0;sample<4;sample++){
  unsigned id=order[sample];clear(guarded);mode(0,0);fence();uint64_t t=ticks(),i=instructions();pair[id](a,scale_a,b,scale_b,guarded+64);fence();i=instructions()-i;t=ticks()-t;
  if(valid())return 3;
  printf("INTEGER_RESULT_ROW id=%u sample=%u cycles=%lu instructions=%lu digest=%lx\n",id,sample,t,i,(unsigned long)hash(guarded+64,45056));
 }
 if(hash(a,180224)!=inputpins[0]||hash(scale_a,22528)!=inputpins[1]||hash(b,180224)!=inputpins[2]||hash(scale_b,22528)!=inputpins[3])return 4;
 printf("INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0\n");return 0;
}
