#include <stdint.h>
extern int printf(const char*,...);
extern void source_case0(float*,float*);
extern void source_case1(float*,float*);
extern void source_case2(float*,float*);
extern void source_case3(float*,float*);
extern void source_case4(float*,float*);
extern void source_case5(float*,float*);
extern void selected_case0(float*,float*);
extern void selected_case1(float*,float*);
extern void selected_case2(float*,float*);
extern void selected_case3(float*,float*);
extern void selected_case4(float*,float*);
extern void selected_case5(float*,float*);
typedef void(*fn)(float*,float*);
int main(void){
 fn source[6]={source_case0,source_case1,source_case2,source_case3,source_case4,source_case5};fn selected[6]={selected_case0,selected_case1,selected_case2,selected_case3,selected_case4,selected_case5};unsigned n[6]={3,5,7,6,5,7};
 const unsigned directed[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x3f800001,0x3f7fffff,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f812345,0xff812345};
 unsigned rng=0x714ce813,total=0;float a[8],s[8],t[8];
 for(unsigned frm=0;frm<5;frm++)for(unsigned op=0;op<6;op++)for(unsigned trial=0;trial<278;trial++){
  for(unsigned j=0;j<8;j++){rng=rng*1664525+1013904223;unsigned raw=trial<22?directed[(trial+j)%22]:rng;__builtin_memcpy(a+j,&raw,4);s[j]=73;t[j]=74;}
  unsigned f0,f1;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");source[op](a,s);asm volatile("csrr %0,fflags":"=r"(f0)::"memory");
  asm volatile("csrw fflags,%0"::"r"(8):"memory");selected[op](a,t);asm volatile("csrr %0,fflags":"=r"(f1)::"memory");
  for(unsigned j=0;j<n[op];j++){unsigned x,y;__builtin_memcpy(&x,s+j,4);__builtin_memcpy(&y,t+j,4);if(x!=y){printf("DRAFT_BITS_FAIL %u %u %u %u %x %x\n",frm,op,trial,j,x,y);return 1;}total++;}
  if(f0!=f1){printf("DRAFT_FLAGS_FAIL %u %u %u %u %u\n",frm,op,trial,f0,f1);return 2;}
  for(unsigned j=n[op];j<8;j++)if(s[j]!=73||t[j]!=74){printf("TAIL_POISON_FAIL\n");return 3;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("DRAFT_GENERIC_FMA PASS %u fivefrm stickyflags sharedSSA tails\n",total);return 0;
}
