#include <stdint.h>
extern int printf(const char*,...);
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


static int check(void){
 typedef void(*fn)(float*,float*,float*);
 fn source[8]={source_probe0,source_probe1,source_probe2,source_probe3,source_probe4,source_probe5,source_probe6,source_probe7};fn selected[8]={selected_probe0,selected_probe1,selected_probe2,selected_probe3,selected_probe4,selected_probe5,selected_probe6,selected_probe7};
 const unsigned directed[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x3f800001,0x3f7fffff,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f812345,0xff812345};
 unsigned rng=0x49e613a7,sa[4],sb[4],outa[4],outb[4],words=0;float a[4],b[4],oa[4],ob[4];
 for(unsigned frm=0;frm<5;frm++)for(unsigned op=0;op<8;op++)for(unsigned trial=0;trial<534;trial++){
  for(unsigned j=0;j<4;j++){rng=rng*1664525+1013904223;sa[j]=trial<22?directed[(trial+j)%22]:rng;rng=rng*1664525+1013904223;sb[j]=trial<22?directed[(trial+2*j)%22]:rng;__builtin_memcpy(a+j,sa+j,4);__builtin_memcpy(b+j,sb+j,4);}
  unsigned f0,f1;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");source[op](a,b,oa);asm volatile("csrr %0,fflags":"=r"(f0)::"memory");
  asm volatile("csrw fflags,%0"::"r"(8):"memory");selected[op](a,b,ob);asm volatile("csrr %0,fflags":"=r"(f1)::"memory");
  for(unsigned j=0;j<4;j++){__builtin_memcpy(outa+j,oa+j,4);__builtin_memcpy(outb+j,ob+j,4);if(outa[j]!=outb[j]){printf("FMA_ALIAS_BITS_FAIL %u %u %u %u %x %x\n",frm,op,trial,j,outa[j],outb[j]);return 1;}words++;}
  if(f0!=f1){printf("FMA_ALIAS_FLAGS_FAIL %u %u %u %u %u\n",frm,op,trial,f0,f1);return 2;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("FMA_ALIAS PASS %u fivefrm stickyflags directed_specials\n",words);return 0;
}
int main(void){return check();}
