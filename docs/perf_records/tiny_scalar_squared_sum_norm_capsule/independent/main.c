#include <stdint.h>
extern int printf(const char*,...);extern void *memcpy(void*,const void*,unsigned long);
struct d1{void*a,*p;long o,s,t;};struct d2{void*a,*p;long o,s[2],t[2];};
extern void _mlir_ciface_control(struct d2*,struct d1*,struct d1*),_mlir_ciface_selected(struct d2*,struct d1*,struct d1*);
static const uint32_t words[]={0,0x80000000,1,0x80000001,0x3f800001,0xbf800001,0x7f800000,0xff800000,0x7fc00123,0x7f800123,0x7f7fffff,0xff7fffff,0x45800000,0xcb800000};
static uint32_t a[84],save[84],seeds[12],reference[12],initial[12],result[12+32];
static unsigned char arena[8192];static unsigned cursor;
static int errno_value;int *__errno(void){return &errno_value;}
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
int main(void){struct d2 in={a,a,0,{12,7},{7,1}};struct d1 s={seeds,seeds,0,12,1},out={result+16,result+16,0,12,1};
 for(unsigned rot=0;rot<14;rot++){
  for(unsigned i=0;i<84;i++)a[i]=save[i]=words[(i+rot)%14];
  for(unsigned i=0;i<12;i++)initial[i]=words[(i+rot+4)%14];
  for(unsigned frm=0;frm<5;frm++){
   unsigned long oldflags,newflags;
   for(unsigned i=0;i<12;i++)seeds[i]=initial[i];for(unsigned i=0;i<44;i++)result[i]=0xdeadbeef;cursor=0;
   asm volatile("csrw frm,%0;csrw fflags,%1"::"r"((unsigned long)frm),"r"(8UL):"memory");
   _mlir_ciface_control(&in,&s,&out);asm volatile("csrr %0,fflags":"=r"(oldflags)::"memory");
   for(unsigned i=0;i<12;i++)reference[i]=result[i+16];
   for(unsigned i=0;i<12;i++)seeds[i]=initial[i];for(unsigned i=0;i<44;i++)result[i]=0xdeadbeef;cursor=0;
   asm volatile("csrw fflags,%0"::"r"(8UL):"memory");
   _mlir_ciface_selected(&in,&s,&out);asm volatile("csrr %0,fflags":"=r"(newflags)::"memory");
   if(oldflags!=newflags){printf("SQUARED_FLAGS_FAIL %u %u %lu %lu\n",rot,frm,oldflags,newflags);return 1;}
   for(unsigned i=0;i<12;i++)if(result[i+16]!=reference[i]){printf("SQUARED_WORD_FAIL %u %u %u %u %u\n",rot,frm,i,result[i+16],reference[i]);return 2;}
   for(unsigned i=0;i<16;i++)if(result[i]!=0xdeadbeef||result[i+28]!=0xdeadbeef){printf("SQUARED_GUARD_FAIL\n");return 3;}
   for(unsigned i=0;i<84;i++)if(a[i]!=save[i]){printf("SQUARED_INPUT_FAIL\n");return 4;}
  }
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");
 printf("INDEPENDENT_SQUARED_SUM_FIVE_FRM_STICKY PASS 840\nDONE\n");return 0;
}