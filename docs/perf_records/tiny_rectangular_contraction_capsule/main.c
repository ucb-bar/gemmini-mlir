#include <stdint.h>
extern int printf(const char*,...);
extern float qk_a[],qk_b[],qk_expected[],pv_a[],pv_b[],pv_expected[];
struct d3 {void*a,*p;long off,size[3],stride[3];};
extern void _mlir_ciface_qk_control(struct d3*,struct d3*,struct d3*),_mlir_ciface_qk_rectangular(struct d3*,struct d3*,struct d3*),_mlir_ciface_pv_control(struct d3*,struct d3*,struct d3*),_mlir_ciface_pv_rectangular(struct d3*,struct d3*,struct d3*);
static union{float f;uint32_t w;} out[16384+128],reference[16384];
static uint8_t arena[512*1024];static unsigned cursor;static int errno_value;
int *__errno(void){return &errno_value;}
void *malloc(unsigned long n){cursor=(cursor+63)&~63u;if(cursor+n>sizeof(arena))return 0;void*p=arena+cursor;cursor+=n;return p;}void free(void*p){(void)p;}
static uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
static void mode(unsigned m){asm volatile("csrw frm,%0"::"r"((uint64_t)m):"memory");}
static unsigned fflags(void){unsigned long v;asm volatile("csrr %0,fflags":"=r"(v));return v;}
static void seedflags(void){unsigned long v=8;asm volatile("csrw fflags,%0"::"r"(v):"memory");}
static uint32_t word(float*p,unsigned i){union{float f;uint32_t w;}v={p[i]};return v.w;}
static uint32_t checksum(float*p,unsigned n){uint32_t s=0;for(unsigned i=0;i<n;i++)s^=word(p,i);return s;}
int main(void){
    float*aa[2]={qk_a,pv_a},*bb[2]={qk_b,pv_b},*ee[2]={qk_expected,pv_expected};
    unsigned an[2]={16384,2048},bn[2]={16384,16384},cn[2]={2048,16384};uint32_t ca[2],cb[2];
    for(unsigned f=0;f<2;f++){ca[f]=checksum(aa[f],an[f]);cb[f]=checksum(bb[f],bn[f]);}
    for(unsigned f=0;f<2;f++){
        unsigned k=f?8:64,n=f?64:8;
        struct d3 a={aa[f],aa[f],0,{32,8,k},{8*k,k,1}},b={bb[f],bb[f],0,{32,k,n},{k*n,n,1}},o={out,out,0,{32,8,n},{8*n,n,1}};
        void(*call[2])(struct d3*,struct d3*,struct d3*)={f?_mlir_ciface_pv_control:_mlir_ciface_qk_control,f?_mlir_ciface_pv_rectangular:_mlir_ciface_qk_rectangular};
#ifdef STRICT
        for(unsigned m=0;m<5;m++){mode(m);unsigned flags0=0;for(unsigned arm=0;arm<2;arm++){
            for(unsigned i=0;i<cn[f]+128;i++)out[i].w=0x7fc12345;
            cursor=0;seedflags();call[arm](&a,&b,&o);unsigned flags1=fflags();
            if(!arm)flags0=flags1;else if(flags1!=flags0){printf("FLAGS_FAIL %u %u %u %u\n",f,m,flags0,flags1);return 1;}
            for(unsigned i=0;i<cn[f];i++){
                if(!arm)reference[i].w=out[i].w;else if(out[i].w!=reference[i].w){printf("MODE_FAIL %u %u %u\n",f,m,i);return 2;}
                if(m==0&&out[i].w!=word(ee[f],i)){printf("ORIGINAL_FAIL %u %u\n",f,i);return 3;}
            }
            for(unsigned i=cn[f];i<cn[f]+128;i++)if(out[i].w!=0x7fc12345){printf("GUARD_FAIL\n");return 4;}
        }}
#else
        mode(0);for(unsigned r=0;r<2;r++)for(unsigned step=0;step<2;step++){
            unsigned arm=r?1-step:step;
            for(unsigned i=0;i<cn[f]+128;i++)out[i].w=0x7fc12345;
            cursor=0;uint64_t t=tick();call[arm](&a,&b,&o);t=tick()-t;
            for(unsigned i=0;i<cn[f];i++)if(out[i].w!=word(ee[f],i)){printf("VALUE_FAIL %u %u %u\n",f,arm,i);return 5;}
            for(unsigned i=cn[f];i<cn[f]+128;i++)if(out[i].w!=0x7fc12345){printf("GUARD_FAIL\n");return 6;}
            printf("RECTANGULAR_CONTRACTION_CYCLES %u %u %u %lu\n",f,r,arm,t);
        }
#endif
        if(ca[f]!=checksum(aa[f],an[f])||cb[f]!=checksum(bb[f],bn[f])){printf("INPUT_FAIL\n");return 7;}
    }
    printf("ORIGINAL_CONTRACTION_COMPLETE PASS\n");return 0;
}
