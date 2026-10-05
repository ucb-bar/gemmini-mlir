#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t gemmini_exact_requant_24_readout_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t gemmini_exact_requant_24_readout_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 if((int64_t)x<680LL)return (int8_t)0;
 if((int64_t)x>=171922LL)return (int8_t)127;
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=gemmini_exact_requant_24_readout_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<gemmini_exact_requant_24_readout_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void gemmini_exact_requant_24_readout(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=gemmini_exact_requant_24_readout_scalar(acc[i+0]);
 out[i+1]=gemmini_exact_requant_24_readout_scalar(acc[i+1]);
 out[i+2]=gemmini_exact_requant_24_readout_scalar(acc[i+2]);
 out[i+3]=gemmini_exact_requant_24_readout_scalar(acc[i+3]);
 out[i+4]=gemmini_exact_requant_24_readout_scalar(acc[i+4]);
 out[i+5]=gemmini_exact_requant_24_readout_scalar(acc[i+5]);
 out[i+6]=gemmini_exact_requant_24_readout_scalar(acc[i+6]);
 out[i+7]=gemmini_exact_requant_24_readout_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=gemmini_exact_requant_24_readout_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
extern void gemmini_exact_requant_24_kernel(int8_t*,int8_t*,int32_t*);
void _mlir_ciface_gemmini_exact_requant_24(memref2 *r,memref4 *a,memref4 *b,memref2 *scratch,memref2 *c) {
 if(!scratch->aligned || scratch->offset<0 || scratch->sizes[0]!=196 || scratch->sizes[1]!=256 || scratch->strides[0]!=256 || scratch->strides[1]!=1 || ((uintptr_t)((int32_t*)scratch->aligned+scratch->offset)&3))__builtin_trap();
 if (a->offset < 0 || !a->aligned || a->sizes[0] != 1 || a->strides[0] != 200704 || a->sizes[1] != 28 || a->strides[1] != 7168 || a->sizes[2] != 28 || a->strides[2] != 256 || a->sizes[3] != 256 || a->strides[3] != 1 || b->offset < 0 || !b->aligned || b->sizes[0] != 3 || b->strides[0] != 196608 || b->sizes[1] != 3 || b->strides[1] != 65536 || b->sizes[2] != 256 || b->strides[2] != 256 || b->sizes[3] != 256 || b->strides[3] != 1 || c->offset < 0 || !c->aligned || c->sizes[0] != 196 || c->strides[0] != 256 || c->sizes[1] != 256 || c->strides[1] != 1) __builtin_trap();
 gemmini_exact_requant_24_kernel((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int32_t*)scratch->aligned+scratch->offset);
 gemmini_exact_requant_24_readout((const int32_t*)scratch->aligned+scratch->offset,(int8_t*)c->aligned+c->offset,50176);*r=*c;
}
