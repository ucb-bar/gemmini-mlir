#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t gemmini_exact_requant_47_readout_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t gemmini_exact_requant_47_readout_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 if((int64_t)x<732LL)return (int8_t)0;
 if((int64_t)x>=185064LL)return (int8_t)127;
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=gemmini_exact_requant_47_readout_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<gemmini_exact_requant_47_readout_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void gemmini_exact_requant_47_readout(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=gemmini_exact_requant_47_readout_scalar(acc[i+0]);
 out[i+1]=gemmini_exact_requant_47_readout_scalar(acc[i+1]);
 out[i+2]=gemmini_exact_requant_47_readout_scalar(acc[i+2]);
 out[i+3]=gemmini_exact_requant_47_readout_scalar(acc[i+3]);
 out[i+4]=gemmini_exact_requant_47_readout_scalar(acc[i+4]);
 out[i+5]=gemmini_exact_requant_47_readout_scalar(acc[i+5]);
 out[i+6]=gemmini_exact_requant_47_readout_scalar(acc[i+6]);
 out[i+7]=gemmini_exact_requant_47_readout_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=gemmini_exact_requant_47_readout_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
#ifndef GEMMINI_DIRECT_CONV_ABI
#define GEMMINI_DIRECT_CONV_ABI
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} memref2;
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[4],strides[4];} memref4;
#endif
extern void gemmini_exact_requant_47_kernel(int8_t*,int8_t*,int32_t*);
void _mlir_ciface_gemmini_exact_requant_47(memref2 *r,memref4 *a,memref4 *b,memref2 *scratch,memref2 *c) {
 if(!scratch->aligned || scratch->offset<0 || scratch->sizes[0]!=49 || scratch->sizes[1]!=512 || scratch->strides[0]!=512 || scratch->strides[1]!=1 || ((uintptr_t)((int32_t*)scratch->aligned+scratch->offset)&3))__builtin_trap();
 if (a->offset < 0 || !a->aligned || a->sizes[0] != 1 || a->strides[0] != 25088 || a->sizes[1] != 7 || a->strides[1] != 3584 || a->sizes[2] != 7 || a->strides[2] != 512 || a->sizes[3] != 512 || a->strides[3] != 1 || b->offset < 0 || !b->aligned || b->sizes[0] != 3 || b->strides[0] != 786432 || b->sizes[1] != 3 || b->strides[1] != 262144 || b->sizes[2] != 512 || b->strides[2] != 512 || b->sizes[3] != 512 || b->strides[3] != 1 || c->offset < 0 || !c->aligned || c->sizes[0] != 49 || c->strides[0] != 512 || c->sizes[1] != 512 || c->strides[1] != 1) __builtin_trap();
 gemmini_exact_requant_47_kernel((int8_t*)a->aligned+a->offset,(int8_t*)b->aligned+b->offset,(int32_t*)scratch->aligned+scratch->offset);
 gemmini_exact_requant_47_readout((const int32_t*)scratch->aligned+scratch->offset,(int8_t*)c->aligned+c->offset,25088);*r=*c;
}
