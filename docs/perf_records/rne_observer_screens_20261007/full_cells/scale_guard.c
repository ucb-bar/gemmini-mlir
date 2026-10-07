#include <stdint.h>
#include <stddef.h>
static inline int cells_prepared_M8_scan_0(const float *scale,size_t count,ptrdiff_t stride){
 if(!count)return 1;
 if(!scale||stride<0||(stride&&(count-1)>(size_t)PTRDIFF_MAX/(size_t)stride))return 0;
 for(size_t i=0;i<count;i++){uint32_t word;__builtin_memcpy(&word,scale+(ptrdiff_t)i*stride,4);
  if((word&0x7fffffffu)>0x72d46fd5u)return 0;}
 return 1;
}

#include <stdint.h>
#include <stddef.h>
static inline int cells_prepared_M8_scan_1(const float *scale,size_t count,ptrdiff_t stride){
 if(!count)return 1;
 if(!scale||stride<0||(stride&&(count-1)>(size_t)PTRDIFF_MAX/(size_t)stride))return 0;
 for(size_t i=0;i<count;i++){uint32_t word;__builtin_memcpy(&word,scale+(ptrdiff_t)i*stride,4);
  if((word&0x7fffffffu)>0x72d46fd5u)return 0;}
 return 1;
}

extern void cells_M8_rne(void*,void*,void*,void*,void*);
extern void control_M8(void*,void*,void*,void*,void*);
void cells_prepared_M8(void*a0,void*a1,void*a2,void*a3,void*a4){if(cells_prepared_M8_scan_0((const float*)a1,5632,1)&&cells_prepared_M8_scan_1((const float*)a3,5632,1))cells_M8_rne(a0,a1,a2,a3,a4);else control_M8(a0,a1,a2,a3,a4);}
