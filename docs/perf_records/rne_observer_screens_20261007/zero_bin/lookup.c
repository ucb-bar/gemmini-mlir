#include <stdint.h>
extern const float table_b16[65536][2];
extern float cells_source_activation(float);
extern signed char cells_quantize(float);
static inline uint32_t bits(float x){uint32_t w;__builtin_memcpy(&w,&x,4);return w;}
__attribute__((always_inline)) signed char cells_lookup_activation(float x,float up,float scale){
 uint32_t w=bits(x);
 const float*cell=table_b16[w>>16];float lo=cell[0],hi=cell[1];
 if(!(lo<=hi))goto source_fallback;
 float low_product=lo*up,high_product=hi*up;
 float low_scaled=low_product*scale,high_scaled=high_product*scale;
 uint32_t endpoint_magnitude=(bits(low_scaled)|bits(high_scaled))&0x7fffffffu;
 if(endpoint_magnitude<=1056964608u)return 0;
 signed char low_word=cells_quantize(low_scaled),high_word=cells_quantize(high_scaled);
 if(low_word!=high_word)goto source_fallback;
 return low_word;
source_fallback:;
 float original=cells_source_activation(x);
 float product=original*up;
 float scaled=product*scale;
 return cells_quantize(scaled);
}
