#include <stdint.h>
extern const float table_b16[65536][2];
extern float builtinzero_source_activation(float);
extern signed char builtinzero_quantize(float);
static inline uint32_t bits(float x){uint32_t w;__builtin_memcpy(&w,&x,4);return w;}
__attribute__((always_inline)) signed char builtinzero_lookup_activation(float x,float up,float scale){
 uint32_t w=bits(x);
 const float*cell=table_b16[w>>16];float lo=cell[0],hi=cell[1];
 if(!(lo<=hi))goto source_fallback;
 uint32_t zero_product_limit;
 switch(bits(scale)){case 1133693052u: zero_product_limit=987708844u;break;default:goto source_fallback;}
 float endpoint_absolute=__builtin_fmaxf(__builtin_fabsf(lo),__builtin_fabsf(hi));
 float up_absolute=__builtin_fabsf(up);
 float product_bound=endpoint_absolute*up_absolute;
 if((bits(product_bound)&0x7fffffffu)<=zero_product_limit)return 0;
 float low_product=lo*up,high_product=hi*up;
 float low_scaled=low_product*scale,high_scaled=high_product*scale;
 signed char low_word=builtinzero_quantize(low_scaled),high_word=builtinzero_quantize(high_scaled);
 if(low_word!=high_word)goto source_fallback;
 return low_word;
source_fallback:;
 float original=builtinzero_source_activation(x);
 float product=original*up;
 float scaled=product*scale;
 return builtinzero_quantize(scaled);
}
