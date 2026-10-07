#include <stdint.h>
static inline int8_t observer_rne(float x){float lo=-128.0f,hi=127.0f;long n;
 __asm__ volatile("fmax.s %0,%0,%2\n\tfmin.s %0,%0,%3\n\tfcvt.w.s %1,%0,rne":"+&f"(x),"=r"(n):"f"(lo),"f"(hi));return(int8_t)n;}
#define MERLIN_CLOSED_RNE_I8 observer_rne
#define MERLIN_SOURCE_BITCAST_COPY __builtin_memcpy
