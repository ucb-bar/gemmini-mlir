#ifndef GEMMINI_HOST_SOURCE_FMA_BATCH_H
#define GEMMINI_HOST_SOURCE_FMA_BATCH_H
static inline void gemmini_host_source_fma_eight(
 const float fraction[8],float product[8],float coefficient) {
 __asm__("fmadd.s %0,%8,%0,%16\n\tfmadd.s %1,%9,%1,%16\n\tfmadd.s %2,%10,%2,%16\n\tfmadd.s %3,%11,%3,%16\n\tfmadd.s %4,%12,%4,%16\n\tfmadd.s %5,%13,%5,%16\n\tfmadd.s %6,%14,%6,%16\n\tfmadd.s %7,%15,%7,%16"
         : "+&f"(product[0]),"+&f"(product[1]),"+&f"(product[2]),"+&f"(product[3]),"+&f"(product[4]),"+&f"(product[5]),"+&f"(product[6]),"+&f"(product[7])
         : "f"(fraction[0]),"f"(fraction[1]),"f"(fraction[2]),"f"(fraction[3]),"f"(fraction[4]),"f"(fraction[5]),"f"(fraction[6]),"f"(fraction[7]),"f"(coefficient));
}
#define MERLIN_SOURCE_F32_FMA_EIGHT gemmini_host_source_fma_eight
#endif
#define MERLIN_ENABLE_SOURCE_FMA_BATCH_8 1
#if !defined(MERLIN_SOURCE_F32_FMA_EIGHT)
#error "explicit independent source FMA batch provider required"
#endif
