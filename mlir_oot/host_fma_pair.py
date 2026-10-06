"""Explicit RV64GC implementation of Merlin's independent source FMA pair."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFmaPairCapability:
    isa: str
    abi: str
    ieee_f32: bool
    gradual_underflow: bool
    stable_rounding: bool
    nontrapping_unobserved_flags: bool

    def header(self) -> str:
        if self.isa != 'rv64gc' or self.abi != 'lp64d':
            raise ValueError('RV64GC/lp64d source pair capability required')
        if any(value is not True for value in (self.ieee_f32, self.gradual_underflow,
                self.stable_rounding, self.nontrapping_unobserved_flags)):
            raise ValueError('explicit source arithmetic and effect contract required')
        return r'''#ifndef GEMMINI_HOST_SOURCE_FMA_PAIR_H
#define GEMMINI_HOST_SOURCE_FMA_PAIR_H
static inline void gemmini_host_source_fma_pair(
 float al,float bl,float cl,float ah,float bh,float ch,float *low,float *high){
 float l,h;
 __asm__("fmadd.s %0,%2,%3,%4\n\tfmadd.s %1,%5,%6,%7"
         : "=&f"(l),"=&f"(h)
         : "f"(al),"f"(bl),"f"(cl),"f"(ah),"f"(bh),"f"(ch));
 *low=l;*high=h;
}
#define MERLIN_SOURCE_F32_FMA_PAIR gemmini_host_source_fma_pair
#endif
'''
