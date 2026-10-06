"""Explicit dense signed-i8 product pairs sharing one exact i32 destination.

A/B point to plane-major arrays [planes,M,K] and [planes,K,N]. Each pair is
reduced into the same resident accumulator block, then stored once. Callers
provide a proven absolute i32 bound; this is integer sum semantics, no float
reassociation or radix policy is inferred by this target provider.
"""
from .golden_gemm import GoldenGemm, Shape

class GoldenProductSum(GoldenGemm):
    def __init__(self, shape: Shape, *, pairs: tuple[tuple[int, int], ...],
                 lhs_planes: int, rhs_planes: int, absolute_bound: int,
                 lhs_magnitude_bound: int = 128, rhs_magnitude_bound: int = 128):
        if shape.output_dtype != 'i32' or shape.bias:
            raise ValueError('product sums require unbiased i32 output')
        if any((shape.cache_a, shape.cache_b, shape.wide_a, shape.pipeline_m,
                shape.prefetch_b, shape.prefetch_m, shape.banked_m)):
            raise ValueError('product sum currently requires independently loaded K panels')
        if not pairs or min(lhs_planes, rhs_planes) <= 0:
            raise ValueError('nonempty pairs and positive plane counts required')
        if type(absolute_bound) is not int or not 0 <= absolute_bound < 1 << 31:
            raise ValueError('explicit signed-i32 absolute bound required')
        if any(type(v) is not int or not 0 <= v <= 128
               for v in (lhs_magnitude_bound, rhs_magnitude_bound)):
            raise ValueError('signed-i8 operand magnitude domains required')
        required = len(pairs) * shape.k * lhs_magnitude_bound * rhs_magnitude_bound
        if absolute_bound < required:
            raise ValueError('accumulator bound does not cover every product prefix')
        # Caller-owned source proof must establish these operand domains; no
        # value scan or benchmark-specific assumption is emitted here.
        for a,b in pairs:
            if type(a) is not int or type(b) is not int or not (0<=a<lhs_planes and 0<=b<rhs_planes):
                raise ValueError('plane index outside declared input span')
        self.pairs=tuple(pairs)
        self._first_pair=True
        super().__init__(shape)

    def _k_tile(self, m0, n0, k0, mr, nr, kr, first, *args, **kwargs):
        return super()._k_tile(m0,n0,k0,mr,nr,kr,first and self._first_pair,*args,**kwargs)

    def _reduce_output_block(self,m0,n0,mr,nr,slot=0,prefetch=None):
        a,b=self.a,self.b;s=self.shape
        for index,(ap,bp) in enumerate(self.pairs):
            self._first_pair=index==0
            self.a=self._ptr(a,self.fb.const(ap*s.m),s.k,self.fb.const(0))
            self.b=self._ptr(b,self.fb.const(bp*s.k),s.n,self.fb.const(0))
            super()._reduce_output_block(m0,n0,mr,nr,slot,prefetch)
        self.a,self.b=a,b
        self._first_pair=True
