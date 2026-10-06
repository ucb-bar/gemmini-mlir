"""Explicit positive magnitude precision for a private absolute-dot upper bound."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CoarseAbsoluteUpperPlan:
    """Three canonical signed radix-128 digits; positive signed-i8 ceiling."""
    magnitude_bits: int

    def __post_init__(self):
        if type(self.magnitude_bits) is not int or not 1 <= self.magnitude_bits <= 6:
            raise ValueError("positive magnitude precision must be an integer in [1, 6]")

    @property
    def shift(self):
        return 21 - self.magnitude_bits

    def validate_length(self, length):
        if type(length) is not int or length <= 0 or length * (1 << (2*self.magnitude_bits)) > 2**31-1:
            raise ValueError("coarse absolute product exceeds signed-i32 range")


def prepare_coarse_absolute_upper(text, plan):
    """Reuse the admitted source-equality and signed-plane last-use grammar."""
    if not isinstance(plan, CoarseAbsoluteUpperPlan):
        raise ValueError("typed coarse absolute upper plan required")
    from .exact_absolute_products import prepare_exact_absolute_products
    text = prepare_exact_absolute_products(text)
    start = text.index('  for(int i=0;i<3*m*k;i++){int v=w->ap[i];')
    end = text.index('  merlin_fma_bound env=merlin_fma_bound_begin();', start)
    shift=plan.shift
    replacement = f"""  /* Each canonical sign-coherent magnitude is <= 2^21-1. Ceiling
   * division yields U <= 2^bits <= 64 and |source| <= step*U*2^shift.
   * Plane zero is overwritten only after all signed products complete.
   * B retains the callback's [plane,k,column] indexing. */
  if(k>INT32_MAX/{1 << (2*plan.magnitude_bits)})return 0;
  for(int side=0;side<2;side++){{
   int8_t *p=side?w->bp:w->ap;int count=side?n*k:m*k;
   for(int i=0;i<count;i++){{
    unsigned magnitude=0;
    for(int d=0;d<3;d++){{int v=p[d*count+i];if(v==INT8_MIN)return 0;
     magnitude+=(unsigned)(v<0?-v:v)<<(7*d);}}
    p[i]=(int8_t)((magnitude+{(1<<shift)-1}u)>>{shift});
   }}
  }}
  if(!product(opaque,w->ap,w->bp,w->readout,m,n,k,0))return 0;
  for(int r=0;r<m;r++)for(int c=0;c<n;c++){{
   int i=r*n+c;
   if(w->readout[i]<0)return 0;
   w->absolute_center[i]=(double)w->readout[i]*0x1p{2*shift};
   w->absolute_center[i]*=w->astep[r]*w->bstep[c];
  }}
"""
    text=text[:start]+replacement+text[end:]
    return text.replace('exact_absolute_dot_bounds.h','absolute_upper_dot_bounds.h').replace('merlin_exact_absolute_dot','merlin_absolute_upper_dot')
