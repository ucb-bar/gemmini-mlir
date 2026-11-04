"""Explicit exact weighted-integer reconstruction using encoded panel support.

No arbitrary floating zero-elimination permission is granted. Callers provide
positive-zero-seeded, RNE reconstruction of all groups of the validated radix
plan, unchanged complete encoded-i8 metadata, nonaliasing fully written i32
group outputs, and the original signed-digit magnitude/range proof with at most
the plan's reduction-length terms. Under that
contract every weighted term and prefix is an exact binary64 integer; omitted
positive-zero additions preserve both the result and the sign of zero.
"""

from .radix_product_groups import RadixProductPlan, plan_radix_product_groups


def c_header(plan: RadixProductPlan) -> str:
    canonical = plan_radix_product_groups(
        radix_bits=plan.radix_bits, digits=plan.digits, reduction_length=plan.reduction_length
    )
    if plan != canonical or plan.radix_bits > 7:
        raise ValueError("canonical signed-byte/i32/exact-binary64 radix plan required")
    declarations = []
    cases = []
    for ordinal, group in enumerate(plan.groups):
        pairs = ",".join("{" + str(a) + "," + str(b) + "}" for a, b in group.pairs)
        declarations.append(f"static const uint8_t merlin_radix_pairs_{ordinal}[][2]={{{pairs}}};")
        cases.append(
            f"case {ordinal}:pairs=merlin_radix_pairs_{ordinal};count={len(group.pairs)};weight=0x1p{group.exponent};break;"
        )
    return (
        """#ifndef MERLIN_ENCODED_RADIX_RECONSTRUCT_H
#define MERLIN_ENCODED_RADIX_RECONSTRUCT_H
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53
#error exact_radix_reconstruction_requires_binary64
#endif
"""
        + f"#define MERLIN_EXACT_RADIX_MAX_REDUCTION_LENGTH {plan.reduction_length}\n"
        + "\n".join(declarations)
        + r"""
static inline int merlin_radix_panel_supported(const uint8_t *az,const uint8_t *bz,
    size_t ablocks,size_t bblocks,size_t ai,size_t bi,const uint8_t (*pairs)[2],size_t count){
  for(size_t p=0;p<count;p++)if(az[pairs[p][0]*ablocks+ai]&&bz[pairs[p][1]*bblocks+bi])return 1;
  return 0;
}
/* Runtime obligation: RNE, complete immutable encoded metadata, source i32
 * outputs exact, positive-zero-seeded exact prefixes under this emitted plan.
 * Blocks/layout are caller parameters, with no device identity in this code. */
static inline void merlin_radix_group_accumulate_exact_f64(double *dst,
    const int32_t *source,const uint8_t *az,const uint8_t *bz,size_t m,size_t n,
    size_t a_block_rows,size_t b_block_rows,unsigned group){
  assert(m&&n&&a_block_rows&&b_block_rows);
  const uint8_t (*pairs)[2]=0;size_t count=0;double weight=0;
  switch(group){
"""
        + "\n".join(cases)
        + r"""
  default:assert(!"group outside emitted exact radix plan");return;}
  const size_t ablocks=(m-1)/a_block_rows+1,bblocks=(n-1)/b_block_rows+1;
  /* One complete supported pair proves every output panel may be nonzero.
   * Keep the original flat update, avoiding tiled traversal on dense groups. */
  for(size_t p=0;p<count;p++){
    int complete=1;
    for(size_t a=0;a<ablocks;a++)complete&=az[pairs[p][0]*ablocks+a]!=0;
    for(size_t b=0;b<bblocks;b++)complete&=bz[pairs[p][1]*bblocks+b]!=0;
    if(complete){for(size_t t=0;t<m*n;t++)dst[t]+=(double)source[t]*weight;return;}
  }
  for(size_t a=0;a<ablocks;a++)for(size_t b=0;b<bblocks;b++){
    if(!merlin_radix_panel_supported(az,bz,ablocks,bblocks,a,b,pairs,count))continue;
    const size_t re=(a+1)*a_block_rows<m?(a+1)*a_block_rows:m;
    const size_t ce=(b+1)*b_block_rows<n?(b+1)*b_block_rows:n;
    for(size_t r=a*a_block_rows;r<re;r++)for(size_t c=b*b_block_rows;c<ce;c++){
      const size_t at=r*n+c;dst[at]+=(double)source[at]*weight;
    }
  }
}
#endif
"""
    )
