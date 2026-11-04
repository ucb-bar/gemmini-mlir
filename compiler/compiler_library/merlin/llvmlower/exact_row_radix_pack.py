"""Prepare one source-grid proof per immutable row before canonical packing.

A finite normal BF16 row without zeros, with exponent span <=7*digits-8,
needs no rounding or saturation in the canonical signed-magnitude radix128
representation. The mandatory source scan proves this sufficient condition.
The admitted row loop writes identical planes and source-widened reconstruction;
all other rows retain the complete original encoder. No per-word fast/slow
branch, sampled-data rule, or approximate source certificate is introduced.
"""

from merlin.llvmlower.fused_encoded_witness import fused_encoded_row_header


def c_header() -> str:
    source = fused_encoded_row_header()
    changes = (
        ("  uint32_t maximum = 0;", "  uint32_t maximum = 0,minimum=UINT32_C(0x7f800000);"),
        (
            "    if (magnitude > maximum) maximum = magnitude;",
            "    if (magnitude > maximum) maximum = magnitude;\n    if (magnitude < minimum) minimum = magnitude;",
        ),
        (
            "  int same=1;",
            r"""  /* BF16 normal significands have eight bits. The source max fixes
   * step, so min exponent >= max exponent-(7*digits-8) admits every
   * coefficient as an exact unsaturated integer with <=7*digits bits.
   * No zero/subnormal or uncertain endpoint is admitted by this row proof. */
  if(!lower && digits>=2 && minimum>=UINT32_C(0x00800000) &&
     (maximum>>23)-(minimum>>23)<=7*digits-8){
    for(size_t z=0;z<length;z++){
      uint32_t raw;MERLIN_SOURCE_BITCAST_COPY(&raw,source+z*source_stride,4);
      unsigned biased=(raw>>23)&255;
      uint32_t coefficient=(((raw>>16)&127)|128)<<((int)biased-134-step_exponent);
      const int negative=(raw>>31)!=0;
      float original;MERLIN_SOURCE_BITCAST_COPY(&original,&raw,4);
      reconstructed[z*reconstructed_stride]=(double)original;
      for(unsigned digit=0;digit<digits;digit++){
        int value=(int)((coefficient>>(7*digit))&127);
        planes[digit*digit_stride+z*element_stride]=(int8_t)(negative?-value:value);
      }
    }
    *exact=1;return 1;
  }
  int same=1;""",
        ),
    )
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("canonical finite row producer changed")
        source = source.replace(before, after)
    return source


def prepare_exact_row_radix(text: str) -> str:
    original = fused_encoded_row_header()
    if text.count(original) != 1:
        raise ValueError("owned canonical encoder header is not present exactly once")
    return text.replace(original, c_header())
