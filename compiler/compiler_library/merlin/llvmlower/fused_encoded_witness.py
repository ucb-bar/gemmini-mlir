"""Fuse private canonical radix reconstruction and its equality witness.

The canonical encoder remains the single arithmetic implementation. This
specialization changes only its output consumer: preserve the binary32 product,
then widen it and record equality while the original source word is live.
"""

from merlin.common.paths import data_path


def fused_encoded_row_header() -> str:
    source = (data_path("runtime", "c") / "bf16_radix_pack.h").read_text()
    start = source.index("static inline int merlin_bf16_radix_row(")
    body = source[start : source.rindex("\n#endif")]
    replacements = (
        ("merlin_bf16_radix_row(", "merlin_bf16_radix_row_widen("),
        (
            "size_t source_stride, float *reconstructed, size_t reconstructed_stride,",
            "size_t source_stride, double *reconstructed, size_t reconstructed_stride,",
        ),
        (
            "unsigned digits, float *step) {",
            "unsigned digits, float *step, const float *lower, const float *upper, unsigned char *exact) {",
        ),
        (
            "  if (!eligibility || !eligibility->valid || !source || !reconstructed || !planes || !step ||",
            "  if (!exact || (lower==0)!=(upper==0) || !eligibility || !eligibility->valid || !source || !reconstructed || !planes || !step ||",
        ),
        (
            "  for (size_t z=0; z<length; ++z) {\n    uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw,source+z*source_stride,sizeof(raw));\n    const unsigned biased",
            "  int same=1;\n  for (size_t z=0; z<length; ++z) {\n    uint32_t raw; MERLIN_SOURCE_BITCAST_COPY(&raw,source+z*source_stride,sizeof(raw));\n    const unsigned biased",
        ),
        (
            "    reconstructed[z*reconstructed_stride] = (float)signed_coefficient * *step;",
            "    float value=(float)signed_coefficient * *step;\n"
            "    reconstructed[z*reconstructed_stride]=(double)value;\n"
            "    float original;MERLIN_SOURCE_BITCAST_COPY(&original,&raw,sizeof(original));\n"
            "    int equal=original==value;\n"
            "    if(lower)equal=equal&&lower[z*source_stride]==original&&upper[z*source_stride]==original;\n"
            "    same=same&&equal;",
        ),
        ("  return 1;", "  *exact=(unsigned char)same;\n  return 1;"),
    )
    for before, after in replacements:
        if body.count(before) != 1:
            raise ValueError("canonical radix producer grammar changed")
        body = body.replace(before, after)
    return (
        "#ifndef MERLIN_FUSED_ENCODED_ROW_H\n#define MERLIN_FUSED_ENCODED_ROW_H\n"
        '#include "bf16_radix_pack.h"\n'
        "/* Finite BF16 source admitted by original scan. For d<=3, signed\n"
        " * coefficient has <=21 magnitude bits and power-of-two step.\n"
        " * Original binary32 reconstruction is finite and exactly widened.\n"
        " * Source/endpoints/planes/reconstruction/flag owners are disjoint;\n"
        " * source and endpoint spans stay immutable through this row epoch. */\n" + body + "\n#endif\n"
    )


def prepare_fused_encoded_witness(text: str) -> str:
    before = """ for(int row=0;row<m;row++){
  float step;if(!merlin_bf16_radix_row(&environment,source+row*k,k,1,rf+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step))return 0;
  steps[row]=step;
 }
 *proof=merlin_encoded_rows_widen(&environment,source,rf,lower,upper,rd,flags,m,k);return proof->valid;"""
    after = """ if(m<=0||k<=0||!proof||!flags||(lower==0)!=(upper==0))return 0;
 for(int row=0;row<m;row++){
  float step;if(!merlin_bf16_radix_row_widen(&environment,source+row*k,k,1,rd+row*k,1,
    planes+(transpose?row:row*k),m*k,transpose?m:1,3,&step,
    lower?lower+row*k:0,upper?upper+row*k:0,flags+row))return 0;
  steps[row]=step;
 }
 *proof=(merlin_encoded_row_equality){source,lower,upper,rd,flags,(size_t)m,(size_t)k,1};return 1;"""
    if text.count(before) != 1 or text.count("w->arf") != 1 or text.count("w->brf") != 1:
        raise ValueError("private reconstruction scratch has unknown producer/consumer")
    text = text.replace(before, after)
    position = text.index("#include ")
    return text[:position] + fused_encoded_row_header() + text[position:]
