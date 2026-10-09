# Fixed-prefix source quantizer: rejected complete cost

Stock FireSim jobs 2110/2111 completed with the original 150,528 quantized values,
158,700 padded output bytes, immutable 602,112 input bytes, 16,384 guard bytes and
all ten rounding-mode/sticky-flag observations passing. Both actual staged
ELFs and stock bitstream identities were retained; every executable section
contains zero custom instructions. The original 195-pin qualification is unchanged.

| Arm | First complete window | Second complete window | Mean cycles |
| --- | ---: | ---: | ---: |
| Original eight-lane source quantizer | 3,361,643 | 3,327,864 | 3,344,753.5 |
| Fixed 18-bit integer-observation prefix | 5,430,255 | 5,395,478 | 5,412,866.5 |

The prefix arm regresses 61.831552%. Its complete source-derived interval proof
remains valid, including the unchanged original fallback, and actual linked LLVM
contains no hot lookup calls. The 96.4765% source-hit coverage and 512 KiB requested
table do not predict complete cost. These are two consecutive windows per arm;
no ABBA, stability, cache-miss or pure memory-service claim is made. The isolated
allocation/layout/copy graph differs from whole-model bufferization.

The fixed policy is not promoted, no whole-model arm was built, and no additional
partition search follows. Generic LLVM helper linking is qualified separately;
it grants no numerical permission or automatic table selection.

The sibling JSON is a byte-identical copy of the root's terminal authority:
`/scratch/agustin/tmp/gemmini-current-profile-20261007/out/quant_prefix_stock/qualification.json`,
SHA256 `6f3e0d42b5392109cc0be667b3fbda79af391e098ccceaa9ce04c59a9f61fff3`.
The original packet remains
`out/current_quant_prefix_seal_v2/qualification.json`,
SHA256 `1bf3258b7ff9f3fc19080e1c15f8f8b828125bacf23b4f0b3822440546d9b88d`.
