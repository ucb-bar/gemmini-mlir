
## Source-proven residual domains (2026-10-05; experiment only)

`golden_residual_domains` traces verified static reshape/transpose operations and
recognizes finite source DQ/add/ReLU/positive symmetric quantization. Unknown
producers retain [-128,127]. No sample values or region labels establish a range.
The closed recipe has 12 skips proven [0,127]; four projection residuals retain
both full signed domains. Every admissible pair is checked by the C++ interval
search and independently by NumPy with original float32 ordering. The source
MLIR, proof module, search source, and admissible output bytes are hashed.

The first exact representatives in the explicitly limited ratio neighborhood
reduce coefficient chunks in quantize_14 (9 to 3), quantize_27 (17 to 16), and
quantize_48 (8 to 7). This is not a globally optimal coefficient search. Residual
array issue floor falls from 5,193,216 to 4,854,528 cycles, saving 338,688 (6.52%).
The selected whole-model floor would remain 22,466,944 before DMA and host work.
The first residual still requires its full signed domain and 39 chunks.
No graph rewrite, current proof replacement, or hardware promotion is made.
Three tests cover finite source proof, refusal of unknown/no-ReLU/nonzero-threshold/
overflow cases, and propagation through verified reshape/permutation.

## Stock FireSim whole ResNet 1775

Exact virtual-padding candidate completed in 62,441,162 forward cycles, all 1,000
outputs bit-exact to the unchanged closed-recipe golden. ELF, staged ELF,
bitstream, and job-owned UART hashes are pinned in the receipt. This is 8.69%
below the corresponding padded candidate1774 (68,385,997); the unprofiled result
is retained independently of the separate boundary profiling job1777.
