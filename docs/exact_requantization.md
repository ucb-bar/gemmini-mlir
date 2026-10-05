# Exact scalar epilogue fusion

`golden_requant` proves whether a scalar f32 dequantization/bias/requantization
chain can be represented by Gemmini integer bias preload plus one CONFIG_ST
scale, nearest-even rounding and signed int8 saturation. It does not silently
reassociate floating-point arithmetic or round arbitrary float biases.

For finite positive scalar scales, the entire scalar expression is monotone in
the integer accumulator. An int8 output has only 255 transition boundaries
(127 with ReLU). Binary-searching each transition over the complete accumulator
domain, then comparing their exact locations, proves equivalence for every
accumulator in that domain. The domain is conservatively bounded by signed int8
GEMM K, or all i32 values if overflow is possible. No sampled values establish
the proof. For example, folding f32 multiplications by 0.1 then 0.1 is refused:
accumulator -12,650 produces -126 originally and -127 after reassociation.

For captured float channel biases, the source evaluator preserves the original
order: i32→f32, each f32 dequant multiplication, f32 bias addition, optional ReLU,
f32 output reciprocal multiplication, nearest-even and saturation. The target
has integer bias before one f32 scale. Each output transition gives an equality
or inequality on that integer bias. Their intersection either supplies a valid
integer preload value or refuses the channel. The feasible range also excludes
i32 accumulator overflow. Bias 0.01 following a 1/32 scale is correctly refused;
bias 1.0 synthesizes integer bias 32 exactly.

The structural `match` accepts a canonical pointwise linalg epilogue with exact
identity tensor maps, a signed from-zero integer GEMM, finite uniform scale
constants, optional immutable dense channel-bias constants, roundeven and exact
int8 clamp. Runtime bias, residual additions, nonuniform scales, other rounding,
fastmath, shared accumulator consumers and unprovable scale/bias combinations
are refused. `compile_selected` emits the primitive xDSL GEMM with MVIN3 integer
bias preload and scaled store. The receipt publishes the required integer bias
table; callers must bind that table rather than passing the original float bias.

The 17×20×19 fixture proves float biases [-.0625,-.03125,0,.03125,.0625,…] become
integer biases [-2,-1,0,1,2,…] with store scale .03125. GSIM validated all outputs
and a 2KB guard at **776 kernel cycles**; the final linked ELF passes the static
no-FSM audit. This is a synthetic correctness/performance probe, not a model
benchmark. Tests cross-check threshold proofs against exhaustive small domains
and exercise concrete refusal cases.

## Current ResNet capture

The exact current capture has **54 scalar dequant chains**, each with two scalar
f32 multiplications. It does not have unequal per-channel dequant scales.
All 7,552 bias entries across its 32 direct unary requant candidates are zero in
the hash-pinned captured weights. Channel scale specialization is therefore not
the obstacle for this capture; original f32 rounding, residuals and layout/host
work are the remaining obstacles.

The exhaustive threshold census finds:

* **27 complete unary epilogues / 5,504 channels proved**, all integer biases 0.
* **5 unary epilogues / 2,048 channels refused**: matmul_25, matmul_29,
  matmul_43, matmul_48 and matmul_51. No integer bias with the selected combined
  scale preserves every output transition.
* **20 residual branches** require a two-input expression and are outside the
  unary proof; the stem pool path and FC float return are also retained.

The 27 proved regions comprise 12 direct 3×3 convolutions and 15 other GEMMs.
Freezing their captured zero biases requires the source/capture weight pins to
remain bound to the compiled artifact. The analysis does not itself mutate a
runtime parameter into a constant or constitute whole-model validation.
The compact census is `perf_records/resnet_scalar_epilogue_census.json`; the
full local report and bias tables are `out/requant_bias_census.json`.

For future captures, a target quantization recipe can expose integer bias and a
single per-tensor requant scale directly. Such a recipe changes the numerical
program unless independently proved equivalent to the prior float expression;
it therefore needs its own reference/golden and accuracy validation. Unequal
per-channel scales cannot be represented by one CONFIG_ST value. A compiler may
split output-channel groups only when each group's scale is proven equal and
must account for the added configuration/DMA costs. Current exact proofs do
not grant permission to approximate the five refused layers or residual paths.

## Source-bound whole-graph bundle

`python -m mlir_oot.captured_requant_bundle CAPTURE --llvm-bin LLVM_BIN --output OUTPUT`
verifies the capture source receipt, manifest, and safetensors identity before
specializing immutable zero bias channels. This must run before normal model
preparation/offload. `rewritten.mlir` contains 27 fused calls: 12 direct spatial
convolutions and 15 dense GEMMs. It leaves 4 direct convolutions and 23 other
GEMMs for the existing binders. Link all three device objects. `native_oracle.c`
provides corresponding reference adapters for complete host numeric validation.

Each fused call produces spatial-major i8 `[M,N]`; original reshape/transpose
operations are replayed in i8 to preserve exact source output layout. Persistent
NHWC propagation remains a separate graph transform. The bundle removes the
accepted float dequantization, zero-bias addition, ReLU, and quantization chains;
it does not remove residual branches or approximate the five failed scale proofs.
Changing the captured weights requires rebuilding, even though input images may
vary. Nonzero captured biases are conservatively refused by this bundle until
its ABI binds a synthesized preload table explicitly.

Current source-bound build: `out/current_requant_bundle`, object SHA256
`4a0bfb4d64f2e13ed3d8919aef42dea8eae4b77528a765e284594ad66173fedc`.
All 27 routes compile, the combined object has no undefined symbols and passes
no-FSM audit; rewritten IR verifies before and after serialization. Native
selected dense and direct kernels match the original ordered f32 epilogue for
200,704 values each, including nonzero direct-convolution halo values. This
checks scalar kernels and source semantics; whole graph device execution must
still be verified by the integration build. Existing primitive i8 GEMM GSIM and
direct-convolution FireSim evidence cover their underlying instruction schedules.

### Complete mixed catalog integration

`mlir_oot.fused_mixed_catalog.stage_capture` produces a derived capture after
checking the original model, weights, weight manifest, rewritten model, and
compiled fused object hashes. The original artifacts remain unchanged; the new
receipt records its source and fusion manifest. `merlin_callbacks` then composes
remaining direct convolutions and dense contractions with the precompiled fused
object, recording all 54 routes and native reference source paths.

Reproducible full build and correctness gates:

```
MERLIN_GENERALIZE_BEFORE_FUSE=1 MERLIN_FUSE_POST=1 \
MERLIN_CLANG=/path/to/llvm/bin/clang \
PYTHONPATH=.:/path/to/merlin/src python tests/fused_whole_model_probe.py \
  /path/to/original-capture /path/to/requant-bundle \
  --work /path/to/fresh-work --llvm-bin /path/to/llvm/bin \
  --spike /path/to/gemmini-spike
```

Use `--validate-existing` to validate an already built work directory. The native
gate checks the full host graph with scalar device stand-ins; the subsequent
Spike gate executes actual Gemmini instructions and requires bitexact complete
output equality. The reported Spike counter is retired instructions and must
not be described as measured FireSim cycles.

Current full executable: `out/whole_requant/build_direct/model.elf`, SHA256
`2f8f4d0341791b5278dbc66d91e4cc6787e9361ad6ab40193acedfee69da0ac7`.
All 54 contractions are bound: fused27/direct4/dense23, with 10 remaining dense
signatures. Final executable no-FSM scan passes. Whole-model native execution
matches all 1,000 reference outputs exactly (maximum absolute error zero).
This variant leaves the experimental layout propagation pass disabled.

The first executable above also passed actual Gemmini Spike execution for all
1,000 outputs. It retired 1,534,794,066 instructions, but omitted the host
generalize-before-fuse and post-fusion environment flags used in the earlier
1.193B baseline; these counters are not a controlled performance comparison.
An explicitly flagged build is required before evaluating the optimization.
