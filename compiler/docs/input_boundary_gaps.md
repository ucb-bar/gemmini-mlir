# Input boundary evidence for residual-state convolution

This records the investigation of a native caller boundary disagreement, resolved by preserving the public source output result order. The input declaration discrepancy below was not the cause.

For `SY_residual_state_conv_k3x3_s1x1_d1x1_pad1x1`, the public `layers/.../capsule.yaml` input declarations are ordered `P_W, P_A, W, IFM`. Its `capsule.interface.mlir` declares tensors in order `P_W, P_A, IFM, W`. The command-buffer artifact preserves the latter input order. The selected logical ABI uses logical inputs in declaration order, then logical outputs in result order. The emitted function therefore receives `P_W, P_A, IFM, W, P_Y, Y0`; the prelude store uses argument 4 and the convolution store uses argument 5. No capsule-order metadata is present in the supplied interface to resolve a caller using the YAML order.

Both output declarations and original operation attributes specify signed i32, including prelude `P_Y`. The old native console nevertheless contains `OUT_B64_BEGIN v1 P_Y 16 16 1 s` and one chunk of length `0100` (256 bytes), while the logical i32 result occupies 1024 bytes. This is evidence of a boundary disagreement worth investigating; the dump alone does not identify the caller's complete pointer mapping.

The frozen old result is request `3_1793175341780872471_b9572434c7393530d7bf69e663981584`, case detail `qa/cases/SY_residual_state_conv_k3x3_s1x1_d1x1_pad1x1/680578e73b80830c446f0ac333db18ebf8fe11016f2ef4aa3a940c6d5aa1a5ef.json`. It reports exact command-buffer numerics for both outputs, native `functional_mismatch`, and the explicit structural violation `trace does not open with a FENCE`. Command-buffer correctness does not prove native numerical correctness.

An explicit execution configuration reset was added at convolution entry after its drain. Focused request `2_1793176008083344760_f3bc7e14780f3328625b8d66d03cccd3` retained native failures for both k3 stride variants; the k7 sibling failed with a UTF-8 decode error. Reset alone therefore does not resolve the cluster. Opening-fence repair and a matching native ABI remain separate checks. No capsule declarations, synthetic traces or grading metadata were changed.

## Resolution

The command-buffer output declarations must preserve the source-published result order. The parent backend corrected that order without changing the encoder or guessing a capsule input permutation. Request `3_1793177243870274671_14af931a42d7ec16d87f1b178276bf18`, saved as `ab/stateconv-ordered1.json`, passed native numerics and structural checks for all three residual-state convolution cases (k3 stride one, k3 stride two, and k7 stride two). The YAML/source input order difference was not the failure cause. The earlier console and reset-only results remain diagnostic evidence, not an unresolved failure claim.


## Unresolved original input and trace boundaries

Three original edge chains declare an `acc_scale` stage with explicit f32 scale 1.0. Their command-buffer L0/L1 checks pass, while native feedback reports `mode acc_scale declared but no CONFIG_ST has a non-identity scale`. The compiler preserves the identity scale. An artificial nonidentity store scale would change the source. These remain failures; this trace disagreement does not establish native numerical correctness.

The distinct-column-scale layer publishes all 48 column scales in its source, and the emitted command buffer preserves them. Its YAML separately declares a custom `merlin.component_input_palette.v1` for A0/W/B that is absent from the compiler interface. L0 reports a numerical divergence. The missing palette is a suspected input boundary gap, not a demonstrated cause; no source-specific stimulus or answer is compiled. Exact public source hashes, result pointers and failure diagnostics are in `ab/input-boundary-evidence.json`.

Two large residual chains and the sibling-feed chain have also reported native disagreement. Independent evaluation of public deterministic stimuli matches their own published output arrays in `ab/chain_public_stimulus_numeric.json`; this does not override the failing native verdict. Newly fixed elaboration in the wide-transfer proof exposes the sibling failure again, so it remains an unresolved correctness obligation.
