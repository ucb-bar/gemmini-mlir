# Reject a locally bounded residual approximation

The exact wide residual requires 39 diagonal coefficient chunks for the first
matched source operation. Its source scales can also derive a two chunk integer
predictor: p=73, q=61, binary32 readout=0.013245166279375553. Without correction,
this predictor differs on 53 of all 65,536 signed int8 input pairs, by at most one
output step. The earlier count of 45 refers to differences among 802,816 actual
ABI values for the original capture; it is not the complete domain count.

We ran one native feasibility experiment using the unchanged original source,
inputs, weights and golden output. The driver re-parses each residual's scalar
source scales, checks the source and catalog bindings, and derives bounded
coefficient candidates from those scales. It chooses one operation by shape and
coefficient chunk work saved. Runtime values and golden outputs do not choose
the coefficients or operation.

The actual native predictor passes the complete signed pair domain and 4,096
output guard bytes. Only that one residual body changes. The native host object,
adapters, other 15 residual bodies, contractions, stem, mean and runtime remain
the control implementations. This changes the local numeric contract and does
not establish source equivalence.

The original 1897 native validation recipe requires every one of the 1,000 final
binary32 output words to match the original golden exactly, with atol=rtol=0.
The control passes that same gate again. The approximation fails:

| Whole model gate | Control | One approximate residual |
| --- | ---: | ---: |
| Changed f32 words | 0 | 1,000 |
| Maximum absolute error | 0 | 1.230414867401123 |
| Relative L2 error | 0 | 0.013461263406102052 |
| Original exact gate | Pass | Fail |

The candidate is rejected. A small local integer error does not bound error
after later quantization and model stages. The exact default remains enabled;
the whole tolerance and golden output remain unchanged. No target object or
FireSim job was built for this branch, and no cycle benefit is claimed.

Source, gate recipe, native library, every frozen forward argument, both full
outputs, coefficient derivation and rejection receipts are pinned in
[residual_affine_approx_native_rejected.json](perf_records/residual_affine_approx_native_rejected.json).
The owned driver and raw artifacts remain under
`out/residual_affine_approx_native`. Root records token accounting separately.
