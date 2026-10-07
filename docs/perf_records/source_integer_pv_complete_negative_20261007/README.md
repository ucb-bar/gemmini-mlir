# Exact source probability and original i32 V: complete negative

The original source V is its i32 projection followed by two separately rounded
binary32 scale products. This experiment preserves those codes and source
probabilities through a runtime exact bit lattice. Four P planes and three V
planes are grouped into six equal-exponent signed-byte products, with explicit
i32 and i64 prefix-range proofs. The Gemmini provider executes six batched
calls across four KV groups and writes 98,304 exact i32 words (384 KiB).

All 360,448 original compiled i8 observations across 22 contexts pass natively.
The actual device context checks all 98,304 readout words, all 16,384 original
compiled i8 outputs, dirty storage/input guards, five source rounding modes,
and a fresh final-executable no-FSM audit. Native products are an explicit
integer stand-in; the target counterpart uses actual xDSL primitive GEMMs.

| Complete PV/quantizer alternative | Source instructions | Candidate instructions |
| --- | ---: | ---: |
| Portable point certificate | 635,401 | 6,234,123 |
| Prepared exact column coefficients | 635,401 | 4,833,777 |
| Prepared coefficients and directed scalar bounds | 635,401 | 3,541,881 |

These are functional Spike retired instructions, **not hardware cycles**. The
best zero-replay context remains 5.57 times the source path, so all three
alternatives are rejected for promotion and FPGA admission. No normal source
route or whole-model candidate was installed. The exact observed-i8 policy is
this experiment's choice; the user's original whole Torch tolerance remains
unchanged and permits separately declared approximate policies.

The complete interval contains encoding, source V preparation, packing, actual
products/readback, reconstruction, certificates, source fallback, final
quantization, frames and stores. QK, softmax, projection and GQA views precede
both arms. Caller-owned workspace allocation and dirty poisoning precede both
timed intervals; no implicit allocation/cache or persistent prepared state is
used. The prepared workspace is 259,584 bytes.

The first same-ELF PC census places approximately 6.220 million instructions
per call in the portable executor, including 16,640 floating divisions. The
prepared/directed executor removes the per-output division, leaving 256 row
norm divisions, but integer reconstruction, adjacency checks, branches and
memory still cost approximately 3.528 million instructions per call. Device
issue costs and physical mesh/DMA cycles are distinct and remain unpriced.

The first [627-pin receipt](first_receipt.json) is unchanged. The separate
[1,070-pin successor receipt](prepared_receipt.json) retains the negative,
binds actual source/compiler/objects and re-emits every context's C exactly
after Python formatting. Its original measured generator bytes are preserved
in [the snapshot](measured_prepared_codegen_snapshot.py); the source-pin
resolution is explicit, with no primary receipt rewritten. The core's 104
independent compiled/range/tie/subnormal/overflow/refusal tests and structure
gate pass. Token attribution is unavailable to this child; root attaches
shared campaign snapshots.

Generic representation, range mathematics and portable executor scheduling
live in the isolated Merlin prototype. Target ABI, products and scalar ISA
capabilities live OOT. Local core heads are `c7ab7ae74` and `263f78fdd`; they
are **unpromoted prototypes**, not a request to install unwired helpers on main.
The requested current whole-model FireSim profile should identify the next
larger host/device opportunity before further exact-PV certificate tuning.
