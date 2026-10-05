# Residual affine prediction and exact correction

The standalone CPU correction route is rejected for model promotion. It preserves
the original arithmetic, but scanning the inputs costs more than the device work
it removes. The normal model policies and original numeric gates are unchanged.

## Hypothesis and numeric contract

The first source residual has positive binary32 scales `0.011258588172495365`,
`0.00940733402967453`, and output scale `0.011643771082162857`, followed by ReLU.
Its existing exact integer implementation uses coefficients 2609 and 2180 with
readout scale `0.00037060913746245205`, requiring 39 signed byte chunks.

A two-chunk predictor uses coefficients 73 and 61 with readout scale
`0.013245166279375553`. Complete comparison against the ordered source expression
finds 53 differing pairs out of 65,536, each differing by one output unit. The
corrective mechanism uses a certified pair bitmap and replays the original
separate binary32 products, addition, ReLU, and reciprocal multiplication only
when a pair differs. It changes no numeric tolerance.

Merlin owns this generic certificate and CPU correction utility in commit
`54ad7773f`. The OOT experiment delegates to it and uses the existing xDSL residual
provider for device prediction. No model or provenance identifier selects the
strategy.

## Complete measured cost

All qualified GSIM rows below check every output and 2,048 guard bytes. Their
final ELFs pass the zero-FSM audit. These are capsule cycles, with the complete
device invocation and CPU correction included where enabled.

| Arm | Inputs | Cycles | Outcome |
| --- | --- | ---: | --- |
| Exact 39-chunk control | All 65,536 pairs | 164,566 | Control |
| Two-chunk predictor alone | All 65,536 pairs | 19,817 | Provider arithmetic proof only |
| Unary word guard and correction | All 65,536 pairs | 1,319,205 | Reject |
| Unary and predictor-bit guards with lane gathering | All 65,536 pairs | 687,416 | Reject |
| Exact control | First 65,536 captured pairs | 164,566 | Control |
| Packed guards with lane gathering | First 65,536 captured pairs | 303,082 | Reject: 84.17% slower |
| Inlined bitmap check and cold floating replay | First 65,536 captured pairs | 293,843 | Reject: 78.56% slower |

The actual target-module CFG trace reduces compute and preload commands from
9,984 each to 512 each for 65,536 elements. Input transfers and output stores
retain their original counts. Command count reductions therefore do not establish
a complete performance win.

A native diagnostic captures all 802,816 original first-residual operand pairs
while retaining all 1,000 original model output bits. Only 45 pairs require
correction. Nevertheless, the combined packed predicate flags 14,532 lanes across
13,893 words, and the implementation scans all 100,352 words. Those frequencies
describe this immutable capture; they establish no generic input bound.

Full 802,816-element GSIM arms report kernel counters of 1,997,715 for the control
and 4,375,579 for packed correction, then exhaust their 600-second budgets during
output verification. Neither reaches its final correctness marker. These
counters are retained as unqualified observations; the qualified prefix and
complete pair-domain results already justify rejecting the candidate.

## Correctness and artifact limits

The generic utility has 16 passing native checks, including three independent
numeric contracts, signed outputs, ReLU, no-exception tables, all operand pairs,
unaligned arrays, scalar tails, and guards. Independent compiled predicates cover
all 72 centered radius pairs, every byte value in every lane, all 256 gathered
flag masks, and adjacent zero-byte borrow cases. Prediction must independently
match the target contract; the software proof alone is insufficient.

Original GSIM capsule ELFs use the `cycle` CSR, which strict RV64GC Spike refuses
before the target invocation. Separate diagnostics replace exactly two counter
reads with `mcycle`; complete byte ledgers and zero-FSM audits are retained. Those
diagnostics pass full predictor and corrected pair-domain checks and the captured
prefix. They prove the same arithmetic path while retaining their distinct ELF
identities. New probe generation uses `mcycle` directly.

The original capsule builder places caller flags before provider flags. Its
`-ffast-math` recipe can override requested strict floating options. The original
negative artifacts retain their actual flags; their complete source-specific
output checks are not a generic strict-floating proof. The separate native tests
disable contraction. The parent owns the generic compiler-flag ordering fix.

Any caller must preserve original inputs through correction, prove output/input
nonoverlap, and retain RN-even binary32 operations with gradual underflow. The
packed path asserts little endian storage. Production admission still requires
normal whole-model source, object, accuracy, and hardware gates.

## Next direction

The parent is checking exact coefficient, accumulator seed, and store-scale
feasibility over the actual source domains. The earlier source-domain study
already proves twelve nonnegative shortcut inputs; it does not narrow the four
signed projection shortcuts. A future correction route needs either an exact
cheap predictor or fusion with an already required host operation, followed by
a complete measured cost comparison.

The full pins, negative revisions, raw receipt paths, and token-accounting scope
are in [the experiment receipt](perf_records/residual_affine_cpu_correction_rejected.json).
