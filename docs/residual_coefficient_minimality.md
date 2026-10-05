# Exact residual coefficient and readout derivation

The first large residual's existing `p=2609, q=2180` choice is exact on all
65,536 signed-i8 input pairs. Its original solver returned the first accepted
candidate within a ratio neighborhood of ±2. Conservative interval pruning
preceded full binary32 comparison; an error tolerance did not select the
coefficients. That search alone did not establish minimality.

The new [certificate](perf_records/residual_coefficient_minimality.json) establishes
**39 chunks as the minimum in the existing positive coefficient, one binary32
scaled-store family**, even allowing a constant integer accumulator bias. This
is a statement about that arithmetic/readout contract, not every device algorithm.
No production rewrite or new hardware candidate resulted from this investigation.

## Source and proof scope

The original immutable capture was structurally reparsed. It supplies the two
finite positive binary32 dequantization scales, the output scale, ReLU and the
complete signed-i8 operand domains. Neither operand of this first residual has a
source proof of nonnegativity. Original separate f32 multiplications, addition,
ReLU, reciprocal-scale multiplication, nearest-even rounding and clipping define
the full input-pair table.

An independent native C oracle preserves those operation boundaries with
`-ffp-contract=off`. Its complete table equals the original table hash
`f98b6bba389c422605814dfededc0af4345d35c8b281221c906c57d1056ed7bf`.
It verifies both the existing 39-chunk readout and the alternative threshold
readout with zero mismatches. This is a local arithmetic proof; no new whole-model
quality or hardware-performance result is claimed.

## Exhaustive coefficient/bias certificate

For any monotone readout of `p*a+q*b`, two source output transitions require:

```
73/61 < p/q < 152/127
```

The lower witness compares source outputs 0 and 1 at `(-68,82)` and `(-7,9)`.
The upper witness compares outputs 1 and 2 at `(45,-52)` and `(-82,100)`.
Exact convex hulls of every output-label group reproduce the same cone. Its
complete set of positive coefficients satisfying
`ceil(p/127)+ceil(q/127) <= 38` contains only **276 candidates**.

The target readout is `clip_RNE(f32(f32(p*a+q*b+r)*s), 0, 127)` with positive
finite binary32 `s`, integer `r` and no signed32 accumulation overflow. Arbitrarily
large bias does not bypass the proof: source `(0,0)=0` and `(127,127)=127`, together
with the signed32-to-binary32 conversion error bound, force
`-612901 <= r <= 2561` for every candidate. All possible shifted accumulators are
then within `[-1230629,615463]`, so their binary32 integer conversion is exact.

For each output label, its minimum and maximum integer accumulators constrain
the real product to closed dyadic intervals. The closed intervals conservatively
include midpoint ties that may actually be forbidden. Set `v=1/(D*s)`, where `D`
is their common dyadic denominator. Eliminating bias from

```
L[y]*v - r <= min_acc[y]
U[z]*v - r >= max_acc[z]
```

gives exact rational bounds on `v` and a necessary integer-bias range. Results:

| Rejection | Candidates |
| --- | ---: |
| No continuous scale/bias satisfies the transition constraints | 267 |
| Continuous phase interval contains no integer bias | 8 |
| Remaining scale interval contains no binary32 value | 1 |

The last case is `p=2311,q=1931,r=0`, requiring a scale between approximately
`0.00041839862759524986` and `0.0004183986427055859`. The adjacent binary32 values
have bit patterns `970677379` and `970677380`; both lie outside that interval.
Independent native evaluation finds respectively 2 and 1 mismatched pairs.
The current exact 39-chunk candidate provides the matching positive control.

## Smaller exact threshold contract and cost

An arbitrary monotone integer-threshold readout permits **`p=225,q=188`**, only
four chunks. Its 127 thresholds use 508 bytes and decode every source pair
exactly. A dense decode table spanning the resulting integer range would use
105,316 bytes. These are generic input-pair-derived contracts that a backend
could select from semantics and capabilities; they do not require a model name.

The current scalar execution policy does not justify implementing that route.
For 802,816 outputs:

| Cost | Current scaled store | Threshold readout |
| --- | ---: | ---: |
| Mesh issue floor | 1,956,864 | 200,704 |
| Readout bytes | 802,816 | 3,211,264 |
| Optimistic additional scalar decode instructions | 0 | 3,211,264 |

The optimistic decoder assumes only four instructions per value: load the i32
accumulator, form a table address, load the result byte and store it. It excludes
loop/address overhead, cache misses, synchronization and additional DMA; a
threshold search usually costs more. The current first residual measured
2,198,800 device cycles in profile 1801. Thus even this optimistic CPU decode
cost exceeds the measured existing device interval, before adding its mesh work
or extra traffic. This is a cost screen, not measured candidate performance.

The threshold contract remains useful for a future vector host or device
threshold-readout capability. Any promoted generic numeric derivation and CPU
decoder belong in Merlin; target legality, schedules and resource costs belong
in the OOT backend. No such production implementation was added here.

The prior [full-width execute refusal](full_width_execute_read_refusal.md) remains
unchanged. Legal full-width readout does not establish full-width D feedback for
compute/preload operands.
