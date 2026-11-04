# Authored Gemmini compiler

One source-driven compiler preserves typed graph axes, addresses, effects, numerical boundaries, output ABI and completion edges through scheduling and ordinary Gemmini encoding. It reuses the admitted Merlin host builders and runtime contracts. Integer contraction, convolution, residual arithmetic, saturating narrow, pooling and proven movement execute on the accelerator. Unsupported numerical contracts are refused. Hardware sequencers and case-name dispatch are absent.

Current strategies include bounded operand residency, geometry-based matrix panels, device convolution preparation, shared-RHS sibling packing, accumulator panels, and exact per-column scaled readout. Alternatives invoke real shared lowering paths; registration and compilation alone are not performance evidence.

## Demonstrated coverage

| Scope | Observations | Limits |
|---|---|---|
| 109 core cases | 108 numerical executions pass; required refusal passes | Development checks do not certify unrun physical tiers |
| 101 original correctness cases | 100 reported L3 passes: 94 RTL executions and six source-verification records; required refusal passes | Three unity-gain structural trace conflicts remain |
| 40 scalability cases | All 40 numerical case rows pass | Aggregate response remains incomplete/false without a failure detail; no aggregate qualification claim |
| 3 public programs | All three pass; the changed residual program was checked again | These do not certify the larger complete model graphs |
| Selected model kernels | 68 kernels build; 159 calls replay exactly on one input | Complete-chain execution/timing and shape variants remain unverified |

The explicit legacy pointer ABI resolved the six connected-chain numerical failures and distinct-column input transport. It publishes source declaration order, not capsule input-list order. All output pointers remain visible.

The original physical ledger reports 92 structural trace passes, six source-verification records, and three trace failures. Those three sources declare gain 1.0 while the trace predicate requires a non-identity CONFIG_ST scale. Numerics and L3 barriers pass; no unused configuration or fabricated gain was inserted. The six source-verification cases pass native source-placement, host-compute and completed-dispatch audits. See [physical ledger](../artifacts/physical-current-ledger.json) and [unity-gain evidence](../artifacts/identity-scale-qualification.json).

Five short-K convolutions perform actual zero initialization and accumulating compute and pass physical trace checks. The capacity-boundary specialization passes at 311,591 measured cycles; its full-row/tail split preserves device commands. This is a qualification improvement, not a performance gain. Partial-column trials were reverted after a measured regression.

| Model program | Built kernels | Exact selected calls | Reported contraction census |
|---|---:|---:|---:|
| ResNet-50 | 24 | 45 | 54/54 |
| TinyLlama prefill | 8 | 22 | 3675/3675 |
| TinyLlama decode | 8 | 22 | 3675/3675 |
| SmolVLA prefix | 14 | 36 | 3259/3259 |
| SmolVLA flow | 14 | 34 | 2036/2036 |
| SmolVLA action decode | 0 | 0 | 0/0 |

The contraction census is separate from all-region coverage. The action-decode endpoint selects no kernels and gives no complete-graph proof. The `whole_model_regions` opt-in does not establish selection of linked regions.

## Performance evidence

Comparisons below use matched inputs and completed-call scopes. Spike results use the same provisional, untuned timing model; complete accounting is not an accuracy certificate. Fast estimates are calibrated approximations with wide bands, not measurements.

| Change and representative | Before → after cycles | Evidence |
|---|---:|---|
| Joint activation/M panels, 512³ matrix | 4,964,592 → 744,716 | Complete Spike model |
| Thin raw N panels, M16/K2048/N2048 validation | 3,346,029 → 1,937,379 | Complete Spike model |
| Scaled N panels, M16/K2048/N11264 validation | 18,008,819 → 10,412,245 | Complete Spike model |
| Wide RHS and activation cache, scaled M16/K2048/N11264 | 10,412,245 → 6,526,944 | Complete Spike model |
| Wide RHS, raw M16/K5632/N2048 validation | 5,195,275 → 2,782,320 | Complete Spike model |
| Wide RHS, scaled M16/K2048/N2560 validation | 2,409,169 → 1,537,285 | Complete Spike model |
| Wide RHS, raw M16/K2048/N2048 projection | 1,927,326 → 1,239,687 | Complete Spike model |
| Wide RHS, connected sibling chain | 14,625,831 → 12,676,787 | Complete Spike model |
| Pool preparation, connected CNN stem | 393,945 → 34,867 | Measured gSIM |
| Device preparation and matrix panels, CI64/CO80 | 524,426 → 33,596 | Measured gSIM |
| Row-run producer versus prior preparation, CI64/CO80 | 36,964 → 33,596 | Measured gSIM |
| Per-column staging, aligned matrix | 5,952 → 3,986 | Measured gSIM |
| Per-column staging, tail matrix | 6,207 → 4,524 | Measured gSIM |
| Per-column staging, aligned convolution | 4,497 → 2,915 | Measured gSIM |

The joint matrix rule has six development and three fresh geometry comparisons. For 512³, the structural bound is 524,288 cycles, the modeled roofline gap is 1.4205, useful compute peak attainment is 70.401% of the whole modeled call, and padding efficiency is 100%. Modeled execute/CPU/storage reservations overlap and cannot be summed into elapsed cycles. They do not establish measured processing-element occupancy.

The base thin raw N-panel rule selects M1..16, 16<N<=4096 and useful MACs>=2²⁰, with independent semantic/storage admission. Full-M16 scalar-i8 readouts additionally admit N<=16384 after reserved validation. M1 scaled readouts retain incumbents outside the separately validated large wide-RHS region below. The later wide-RHS region below extends raw width only at large K. A forced M1 scaled connected trial regressed 14,628,032→14,664,631 modeled cycles and was rejected. Its source precision was corrected before selection.

For the scaled validation, useful compute peak attainment rises 8.006%→13.847% of the whole modeled call with 100% padding efficiency; modeled DRAM traffic falls 91,987,136→24,323,584 bytes. The 1,441,792-cycle structural compute floor is not a complete physical roofline. Detailed model/resource identities are in [the selection record](../artifacts/scaled-n-heldout-selection.json).

Raw N panels with N>1024 additionally retain full-K activations when scratch capacity permits. Complete modeled gains are small: TL down 5,221,332→5,195,275 and reserved projection 1,937,379→1,927,326. The single-panel control regressed 849,978→850,404, so it retains reload. These differences are below established model-accuracy resolution; no measured gain or optimal cutoff is claimed.

The selected wide-RHS bundle prepares bounded scratch panels with ordinary 64-column MVIN2 commands and consumes their actual rows. It retains full-K activations when independent storage admission fits, including scalar M16 calls. Within the existing readout/M/MAC region, K>=2048 and N>=1024 select this bundle; raw widths extend to N<=32768 only at K>=2048. Smaller K/N and M2..15 scaled paths preserve their previous schedules. Seven initially changed public programs pass numerical checks; five matrix calls and one complete connected call have favorable complete-model comparisons. These are conservative cost predicates with limited boundary timing coverage, not learned optimal crossovers.

For the wide scaled call, useful compute peak attainment increases 13.847%→22.090% of the whole modeled call, with 100% padding efficiency. Its 1,441,792-cycle compute floor implies a partial compute gap 7.222→4.527; the complete physical roofline remains UNKNOWN. Modeled CPU issue reservation rises 7.060%→12.008% as total elapsed cycles fall. Modeled DRAM requests change 24,323,584→23,968,704 bytes. Load/compute/store bookings and overlap/queue reports remain modeled, not measured occupancy; host wait attribution is UNKNOWN. Full resource and identity evidence is in [wide-RHS evidence](../artifacts/wide-rhs-utilization.json).

A separate fixed rule now admits scalar-i8 M1 calls with K>=2048 and 2048<=N<=16384, retaining independent semantics/storage checks and the MAC threshold. The sibling chain used for development improves 12,676,787→9,960,256 complete modeled cycles; the reserved residual chain, with the opposite K/N balance, improves 9,246,523→3,316,735. Both pass numerics under the same model/scope/domain. Fresh native repeated-input checks cover ragged M1/K2049/N2053 with bias/ReLU and padded strides, both sides of the K/N lower limits, and the aligned corner; these establish legality, not an optimal timing cutoff. Two exact selected products match their checked candidate, all 112 core/program emissions remain unchanged. Useful compute peak attainment is about 1.38% of each complete modeled call and nominal padding efficiency is 6.25%; neither is measured PE occupancy. These chains still have substantial headroom and UNKNOWN complete physical rooflines. [Resource evidence](../artifacts/wide-m1-utilization.json). See [M1 selection](../artifacts/wide-m1-scaled-hypothesis.json).

An ablation keeps the 64-column RHS producer and removes retained activations. All numerics pass, but complete modeled cycles rise 2,782,320→3,307,395 for raw high-K, 6,526,944→6,587,228 for wide scaled, and 1,537,285→1,548,996 for the smaller scaled call. The selected bundle remains. The smaller differences remain below established model-accuracy resolution; no extra cutoff is inferred. See [ablation](../artifacts/wide-rhs-ablation-hypothesis.json).

The M1/K2048/N32000 raw call has UNKNOWN complete timing because its observation budget is insufficient. Fully priced fast estimates favor 24,644,402 [12,642,578–34,551,452]→18,671,014 [9,335,507–25,747,328]. Rolling calibration windows differ; both use 32 cycles per command and the changed baseline fence coefficient affects fewer than 11 cycles. Bands overlap broadly and no measured or complete-model gain is claimed. The real wide-DMA RTL witness passes numerics and trace checks at 1,381 cycles after disabling fitting full-weight residency. The first 1,259-cycle witness retained that cache and did not exercise the new producer. Small selected programs keep the faster existing cache. See [trial and selection](../artifacts/wide-rhs-hypothesis.json).

Per-column staging computes raw i32 once, then applies exact source f32 gains through device accumulator DMA. Five original physical readouts improve about 27–35% and pass numerical/structural checks. For the large per-column convolution, grouped loops versus staged loops have fully priced fast estimates 30,481,726 [18,228,072–53,190,612] versus 3,773,047 [2,256,282–6,583,967]. The earlier grouped 401,408 estimate was an incomplete lower bound and is not a valid baseline. Complete timing remains UNKNOWN at the raw-ACC packet writer.

Within the staged region, M>16 uses disjoint accumulator panels provisionally: a matched fast comparison favors 3,790,794→3,773,416 with overlapping wide bands. Native changing-state and lifetime checks pass, including the 1024-row boundary. The five physical M<=16 programs retain their tested paths. No large heldout timing or accuracy claim is made; contrary complete/physical evidence warrants reversion.

Shared-RHS single-row packing is selected provisionally from matched fast estimates 4,341,507→978,624, with bands 2,730,808–8,431,206 and 615,554–1,900,487. Complete packed timing remains UNKNOWN. Multirow packing remains optional after unfavorable attention estimates.

The selected regional convolution producer splits activation DMA at output-row, image and padding boundaries, feeding actual prepared scratch to the shared matrix consumer. Aligned CI, M>16, N>16, useful MACs>=2²⁰ and a fitting K panel select it outside pooling/per-column paths. Its CI64/CO80 physical witness improves 36,964→33,596 cycles; exact selected LLVM matches that witness. Two large development and two reserved row-run validation comparisons favor it. Validation fast estimates improve 2,649,929→2,397,799 and 11,344,727→9,235,531, with broad approximate .588–1.745 ratio bands. Candidate complete estimates are 750,803 and 3,656,750; incumbents have UNKNOWN complete timing, so these are not matched-complete speedups. No optimal crossover claim is made. See [selection](../artifacts/conv-row-runs-hypothesis.json).

For those two candidates, useful compute peak attainment is 60.15% and 49.40% of the complete modeled call, with 100% useful/padded work. The structural compute floors are 451,584 and 1,806,336 cycles; complete physical rooflines remain UNKNOWN. Modeled CPU issue reservations are 45.46%/45.93%, execute-controller reservations 61.62%/49.48%; overlapping reservations are not elapsed-time components or measured engine occupancy. Detailed traffic, queue and overlap fields are retained in [resource evidence](../artifacts/conv-row-runs-utilization.json). Earlier scratch alternatives remain available.

Optional bounded upstream loop unrolling reaches actual optimized IR and RV64 code; native changing-input numerics and exact ordered device commands pass. All three paired cases pass numerics. Complete modeled incumbent/4/8 cycles are 743,691/744,848/745,369 for 512³, 5,195,275/5,195,261/5,195,654 for the thin matrix, and 2,452,109/2,451,669/2,450,227 for the convolution. Global incumbent remains selected because results are mixed and gains are small relative to model uncertainty; no regional rule is inferred from one convolution. Optional host fusion/storage/request routes have no demonstrated native performance application and retain their defaults.

## Evidence boundaries

The actual public source map includes host regions, shapes, layouts, numerical seams and invocation counts. Emission identity censuses identify changed programs; focused checks replace their older observations while identical programs retain their evidence. All 94 ordinary physical-program LLVM identities match the final selected compiler; six source verification records and the required refusal are tracked separately. See [emission reuse](../artifacts/final-physical-emission-reuse.json). Model replay after the initial wide-RHS selection passes at source 0246df670a675644: 68 built kernels and 159 exact calls. Final replay after M1-scaled promotion also passes at source e1f1ef81b7c0b1d1, again 68 kernels and 159 exact calls. All 112 core/program emissions remain byte-identical to the tested prior checkpoint; eight scalability emissions change and match passing checked candidates. See [final evidence reuse](../artifacts/final-evidence-reuse.json). Complete-chain timing remains UNKNOWN.

Trial records pin source/library identities, actual pass order/options, before/after command buffers and LLVM, executable/model identities and keep/reject decisions.

Host preparation/dispatch/wait attribution, measured engine occupancy, complete large-model timing, and physical rooflines for UNKNOWN cases remain UNKNOWN. A model question is pending trusted diagnostic review; it grants no calibration dispatch permission. No fitted timing model is claimed.

Detailed evidence is under `artifacts/`, with integration decisions in `ITERATION_NOTES.md` and remaining work in `READY_WORK.md` outside the submission.
