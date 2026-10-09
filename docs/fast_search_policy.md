# CPU simulation and analytical performance search

## Answer and present evidence

Native execution, Spike, GSIM and analytical models can support many more
independent candidate evaluations than a single FPGA. They could drive a compiler
toward the handwritten schedules. We have not demonstrated an independent
automatic campaign reaching that performance, nor a generally calibrated whole-
model predictor. Final requested-stock FireSim evaluation remains necessary.

The evidence already shows why one universal simulator conversion is insufficient:

| Controlled evidence | Observed result | Consequence |
| --- | --- | --- |
| CPU service battery | 45 retired-instruction windows agree across Spike, GSIM and stock | Retirements are useful conserved features; they are not Rocket cycles |
| Held FMA dependency case | Count-only error 91.17%; `retired_instructions + stream_dependent_fp_ops` maximum held stock error 0.259553% | Include ordering and latency dependencies |
| Held memory service cases | Maximum stock error 38.29%; GSIM error 11.60% | Separate host memory/cache regimes and simulator configurations |
| Operational service pilot | One held middle case: compute 1.88%, requested loads 8.98%, i32 readback 6.59% errors | Useful local screening, not whole-model qualification |
| Crossed gather pilot | Initial 2.83% error; broader crossed holdouts 40.76% / 40.87% | Withhold combinations as well as interpolated sizes |
| Quantizer spacing | Equal arithmetic counts; roughly 30–31% stock improvement; held dependency model errors 2.524% / 3.892% | Count equality cannot tie schedules; one held pair is insufficient for broad search |
| Retained command loops | 1.7874% GSIM improvement versus 10.2899% stock improvement | Code/branch/cache effects need configuration-specific prices |
| Complete attention sections | Frozen conditional forecasts near 0.5–0.65% error within one source family | Keep the original domain refusal; close local forecasts do not authorize whole-model export |
| Interval tables | Section improves 6.531%; separately composed whole regresses 0.6878% | Price live working sets, initialization and complete composition |

Sources: [three-engine battery](perf_records/rv64gc_cpu_service_battery_three_engines_qualified.json),
[CPU pilot](perf_records/root_cpu_stream_model_pilot_20261006.json),
[operational pilot](perf_records/root_operational_service_model_pilot_20261006.json),
[crossed gather](perf_records/root_gather_crossed_model_pilot_20261006.json),
[quantizer spacing](perf_records/root_source_host_quant_spacing_stock2063_terminal_review_20261007.json),
[command loops](perf_records/root_resnet_source_stride_stock2083_2084_stock2083_2084_terminal.json),
[blinded section](perf_records/root_smol_complete_group_blinded_retirement_screen_20261007.json),
[floor score](perf_records/root_bounded_floor_stock2064_terminal_review_20261007.json),
[fused readout](perf_records/root_smol_fused_radix_stock2069_terminal_review_20261007.json),
[whole table negative](perf_records/root_tiny_source_interval_terminal_review_20261006.json).

The latest composed Tiny candidate reinforces this requirement: restoring an
omitted late legalizer reduced retirements versus its uncomposed approximate
carrier, but that baseline was not the champion. The complete candidate actually
retires 48.154% more instructions than the champion and costs 573,452,525 stock
cycles versus 378,946,263. Keep this as a negative composition, not a new best. [Diagnosis and scope](tiny_composed_regression.md).

## A parallel search funnel

1. **Static legality and applicability.** Reject resource violations, unsupported
   effects, escaped uses, invalid numeric policies, ISA restrictions and incomplete
   source bindings. Deduplicate identical emitted objects before simulation.
2. **Native complete components.** Check the original functional budget and
   fallback. Measure inclusive preparation/proof/replay/publication costs. Native
   wall time does not stand in for Rocket timing.
3. **Spike.** Run the linked candidate, all-output comparison and ISA/FCSR checks;
   collect exclusive retired-PC roles, instruction dependencies, code/frame/table
   footprints, requests and replay counts. Its cycle counter is a retirement proxy.
4. **Analytical intervals.** Compute resource feasibility and lower bounds; price
   actual features only inside independently qualified calibration regimes. Refuse
   unknown physical traffic, cache, issue or overlap terms. Do not turn UNKNOWN into
   a zero-cost advantage. Preserve tied or overlapping rankings.
5. **GSIM complete components.** Run a small, diverse shortlist and controls using
   the selected RTL engine, positive completion, unchanged outputs and complete
   timers. Respect the shared runtime limiter and explicit cycle/wall budgets.
   Initialization and validation outside the timer still consume simulator work.
6. **FireSim compositions and final freeze.** Resolve simulator disagreement,
   new regimes and important whole-program interactions during development. After
   freezing the compiler, run protected held-out whole models on the requested
   configuration with original output budgets and full executable ISA audit.

CPU worker count must be bounded by memory, engine leases and useful throughput.
A large model can exhaust memory or slow all workers. Parallelize independent
experiments and executable snapshots; never share mutable candidates, target
scratch images or result paths. Cancellation/timeouts leave incomplete evidence,
not a partial performance score.

This can reduce the number of intermediate FPGA runs substantially. A rule that
permits **only** a final FPGA run needs a qualified simulator/configuration and
held-out ranking evidence spanning every relevant mechanism. Current stock/cache/
memory disagreement does not support that rule yet.

## Roofline: a bound and a bottleneck diagnosis

For each resource, use operation counts and data demand from the candidate's
actual semantics and emitted program, plus rates derived from selected hardware:

```
compute_floor = required arithmetic / declared peak arithmetic per cycle
movement_floor = required physical movement / declared peak bytes per cycle
```

Use a dependence graph and declared concurrency to compose bounds. Independent
resources may overlap; serial stages cannot be collapsed into one `max`.
Packing, quantization, observers, certificates, allocation, dispatch, reconstruction,
fallback/replay and drain remain stages in that graph. Avoid counting a measured
inclusive callback again as separate contained costs.

Requested DMA bytes and optimistic unique bytes describe different quantities.
Unique bytes are not physical DDR traffic without an established cache/reuse law.
Padded issue geometry is not necessarily a mandatory compute bound for every
stationary/tail schedule. Short waves need the actual state/D-port legality proof.
Expose every assumption and UNKNOWN rather than printing one unjustified cycle goal.

A useful capsule report gives:

- complete measured cycles and critical-path stage shares;
- useful arithmetic versus executed/padded arithmetic;
- requested, unique and measured physical bytes separately;
- compute/movement/host bounds, fixed setup and observed overlap;
- distance to each bound and the resource responsible;
- calibrated interval, applicable regime and unresolved ranking;
- a concrete next experiment that can distinguish the competing explanations.

An analytical reproduction of Jack's roughly 22M ResNet result would be a useful
external check **after** independently fitting the model. Matching that one label
alone is not validation. Compare the identical code, input representation and timer;
his INT8-input timer and our FP32-input quantization-inclusive timer differ.
Keep that program and label out of independent participant inputs and holdout fits.

## Calibration that permits wider search

Use the [capsule matrix](capsule_optimization_coverage.md) to choose controlled,
workload-independent mechanisms. Vary one relevant decision while keeping work,
outputs and complete cost boundary fixed. Include:

- FP dependency spacing, integer dependency/branch chains and spill/code regimes;
- contiguous/strided/gather memory, cold/warm reuse and capacity transitions;
- command issue/configuration/fences, transfers and exact readout/decoder;
- resident A/B, output blocking, tail order and bank/prefetch interaction;
- table/certificate initialization, locality, replay and source fallback;
- nested and repeated producer/consumer compositions.

Withhold alternative schedules, seeds, shapes and crossed regimes; repeats estimate
variability, while independent families test transfer. Fit and score by selected
target, engine, ABI, numeric policy and complete cost scope. Freeze coefficients
before opening the held label. Report ranking errors as well as absolute error:
an accurate average can still choose the wrong optimization. Promote a ranking
domain only with calibrated uncertainty and independent comparison evidence.

If a new candidate leaves that domain, use the model to choose the next calibration
experiment, not to extrapolate a confident winner. Preserve failed hypotheses;
otherwise the agent repeatedly rediscovers tables, fewer callbacks or nominal MAC
savings that actually lose.

## Existing code and missing integration

OOT `golden_tuning.py` already enumerates legal blocks and reports peak floors,
padded work, requested/unique traffic and command estimates. Those are analytical
features, not calibrated complete cycle predictions. Target RTL/resource/command
models and their simulator adapters belong here.

Merlin already provides `perf/layer_bench/run.py::run_on_gsim`, engine selection and
runtime slots, calibration/feature-calibration/bundle tools, cycle intervals and
the `phase2_analytical_provider`. These generic tools bind input and provider
identities and retain UNKNOWN. Their existence is not proof of a calibrated
whole-host/cache/DMA model or an autonomous qualified CPU portfolio.

The legacy `build_fast_evaluator_installation` factory serves the global model
portfolio and its held-out quality observer; it is not an installed independent
component analytical adapter. Component-only feedback accepts an explicitly
installed `ComponentAnalyticalProvider` callback with exact-member cycle intervals.
The general native/Spike-feature-to-calibrated-component callback is still missing.

There is already bounded parallel machinery: certified GSIM feedback can prefetch
sweep jobs through `DevelopmentGsimFeedback._prefetch_wave` and paired-measurement
schedule fan-out; Phase1 sim jobs also have asynchronous process/lease controls.
Reuse these normal engine contracts. A unified supervised independent native/
Spike/analytical/RTL campaign and qualified fresh launch remain missing. The
legacy whole-portfolio thread pool is a separate workflow.

Still required: approved independent domain inputs and semantic scenario coverage;
matched complete-cost calibration over this matrix; content-bound baseline and
candidate comparison; prediction uncertainty and active experiment selection;
bounded parallel workers with cancellation; fresh author transport/isolation and
protected final evaluation. The separate upstream schedule proxy reports useful
pairwise ranking on its measured set, but was fit on that set and has large
absolute errors; do not count it as independent validation from this journey.

The implementation ownership rule remains: generic workers, evidence, calibration,
ranking, cost composition and host transforms in Merlin; target hardware facts,
device schedules, instruction models and execution support in OOT. Record reusable
setup cost separately and total elapsed cost including setup. A 20x convergence
claim requires a matched authenticated campaign; none has been completed here.
