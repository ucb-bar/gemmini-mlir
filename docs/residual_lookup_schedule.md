# Exact scalar residual lookup schedule

Optional `cpu_lut_schedule="raw_u8_x4"` in the captured residual bundle selects
a scalar RV64GC schedule. CLI `--cpu-lut-schedule raw_u8_x4`; whole-model helper
`--residual-lut-schedule raw_u8_x4`. Default remains `scalar`.

The original65536-byte table is indexed by signed-i8 operands plus128. Rotate
both table axes by128 so raw uint8 bit patterns select identical entries, then
unroll four output elements per iteration. This removes offset arithmetic and
amortizes loop/pointer updates. No RVV, Gemmini FSM, numeric approximation or
new qparams are used. Proof identity remains the original source float32 table;
the schedule and compiled-source hash are recorded separately.

Exhaustive65536input-pair tests compare independently compiled sequential
float32 operations, including round-to-even, with and without ReLU; all output
values and a16-byte guard match. Same-source full exact50ResNet, pooled stem,
complete weight hoist and layout propagation passes native and actual Spike
for every1,000preserved golden bits, rank mismatch0, final no-FSM:

| Schedule | Spike retired instructions |
|---|---:|
| scalar |341,424,822|
| raw byte /unroll4 |316,587,699|

This removes24,837,123instructions (7.27%whole-model). These are functional
Spike counters, not measured FireSim cycles. No hardware speedup claim yet.
Receipt: `docs/perf_records/resnet_exact_cpu_lut_schedule.json`.
